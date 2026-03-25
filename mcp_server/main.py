from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from mcp_server.config import get_settings
from mcp_server.tools.diagnose import diagnose_failure, generate_patch_with_llm
from mcp_server.tools.fix_generator import build_fix_proposal_from_patch
from mcp_server.tools.github_client import GitHubClient
from mcp_server.tools.code_indexing import build_and_query
from mcp_server.tools.logs_analyzer import (
    extract_failing_tests,
    extract_python_file_candidates,
    normalize_logs,
)
from mcp_server.tools.pr_bot import open_autofix_pr
from mcp_server.tools.risk_guard import decide_governance
from mcp_server.tools.schemas import DiagnosisResult, PipelineContext, RepairAttempt


mcp = FastMCP("agentic-cicd-orchestrator")


@mcp.tool()
async def inspect_pipeline_failure(repository: str, run_id: int) -> dict[str, Any]:
    """Collect logs, failing tests, commit context, then produce diagnosis."""
    settings = get_settings()
    github = GitHubClient(settings.GITHUB_TOKEN)

    run = await github.get_workflow_run(repository, run_id)
    commit_sha = run.get("head_sha", "")
    jobs = await github.list_run_jobs(repository, run_id)
    changed_files = await github.get_commit_files(repository, commit_sha)
    logs_text = await github.get_run_logs_text(repository, run_id)
    if not logs_text:
        logs_text = f"No logs downloaded for run {run_id}."
    failed_jobs = [job.get("name", "unknown") for job in jobs if job.get("conclusion") == "failure"]
    if failed_jobs:
        logs_text = f"Failed jobs: {failed_jobs}\n\n{logs_text}"
    logs_excerpt = normalize_logs(logs_text)
    failing_tests = extract_failing_tests(logs_excerpt)

    context = PipelineContext(
        repository=repository,
        run_id=run_id,
        commit_sha=commit_sha,
        logs_excerpt=logs_excerpt,
        failing_tests=failing_tests,
        changed_files=changed_files,
    )
    diagnosis = await diagnose_failure(settings, context)
    decision = decide_governance(settings, diagnosis, changed_files)

    return {
        "context": context.model_dump(),
        "diagnosis": diagnosis.model_dump(),
        "governance": decision.model_dump(),
    }


@mcp.tool()
async def orchestrate_autofix(
    repository: str,
    run_id: int,
    base_branch: str = "main",
) -> dict[str, Any]:
    """Attempt low-risk autonomous fix and open PR."""
    analysis = await inspect_pipeline_failure(repository, run_id)
    governance = analysis["governance"]
    if governance["decision"] != "auto_fix":
        return {
            "status": "requires_review",
            "reason": governance["rationale"],
            "analysis": analysis,
        }

    settings = get_settings()
    if settings.PATCH_STRATEGY != "unified_diff":
        return {
            "status": "requires_review",
            "reason": f"Unsupported PATCH_STRATEGY={settings.PATCH_STRATEGY}. Expected unified_diff.",
            "analysis": analysis,
        }
    github = GitHubClient(settings.GITHUB_TOKEN)

    context = PipelineContext.model_validate(analysis["context"])
    diagnosis_model = DiagnosisResult.model_validate(analysis["diagnosis"])

    # Fetch candidate file contents at the failing commit SHA.
    candidates = [p for p in context.changed_files if p.endswith(".py")]
    candidates.extend(extract_python_file_candidates(context.failing_tests))
    if ("no module named 'app'" in context.logs_excerpt.lower()) and ("app.py" not in candidates):
        candidates.append("app.py")
    if "tests/test_app.py" not in candidates:
        candidates.append("tests/test_app.py")

    prioritized: list[str] = []
    for p in ["app.py", "tests/test_app.py", "test_app.py"]:
        if p in candidates and p not in prioritized:
            prioritized.append(p)
    for p in candidates:
        if p not in prioritized:
            prioritized.append(p)
    candidates = prioritized

    file_contents: dict[str, str] = {}
    for path in candidates[: settings.MAX_AUTOFIX_FILES + 2]:
        content = await github.get_file_content(repository=repository, path=path, ref=context.commit_sha or "main")
        if content:
            file_contents[path] = content

    retrieval_context = "Indexing disabled."
    retrieval_snippet_count = 0
    if settings.INDEXING_ENABLED:
        query_text = (
            f"Failure summary: {diagnosis_model.summary}\n"
            f"Root cause: {diagnosis_model.root_cause}\n"
            f"Failing tests: {context.failing_tests}\n"
            f"Logs excerpt:\n{context.logs_excerpt[:4000]}"
        )
        retrieval_context, retrieval_snippet_count, indexing_debug = build_and_query(
            file_contents=file_contents,
            query_text=query_text,
            rebuild=settings.INDEXING_REBUILD,
            top_k=settings.INDEXING_TOP_K,
            max_query_chars=settings.INDEXING_MAX_QUERY_CHARS,
            max_query_tokens=settings.INDEXING_MAX_QUERY_TOKENS,
            skip_symbol_token_threshold=settings.INDEXING_SKIP_SYMBOL_TOKEN_THRESHOLD,
        )
    else:
        indexing_debug = {"enabled": False}

    attempt_logs: list[RepairAttempt] = []
    feedback = f"Indexing enabled={settings.INDEXING_ENABLED}; retrieved_snippets={retrieval_snippet_count}."
    fix_proposal = None
    for attempt in range(1, settings.MAX_REPAIR_ATTEMPTS + 1):
        try:
            patch_result = await generate_patch_with_llm(
                settings=settings,
                context=context,
                diagnosis=diagnosis_model,
                file_contents=file_contents,
                retrieval_context=retrieval_context,
                previous_attempt_feedback=feedback,
            )
            attempt_logs.append(
                RepairAttempt(
                    attempt=attempt,
                    status="generated",
                    message=f"Generated patch for {len(patch_result.touched_files)} files.",
                )
            )
            fix_proposal = build_fix_proposal_from_patch(settings, patch_result, file_contents)
            if fix_proposal.file_changes:
                attempt_logs.append(
                    RepairAttempt(
                        attempt=attempt,
                        status="applied",
                        message=f"Prepared {len(fix_proposal.file_changes)} file changes.",
                    )
                )
                break
            feedback = "Patch produced no effective file changes. Provide a different unified diff."
            attempt_logs.append(
                RepairAttempt(
                    attempt=attempt,
                    status="failed",
                    message=feedback,
                )
            )
        except Exception as exc:
            feedback = f"Patch application failed: {exc}"
            attempt_logs.append(
                RepairAttempt(
                    attempt=attempt,
                    status="failed",
                    message=feedback,
                )
            )

    if not fix_proposal or not fix_proposal.file_changes:
        return {
            "status": "failed_after_retries",
            "reason": "Could not generate/apply a valid patch within retry limit.",
            "analysis": analysis,
            "attempts": [a.model_dump() for a in attempt_logs],
        }

    pr = await open_autofix_pr(
        github=github,
        repository=repository,
        base_sha=context.commit_sha,
        base_branch=base_branch,
        fix_proposal=fix_proposal,
        notes_path=f".agentic/notes/autofix-{context.run_id}.md",
        attempt_logs=[a.model_dump() for a in attempt_logs],
        indexing_debug=indexing_debug,
    )
    return {
        "status": "fixed",
        "pull_request": pr,
        "analysis": analysis,
        "indexing_debug": indexing_debug,
        "attempts": [a.model_dump() for a in attempt_logs],
    }


if __name__ == "__main__":
    mcp.run()

