from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from mcp_server.tools.github_client import GitHubClient
from mcp_server.tools.schemas import CodeFixProposal


async def open_autofix_pr(
    github: GitHubClient,
    repository: str,
    base_sha: str,
    base_branch: str,
    fix_proposal: CodeFixProposal,
    notes_path: str = "AUTOFIX_NOTES.md",
    attempt_logs: list[dict[str, Any]] | None = None,
    indexing_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    branch_name = f"agentic/autofix-{base_sha[:7]}"
    try:
        await github.create_branch(repository, branch_name, base_sha)
    except httpx.HTTPStatusError as exc:
        # Common case: rerunning same SHA where branch already exists.
        if exc.response.status_code != 422:
            raise
        unique_suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        branch_name = f"{branch_name}-{unique_suffix}"
        await github.create_branch(repository, branch_name, base_sha)

    await github.create_or_update_file(
        repository=repository,
        path=notes_path,
        branch=branch_name,
        content=_build_notes_content(fix_proposal, attempt_logs or [], indexing_debug=indexing_debug),
        message=fix_proposal.title,
    )

    for fc in fix_proposal.file_changes:
        await github.create_or_update_file(
            repository=repository,
            path=fc.path,
            branch=branch_name,
            content=fc.content,
            message=fix_proposal.title,
        )

    return await github.create_pull_request(
        repository=repository,
        title=fix_proposal.title,
        body=fix_proposal.description,
        head=branch_name,
        base=base_branch,
    )


def _build_notes_content(
    fix_proposal: CodeFixProposal,
    attempt_logs: list[dict[str, Any]],
    *,
    indexing_debug: dict[str, Any] | None = None,
) -> str:
    attempts = "\n".join(
        f"- attempt {a.get('attempt')}: {a.get('status')} - {a.get('message')}"
        for a in attempt_logs
    )
    if not fix_proposal.file_changes:
        return f"# Autofix Notes\n\n{fix_proposal.description}\n\n## Attempts\n{attempts}\n"
    changed = "\n".join(f"- {fc.path}" for fc in fix_proposal.file_changes)

    indexing_lines: list[str] = []
    if indexing_debug and indexing_debug.get("enabled"):
        indexing_lines = [
            "## Indexing",
            f"- token_count: {indexing_debug.get('token_count')}",
            f"- max_query_tokens: {indexing_debug.get('max_query_tokens')}",
            f"- skip_symbol_decision: {indexing_debug.get('skip_symbol_decision')}",
            f"- fallback_used: {indexing_debug.get('fallback_used')}",
            f"- final_skip_symbol: {indexing_debug.get('final_skip_symbol')}",
            f"- results_count: {indexing_debug.get('results_count')}",
        ]

    indexing_block = ("\n".join(indexing_lines) + "\n\n" if indexing_lines else "")
    return (
        "# Autofix Notes\n\n"
        f"## Summary\n{fix_proposal.description}\n\n"
        f"## Attempts\n{attempts}\n\n"
        + indexing_block
        + "## Files updated\n"
        + f"{changed}\n"
    )

