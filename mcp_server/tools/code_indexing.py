from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def write_candidate_files(target_root: Path, file_contents: dict[str, str]) -> None:
    """
    Write retrieved candidate files into a temp folder preserving directories.

    Keys in file_contents are expected to be repo-like paths: e.g.
    "tests/test_app.py" or "app.py".
    """
    for rel_path, content in file_contents.items():
        safe_rel = rel_path.replace("\\", "/").lstrip("/")
        out_path = target_root / safe_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")


def _run_python_script(script_path: Path, args: list[str], cwd: Path) -> tuple[int, str]:
    # Use the current interpreter to ensure consistent dependencies.
    cmd = [sys.executable, str(script_path), *args]
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def _split_identifier_for_indexing(token: str) -> list[str]:
    """
    Approximate the tokenizer used by `indexing/index_schema.py` so our token caps
    reliably bound the SQL complexity in `symbol_retrieval()`.
    """
    # 1) Split CamelCase / digits (matches index_schema.split_identifier idea).
    pieces = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", token)
    # 2) Split snake_case and non-word separators.
    pieces.extend(re.split(r"[_\\W]+", token))
    out: list[str] = []
    for p in pieces:
        lowered = (p or "").strip().lower()
        if len(lowered) >= 2:
            out.append(lowered)
    return out


def sanitize_query_text(
    query_text: str,
    *,
    max_chars: int,
    max_tokens: int,
) -> tuple[str, int]:
    """
    Produce a capped query string made only of identifier-like tokens.

    This keeps the downstream `indexing/query_code.py` tokenization bounded,
    preventing `sqlite3.OperationalError: Expression tree is too large`.
    """
    # Extract "identifier-ish" tokens first, then apply the same sub-splitting.
    raw_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query_text or "")
    all_tokens: list[str] = []
    for tok in raw_tokens:
        all_tokens.extend(_split_identifier_for_indexing(tok))

    if not all_tokens:
        safe = (query_text or "").strip()[:max_chars]
        return safe, 0

    capped_tokens = all_tokens[:max_tokens]
    safe = " ".join(capped_tokens)
    if len(safe) > max_chars:
        safe = safe[:max_chars]
    return safe, len(capped_tokens)


def build_index(temp_root: Path, rebuild: bool) -> Path:
    """
    Build the hybrid local code index by calling:
    - indexing/index_code.py

    Returns the index directory path used by the indexer.
    """
    index_dir = temp_root / ".code_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[2] / "indexing" / "index_code.py"

    # NOTE: index_code.py uses ensure_index_paths(...), defaulting index_dir to <folder>/.code_index
    args = [
        "--folder",
        str(temp_root),
        "--index-dir",
        str(index_dir),
    ]
    if rebuild:
        args.append("--rebuild")

    rc, output = _run_python_script(script, args, cwd=Path(__file__).resolve().parents[2])
    if rc != 0:
        raise RuntimeError(f"Index build failed (rc={rc}). Output:\n{output}")

    return index_dir


def query_index(
    index_dir: Path,
    temp_root: Path,
    query_text: str,
    top_k: int,
    *,
    skip_symbol: bool = False,
) -> list[dict[str, Any]]:
    """
    Query the local index by calling:
    - indexing/query_code.py --json
    """
    script = Path(__file__).resolve().parents[2] / "indexing" / "query_code.py"
    args = [
        "--folder",
        str(temp_root),
        "--index-dir",
        str(index_dir),
        "--query",
        query_text,
        "--top-k",
        str(top_k),
        "--json",
    ]
    if skip_symbol:
        args.append("--skip-symbol")
    rc, output = _run_python_script(script, args, cwd=Path(__file__).resolve().parents[2])
    if rc != 0:
        raise RuntimeError(f"Index query failed (rc={rc}). Output:\n{output}")

    parsed = json.loads(output)
    return parsed.get("results", [])


def render_retrieval_context(results: list[dict[str, Any]], max_snippets: int = 8) -> str:
    if not results:
        return "No indexed context retrieved."

    lines: list[str] = []
    count = 0
    for r in results:
        if count >= max_snippets:
            break
        file_path = r.get("file_path") or r.get("file_rel_path") or "<unknown>"
        line_range = r.get("line_range") or r.get("lineRange") or [0, 0]
        snippet = r.get("snippet") or ""

        lines.append(f"### {file_path}:{line_range[0]}-{line_range[1]}")
        lines.append(snippet.rstrip())
        lines.append("")
        count += 1
    return "\n".join(lines).strip()


def build_and_query(
    file_contents: dict[str, str],
    query_text: str,
    rebuild: bool,
    top_k: int,
    *,
    max_query_chars: int = 2000,
    max_query_tokens: int = 120,
    skip_symbol_token_threshold: int = 80,
) -> tuple[str, int, dict[str, Any]]:
    """
    Convenience wrapper:
    - create temp folder
    - write candidate files
    - build index
    - query index

    Returns:
      (rendered_context, number_of_results, debug_info)
    """
    if not file_contents:
        return ("No indexed context retrieved.", 0, {"enabled": False, "results_count": 0})

    with tempfile.TemporaryDirectory(prefix="agentic-index-") as tmp:
        temp_root = Path(tmp)
        write_candidate_files(target_root=temp_root, file_contents=file_contents)
        index_dir = build_index(temp_root=temp_root, rebuild=rebuild)

        safe_query, token_count = sanitize_query_text(
            query_text,
            max_chars=max_query_chars,
            max_tokens=max_query_tokens,
        )
        skip_symbol = token_count > skip_symbol_token_threshold
        fallback_used = False
        final_skip_symbol = skip_symbol

        # Primary attempt uses capped tokens; if SQLite still complains, fall back to skip-symbol.
        try:
            results = query_index(
                index_dir=index_dir,
                temp_root=temp_root,
                query_text=safe_query,
                top_k=top_k,
                skip_symbol=skip_symbol,
            )
        except RuntimeError:
            if not skip_symbol:
                fallback_used = True
                results = query_index(
                    index_dir=index_dir,
                    temp_root=temp_root,
                    query_text=safe_query,
                    top_k=top_k,
                    skip_symbol=True,
                )
                final_skip_symbol = True
            else:
                raise
        rendered = render_retrieval_context(results=results)
        debug = {
            "enabled": True,
            "token_count": token_count,
            "max_query_tokens": max_query_tokens,
            "max_query_chars": max_query_chars,
            "skip_symbol_token_threshold": skip_symbol_token_threshold,
            "skip_symbol_decision": skip_symbol,
            "fallback_used": fallback_used,
            "final_skip_symbol": final_skip_symbol,
            "results_count": len(results),
        }
        return rendered, len(results), debug

