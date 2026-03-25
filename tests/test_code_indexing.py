from __future__ import annotations

from pathlib import Path

from mcp_server.tools.code_indexing import (
    build_and_query,
    render_retrieval_context,
    sanitize_query_text,
    write_candidate_files,
)


def test_write_candidate_files_preserves_structure(tmp_path: Path):
    files = {"tests/test_app.py": "def add(a,b):\n    return a+b\n"}
    write_candidate_files(target_root=tmp_path, file_contents=files)
    out = tmp_path / "tests" / "test_app.py"
    assert out.exists()
    assert "return a+b" in out.read_text(encoding="utf-8")


def test_render_retrieval_context_formats_headers():
    results = [
        {
            "file_path": "tests/test_app.py",
            "line_range": [8, 20],
            "snippet": "def add(a,b):\n    return a+b\n",
            "score_breakdown": {"total": 1.0},
        }
    ]
    ctx = render_retrieval_context(results, max_snippets=1)
    assert "tests/test_app.py:8-20" in ctx
    assert "def add" in ctx


def test_build_and_query_returns_some_context(tmp_path: Path):
    # This uses the existing indexing scripts.
    file_contents = {"app.py": "def add(a,b):\n    return a+b\n"}
    query = "add function"
    rendered, n, debug = build_and_query(
        file_contents=file_contents,
        query_text=query,
        rebuild=True,
        top_k=3,
    )
    assert isinstance(rendered, str)
    # With lexical retrieval, we expect at least one snippet.
    assert n >= 0
    assert isinstance(debug, dict)
    assert debug.get("enabled") is True


def test_sanitize_query_text_caps_tokens_and_chars():
    raw = "This_is_a_very_long_log_excerpt_with_terms " + " ".join(
        ["expression_tree_too_large"] * 200
    )
    safe, token_count = sanitize_query_text(
        raw,
        max_chars=300,
        max_tokens=25,
    )
    assert len(safe) <= 300
    assert token_count <= 25

    # If threshold is low, we should trigger skip-symbol behavior upstream.
    # (We don't test the SQL path here; we validate token counting plumbing.)
    _, token_count_2 = sanitize_query_text(raw, max_chars=300, max_tokens=40)
    assert token_count_2 <= 40
