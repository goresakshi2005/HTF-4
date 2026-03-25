import pytest

from mcp_server.config import Settings
from mcp_server.tools.fix_generator import (
    apply_patch_to_text,
    build_fix_proposal_from_patch,
    normalize_unified_diff,
    parse_unified_diff,
)
from mcp_server.tools.schemas import PatchGenerationResult


def test_parse_unified_diff_extracts_files():
    patch = """--- a/tests/test_app.py
+++ b/tests/test_app.py
@@ -1,3 +1,3 @@
-assert add(2, 2) == 5
+assert add(2, 2) == 4
 """
    files = parse_unified_diff(patch)
    assert len(files) == 1
    assert files[0].path == "tests/test_app.py"
    assert len(files[0].hunks) == 1


def test_apply_patch_to_text_replaces_line():
    original = "def test_add():\n    assert add(2, 2) == 5\n"
    hunks = [["@@ -1,2 +1,2 @@", " def test_add():", "-    assert add(2, 2) == 5", "+    assert add(2, 2) == 4"]]
    updated = apply_patch_to_text(original, hunks)
    assert "== 4" in updated
    assert "== 5" not in updated


def test_build_fix_proposal_from_patch_applies_changes():
    settings = Settings(OPENAI_API_KEY="x", GITHUB_TOKEN="x")
    patch_result = PatchGenerationResult(
        patch="--- a/tests/test_app.py\n+++ b/tests/test_app.py\n@@ -1,2 +1,2 @@\n def test_add():\n-    assert add(2, 2) == 5\n+    assert add(2, 2) == 4\n",
        rationale="Fix incorrect expected value.",
        touched_files=["tests/test_app.py"],
    )
    files = {"tests/test_app.py": "def test_add():\n    assert add(2, 2) == 5\n"}
    proposal = build_fix_proposal_from_patch(settings, patch_result, files)
    assert len(proposal.file_changes) == 1
    assert proposal.file_changes[0].path == "tests/test_app.py"
    assert "== 4" in proposal.file_changes[0].content


def test_build_fix_proposal_rejects_unknown_file():
    settings = Settings(OPENAI_API_KEY="x", GITHUB_TOKEN="x")
    patch_result = PatchGenerationResult(
        patch="--- a/unknown.py\n+++ b/unknown.py\n@@ -1,1 +1,1 @@\n-print('x')\n+print('y')\n",
        rationale="Try update unknown file.",
        touched_files=["unknown.py"],
    )
    with pytest.raises(ValueError):
        build_fix_proposal_from_patch(settings, patch_result, {"tests/test_app.py": "x\n"})


def test_normalize_unified_diff_removes_code_fences():
    raw = """```diff
--- a/tests/test_app.py
+++ b/tests/test_app.py
@@ -1,1 +1,1 @@
-x
+y
```"""
    norm = normalize_unified_diff(raw)
    assert "```" not in norm
    assert norm.startswith("--- a/tests/test_app.py")


def test_fuzzy_fallback_applies_when_hunk_line_mismatch():
    settings = Settings(OPENAI_API_KEY="x", GITHUB_TOKEN="x")
    patch_result = PatchGenerationResult(
        patch="--- a/tests/test_app.py\n+++ b/tests/test_app.py\n@@ -8,1 +8,1 @@\n-    assert add(2, 2) == 5\n+    assert add(2, 2) == 4\n",
        rationale="Fix assertion",
        touched_files=["tests/test_app.py"],
    )
    # Line numbers intentionally don't match this short file; fuzzy fallback should still succeed.
    files = {"tests/test_app.py": "def test_add():\n    assert add(2, 2) == 5\n"}
    proposal = build_fix_proposal_from_patch(settings, patch_result, files)
    assert len(proposal.file_changes) == 1
    assert "== 4" in proposal.file_changes[0].content

