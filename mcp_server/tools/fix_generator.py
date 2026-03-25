from __future__ import annotations

import re
from dataclasses import dataclass

from mcp_server.config import Settings
from mcp_server.tools.schemas import CodeFixProposal, FileChange, PatchGenerationResult


HUNK_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")


@dataclass
class ParsedPatchFile:
    path: str
    hunks: list[list[str]]


def build_fix_proposal_from_patch(
    settings: Settings,
    patch_result: PatchGenerationResult,
    file_contents: dict[str, str],
) -> CodeFixProposal:
    parsed_files = parse_unified_diff(normalize_unified_diff(patch_result.patch))
    if not parsed_files:
        raise ValueError("No valid file patch blocks found in LLM patch output.")
    if len(parsed_files) > settings.MAX_AUTOFIX_FILES:
        raise ValueError("Patch touches too many files.")
    file_changes: list[FileChange] = []
    total_hunks = 0
    for parsed in parsed_files:
        if parsed.path not in file_contents:
            raise ValueError(f"Patch references unavailable file: {parsed.path}")
        total_hunks += len(parsed.hunks)
        if total_hunks > settings.MAX_AUTOFIX_HUNKS:
            raise ValueError("Patch has too many hunks.")
        try:
            updated = apply_patch_to_text(file_contents[parsed.path], parsed.hunks)
        except ValueError:
            updated = apply_patch_with_fuzzy_match(file_contents[parsed.path], parsed.hunks)
        if updated != file_contents[parsed.path]:
            file_changes.append(FileChange(path=parsed.path, content=updated))

    return CodeFixProposal(
        title="ci: llm auto-repair patch",
        description=patch_result.rationale,
        file_changes=file_changes,
    )


def parse_unified_diff(patch_text: str) -> list[ParsedPatchFile]:
    lines = patch_text.splitlines()
    files: list[ParsedPatchFile] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- a/") and i + 1 < len(lines) and lines[i + 1].startswith("+++ b/"):
            path = lines[i + 1][6:].strip()
            i += 2
            hunks: list[list[str]] = []
            current_hunk: list[str] = []
            while i < len(lines) and not lines[i].startswith("--- a/"):
                if lines[i].startswith("@@ "):
                    if current_hunk:
                        hunks.append(current_hunk)
                    current_hunk = [lines[i]]
                elif current_hunk:
                    current_hunk.append(lines[i])
                i += 1
            if current_hunk:
                hunks.append(current_hunk)
            files.append(ParsedPatchFile(path=path, hunks=hunks))
            continue
        i += 1
    return files


def normalize_unified_diff(patch_text: str) -> str:
    # Remove markdown code fences and trim noisy wrappers.
    text = patch_text.strip()
    if text.startswith("```"):
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("```"))
    # Normalize some malformed hunk headers often emitted by models.
    fixed_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("@@") and not line.endswith("@@"):
            # keep only canonical header prefix until closing @@ if present
            if "@@" in line[2:]:
                idx = line[2:].find("@@")
                line = line[: idx + 4]
        fixed_lines.append(line.rstrip("\r"))
    return "\n".join(fixed_lines)


def apply_patch_to_text(original: str, hunks: list[list[str]]) -> str:
    lines = original.splitlines()
    offset = 0
    for hunk in hunks:
        header = hunk[0]
        match = HUNK_RE.match(header)
        if not match:
            raise ValueError(f"Invalid hunk header: {header}")
        old_start = int(match.group("old_start"))
        idx = max(0, old_start - 1 + offset)
        for line in hunk[1:]:
            if not line:
                marker = " "
                content = ""
            else:
                marker = line[0]
                content = line[1:]
            if marker == " ":
                if idx >= len(lines) or lines[idx] != content:
                    raise ValueError("Context mismatch while applying patch.")
                idx += 1
            elif marker == "-":
                if idx >= len(lines) or lines[idx] != content:
                    raise ValueError("Delete mismatch while applying patch.")
                lines.pop(idx)
                offset -= 1
            elif marker == "+":
                lines.insert(idx, content)
                idx += 1
                offset += 1
    return "\n".join(lines) + ("\n" if original.endswith("\n") else "")


def apply_patch_with_fuzzy_match(original: str, hunks: list[list[str]]) -> str:
    lines = original.splitlines()
    for hunk in hunks:
        minus_lines = [l[1:] for l in hunk[1:] if l.startswith("-")]
        plus_lines = [l[1:] for l in hunk[1:] if l.startswith("+")]
        if not minus_lines and plus_lines:
            # Pure insertion fallback: append near EOF.
            lines.extend(plus_lines)
            continue
        start_idx = _find_subsequence(lines, minus_lines)
        if start_idx == -1:
            raise ValueError("Context mismatch while applying patch.")
        end_idx = start_idx + len(minus_lines)
        lines = lines[:start_idx] + plus_lines + lines[end_idx:]
    return "\n".join(lines) + ("\n" if original.endswith("\n") else "")


def _find_subsequence(lines: list[str], pattern: list[str]) -> int:
    if not pattern:
        return -1
    max_start = len(lines) - len(pattern)
    for i in range(max_start + 1):
        if lines[i : i + len(pattern)] == pattern:
            return i
    return -1

