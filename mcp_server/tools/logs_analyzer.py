from __future__ import annotations

import re


TEST_FAIL_PATTERNS = [
    re.compile(r"FAILED\s+(.+)", re.IGNORECASE),
    re.compile(r"AssertionError[:\s]+(.+)", re.IGNORECASE),
    re.compile(r"error[:\s]+(.+)", re.IGNORECASE),
]


def normalize_logs(raw_logs: str, max_chars: int = 6000) -> str:
    squashed = "\n".join(line.rstrip() for line in raw_logs.splitlines())
    if len(squashed) <= max_chars:
        return squashed
    return squashed[-max_chars:]


def extract_failing_tests(raw_logs: str, max_items: int = 20) -> list[str]:
    failures: list[str] = []
    for line in raw_logs.splitlines():
        for pattern in TEST_FAIL_PATTERNS:
            match = pattern.search(line)
            if match:
                item = match.group(1).strip()
                if item and item not in failures:
                    failures.append(item)
            if len(failures) >= max_items:
                return failures
    return failures


def extract_python_file_candidates(items: list[str]) -> list[str]:
    candidates: list[str] = []
    for item in items:
        normalized = item.replace("\\", "/")
        for token in normalized.replace(":", " ").split():
            token = token.strip(" ,.;:()[]{}'\"")
            if token.endswith(".py") and "/" in token and token not in candidates:
                candidates.append(token)
    return candidates

