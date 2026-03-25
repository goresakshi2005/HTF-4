from mcp_server.tools.logs_analyzer import extract_failing_tests, normalize_logs


def test_extract_failing_tests():
    logs = """
    test_a.py::test_x FAILED
    AssertionError: expected 1 got 0
    """
    failures = extract_failing_tests(logs)
    assert failures


def test_normalize_logs_limits_chars():
    raw = "a" * 10000
    normalized = normalize_logs(raw, max_chars=100)
    assert len(normalized) == 100

