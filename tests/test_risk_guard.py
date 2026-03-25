from mcp_server.config import Settings
from mcp_server.tools.risk_guard import decide_governance
from mcp_server.tools.schemas import DiagnosisResult


def _settings() -> Settings:
    return Settings(
        OPENAI_API_KEY="x",
        GITHUB_TOKEN="x",
        FORCE_AUTOFIX_ALL=False,
    )


def test_low_risk_autofix():
    diagnosis = DiagnosisResult(
        summary="unit test failure",
        root_cause="minor typo",
        confidence=0.9,
        proposed_fix="update tests/test_file.py assertion",
        risk_score=0.2,
        reason_codes=["TEST_FAILURE"],
    )
    decision = decide_governance(_settings(), diagnosis, ["tests/test_file.py"])
    assert decision.decision == "auto_fix"


def test_high_risk_block():
    diagnosis = DiagnosisResult(
        summary="workflow failure",
        root_cause="deploy permission missing",
        confidence=0.7,
        proposed_fix="edit .github/workflows/deploy.yml",
        risk_score=0.4,
        reason_codes=["WORKFLOW_CHANGE"],
    )
    decision = decide_governance(_settings(), diagnosis, [".github/workflows/deploy.yml"])
    assert decision.decision in {"human_review", "block"}


def test_force_autofix_bypasses_thresholds():
    settings = _settings()
    settings.FORCE_AUTOFIX_ALL = True
    diagnosis = DiagnosisResult(
        summary="workflow risk",
        root_cause="sensitive workflow modification",
        confidence=0.7,
        proposed_fix="edit .github/workflows/deploy.yml",
        risk_score=0.99,
        reason_codes=["HIGH_RISK_CHANGE"],
    )
    decision = decide_governance(settings, diagnosis, [".github/workflows/deploy.yml"])
    assert decision.decision == "auto_fix"

