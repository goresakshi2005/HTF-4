from __future__ import annotations

from mcp_server.config import Settings
from mcp_server.tools.schemas import DiagnosisResult, GovernanceDecision


HIGH_RISK_KEYWORDS = [
    "terraform",
    "kubernetes",
    "helm",
    "secrets",
    ".github/workflows",
    "auth",
    "permission",
]


def decide_governance(
    settings: Settings,
    diagnosis: DiagnosisResult,
    changed_files: list[str],
) -> GovernanceDecision:
    if settings.FORCE_AUTOFIX_ALL:
        return GovernanceDecision(
            decision="auto_fix",
            rationale="FORCE_AUTOFIX_ALL is enabled; bypassing governance thresholds.",
            risk_score=diagnosis.risk_score,
            requires_approval=False,
        )

    risky_change = any(
        keyword in path.lower()
        for path in changed_files
        for keyword in HIGH_RISK_KEYWORDS
    )
    score = diagnosis.risk_score
    if risky_change:
        score = max(score, settings.RISK_HUMAN_REVIEW_THRESHOLD)

    if score < settings.RISK_AUTO_FIX_THRESHOLD:
        return GovernanceDecision(
            decision="auto_fix",
            rationale="Risk score below autonomous fix threshold.",
            risk_score=score,
            requires_approval=False,
        )

    if score < settings.RISK_HUMAN_REVIEW_THRESHOLD:
        return GovernanceDecision(
            decision="human_review",
            rationale="Moderate risk requires operator approval.",
            risk_score=score,
            requires_approval=True,
        )

    return GovernanceDecision(
        decision="block",
        rationale="High risk remediation path blocked for autonomous action.",
        risk_score=score,
        requires_approval=True,
    )

