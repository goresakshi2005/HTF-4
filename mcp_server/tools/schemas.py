from typing import Literal
from pydantic import BaseModel, Field


class PipelineContext(BaseModel):
    repository: str
    run_id: int
    commit_sha: str = ""
    logs_excerpt: str = ""
    failing_tests: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)


class DiagnosisResult(BaseModel):
    summary: str
    root_cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_fix: str
    risk_score: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)


class GovernanceDecision(BaseModel):
    decision: Literal["auto_fix", "human_review", "block"]
    rationale: str
    risk_score: float = Field(ge=0.0, le=1.0)
    requires_approval: bool = False


class FileChange(BaseModel):
    path: str
    content: str


class CodeFixProposal(BaseModel):
    title: str
    description: str
    file_changes: list[FileChange]


class PatchGenerationResult(BaseModel):
    patch: str
    rationale: str
    touched_files: list[str] = Field(default_factory=list)


class RepairAttempt(BaseModel):
    attempt: int
    status: Literal["generated", "applied", "failed"]
    message: str


class RepairResult(BaseModel):
    status: Literal["fixed", "requires_review", "failed_after_retries"]
    attempts: list[RepairAttempt] = Field(default_factory=list)
    fix_proposal: CodeFixProposal | None = None

