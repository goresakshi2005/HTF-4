from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")

    GITHUB_TOKEN: str = Field(default="")
    GITHUB_APP_ID: str = Field(default="")
    GITHUB_APP_PRIVATE_KEY: str = Field(default="")

    RISK_AUTO_FIX_THRESHOLD: float = Field(default=0.30, ge=0.0, le=1.0)
    RISK_HUMAN_REVIEW_THRESHOLD: float = Field(default=0.70, ge=0.0, le=1.0)

    MAX_AUTOFIX_FILES: int = Field(default=5, ge=1)
    MAX_AUTOFIX_HUNKS: int = Field(default=20, ge=1)
    MAX_REPAIR_ATTEMPTS: int = Field(default=3, ge=1, le=10)
    PATCH_STRATEGY: str = Field(default="unified_diff")
    LLM_PATCH_MAX_CHARS: int = Field(default=24000, ge=2000, le=200000)
    INDEXING_ENABLED: bool = Field(default=False)
    INDEXING_REBUILD: bool = Field(default=False)
    INDEXING_TOP_K: int = Field(default=8, ge=1, le=50)
    INDEXING_MAX_QUERY_CHARS: int = Field(default=2000, ge=200, le=100000)
    INDEXING_MAX_QUERY_TOKENS: int = Field(default=120, ge=10, le=2000)
    INDEXING_SKIP_SYMBOL_TOKEN_THRESHOLD: int = Field(default=80, ge=1, le=2000)
    FORCE_AUTOFIX_ALL: bool = Field(default=False)

    ALLOW_AUTOMERGE_LOW_RISK: bool = Field(default=True)
    REQUIRE_APPROVAL_FOR_RELEASE: bool = Field(default=True)
    LOG_LEVEL: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

