"""
src/utils/config.py
───────────────────
Centralised configuration loaded once from environment variables.
All agents import `settings` from here — no scattered os.getenv calls.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator

# Prefer pydantic-settings; fall back to plain pydantic BaseModel if absent.
try:
    from pydantic_settings import BaseSettings as _Base
except ImportError:
    from pydantic import BaseModel as _Base  # type: ignore


load_dotenv()


class Settings(_Base):
    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="claude-sonnet-4-20250514", alias="LLM_MODEL")

    # ── Search ────────────────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    max_search_results: int = Field(default=5, alias="MAX_SEARCH_RESULTS")
    max_url_content_chars: int = Field(default=4000, alias="MAX_URL_CONTENT_CHARS")

    # ── Agent Tuning ─────────────────────────────────────────────────────────
    max_subtasks: int = Field(default=4, alias="MAX_SUBTASKS")
    max_retries_per_subtask: int = Field(default=2, alias="MAX_RETRIES_PER_SUBTASK")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: Path = Field(default=Path("./logs"), alias="LOG_DIR")

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("log_dir", mode="before")
    @classmethod
    def _ensure_log_dir(cls, v) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_required_keys(self) -> None:
        """Raise early with clear messages if critical keys are missing."""
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example → .env and fill in your keys."
            )


# Singleton — imported everywhere as `from src.utils.config import settings`
settings = Settings()
