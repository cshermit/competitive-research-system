"""
src/utils/logger.py
───────────────────
Two-channel logging:
  • Console  — Rich coloured, human-readable
  • File     — JSONL (one JSON object per line), machine-parseable

Every agent transition and tool call flows through here, making the
LangGraph state machine fully visible and debuggable.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ── Console setup ─────────────────────────────────────────────────────────────
_THEME = Theme(
    {
        "transition": "bold cyan",
        "tool_call": "bold yellow",
        "retry": "bold red",
        "success": "bold green",
        "plan": "bold magenta",
    }
)
_console = Console(theme=_THEME, stderr=False, legacy_windows=False, force_terminal=True)


# ── JSON file handler ──────────────────────────────────────────────────────────
class _JSONLHandler(logging.Handler):
    """Appends one JSON object per log record to a .jsonl file."""

    def __init__(self, filepath: Path) -> None:
        super().__init__()
        self._filepath = filepath

    def emit(self, record: logging.LogRecord) -> None:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Forward any extra structured fields attached by our helpers
        for key in ("agent", "event", "data"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val

        try:
            with self._filepath.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except Exception:  # noqa: BLE001 — logging must never crash the app
            self.handleError(record)


# ── Public factory ─────────────────────────────────────────────────────────────
def get_logger(name: str, *, log_dir: Path | None = None, level: str = "INFO") -> logging.Logger:
    """
    Return (and configure once) a named logger with both handlers attached.

    Usage
    -----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello", extra={"agent": "planner", "event": "subtask_created"})
    """
    logger = logging.getLogger(name)

    if logger.handlers:          # already configured — return as-is
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # ── Console handler (Rich) ──────────────────────────────────────────────
    rich_handler = RichHandler(
        console=_console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.DEBUG)
    logger.addHandler(rich_handler)

    # ── JSONL file handler ──────────────────────────────────────────────────
    if log_dir is None:
        log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    jsonl_path = log_dir / f"agent_run_{date_str}.jsonl"

    file_handler = _JSONLHandler(jsonl_path)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger


# ── Structured event helpers ───────────────────────────────────────────────────
class AgentLogger:
    """
    Thin wrapper that attaches structured metadata to every log call.

    Example
    -------
    alog = AgentLogger("planner")
    alog.transition("researcher", reason="subtasks ready")
    alog.tool_call("web_search", query="OpenAI competitors 2025")
    alog.retry(subtask_id="t1", attempt=2, revised_query="...")
    """

    def __init__(self, agent_name: str, log_dir: Path | None = None, level: str = "INFO") -> None:
        self._agent = agent_name
        self._log = get_logger(f"agent.{agent_name}", log_dir=log_dir, level=level)

    # ── Transition ──────────────────────────────────────────────────────────
    def transition(self, to_agent: str, *, reason: str = "") -> None:
        msg = f"[transition]  {self._agent} → {to_agent}"
        if reason:
            msg += f"  ({reason})"
        self._log.info(
            msg,
            extra={"agent": self._agent, "event": "transition", "data": {"to": to_agent, "reason": reason}},
        )

    # ── Tool calls ───────────────────────────────────────────────────────────
    def tool_call(self, tool_name: str, **kwargs: Any) -> None:
        params_str = "  ".join(f"{k}={v!r}" for k, v in kwargs.items())
        self._log.info(
            f"[tool_call]   {tool_name}  {params_str}",
            extra={"agent": self._agent, "event": "tool_call", "data": {"tool": tool_name, **kwargs}},
        )

    def tool_result(self, tool_name: str, *, result_summary: str) -> None:
        self._log.info(
            f"[tool_result] {tool_name} → {result_summary}",
            extra={"agent": self._agent, "event": "tool_result", "data": {"tool": tool_name, "summary": result_summary}},
        )

    # ── Retry ────────────────────────────────────────────────────────────────
    def retry(self, subtask_id: str, *, attempt: int, reason: str, revised_query: str) -> None:
        self._log.warning(
            f"[retry]       subtask={subtask_id}  attempt={attempt}  reason={reason!r}",
            extra={
                "agent": self._agent,
                "event": "retry",
                "data": {"subtask_id": subtask_id, "attempt": attempt, "reason": reason, "revised_query": revised_query},
            },
        )

    # ── Generic helpers ───────────────────────────────────────────────────────
    def info(self, msg: str, **data: Any) -> None:
        self._log.info(msg, extra={"agent": self._agent, "event": "info", "data": data or None})

    def warning(self, msg: str, **data: Any) -> None:
        self._log.warning(msg, extra={"agent": self._agent, "event": "warning", "data": data or None})

    def error(self, msg: str, **data: Any) -> None:
        self._log.error(msg, extra={"agent": self._agent, "event": "error", "data": data or None})

    def debug(self, msg: str, **data: Any) -> None:
        self._log.debug(msg, extra={"agent": self._agent, "event": "debug", "data": data or None})
