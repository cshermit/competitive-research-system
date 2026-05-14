"""
src/graph/state.py
──────────────────
Defines the shared state that flows through every node in the LangGraph
state machine.  TypedDicts give us:
  • Full type-safety and IDE autocomplete
  • LangGraph-compatible schema (no extra deps)
  • Annotated fields with reducer functions for safe list accumulation

State lifecycle
───────────────
START → plan → research → validate → research (retry / next)
                                    ↘ synthesize → END
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Optional
from typing import List  # keep explicit for Python 3.10 compat

from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# Sub-types (plain TypedDicts — used inside ResearchState)
# ─────────────────────────────────────────────────────────────────────────────

class SubTask(TypedDict):
    """A single research question carved out of the original query."""
    id: str                         # e.g. "t1", "t2" …
    query: str                      # current (possibly revised) search query
    original_query: str             # never mutated — for audit trail
    rationale: str                  # why the Planner created this subtask
    status: str                     # pending | researching | retrying | completed | failed
    retry_count: int                # how many times we've retried this subtask
    revised_query: Optional[str]    # Planner's revised query on retry


class ResearchNote(TypedDict):
    """Evidence collected by the Researcher for one SubTask."""
    subtask_id: str
    query: str
    content: str                    # synthesised text from search + fetched URLs
    sources: List[str]              # list of URLs
    result_count: int               # raw number of search hits
    timestamp: str


class AgentTransition(TypedDict):
    """Immutable audit record of one node-to-node hop."""
    from_agent: str
    to_agent: str
    reason: str
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# Master state — everything that flows between nodes
# ─────────────────────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    original_query: str

    # ── Planner outputs ───────────────────────────────────────────────────────
    subtasks: List[SubTask]
    current_subtask_index: int      # pointer into subtasks list
    max_retries_per_subtask: int

    # ── Researcher outputs (Annotated → lists are APPENDED, not replaced) ────
    research_notes: Annotated[List[ResearchNote], operator.add]

    # ── Audit trail (Annotated → always appended) ─────────────────────────────
    transitions: Annotated[List[AgentTransition], operator.add]

    # ── Synthesizer output ────────────────────────────────────────────────────
    final_report: Optional[str]

    # ── Control flow ──────────────────────────────────────────────────────────
    # Set by validate_node so the conditional edge can route without
    # re-inspecting the whole state.
    next_action: str                # "research" | "synthesize"

    # ── Health ────────────────────────────────────────────────────────────────
    status: str                     # planning | researching | validating | synthesizing | complete | failed
    error: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def make_transition(from_agent: str, to_agent: str, reason: str = "") -> AgentTransition:
    return AgentTransition(
        from_agent=from_agent,
        to_agent=to_agent,
        reason=reason,
        timestamp=now_iso(),
    )


def initial_state(query: str, max_retries: int = 2) -> ResearchState:
    """Return a clean starting state for a new research run."""
    return ResearchState(
        original_query=query,
        subtasks=[],
        current_subtask_index=0,
        max_retries_per_subtask=max_retries,
        research_notes=[],
        transitions=[],
        final_report=None,
        next_action="research",
        status="planning",
        error=None,
    )
