"""
src/agents/planner.py
─────────────────────
The Planner is the "control plane" of the multi-agent system.
It has two responsibilities:

  1. plan()   — decompose the original query into focused subtasks
  2. validate() — inspect Researcher output; revise query on empty results

All retry logic lives here; the Planner decides whether to ask the
Researcher to try again, move on, or hand off to the Synthesizer.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.graph.state import ResearchState, SubTask, make_transition, now_iso
from src.utils.config import settings
from src.utils.logger import AgentLogger

_log = AgentLogger("planner", level=settings.log_level)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas for structured LLM output
# ─────────────────────────────────────────────────────────────────────────────

class SubTaskSchema(BaseModel):
    id: str = Field(description="Short unique ID like t1, t2 …")
    query: str = Field(description="A focused web-search query (≤ 15 words)")
    rationale: str = Field(description="Why this subtask is needed")


class PlanSchema(BaseModel):
    subtasks: List[SubTaskSchema] = Field(
        description="Ordered list of research subtasks (2–5 items)"
    )
    overall_approach: str = Field(description="One-sentence research strategy")


class RevisionSchema(BaseModel):
    revised_query: str = Field(description="Improved web-search query")
    reasoning: str = Field(description="Why this revision should yield better results")


# ─────────────────────────────────────────────────────────────────────────────
# Planner node
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    Stateless agent — receives state dict, returns partial state update.
    """

    def __init__(self) -> None:
        self._llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.3,
        )
        self._plan_llm = self._llm.with_structured_output(PlanSchema)
        self._revise_llm = self._llm.with_structured_output(RevisionSchema)

    # ── Public node entry-points ─────────────────────────────────────────────

    def plan_node(self, state: ResearchState) -> dict:
        """
        LangGraph node: decompose the original query into SubTasks.
        Returns a partial state dict that LangGraph merges into the full state.
        """
        query = state["original_query"]
        max_tasks = settings.max_subtasks

        _log.info(f"Planning research for: {query!r}")
        _log.transition("researcher", reason="subtasks ready")

        prompt = (
            f"You are a competitive intelligence research planner.\n"
            f"Original research request: {query}\n\n"
            f"Break this into {max_tasks} or fewer focused subtasks. "
            f"Each subtask should be a precise web-search query that, combined with others, "
            f"will produce a comprehensive competitive briefing.\n"
            f"Avoid overlapping queries."
        )

        try:
            plan: PlanSchema = self._plan_llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            _log.error(f"Planner LLM failed: {exc}")
            # Fallback: treat the original query as a single subtask
            plan = PlanSchema(
                subtasks=[SubTaskSchema(id="t1", query=query, rationale="fallback — planner LLM unavailable")],
                overall_approach="Direct search on original query.",
            )

        subtasks: List[SubTask] = [
            SubTask(
                id=s.id,
                query=s.query,
                original_query=s.query,
                rationale=s.rationale,
                status="pending",
                retry_count=0,
                revised_query=None,
            )
            for s in plan.subtasks
        ]

        _log.info(
            f"Plan created: {len(subtasks)} subtasks  |  strategy: {plan.overall_approach}",
        )
        for st in subtasks:
            _log.debug(f"  [{st['id']}] {st['query']}")

        return {
            "subtasks": subtasks,
            "current_subtask_index": 0,
            "status": "researching",
            "transitions": [make_transition("planner", "researcher", reason="subtasks ready")],
        }

    def validate_node(self, state: ResearchState) -> dict:
        """
        LangGraph node: inspect what the Researcher returned for the current
        subtask and decide what happens next.

        Decision tree
        ─────────────
        Research sufficient  → mark completed, advance index
          ├─ more subtasks  → next_action = "research"
          └─ all done       → next_action = "synthesize"
        Research empty/thin
          ├─ retries left   → revise query, keep index, next_action = "research"
          └─ max retries    → mark failed, advance index, next_action = "research/synthesize"
        """
        idx = state["current_subtask_index"]
        subtasks = list(state["subtasks"])   # mutable copy
        current = subtasks[idx]
        max_retries = state["max_retries_per_subtask"]

        # Collect notes produced for this subtask
        notes = [n for n in state["research_notes"] if n["subtask_id"] == current["id"]]
        combined_content = " ".join(n["content"] for n in notes)
        is_sufficient = len(combined_content.strip()) >= 150  # at least ~2 sentences

        if is_sufficient:
            subtasks[idx] = {**current, "status": "completed"}
            new_idx = idx + 1
            _log.info(f"Subtask {current['id']} completed  ({len(combined_content)} chars)")
            next_action = "research" if new_idx < len(subtasks) else "synthesize"
            reason = "subtask complete, fetching next" if next_action == "research" else "all subtasks done"
            _log.transition("researcher" if next_action == "research" else "synthesizer", reason=reason)
            return {
                "subtasks": subtasks,
                "current_subtask_index": new_idx,
                "next_action": next_action,
                "status": "researching" if next_action == "research" else "synthesizing",
                "transitions": [make_transition("validator", next_action, reason=reason)],
            }

        # ── Research was empty / insufficient ──────────────────────────────
        retry_count = current["retry_count"]

        if retry_count < max_retries:
            revised = self._revise_query(current["query"], current["rationale"], retry_count + 1)
            subtasks[idx] = {
                **current,
                "status": "retrying",
                "retry_count": retry_count + 1,
                "query": revised,
                "revised_query": revised,
            }
            _log.retry(
                current["id"],
                attempt=retry_count + 1,
                reason="insufficient research results",
                revised_query=revised,
            )
            return {
                "subtasks": subtasks,
                "next_action": "research",
                "transitions": [make_transition("validator", "researcher", reason=f"retry {retry_count + 1}")],
            }
        else:
            # Give up on this subtask — mark failed, move on
            subtasks[idx] = {**current, "status": "failed"}
            new_idx = idx + 1
            _log.warning(
                f"Subtask {current['id']} FAILED after {max_retries} retries — skipping"
            )
            next_action = "research" if new_idx < len(subtasks) else "synthesize"
            reason = "max retries reached, advancing"
            return {
                "subtasks": subtasks,
                "current_subtask_index": new_idx,
                "next_action": next_action,
                "transitions": [make_transition("validator", next_action, reason=reason)],
            }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _revise_query(self, original_query: str, rationale: str, attempt: int) -> str:
        """Ask the LLM to reformulate a failing query."""
        prompt = (
            f"A web search for the following query returned no useful results.\n"
            f"Original query: {original_query}\n"
            f"Research goal: {rationale}\n"
            f"Attempt number: {attempt}\n\n"
            f"Provide a revised query that is more likely to find relevant information. "
            f"Try different keywords, synonyms, or a narrower/broader scope."
        )
        try:
            result: RevisionSchema = self._revise_llm.invoke(prompt)
            return result.revised_query
        except Exception as exc:  # noqa: BLE001
            _log.error(f"Query revision LLM failed: {exc}")
            # Simple fallback: append "market analysis report"
            return f"{original_query} detailed analysis report"


# ── Module-level singleton (used by graph.py) ─────────────────────────────────
_planner = PlannerAgent()

plan_node = _planner.plan_node
validate_node = _planner.validate_node
