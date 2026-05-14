"""
tests/test_planner.py
─────────────────────
Tests for PlannerAgent.plan_node and PlannerAgent.validate_node.
All LLM calls are mocked — no API keys required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import SubTask, initial_state


# ─────────────────────────────────────────────────────────────────────────────
# plan_node tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanNode:

    def _make_plan_response(self, subtask_queries: list[str]):
        """Build a mock PlanSchema-like object."""
        class FakeSubTask:
            def __init__(self, i, q):
                self.id = f"t{i}"
                self.query = q
                self.rationale = f"Rationale for {q}"

        class FakePlan:
            subtasks = [FakeSubTask(i + 1, q) for i, q in enumerate(subtask_queries)]
            overall_approach = "Test approach"

        return FakePlan()

    @patch("src.agents.planner.ChatAnthropic")
    def test_plan_node_creates_subtasks(self, mock_llm_cls, base_state):
        """plan_node should return a list of SubTask dicts."""
        plan_response = self._make_plan_response([
            "OpenAI competitors 2025",
            "Anthropic Claude market share",
        ])

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = plan_response
        mock_llm_cls.return_value = mock_llm

        # Import after mocking
        from src.agents.planner import PlannerAgent
        agent = PlannerAgent()
        result = agent.plan_node(base_state)

        assert "subtasks" in result
        assert len(result["subtasks"]) == 2
        assert result["subtasks"][0]["id"] == "t1"
        assert result["subtasks"][0]["status"] == "pending"
        assert result["subtasks"][0]["retry_count"] == 0
        assert result["current_subtask_index"] == 0
        assert result["status"] == "researching"

    @patch("src.agents.planner.ChatAnthropic")
    def test_plan_node_fallback_on_llm_error(self, mock_llm_cls, base_state):
        """plan_node should fall back to original query when LLM errors."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("API down")
        mock_llm_cls.return_value = mock_llm

        from src.agents.planner import PlannerAgent
        agent = PlannerAgent()
        result = agent.plan_node(base_state)

        # Should still return at least one subtask (the fallback)
        assert len(result["subtasks"]) >= 1
        assert result["subtasks"][0]["query"] == base_state["original_query"]

    @patch("src.agents.planner.ChatAnthropic")
    def test_plan_node_records_transition(self, mock_llm_cls, base_state):
        """plan_node should append a transition to the audit trail."""
        plan_response = self._make_plan_response(["query one"])
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = plan_response
        mock_llm_cls.return_value = mock_llm

        from src.agents.planner import PlannerAgent
        agent = PlannerAgent()
        result = agent.plan_node(base_state)

        assert "transitions" in result
        assert len(result["transitions"]) >= 1
        t = result["transitions"][0]
        assert t["from_agent"] == "planner"
        assert t["to_agent"] == "researcher"


# ─────────────────────────────────────────────────────────────────────────────
# validate_node tests
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateNode:

    @patch("src.agents.planner.ChatAnthropic")
    def test_validate_marks_completed_on_sufficient_notes(self, mock_llm_cls, state_with_notes):
        """When research notes are sufficient, current subtask should be 'completed'."""
        mock_llm_cls.return_value = MagicMock()

        from src.agents.planner import PlannerAgent
        agent = PlannerAgent()

        # state_with_notes has index=1 (t1 done), so validate checks t1's notes
        # But let's set index back to 0 for clarity
        test_state = {**state_with_notes, "current_subtask_index": 0}
        result = agent.validate_node(test_state)

        assert result["subtasks"][0]["status"] == "completed"
        # Should advance the index
        assert result["current_subtask_index"] == 1

    @patch("src.agents.planner.ChatAnthropic")
    def test_validate_retries_on_empty_notes(self, mock_llm_cls, state_with_subtasks):
        """When research returned nothing, validate should trigger a retry."""
        mock_revise = MagicMock()
        mock_revise.invoke.return_value = MagicMock(revised_query="revised OpenAI competitor list")
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_revise
        mock_llm_cls.return_value = mock_llm

        from src.agents.planner import PlannerAgent
        agent = PlannerAgent()

        # No notes in state → empty research
        result = agent.validate_node(state_with_subtasks)

        assert result["next_action"] == "research"
        assert result["subtasks"][0]["status"] == "retrying"
        assert result["subtasks"][0]["retry_count"] == 1

    @patch("src.agents.planner.ChatAnthropic")
    def test_validate_fails_subtask_after_max_retries(self, mock_llm_cls, state_with_subtasks):
        """After max retries, validate should mark subtask as failed and move on."""
        mock_llm_cls.return_value = MagicMock()

        from src.agents.planner import PlannerAgent
        agent = PlannerAgent()

        # Simulate already at max retries
        subtasks = list(state_with_subtasks["subtasks"])
        subtasks[0] = {**subtasks[0], "retry_count": 2}
        exhausted_state = {**state_with_subtasks, "subtasks": subtasks}

        result = agent.validate_node(exhausted_state)

        assert result["subtasks"][0]["status"] == "failed"
        assert result["current_subtask_index"] == 1  # advanced past failed subtask

    @patch("src.agents.planner.ChatAnthropic")
    def test_validate_routes_to_synthesize_when_all_done(self, mock_llm_cls, state_all_done):
        """When index is past end of subtask list, should route to synthesize."""
        mock_llm_cls.return_value = MagicMock()

        from src.agents.planner import PlannerAgent
        agent = PlannerAgent()

        # all subtasks completed, index=2 (past end)
        # We need to set up a state where the last subtask has sufficient notes
        # Let's use a state at the last subtask with a note
        last_idx = len(state_all_done["subtasks"]) - 1
        test_state = {
            **state_all_done,
            "current_subtask_index": last_idx,
        }
        result = agent.validate_node(test_state)

        # Should route to synthesize since all subtasks will be done
        assert result["next_action"] == "synthesize"
