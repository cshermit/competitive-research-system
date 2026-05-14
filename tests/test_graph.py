"""
tests/test_graph.py
───────────────────
Integration tests for the full LangGraph state machine.
Each agent node is mocked to return predictable outputs so we can
verify routing logic without making real LLM / API calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import initial_state, make_transition, ResearchNote, SubTask


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — canned node outputs
# ─────────────────────────────────────────────────────────────────────────────

def _plan_output():
    """Minimal valid plan_node output."""
    return {
        "subtasks": [
            SubTask(id="t1", query="q1", original_query="q1", rationale="r1",
                    status="pending", retry_count=0, revised_query=None),
        ],
        "current_subtask_index": 0,
        "status": "researching",
        "transitions": [make_transition("planner", "researcher")],
    }


def _research_output(subtask_id="t1", content="Lots of good research content here."):
    return {
        "research_notes": [
            ResearchNote(subtask_id=subtask_id, query="q1", content=content,
                         sources=["https://x.com"], result_count=3, timestamp="2025-01-01T00:00:00+00:00"),
        ],
        "subtasks": [
            SubTask(id="t1", query="q1", original_query="q1", rationale="r1",
                    status="researching", retry_count=0, revised_query=None),
        ],
        "transitions": [make_transition("researcher", "validator")],
    }


def _validate_output_done():
    return {
        "subtasks": [
            SubTask(id="t1", query="q1", original_query="q1", rationale="r1",
                    status="completed", retry_count=0, revised_query=None),
        ],
        "current_subtask_index": 1,   # past end → synthesize
        "next_action": "synthesize",
        "status": "synthesizing",
        "transitions": [make_transition("validator", "synthesize")],
    }


def _synthesize_output():
    return {
        "final_report": "# Report\n\nFindings here.",
        "status": "complete",
        "transitions": [make_transition("synthesizer", "END")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphRouting:

    def test_graph_compiles_without_error(self):
        """The graph should compile with no import or structural errors."""
        from src.graph.graph import build_graph
        graph = build_graph()
        app = graph.compile()
        assert app is not None

    def test_graph_has_expected_nodes(self):
        """Graph should contain all four required nodes."""
        from src.graph.graph import build_graph
        graph = build_graph()
        # Access the internal graph structure
        node_names = set(graph.nodes.keys())
        for expected in ("plan", "research", "validate", "synthesize"):
            assert expected in node_names, f"Missing node: {expected}"

    @patch("src.graph.graph.synthesize_node")
    @patch("src.graph.graph.validate_node")
    @patch("src.graph.graph.research_node")
    @patch("src.graph.graph.plan_node")
    def test_happy_path_end_to_end(
        self, mock_plan, mock_research, mock_validate, mock_synthesize
    ):
        """
        Full graph invocation with all nodes mocked.
        Verifies: plan → research → validate → synthesize → END.
        """
        mock_plan.return_value = _plan_output()
        mock_research.return_value = _research_output()
        mock_validate.return_value = _validate_output_done()
        mock_synthesize.return_value = _synthesize_output()

        from src.graph.graph import compile_graph
        from src.graph.state import initial_state

        app = compile_graph()
        state = initial_state("test query")
        result = app.invoke(state)

        assert result["status"] == "complete"
        assert result["final_report"] is not None
        assert "Report" in result["final_report"]

        # All four nodes should have been called once
        mock_plan.assert_called_once()
        mock_research.assert_called_once()
        mock_validate.assert_called_once()
        mock_synthesize.assert_called_once()

    @patch("src.graph.graph.synthesize_node")
    @patch("src.graph.graph.validate_node")
    @patch("src.graph.graph.research_node")
    @patch("src.graph.graph.plan_node")
    def test_retry_path_calls_research_twice(
        self, mock_plan, mock_research, mock_validate, mock_synthesize
    ):
        """
        When validate routes back to 'research' (retry), research_node is called twice.
        """
        mock_plan.return_value = _plan_output()

        # First research returns empty; second returns good content
        mock_research.side_effect = [
            _research_output(content=""),        # empty → triggers retry
            _research_output(content="Good research data from second attempt."),
        ]

        # First validate → retry; second validate → synthesize
        retry_validate = {
            "subtasks": [
                SubTask(id="t1", query="revised q1", original_query="q1", rationale="r1",
                        status="retrying", retry_count=1, revised_query="revised q1"),
            ],
            "next_action": "research",
            "transitions": [make_transition("validator", "researcher", reason="retry 1")],
        }
        mock_validate.side_effect = [retry_validate, _validate_output_done()]
        mock_synthesize.return_value = _synthesize_output()

        from src.graph.graph import compile_graph
        from src.graph.state import initial_state

        app = compile_graph()
        result = app.invoke(initial_state("test query"))

        assert result["status"] == "complete"
        assert mock_research.call_count == 2     # called twice due to retry
        assert mock_validate.call_count == 2

    def test_transitions_accumulate_across_nodes(self):
        """Annotated list reducer should accumulate transitions, not replace them."""
        from src.graph.state import ResearchState, make_transition

        state: ResearchState = {
            "original_query": "q",
            "subtasks": [],
            "current_subtask_index": 0,
            "max_retries_per_subtask": 2,
            "research_notes": [],
            "transitions": [make_transition("planner", "researcher")],
            "final_report": None,
            "next_action": "research",
            "status": "researching",
            "error": None,
        }

        # Simulate LangGraph's reducer applying a new transition
        import operator
        new_t = [make_transition("researcher", "validator")]
        combined = operator.add(state["transitions"], new_t)

        assert len(combined) == 2
        assert combined[0]["from_agent"] == "planner"
        assert combined[1]["from_agent"] == "researcher"
