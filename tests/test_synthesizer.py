"""
tests/test_synthesizer.py
─────────────────────────
Tests for SynthesizerAgent.synthesize_node.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import ResearchNote


class TestSynthesizerNode:

    def _make_llm_response(self, content: str):
        class FakeMsg:
            pass
        m = FakeMsg()
        m.content = content
        return m

    @patch("src.agents.synthesizer.ChatAnthropic")
    def test_synthesize_node_returns_report(self, mock_llm_cls, state_all_done):
        """synthesize_node should populate final_report."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response(
            "## Executive Summary\n- OpenAI dominates with 60% share.\n"
            "## Key Findings\nAnthropic is the fastest growing challenger."
        )
        mock_llm_cls.return_value = mock_llm

        from src.agents.synthesizer import SynthesizerAgent
        agent = SynthesizerAgent()
        result = agent.synthesize_node(state_all_done)

        assert "final_report" in result
        assert result["final_report"] is not None
        assert len(result["final_report"]) > 100
        assert result["status"] == "complete"

    @patch("src.agents.synthesizer.ChatAnthropic")
    def test_synthesize_report_contains_quality_table(self, mock_llm_cls, state_all_done):
        """Report should always include a quality/coverage table."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response("Report content here.")
        mock_llm_cls.return_value = mock_llm

        from src.agents.synthesizer import SynthesizerAgent
        agent = SynthesizerAgent()
        result = agent.synthesize_node(state_all_done)

        report = result["final_report"]
        # Quality table should be present
        assert "Research Quality" in report
        assert "t1" in report or "t2" in report  # subtask IDs appear

    @patch("src.agents.synthesizer.ChatAnthropic")
    def test_synthesize_empty_notes_returns_warning(self, mock_llm_cls, base_state):
        """When there are no notes, return a clear warning report."""
        mock_llm_cls.return_value = MagicMock()

        from src.agents.synthesizer import SynthesizerAgent
        agent = SynthesizerAgent()
        result = agent.synthesize_node(base_state)  # base_state has no notes

        assert result["final_report"] is not None
        assert "⚠️" in result["final_report"] or "No research" in result["final_report"]
        assert result["status"] == "complete"

    @patch("src.agents.synthesizer.ChatAnthropic")
    def test_synthesize_records_transition(self, mock_llm_cls, state_all_done):
        """synthesize_node should append a transition to END."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response("Report body.")
        mock_llm_cls.return_value = mock_llm

        from src.agents.synthesizer import SynthesizerAgent
        agent = SynthesizerAgent()
        result = agent.synthesize_node(state_all_done)

        assert "transitions" in result
        assert result["transitions"][0]["to_agent"] == "END"

    @patch("src.agents.synthesizer.ChatAnthropic")
    def test_synthesize_llm_failure_returns_raw_notes(self, mock_llm_cls, state_all_done):
        """If the LLM fails, synthesizer should still return a report with raw notes."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        mock_llm_cls.return_value = mock_llm

        from src.agents.synthesizer import SynthesizerAgent
        agent = SynthesizerAgent()
        result = agent.synthesize_node(state_all_done)

        # Should not propagate the exception; should return something
        assert result["final_report"] is not None
        assert result["status"] == "complete"

    @patch("src.agents.synthesizer.ChatAnthropic")
    def test_report_includes_query_and_timestamp(self, mock_llm_cls, state_all_done):
        """The report header should include the original query."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response("Body text.")
        mock_llm_cls.return_value = mock_llm

        from src.agents.synthesizer import SynthesizerAgent
        agent = SynthesizerAgent()
        result = agent.synthesize_node(state_all_done)

        report = result["final_report"]
        assert state_all_done["original_query"] in report
        assert "Generated" in report   # timestamp header
