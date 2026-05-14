"""
tests/test_researcher.py
────────────────────────
Tests for ResearcherAgent.research_node.
Mocks both Tavily and httpx so no network calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestResearcherNode:

    def _make_llm_response(self, content: str):
        class FakeMsg:
            pass
        msg = FakeMsg()
        msg.content = content
        return msg

    @patch("src.agents.researcher.fetch_url")
    @patch("src.agents.researcher.web_search")
    @patch("src.agents.researcher.ChatAnthropic")
    def test_research_node_appends_note(
        self, mock_llm_cls, mock_search, mock_fetch, state_with_subtasks
    ):
        """research_node should append exactly one ResearchNote."""
        mock_search.return_value = {
            "query": "OpenAI competitors AI market 2025",
            "results": [
                {"title": "Test", "url": "https://example.com", "content": "Google competes with OpenAI.", "score": 0.9},
            ],
            "answer": "Google, Anthropic, Meta.",
        }
        mock_fetch.return_value = {"url": "https://example.com", "content": "Full article text here.", "success": True, "error": None}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response(
            "• Google DeepMind is a top competitor.\n• Anthropic raised $4B from Amazon."
        )
        mock_llm_cls.return_value = mock_llm

        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        result = agent.research_node(state_with_subtasks)

        assert "research_notes" in result
        assert len(result["research_notes"]) == 1
        note = result["research_notes"][0]
        assert note["subtask_id"] == "t1"
        assert "Google" in note["content"] or len(note["content"]) > 0
        assert len(note["sources"]) >= 1

    @patch("src.agents.researcher.fetch_url")
    @patch("src.agents.researcher.web_search")
    @patch("src.agents.researcher.ChatAnthropic")
    def test_research_node_empty_search_returns_empty_content(
        self, mock_llm_cls, mock_search, mock_fetch, state_with_subtasks
    ):
        """When Tavily returns no results, content should be empty (signals retry)."""
        mock_search.return_value = {"query": "test", "results": [], "answer": ""}
        mock_fetch.return_value = {"url": "", "content": "", "success": False, "error": "no url"}

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response("")
        mock_llm_cls.return_value = mock_llm

        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        result = agent.research_node(state_with_subtasks)

        note = result["research_notes"][0]
        # Empty content signals to Planner that retry is needed
        assert note["content"] == "" or len(note["content"]) < 150

    @patch("src.agents.researcher.fetch_url")
    @patch("src.agents.researcher.web_search")
    @patch("src.agents.researcher.ChatAnthropic")
    def test_research_node_records_transition(
        self, mock_llm_cls, mock_search, mock_fetch, state_with_subtasks
    ):
        """research_node should log a transition to the audit trail."""
        mock_search.return_value = {
            "query": "test", "results": [
                {"title": "T", "url": "https://x.com", "content": "Some content.", "score": 0.8}
            ], "answer": ""
        }
        mock_fetch.return_value = {"url": "https://x.com", "content": "text", "success": True, "error": None}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response("Some research findings here about AI.")
        mock_llm_cls.return_value = mock_llm

        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        result = agent.research_node(state_with_subtasks)

        assert "transitions" in result
        assert len(result["transitions"]) >= 1
        t = result["transitions"][0]
        assert t["from_agent"] == "researcher"

    @patch("src.agents.researcher.fetch_url")
    @patch("src.agents.researcher.web_search")
    @patch("src.agents.researcher.ChatAnthropic")
    def test_research_uses_revised_query_when_set(
        self, mock_llm_cls, mock_search, mock_fetch, state_with_subtasks
    ):
        """If subtask has a revised_query, the researcher should use it."""
        subtasks = list(state_with_subtasks["subtasks"])
        subtasks[0] = {**subtasks[0], "query": "revised query terms", "revised_query": "revised query terms"}
        revised_state = {**state_with_subtasks, "subtasks": subtasks}

        mock_search.return_value = {"query": "revised query terms", "results": [], "answer": ""}
        mock_fetch.return_value = {"url": "", "content": "", "success": False, "error": None}
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._make_llm_response("")
        mock_llm_cls.return_value = mock_llm

        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        agent.research_node(revised_state)

        # The search should have been called with the revised query
        call_args = mock_search.call_args
        assert "revised query terms" in call_args[0][0]

    @patch("src.agents.researcher.fetch_url")
    @patch("src.agents.researcher.web_search")
    @patch("src.agents.researcher.ChatAnthropic")
    def test_research_node_llm_distil_failure_falls_back(
        self, mock_llm_cls, mock_search, mock_fetch, state_with_subtasks
    ):
        """If the distillation LLM fails, fallback to raw snippets."""
        mock_search.return_value = {
            "query": "test", "results": [
                {"title": "T", "url": "https://x.com", "content": "Raw snippet content here.", "score": 0.8}
            ], "answer": ""
        }
        mock_fetch.return_value = {"url": "https://x.com", "content": "Fetched content.", "success": True, "error": None}

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_llm_cls.return_value = mock_llm

        from src.agents.researcher import ResearcherAgent
        agent = ResearcherAgent()
        result = agent.research_node(state_with_subtasks)

        note = result["research_notes"][0]
        # Should still have some content from raw fallback
        assert len(note["content"]) > 0
