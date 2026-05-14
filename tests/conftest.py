"""
tests/conftest.py
─────────────────
Shared pytest fixtures.
All external calls (LLM, Tavily, httpx) are mocked so tests run offline.
"""
from __future__ import annotations

import pytest

from src.graph.state import ResearchNote, SubTask, initial_state


# ─────────────────────────────────────────────────────────────────────────────
# State fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    """Minimal initial state for a fresh research run."""
    return initial_state("Who are OpenAI's top competitors?", max_retries=2)


@pytest.fixture
def state_with_subtasks(base_state):
    """State that already has subtasks planned."""
    subtasks = [
        SubTask(
            id="t1",
            query="OpenAI competitors AI market 2025",
            original_query="OpenAI competitors AI market 2025",
            rationale="Identify direct competitors",
            status="pending",
            retry_count=0,
            revised_query=None,
        ),
        SubTask(
            id="t2",
            query="Anthropic Claude vs OpenAI GPT comparison",
            original_query="Anthropic Claude vs OpenAI GPT comparison",
            rationale="Compare leading models",
            status="pending",
            retry_count=0,
            revised_query=None,
        ),
    ]
    return {**base_state, "subtasks": subtasks, "current_subtask_index": 0, "status": "researching"}


@pytest.fixture
def state_with_notes(state_with_subtasks):
    """State that has one completed research note."""
    note = ResearchNote(
        subtask_id="t1",
        query="OpenAI competitors AI market 2025",
        content=(
            "• Google DeepMind released Gemini Ultra in late 2024 with strong benchmark scores.\n"
            "• Anthropic raised $4B from Amazon; Claude 3.5 Sonnet outperforms GPT-4o on coding.\n"
            "• Meta's Llama 3.1 405B is the leading open-source alternative.\n"
            "• Mistral AI (France) raised €600M and competes on efficiency.\n"
            "• xAI's Grok 2 targets enterprise market with X platform integration."
        ),
        sources=["https://example.com/ai-market", "https://example.com/claude"],
        result_count=5,
        timestamp="2025-01-01T00:00:00+00:00",
    )
    subtasks = list(state_with_subtasks["subtasks"])
    subtasks[0] = {**subtasks[0], "status": "completed"}
    return {
        **state_with_subtasks,
        "subtasks": subtasks,
        "research_notes": [note],
        "current_subtask_index": 1,
    }


@pytest.fixture
def state_all_done(state_with_notes):
    """State where all subtasks are done and we're ready to synthesize."""
    note2 = ResearchNote(
        subtask_id="t2",
        query="Anthropic Claude vs OpenAI GPT comparison",
        content=(
            "• Claude 3.5 Sonnet scores 92% on HumanEval vs GPT-4o's 90.2%.\n"
            "• OpenAI leads on market share (~60%) while Anthropic grows 3x YoY.\n"
            "• Enterprise pricing: Claude Pro $20/mo vs ChatGPT Plus $20/mo.\n"
        ),
        sources=["https://example.com/benchmark"],
        result_count=3,
        timestamp="2025-01-01T00:01:00+00:00",
    )
    subtasks = list(state_with_notes["subtasks"])
    subtasks[1] = {**subtasks[1], "status": "completed"}
    return {
        **state_with_notes,
        "subtasks": subtasks,
        "research_notes": state_with_notes["research_notes"] + [note2],
        "current_subtask_index": 2,      # past end of list
        "next_action": "synthesize",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LLM / API mock helpers
# ─────────────────────────────────────────────────────────────────────────────

class _FakeLLMResponse:
    """Mimics a LangChain ChatMessage."""
    def __init__(self, content: str):
        self.content = content


@pytest.fixture
def fake_llm_response():
    return _FakeLLMResponse


@pytest.fixture
def mock_tavily_results():
    """Realistic-looking Tavily search results."""
    return {
        "results": [
            {
                "title": "Top OpenAI Competitors 2025",
                "url": "https://example.com/openai-competitors",
                "content": "Google, Anthropic, and Meta are the leading OpenAI competitors with strong model performance.",
                "score": 0.92,
            },
            {
                "title": "AI Market Analysis",
                "url": "https://example.com/ai-market",
                "content": "The generative AI market is expected to reach $1.3T by 2032.",
                "score": 0.85,
            },
        ],
        "answer": "OpenAI's top competitors include Google DeepMind, Anthropic, Meta AI, and Mistral.",
    }
