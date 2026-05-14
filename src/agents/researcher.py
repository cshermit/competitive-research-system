"""
src/agents/researcher.py
────────────────────────
The Researcher executes one SubTask at a time:
  1. Run a Tavily web search for the subtask query.
  2. Fetch the top-N URLs for richer content.
  3. Ask the LLM to distil the raw text into a concise research note.
  4. Append the note to state["research_notes"].

The Researcher does NOT decide what to do next — that's the Planner's job.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src.graph.state import ResearchNote, ResearchState, make_transition, now_iso
from src.tools.url_fetcher import fetch_url
from src.tools.web_search import web_search
from src.utils.config import settings
from src.utils.logger import AgentLogger

_log = AgentLogger("researcher", level=settings.log_level)

_URLS_TO_FETCH = 2          # fetch the top N URLs for deeper content


class ResearcherAgent:

    def __init__(self) -> None:
        self._llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.1,    # low temp — we want factual synthesis
        )

    def research_node(self, state: ResearchState) -> dict:
        """
        LangGraph node: execute the current subtask and return a research note.

        Reads:  state["subtasks"][state["current_subtask_index"]]
        Writes: appends to state["research_notes"]
        """
        idx = state["current_subtask_index"]
        subtasks = list(state["subtasks"])
        current = subtasks[idx]
        query = current["query"]   # may already be revised by Planner

        _log.info(f"Researching subtask [{current['id']}]: {query!r}")

        # ── Step 1: Web search ───────────────────────────────────────────────
        search_resp = web_search(query)
        results = search_resp["results"]

        # ── Step 2: Fetch top-N URLs for full text ───────────────────────────
        fetched_texts: list[str] = []
        sources: list[str] = []

        for result in results[:_URLS_TO_FETCH]:
            url = result["url"]
            fetch = fetch_url(url)
            sources.append(url)
            if fetch["success"] and fetch["content"]:
                fetched_texts.append(f"Source: {url}\n{fetch['content']}")
            else:
                # Fall back to Tavily snippet
                fetched_texts.append(f"Source: {url}\n{result['content']}")

        # Also add remaining results' snippets (no full fetch)
        for result in results[_URLS_TO_FETCH:]:
            sources.append(result["url"])
            fetched_texts.append(f"Source: {result['url']}\n{result['content']}")

        # Include Tavily's synthesised answer if present
        if search_resp["answer"]:
            fetched_texts.insert(0, f"Search synthesis: {search_resp['answer']}")

        raw_text = "\n\n---\n\n".join(fetched_texts)

        # ── Step 3: LLM distillation ─────────────────────────────────────────
        note_content = self._distil(query, raw_text, search_resp["answer"])

        # ── Step 4: Build the ResearchNote ───────────────────────────────────
        note = ResearchNote(
            subtask_id=current["id"],
            query=query,
            content=note_content,
            sources=sources,
            result_count=len(results),
            timestamp=now_iso(),
        )

        # Update subtask status to "researching" while we have the list
        subtasks[idx] = {**current, "status": "researching"}

        _log.info(
            f"Research note for [{current['id']}]: {len(note_content)} chars  "
            f"from {len(sources)} sources"
        )
        _log.transition("validator", reason="research complete, checking quality")

        return {
            "subtasks": subtasks,
            "research_notes": [note],           # Annotated → gets APPENDED
            "transitions": [make_transition("researcher", "validator", reason="research complete")],
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _distil(self, query: str, raw_text: str, search_answer: str) -> str:
        """
        Ask the LLM to compress raw search + fetch text into a
        structured research note (bullet points + key facts).
        """
        if not raw_text.strip():
            return ""   # signal to Planner that research was empty

        prompt = (
            f"You are a competitive intelligence analyst.\n"
            f"Research question: {query}\n\n"
            f"Below is raw text from web search results and fetched pages. "
            f"Extract the most relevant facts, statistics, and insights.\n"
            f"Format as concise bullet points. Be factual, cite specific numbers when available.\n"
            f"Limit to ~300 words.\n\n"
            f"RAW TEXT:\n{raw_text[:8000]}"  # hard cap to avoid token overflow
        )

        try:
            response = self._llm.invoke(prompt)
            return response.content  # type: ignore[attr-defined]
        except Exception as exc:    # noqa: BLE001
            _log.error(f"Distillation LLM failed: {exc}")
            # Return raw snippets as fallback so notes aren't empty
            return raw_text[:2000]


# ── Module-level singleton ─────────────────────────────────────────────────────
_researcher = ResearcherAgent()

research_node = _researcher.research_node
