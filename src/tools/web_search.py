"""
src/tools/web_search.py
───────────────────────
Thin, testable wrapper around the Tavily search API.
Returns a structured dict so agents don't need to know Tavily internals.
"""
from __future__ import annotations

from typing import List, TypedDict

from src.utils.config import settings
from src.utils.logger import AgentLogger

_log = AgentLogger("tool.web_search")


class SearchResult(TypedDict):
    title: str
    url: str
    content: str        # snippet / summary from Tavily
    score: float        # relevance score 0–1


class SearchResponse(TypedDict):
    query: str
    results: List[SearchResult]
    answer: str         # Tavily's synthesised answer (may be empty)


def web_search(query: str, max_results: int | None = None) -> SearchResponse:
    """
    Execute a Tavily web search and return structured results.

    Parameters
    ----------
    query:       The search string.
    max_results: Override the global MAX_SEARCH_RESULTS setting.

    Returns
    -------
    SearchResponse with .results (possibly empty list on failure).
    """
    n = max_results or settings.max_search_results
    _log.tool_call("tavily_search", query=query, max_results=n)

    try:
        from tavily import TavilyClient  # import lazily so tests can patch easily

        client = TavilyClient(api_key=settings.tavily_api_key)
        raw = client.search(
            query=query,
            max_results=n,
            include_answer=True,
            search_depth="advanced",
        )
    except Exception as exc:  # noqa: BLE001
        _log.error(f"Tavily search failed: {exc}", query=query)
        return SearchResponse(query=query, results=[], answer="")

    results: List[SearchResult] = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            score=float(r.get("score", 0.0)),
        )
        for r in raw.get("results", [])
    ]

    answer = raw.get("answer", "") or ""

    _log.tool_result(
        "tavily_search",
        result_summary=f"{len(results)} results  (top score: {results[0]['score']:.2f})" if results else "0 results",
    )

    return SearchResponse(query=query, results=results, answer=answer)
