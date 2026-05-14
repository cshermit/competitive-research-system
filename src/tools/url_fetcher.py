"""
src/tools/url_fetcher.py
────────────────────────
Fetches a URL and returns cleaned text content.
Used by the Researcher to get full article content beyond Tavily snippets.
"""
from __future__ import annotations

import re
from typing import Optional, TypedDict

import httpx

from src.utils.config import settings
from src.utils.logger import AgentLogger

_log = AgentLogger("tool.url_fetcher")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CompetitiveResearchBot/1.0; "
        "+https://github.com/yourorg/competitive-research)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


class FetchResult(TypedDict):
    url: str
    content: str          # cleaned text (truncated to MAX_URL_CONTENT_CHARS)
    success: bool
    error: Optional[str]


def _strip_html(raw: str) -> str:
    """Very light HTML → text (no heavy deps like BeautifulSoup)."""
    # Remove script / style blocks
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    raw = re.sub(r"<[^>]+>", " ", raw)
    # Collapse whitespace
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def fetch_url(url: str, max_chars: int | None = None) -> FetchResult:
    """
    Fetch a URL and return cleaned text.

    Parameters
    ----------
    url:       Target URL (must be http/https).
    max_chars: Truncate text to this many characters (default from settings).

    Returns
    -------
    FetchResult — always returns, never raises.
    """
    limit = max_chars or settings.max_url_content_chars
    _log.tool_call("fetch_url", url=url)

    if not url.startswith(("http://", "https://")):
        return FetchResult(url=url, content="", success=False, error="Non-HTTP URL skipped")

    try:
        with httpx.Client(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        err = f"HTTP {exc.response.status_code}"
        _log.warning(f"fetch_url HTTP error: {err}", url=url)
        return FetchResult(url=url, content="", success=False, error=err)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        _log.warning(f"fetch_url failed: {err}", url=url)
        return FetchResult(url=url, content="", success=False, error=err)

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type or "text/plain" in content_type:
        text = _strip_html(response.text)
    else:
        # PDF, JSON, etc. — skip binary types
        _log.info(f"fetch_url skipped non-text content-type: {content_type}", url=url)
        return FetchResult(url=url, content="", success=False, error=f"Unsupported content-type: {content_type}")

    truncated = text[:limit]
    _log.tool_result("fetch_url", result_summary=f"{len(truncated)} chars from {url}")
    return FetchResult(url=url, content=truncated, success=True, error=None)
