"""
src/agents/synthesizer.py
──────────────────────────
The Synthesizer is the final step in the pipeline.
It receives all research notes and produces a polished competitive
intelligence report in Markdown.

It also appends a "Research Quality" section showing which subtasks
succeeded, failed, or were retried — so the reader knows the confidence
level of each section.
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic

from src.graph.state import ResearchState, make_transition, now_iso
from src.utils.config import settings
from src.utils.logger import AgentLogger

_log = AgentLogger("synthesizer", level=settings.log_level)


class SynthesizerAgent:

    def __init__(self) -> None:
        self._llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.4,     # slightly creative for prose
            max_tokens=4096,
        )

    def synthesize_node(self, state: ResearchState) -> dict:
        """
        LangGraph node: compile all research notes into a final report.
        """
        _log.info("Synthesizing final report …")
        _log.transition("END", reason="report ready")

        notes = state["research_notes"]
        subtasks = state["subtasks"]
        original_query = state["original_query"]

        if not notes:
            _log.warning("No research notes available — returning empty report")
            report = (
                f"# Competitive Intelligence Report\n\n"
                f"**Query:** {original_query}\n\n"
                f"> ⚠️  No research data was collected. All subtasks failed or returned empty results.\n"
                f"> Please verify your TAVILY_API_KEY and retry.\n"
            )
            return {
                "final_report": report,
                "status": "complete",
                "transitions": [make_transition("synthesizer", "END", reason="no data — empty report")],
            }

        # ── Build context for the LLM ────────────────────────────────────────
        notes_block = "\n\n".join(
            f"### Research Note: {n['query']}\n{n['content']}\n**Sources:** {', '.join(n['sources'])}"
            for n in notes
            if n["content"].strip()
        )

        quality_table = self._build_quality_table(subtasks)

        prompt = (
            f"You are a senior competitive intelligence analyst writing an executive briefing.\n\n"
            f"Original research request: {original_query}\n\n"
            f"Below are distilled research notes collected by our research team:\n\n"
            f"{notes_block}\n\n"
            f"---\n\n"
            f"Write a comprehensive competitive intelligence report in Markdown. Structure:\n"
            f"1. **Executive Summary** (3–5 bullet points, the most critical findings)\n"
            f"2. **Key Findings** (detailed sections by theme, use subheadings)\n"
            f"3. **Competitive Landscape** (players, positioning, market dynamics)\n"
            f"4. **Strategic Implications** (what this means for the reader)\n"
            f"5. **Recommended Actions** (3–5 concrete next steps)\n\n"
            f"Be specific: cite numbers, company names, dates. Do NOT make up information "
            f"not present in the notes."
        )

        try:
            response = self._llm.invoke(prompt)
            report_body = response.content   # type: ignore[attr-defined]
        except Exception as exc:         # noqa: BLE001
            _log.error(f"Synthesizer LLM failed: {exc}")
            report_body = f"## Raw Research Notes\n\n{notes_block}"

        # Append quality metadata
        report = (
            f"# Competitive Intelligence Report\n"
            f"**Query:** {original_query}\n"
            f"**Generated:** {now_iso()}\n\n"
            f"---\n\n"
            f"{report_body}\n\n"
            f"---\n\n"
            f"## Research Quality & Coverage\n\n"
            f"{quality_table}\n"
        )

        _log.info(f"Report generated: {len(report)} chars")

        return {
            "final_report": report,
            "status": "complete",
            "transitions": [make_transition("synthesizer", "END", reason="report complete")],
        }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _build_quality_table(self, subtasks: list) -> str:
        """Render a Markdown table showing subtask outcomes."""
        if not subtasks:
            return "_No subtasks were planned._"

        STATUS_ICON = {
            "completed": "✅",
            "failed": "❌",
            "retrying": "🔄",
            "pending": "⏳",
            "researching": "🔍",
        }

        rows = ["| ID | Query | Status | Retries |", "|---|---|---|---|"]
        for st in subtasks:
            icon = STATUS_ICON.get(st["status"], "❓")
            query = st["original_query"][:60] + ("…" if len(st["original_query"]) > 60 else "")
            rows.append(f"| {st['id']} | {query} | {icon} {st['status']} | {st['retry_count']} |")

        return "\n".join(rows)


# ── Module-level singleton ─────────────────────────────────────────────────────
_synthesizer = SynthesizerAgent()

synthesize_node = _synthesizer.synthesize_node
