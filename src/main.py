"""
src/main.py
───────────
CLI entry point for the Competitive Research Multi-Agent System.

Usage
─────
    python -m src.main "Who are the top competitors of Notion in 2025?"

Or import programmatically:
    from src.main import run_research
    report = run_research("Stripe vs Braintree competitive analysis")
    print(report)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.graph.graph import compile_graph
from src.graph.state import ResearchState, initial_state
from src.utils.config import settings
from src.utils.logger import AgentLogger

_console = Console()
_log = AgentLogger("main", level=settings.log_level)


def run_research(
    query: str,
    *,
    max_retries: int | None = None,
    output_file: Optional[Path] = None,
    stream: bool = False,
) -> str:
    """
    Execute a full research run and return the final Markdown report.

    Parameters
    ----------
    query:       The competitive research question.
    max_retries: Override MAX_RETRIES_PER_SUBTASK from settings.
    output_file: If provided, write the report to this path.
    stream:      If True, print intermediate node updates to stdout.
    """
    settings.validate_required_keys()

    retries = max_retries if max_retries is not None else settings.max_retries_per_subtask
    state = initial_state(query, max_retries=retries)
    app = compile_graph()

    _console.print(
        Panel(
            f"[bold cyan]🔍 Competitive Research Agent[/bold cyan]\n\n"
            f"[white]Query:[/white] {query}\n"
            f"[dim]Model: {settings.llm_model}  |  Max subtasks: {settings.max_subtasks}  "
            f"|  Max retries: {retries}[/dim]",
            border_style="cyan",
        )
    )

    final_state: ResearchState

    if stream:
        # ── Stream mode: print each node's delta as it arrives ───────────────
        _console.print("\n[bold]Running in stream mode — node updates:[/bold]\n")
        for chunk in app.stream(state):
            for node_name, node_output in chunk.items():
                _console.print(f"  [dim]▸ node:[/dim] [bold]{node_name}[/bold]", highlight=False)
                _print_node_summary(node_name, node_output)
        # stream() doesn't return the final state, so invoke again
        final_state = app.invoke(state)
    else:
        # ── Normal mode with spinner ─────────────────────────────────────────
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=_console,
            transient=False,
        ) as progress:
            task = progress.add_task("Planning …", total=None)

            # Monkey-patch the progress description via a custom callback approach.
            # (LangGraph doesn't expose streaming in invoke; we use plain invoke.)
            final_state = app.invoke(state)
            progress.update(task, description="[green]✓ Complete")

    report = final_state.get("final_report", "")

    # ── Print report ─────────────────────────────────────────────────────────
    _console.print("\n")
    _console.print(Panel("[bold green]Research Complete[/bold green]", border_style="green"))
    _console.print(Markdown(report))

    # ── Print transition log ──────────────────────────────────────────────────
    transitions = final_state.get("transitions", [])
    if transitions:
        _console.print(f"\n[dim]Agent transitions: {len(transitions)}[/dim]")
        for t in transitions:
            _console.print(
                f"  [dim]{t['timestamp'][:19]}  {t['from_agent']:12s} → {t['to_agent']:12s}  {t['reason']}[/dim]"
            )

    # ── Optionally save to file ───────────────────────────────────────────────
    if output_file:
        output_file.write_text(report, encoding="utf-8")
        _console.print(f"\n[green]Report saved to:[/green] {output_file}")

    return report


def _print_node_summary(node_name: str, output: dict) -> None:
    """Pretty-print a brief summary of what a node returned."""
    if "subtasks" in output:
        n = len(output["subtasks"])
        _console.print(f"    subtasks created: {n}", highlight=False)
    if "research_notes" in output:
        notes = output["research_notes"]
        total_chars = sum(len(n.get("content", "")) for n in notes)
        _console.print(f"    notes appended: {len(notes)}  ({total_chars} chars)", highlight=False)
    if "next_action" in output:
        _console.print(f"    routing → {output['next_action']}", highlight=False)
    if "final_report" in output and output["final_report"]:
        _console.print(f"    report: {len(output['final_report'])} chars", highlight=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        _console.print(
            "[red]Usage:[/red] python -m src.main \"<your research query>\"\n"
            "Example: python -m src.main \"Who are the top AI coding assistant competitors in 2025?\""
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    output_path: Optional[Path] = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = Path(sys.argv[idx + 1])

    stream_mode = "--stream" in sys.argv

    try:
        run_research(query, output_file=output_path, stream=stream_mode)
    except EnvironmentError as e:
        _console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        _console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)
