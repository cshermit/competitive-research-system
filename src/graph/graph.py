"""
src/graph/graph.py
──────────────────
Builds and compiles the LangGraph StateGraph.

State machine topology
──────────────────────

    ┌─────────────┐
    │    START    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │    plan     │  ← PlannerAgent.plan_node
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   research  │◄──────────────────────────────┐
    │             │  ← ResearcherAgent.research_node│
    └──────┬──────┘                                │
           │                                       │
           ▼                                       │
    ┌─────────────┐    next_action == "research"   │
    │  validate   │───────────────────────────────►┘
    │             │  ← PlannerAgent.validate_node
    └──────┬──────┘
           │ next_action == "synthesize"
           ▼
    ┌─────────────┐
    │  synthesize │  ← SynthesizerAgent.synthesize_node
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │     END     │
    └─────────────┘

Retry loop
──────────
When validate_node determines the Researcher returned insufficient data AND
retries remain, it revises the subtask query and routes back to "research".
This prevents garbage from flowing downstream.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.planner import plan_node, validate_node
from src.agents.researcher import research_node
from src.agents.synthesizer import synthesize_node
from src.graph.state import ResearchState
from src.utils.logger import AgentLogger

_log = AgentLogger("graph")


# ─────────────────────────────────────────────────────────────────────────────
# Router — inspects state to pick the next node after "validate"
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_validate(state: ResearchState) -> str:
    """
    Conditional edge function called after validate_node.
    Returns the name of the next node.
    """
    action = state.get("next_action", "synthesize")
    _log.debug(f"[router] validate → {action}")
    return action   # "research" | "synthesize"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct (but do not compile) the StateGraph.
    Useful for testing individual nodes in isolation.
    """
    builder = StateGraph(ResearchState)

    # ── Register nodes ───────────────────────────────────────────────────────
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("validate", validate_node)
    builder.add_node("synthesize", synthesize_node)

    # ── Static edges ─────────────────────────────────────────────────────────
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "validate")
    builder.add_edge("synthesize", END)

    # ── Conditional edge: validate → research | synthesize ───────────────────
    builder.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "research": "research",
            "synthesize": "synthesize",
        },
    )

    return builder


def compile_graph():
    """
    Build + compile the graph. Returns a runnable LangGraph app.

    Usage
    -----
    app = compile_graph()
    result = app.invoke(initial_state("Who are OpenAI's main competitors?"))
    print(result["final_report"])
    """
    _log.info("Compiling LangGraph state machine …")
    graph = build_graph()
    app = graph.compile()
    _log.info("Graph compiled ✓")
    return app
