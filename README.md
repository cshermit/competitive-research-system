# Competitive Research Multi-Agent System

A production-grade **multi-agent orchestration system** built with **LangGraph** that produces competitive intelligence reports. It features an explicit state machine, failure recovery with automatic query revision, and a full audit log of every agent transition and tool call.

---

## Architecture

```
┌─────────────┐
│    START    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   PLANNER   │  Breaks the query into 2–5 focused subtasks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  RESEARCHER │◄─────────────────────────────────────┐
│             │  Tavily search + URL fetch + LLM     │
└──────┬──────┘                                      │
       │                                             │
       ▼                                             │
┌─────────────┐   next_action == "research"          │
│  VALIDATOR  │─────────────────────────────────────►┘
│  (Planner)  │
└──────┬──────┘
       │ next_action == "synthesize"
       ▼
┌─────────────┐
│ SYNTHESIZER │  Compiles Markdown competitive briefing
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     END     │
└─────────────┘
```

### Failure-recovery loop

If the Researcher returns empty or thin results (< 150 characters), the **Planner/Validator** does NOT pass garbage downstream. Instead:

1. It invokes the LLM to **revise the search query** with different keywords or angle.
2. Routes back to the Researcher for a retry.
3. After `MAX_RETRIES_PER_SUBTASK` failed attempts, the subtask is marked `failed` and skipped — the Synthesizer works with whatever data was collected.

This prevents error compounding, the #1 failure mode in naive agent pipelines.

---

## Project structure

```
competitive_research/
├── src/
│   ├── agents/
│   │   ├── planner.py      # Plan + Validate nodes (control plane)
│   │   ├── researcher.py   # Research node (search + fetch + distil)
│   │   └── synthesizer.py  # Synthesize node (final report)
│   ├── graph/
│   │   ├── state.py        # TypedDict state schema + reducers
│   │   └── graph.py        # StateGraph wiring + conditional edges
│   ├── tools/
│   │   ├── web_search.py   # Tavily wrapper
│   │   └── url_fetcher.py  # httpx URL fetcher
│   ├── utils/
│   │   ├── config.py       # Pydantic settings (env vars)
│   │   └── logger.py       # Rich console + JSONL file logger
│   └── main.py             # CLI entry point
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── test_planner.py     # Planner unit tests
│   ├── test_researcher.py  # Researcher unit tests
│   ├── test_synthesizer.py # Synthesizer unit tests
│   └── test_graph.py       # Integration tests (full graph)
├── logs/                   # JSONL audit logs (auto-created at runtime)
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/competitive-research-system.git
cd competitive-research-system

python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> Requires Python 3.10+. Tested on CPython 3.13.

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY=sk-ant-...
#   TAVILY_API_KEY=tvly-...
```

Get a free Tavily key at [app.tavily.com](https://app.tavily.com) (1000 searches/month free tier).

### 3. Run a research query

```bash
python -m src.main "Who are the top competitors of Notion in 2025?"
```

**Options:**

```bash
# Save report to a Markdown file
python -m src.main "Stripe vs Adyen competitive analysis" --output report.md

# Print intermediate node updates (stream mode)
python -m src.main "OpenAI GPT-4o competitors" --stream
```

> On Windows, set `PYTHONIOENCODING=utf-8` if you see encoding errors in the console.

### 4. Run tests

```bash
pip install -r requirements-dev.txt
pytest                          # all tests with coverage
pytest tests/test_planner.py    # single module
pytest -x                       # stop on first failure
```

All tests mock external services (LLM, Tavily, httpx), so no API keys are needed for the test suite.

---

## Programmatic usage

```python
from src.main import run_research

report = run_research(
    "Compare Vercel vs Netlify vs Cloudflare Pages for frontend hosting",
    max_retries=3,
)
print(report)
```

Or use the graph directly:

```python
from src.graph.graph import compile_graph
from src.graph.state import initial_state

app = compile_graph()
state = initial_state("Figma vs Sketch competitive analysis", max_retries=2)

result = app.invoke(state)
print(result["final_report"])

# Or stream node-by-node updates
for chunk in app.stream(state):
    for node_name, output in chunk.items():
        print(f"Node '{node_name}' completed")
```

---

## State machine

The `ResearchState` TypedDict is the single source of truth. Key fields:

| Field                    | Type                    | Description                                     |
| ------------------------ | ----------------------- | ----------------------------------------------- |
| `original_query`         | `str`                   | The user's research question                    |
| `subtasks`               | `List[SubTask]`         | Planned research tasks                          |
| `current_subtask_index`  | `int`                   | Pointer into `subtasks`                         |
| `research_notes`         | `List[ResearchNote]`    | Accumulated evidence (append-only)              |
| `transitions`            | `List[AgentTransition]` | Full audit trail (append-only)                  |
| `next_action`            | `str`                   | Routing hint: `"research"` or `"synthesize"`    |
| `final_report`           | `Optional[str]`         | Markdown report (set by the Synthesizer)        |
| `status`                 | `str`                   | `planning → researching → synthesizing → complete` |

List fields use `Annotated[List[...], operator.add]` — LangGraph's reducer ensures safe accumulation across nodes without race conditions.

---

## Logging

Every run produces two log streams:

**Console** — Rich-formatted, coloured, human-readable:

```
[12:34:01] INFO  Planning research for: 'Who are OpenAI competitors?'
[12:34:03] INFO  Plan created: 4 subtasks | strategy: Cover market, models, pricing, funding
[12:34:03] INFO  [transition]  planner → researcher  (subtasks ready)
[12:34:05] INFO  [tool_call]   tavily_search  query='OpenAI competitors 2025'  max_results=5
[12:34:06] INFO  [tool_result] tavily_search → 5 results  (top score: 0.94)
[12:34:09] WARN  [retry]       subtask=t2  attempt=1  reason='insufficient research results'
```

**File** — JSONL at `logs/agent_run_YYYYMMDD.jsonl`:

```json
{"ts":"2025-01-15T12:34:05Z","level":"INFO","logger":"agent.researcher","message":"[tool_call] tavily_search","agent":"researcher","event":"tool_call","data":{"tool":"tavily_search","query":"OpenAI competitors 2025","max_results":5}}
```

Parse logs for observability:

```bash
cat logs/agent_run_20250115.jsonl | jq 'select(.event == "retry")'
cat logs/agent_run_20250115.jsonl | jq 'select(.event == "transition") | {from: .data.from, to: .data.to, ts: .ts}'
```

---

## Configuration reference

| Variable                    | Default                     | Description                                     |
| --------------------------- | --------------------------- | ----------------------------------------------- |
| `ANTHROPIC_API_KEY`         | —                           | **Required**                                    |
| `TAVILY_API_KEY`            | —                           | **Required**                                    |
| `LLM_MODEL`                 | `claude-sonnet-4-20250514`  | Anthropic model string                          |
| `MAX_SUBTASKS`              | `4`                         | Max subtasks the Planner creates                |
| `MAX_RETRIES_PER_SUBTASK`   | `2`                         | Retries before marking a subtask failed         |
| `MAX_SEARCH_RESULTS`        | `5`                         | Tavily results per search                       |
| `MAX_URL_CONTENT_CHARS`     | `4000`                      | Characters fetched per URL                      |
| `LOG_LEVEL`                 | `INFO`                      | `DEBUG / INFO / WARNING / ERROR`                |
| `LOG_DIR`                   | `./logs`                    | Directory for JSONL log files                   |

---

## Design decisions

**Why LangGraph?** Explicit state machines make control flow visible and debuggable. Compared to ReAct-style agents, every transition is logged and deterministic.

**Why a separate Validator node?** The Planner is the "control plane" — it owns the decision of whether to retry, move on, or synthesize. Keeping this logic separate from the Researcher prevents the Researcher from needing to know about retries.

**Why `Annotated[List, operator.add]`?** LangGraph merges node outputs into the state. For list fields we want *accumulation*, not *replacement*. The reducer ensures research notes from multiple subtasks are never lost.

**Why not async?** Tavily's Python client is synchronous. Adding async complexity for a CLI tool is premature — upgrade to `AsyncTavilyClient` + async nodes if you need concurrent subtask execution.

---

## Extending the system

**Add a new tool** — create `src/tools/my_tool.py` and import it in `researcher.py`.

**Add a new agent** (e.g. a Fact-Checker between Researcher and Synthesizer):

1. Create `src/agents/fact_checker.py`.
2. Register it: `builder.add_node("fact_check", fact_check_node)` in `graph.py`.
3. Rewire edges: `validate → fact_check → synthesize`.

**Parallel subtask research** — use LangGraph's `Send` API to fan out subtasks concurrently.

---

## License

MIT — see [LICENSE](LICENSE).
