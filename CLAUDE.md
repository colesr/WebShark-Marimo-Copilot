# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ds-copilot` is a **plan-first AI copilot for [Marimo](https://marimo.io) reactive data-science notebooks**, built on top of [marimo-pair](https://github.com/marimo-team/marimo-pair). Where marimo-pair gives an autonomous agent that edits cells directly, ds-copilot adds a *propose → human-approve → apply* gate around schema-grounded plans, with a leakage audit and append-only decision provenance. It's a feasibility prototype; the full week-by-week plan and the rationale for the Week-0 pivot live at `~/.claude/plans/this-is-a-plan-enumerated-adleman.md`. `MEMO.md` is the latest internal status memo — read it for current thinking and the "next bet."

## Environment & commands

This is a Windows / PowerShell project. **Nothing is on PATH** — always invoke tools through the project venv at `.\.venv\Scripts\`.

```powershell
# Setup
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Tests (54 tests; run fully offline — no API key or claude CLI needed)
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m pytest tests/test_agent.py -v          # one file
.\.venv\Scripts\python.exe -m pytest tests/test_agent.py::test_name  # one test

# Lint
.\.venv\Scripts\ruff.exe check .          # line-length 100, target py311

# Run a notebook (ALWAYS pass --no-token so the marimo-pair skill can discover the session)
.\.venv\Scripts\marimo.exe edit notebooks\01_demo_churn.py --no-token
```

`MARIMO_CHEATSHEET.md` is the full marimo-CLI reference. Common gotcha: launching with bare `marimo` (wrong Python) gives `ModuleNotFoundError: No module named 'ds_copilot'` — always use `.\.venv\Scripts\marimo.exe`.

## Two LLM backends

`agent.plan()` has two interchangeable backends, selected by `backend=`:

- **`claude-cli`** (default) — shells out to the local `claude` CLI in headless mode (`claude --print --output-format json --json-schema ...`). Bills against the user's **Claude Code subscription**, so no `ANTHROPIC_API_KEY` is needed. The CLI invocation **deliberately strips `ANTHROPIC_API_KEY` from the child env** (`agent.py` `_plan_via_claude_cli`) — if the key is present the CLI prefers API auth, which is exactly what this backend exists to avoid. This is the unlock that lets the demo run with zero API credits.
- **`anthropic`** — Anthropic Python SDK; requires `ANTHROPIC_API_KEY` (read from `.env`). Uses prompt caching.

Both paths use **forced tool use** (`submit_plan` tool) for structured output and populate a `CostReport`. Default model is `claude-opus-4-7`, default effort `medium` (see README's bench table for why Opus+medium is counterintuitively the *fastest* — Claude Code's ~30K-token session cache write/read dominates short calls).

## Architecture — the plan-then-approve pipeline

The data flow is: **DataFrame → Profile → Plan → Widget → applied cells**, with a decision logged at each gate.

1. **`profiler.py`** — `profile(df, target=...)` produces a frozen pydantic `Profile` (per-column dtype, null rate, cardinality, distribution stats, **leakage hints**). Accepts polars or pandas (converts to polars internally). This is the agent's *grounded context* — the system prompt forbids inventing column names, so every plan is anchored to this profile. Leakage hints are heuristic: unique-per-row non-float columns (likely IDs) and numeric columns with `|corr| > 0.9` against the target.

2. **`agent.py`** — `plan(goal, profile, ...)` → frozen pydantic `Plan` of `ProposedCell`s. When a `target` was named, the plan carries a **`LeakageAudit`** (`columns_to_drop`, `transformations_to_audit`, `training_safe`, `rationale`). The large `SYSTEM_PROMPT` and `PLAN_TOOL` JSON schema in this file are the contract — both backends send them. The agent **proposes, never executes**.

3. **`ui.py`** — `planner_widget(df, goal, target=...)` builds the profile, calls `plan()`, and returns a `PlannerWidget` that renders per-cell approval checkboxes. `await widget.apply()` (called from a *downstream* cell) materializes only the checked cells. **Leakage gate:** if `LeakageAudit.training_safe is False` and the user selected a cell matching `_TRAINING_PATTERN` (`.fit(`, `train_test_split`, etc.), Apply is **blocked** unless an override checkbox is ticked.

4. **`decisions.py`** — append-only JSONL log at `.ds_copilot/decisions.jsonl` (relative to CWD). Every widget interaction writes three events: `plan_requested`, `plan_returned`, `cells_applied` (records accepted vs. rejected cells and whether the override was used). Read back with `ds_copilot.history()`.

5. **`dag.py`** — `parse_notebook(path)` AST-parses a marimo `.py` (no kernel required), inferring the cell dependency graph from each `@app.cell` function's args (reads) and return tuple (defines). `dag_widget(path)` renders it as mermaid. Independent of the planning pipeline — a standalone viz tool.

### The `apply()` workaround (important, fragile)

`PlannerWidget.apply()` cannot create cells directly: `marimo._code_mode.get_context()` only works when code enters the kernel through the `/api/kernel/execute` HTTP endpoint, but the widget runs *inside* the kernel. The workaround is a **daemon thread that POSTs a cell-creation script back to `http://127.0.0.1:2718/api/kernel/execute`** so it runs in the proper inbound-HTTP context after the current cell finishes. POST errors are swallowed (logged to a temp file, not raised) to avoid crashing the daemon thread. This is the first thing to revisit if the wedge becomes a real product.

## Testing notes

Tests run **fully offline** — `test_agent.py` monkeypatches `subprocess.run` / the Anthropic client with fakes (`_FakeResponse`, `_FakeToolUseBlock`), so no network, API key, or `claude` CLI is needed. `asyncio_mode = "auto"` is set, so `async def test_*` works without decorators. All pydantic models are `frozen=True` — construct new instances rather than mutating.

`scripts/` holds non-test, network-dependent tools: `bench_plan_speed.py` (latency/quality across model+effort combos — this is what produced the README bench table), `smoke_test_plan.py`, `smoke_test_leakage.py`. These hit the real backend and are run manually, not under pytest.
