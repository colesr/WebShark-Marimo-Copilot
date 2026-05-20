# ds-copilot

A DAG-aware AI copilot for reactive data-science notebooks. The notebook substrate is [Marimo](https://marimo.io); the differentiated piece is a Claude-powered agent with first-class access to the notebook's DAG, cell outputs, and dataframe schemas.

This is a feasibility prototype — see `~/.claude/plans/this-is-a-plan-enumerated-adleman.md` for the full plan, decision gates, and scope.

## What makes the copilot novel

Existing AI-in-notebook tools (Jupyter AI, Hex Magic, Deepnote AI, Cursor) treat notebooks as files of code. This one treats the notebook as a *dataflow graph plus live state*:

1. **DAG-grounded suggestions** — agent knows which cells are upstream/downstream and can propose refactors that respect dependencies.
2. **Schema-aware code generation** — agent introspects live dataframes (columns, dtypes, sample rows) before generating Polars/SQL.
3. **Multi-cell planning** — given a goal ("explore churn drivers"), agent proposes a sequence of cells, shows the plan, executes on approval.

## Quick start

```powershell
# create venv (Windows / PowerShell)
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"

# set your Claude API key
Copy-Item .env.example .env
# edit .env and set ANTHROPIC_API_KEY

# run the gut-check notebook
marimo edit notebooks/00_gut_check.py
```

## Project layout

```
src/ds_copilot/
  agent.py              # Claude agent loop, tool definitions, prompt caching
  dag_introspect.py     # Read Marimo's DAG: cells, deps, outputs
  schema_introspect.py  # Inspect live dataframes (Polars/pandas)
  tools/                # Agent tools (read_cell, execute_cell, propose_cell, ...)
  prompts/              # System prompts, few-shots
  ui.py                 # Marimo UI element (chat panel) — main entry point

notebooks/
  00_gut_check.py       # Week 0 — does Marimo's extension surface host what we need?
  01_demo_churn.py      # Week 3 — agent helps with churn EDA
  02_demo_timeseries.py # Week 5 — forecast feasibility

tests/                  # pytest, against checked-in notebook fixtures
```

## Roadmap

6 weeks part-time, ~10–15 hrs/week. Each week ends with a decision gate. See the plan file for full detail.

| Week | Focus | Gate |
|------|-------|------|
| 0 | Setup, Marimo gut-check | Does `mo.ui.chat` host an agent with cell-modification tools? |
| 1 | DAG + schema introspection | — |
| 2 | Minimum viable agent loop | Is grounded agent visibly better than vanilla Claude? |
| 3 | Multi-cell planning | — |
| 4 | DAG-aware refactoring | Am I excited to use this on my own work? |
| 5 | Hardening + 2nd demo | — |
| 6 | Buffer + write-up | Could I show this to 3 working DSs without embarrassment? |

The point of the gates is to stop early if the thesis is wrong. Shipping isn't the goal — a confident answer to "is this wedge real" is.
