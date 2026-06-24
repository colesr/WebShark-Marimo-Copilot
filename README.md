# WebShark Marimo ds-copilot 🦈

A plan-first AI copilot for [Marimo](https://marimo.io) reactive data-science notebooks. Built on top of [marimo-pair](https://github.com/marimo-team/marimo-pair) — fills the gaps marimo-pair doesn't: structured schema profiling, plan-then-approve workflows, leakage audits before any model fits, and append-only decision provenance.

Feasibility prototype. Full plan + decision gates at `~/.claude/plans/this-is-a-plan-enumerated-adleman.md`.

## What makes it different

- **Plan-then-approve**: Claude proposes a multi-cell plan (pydantic-validated). You see each cell's code + rationale + cost + warnings *before* anything runs, check the boxes for what you want, click Apply.
- **Schema-grounded**: a structured `Profile` (dtypes, null rates, cardinality, distribution stats, leakage hints) is the agent's grounded context. No hallucinated columns.
- **Leakage audit**: when you name a target column, every plan ships with a `LeakageAudit` (columns_to_drop, transformations_to_audit, training_safe bool, rationale). The widget blocks Apply on training cells when `training_safe=False`.
- **Decision provenance**: `.ds_copilot/decisions.jsonl` records every plan request, response, accept/reject. Read it back with `ds_copilot.history()`.
- **Static DAG viz**: `dag_widget("notebook.py")` AST-parses a notebook and renders the cell-dependency graph as mermaid. Useful when plan-first notebooks grow past ~20 cells.

## Quick start

```powershell
# create venv (Windows / PowerShell)
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# default backend is the Claude Code CLI (free via subscription, no API key needed).
# For the anthropic SDK backend, set ANTHROPIC_API_KEY:
Copy-Item .env.example .env  # edit with your key

# run the leakage demo
marimo edit notebooks/01_demo_churn.py --no-token
```

In a marimo cell:

```python
from ds_copilot import planner_widget

planner = planner_widget(
    df,
    goal="Build a baseline churn classifier with proper leakage handling.",
    target="churned",
)
planner
```

Then in a downstream cell:

```python
result = await planner.apply()
result
```

## Tuning the wait

Plan calls block while Claude generates. The default is **Opus 4.8 + medium effort**, ~30-45s on a warm Claude Code cache. Bench results below (run on Opus 4.7, the prior default) on a leakage-prone scenario, all configs caught the leak:

| model              | effort  | latency | cells |
|--------------------|---------|---------|-------|
| claude-opus-4-7    | medium  | ~37s    | 6     |
| claude-sonnet-4-6  | low     | ~52s    | 5     |
| claude-sonnet-4-6  | medium  | ~63s    | 4     |
| claude-haiku-4-5   | medium  | ~76s    | 6     |

Counterintuitively, Opus is often the fastest — Claude Code's session overhead (~30K-token cache write/read) dominates short calls, and Opus's adaptive thinking is most efficient at deciding when *not* to think hard.

Knobs (pass to `planner_widget` or `plan`):

- `effort="low"|"medium"|"high"|"xhigh"|"max"` — Opus 4.8 supports all five. `medium` is the default; `high` adds depth for hard problems; `low` is fastest.
- `model="claude-sonnet-4-6"` — half-price; quality holds for simple plans.
- `backend="anthropic"` — switches to the Python SDK + your `ANTHROPIC_API_KEY`. Faster subsequent calls with prompt caching once you have credits.

## Project layout

```
src/ds_copilot/
  profiler.py    # Profile, ColumnProfile, LeakageHint + profile_widget
  agent.py       # Plan, ProposedCell, LeakageAudit + plan() (claude-cli | anthropic)
  ui.py          # planner_widget; checkbox approval + apply gate on training cells
  decisions.py   # DecisionLog + history() (.ds_copilot/decisions.jsonl)
  dag.py         # parse_notebook + render_mermaid + dag_widget

notebooks/
  00_gut_check.py             # Week 0 — Marimo + mo.ui.chat sanity
  01_marimo_pair_trial.py     # Week 1+2 — autonomous (marimo-pair) vs plan-first side-by-side
  01_demo_churn.py            # Week 3+4 — leakage demo + DAG viz

tests/      # pytest, 50 tests
scripts/    # one-off scripts: live smoke tests, bench
```

## Status (2026-05-20)

| Week | Focus                          | Status | Gate                                |
|------|--------------------------------|--------|-------------------------------------|
| 0    | Gut-check                      | done   | pivoted — `marimo-pair` already serves the autonomous case |
| 1    | Schema profiler + marimo-pair  | done   | —                                   |
| 2    | Plan-then-approve agent + UI   | done   | PASS — "would use this"             |
| 3    | LeakageAudit + decision log    | done   | —                                   |
| 4    | DAG visualization              | done   | PASS — "would reach for it on real work" |
| 5    | Hardening + 2nd demo + 1pp memo| in progress | — |
| 6    | Buffer / spillover             | future | "could show this to 3 working DSs"  |

See `~/.claude/plans/this-is-a-plan-enumerated-adleman.md` for week-by-week scope and the rationale behind the W0 pivot.
