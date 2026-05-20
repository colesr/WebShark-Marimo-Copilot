# ds-copilot — internal memo, 2026-05-20

## What we tried to build

A wedge MVP for a Claude-powered copilot inside Marimo reactive notebooks. Original framing was "DAG-aware copilot" — schema-grounded, multi-cell planning, dependency-aware refactoring.

## What changed in Week 0

The Marimo team shipped [`marimo-pair`](https://github.com/marimo-team/marimo-pair) in early 2026 — a Claude Code skill that drops an external agent inside a running Marimo session. It already had cell CRUD, package install, implicit DAG awareness. The original wedge was *already built*, for free, by the upstream team.

The pivot was the project's most important decision. Rather than abandon, we re-scoped to the gaps `marimo-pair` doesn't address — and specifically to **practitioners doing auditable work** (vs. agentic explorers, who are already well-served).

## What got built (5 commits, 54 tests)

- **`profiler.py`** — structured schema profile (`Profile`, `ColumnProfile`, `LeakageHint`). The agent's grounded context.
- **`agent.py`** — `plan()` returns a pydantic `Plan` of `ProposedCell`s plus a `LeakageAudit`. Two backends: `claude-cli` (default; uses your Claude Code subscription, no API credits needed) and `anthropic` (SDK; needs `ANTHROPIC_API_KEY`). Per-call `CostReport` populated on both paths.
- **`ui.py`** — `planner_widget(df, goal, target=...)` shows the plan with per-cell checkboxes, a leakage-audit pill, and a submit form. `await widget.apply()` materializes only approved cells (via a daemon-thread POST to `/api/kernel/execute` — workaround for `code_mode` requiring inbound-HTTP context).
- **`decisions.py`** — append-only JSONL log of every plan request, response, accept/reject at `.ds_copilot/decisions.jsonl`. `history()` reads back.
- **`dag.py`** — AST-parses a marimo notebook and renders the cell graph as mermaid via `mo.mermaid`. Useful past ~20 cells.
- Two demos: side-by-side autonomous-vs-plan-first (W1+W2) and a deliberate-leak demo (W3) that the agent's `LeakageAudit` catches end-to-end.

## What we learned — gate outcomes

- **W2 (plan vs autonomous):** user verdict *"plan felt grounded, approval gate worth it, would use this."* PASS.
- **W4 (excited to use on real work):** *"yes, would reach for it on real work."* PASS. Top wish: faster planning.
- **Speed fix landed**: bench showed all four model/effort combos catch the deliberate leak; default flipped from `high` → `medium` and typical wait dropped from ~90s to ~37s with no quality loss. (Counterintuitively, Opus + medium was fastest — Claude Code's session overhead dominates short calls.)
- **The `claude-cli` backend is the unlock**: when the user's API credits hit zero we found that the headless `claude --print --json-schema` invocation routes through their existing subscription. Saved the demo, may matter more broadly: it lets *anyone* with Claude Code use the system without funding API credits.
- **`code_mode` context limitation**: `marimo._code_mode.get_context()` only works on inbound HTTP through `/api/kernel/execute`. Cell-internal calls fail. Workaround (daemon-thread HTTP POST) is fine in practice but is the right place to start if the wedge becomes a real product.

## Honest verdict

The pivoted wedge is real. The agent produces materially better artifacts than `marimo-pair`'s autonomous mode for non-trivial, target-bearing work — specifically because the `LeakageAudit` and per-cell rationale make the agent's reasoning *legible and gate-able*. On the leakage demo, the agent independently identified `customer_id` and `churn_score` as columns to drop, explained why (citing the profile's `|corr|=1.0` hint), and set `training_safe=True` only after the plan dropped both before training.

What's *not* yet validated: whether this is sticky over weeks of real use, by someone who isn't me. Self-reported "would use" is a real signal; "used 10 times last week" would be a stronger one.

## Next bet

**Use it on actual work for two weeks.** The whole project points at this. The build is done; the experiment isn't.

Specifically: keep `notebooks/` clean of trial files for now and use `planner_widget` on a real DS problem you'd otherwise solve in Claude Code or marimo-pair. Watch for:

1. *Does the 30-45s wait still feel acceptable when you didn't expect it?*
2. *Does the `LeakageAudit` actually catch anything in real data, or is the synthetic-leak demo all it's good for?*
3. *Do you reach for the decision log when you want to remember what you did, or is `git log` enough?*

If after two weeks you've used it 3+ times unprompted, the wedge is real and the next investment is sharpening — better cell labels in the DAG, streaming the plan as it's generated, error-recovery when a proposed cell fails. If you haven't used it, the wedge is wrong; the most likely standalone value is the profiler + leakage audit as a separate library, not the copilot.

Either answer is fine. The point of decision gates is to know which one to act on.
