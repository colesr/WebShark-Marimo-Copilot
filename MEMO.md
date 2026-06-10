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

## Update — 2026-06-10

Three weeks on. The open questions above have partial answers; recording them here rather than rewriting the original.

**Q2 — does the LeakageAudit catch real leaks, or is the synthetic demo all it's good for? ANSWERED: yes, on real data — now n=2 across two distinct leak classes.**

*Run 1 — temporal leak (Favorita).* Committed in `690042b` (`notebooks/02_favorita_realdata.py` + `scripts/validate_favorita*.py`). Ran the planner on the real Kaggle Favorita store-sales set with a *deliberately neutral goal* — no hint about time-splitting — where the danger is **temporal** (a random train/test split leaks future→past) and the target `sales` is continuous, so the profiler's `|corr|` and target-rate heuristics stay silent. The audit raised the temporal leak on its own reasoning, with no heuristic to lean on.

*Run 2 — semantic post-outcome leak (Bank Marketing).* `scripts/validate_bank_marketing.py`, UCI Bank Marketing (45,211 rows, via `ucimlrepo`). A *different* failure mode: the `duration` column (length of the last contact call) is only known *after* the call ends — the same moment the subscription outcome is decided — so the dataset authors themselves say it must be discarded for a realistic model. It's heuristic-invisible (not unique-per-row, target-corr well under 0.9; the profiler flagged nothing). On a neutral goal, the agent independently dropped `duration` with the correct rationale ("only observable after a call concludes and effectively encodes the outcome"), confined scaling to a train-only Pipeline, and set `training_safe=True` only after dropping it.

Two real datasets, two leak classes (temporal + post-outcome), zero heuristic help, neutral goals both times — the audit caught both. This is no longer the synthetic demo pattern-matching its own planted clue. Caveat still worth honoring: both were single goal-phrasings, and run 2 took ~85s at `effort="high"` (vs the ~37s medium default) — a real hit on Q1. A third run on a *non-leaky* clean real set would be the useful next check: does it avoid false alarms, or does it cry leak on safe data?

**The "3+ unprompted uses" bet — partial.** Beyond Favorita there are two untracked scratch runs that count as genuine unprompted use:
- `notebooks/scratch.py` — Seaborn penguins, baseline `species` classifier through `planner_widget` (encode → stratified split → logreg → confusion matrix → CV). Plan applied end-to-end.
- `notebook.py` (repo root) — `datasets/stores.csv`, profiling + a `perf`-column investigation.

Both deliberately left untracked, per the W5 instruction to keep `notebooks/` clean and just *use* the thing. So the count is ~3 real uses (Favorita + 2 scratch). That clears the bar I set — leaning toward "the wedge is real" — but the scratch uses were lightweight (profiling / one-shot baseline), not the multi-day sticky use that would be the strongest signal. Q1 (is the 30–45s wait acceptable when unexpected?) and Q3 (do you reach for the decision log, or is `git log` enough?) are still unanswered — I haven't caught myself opening `history()` unprompted yet.

**Re-stated bet.** The leakage audit has earned its own follow-up: it cleared the real-data test that was the whole point of W5, and it's the component most likely to have standalone value. Next concrete step is *not* more building — it's 2–3 more neutral-goal audit runs on different real datasets to turn n=1 into a pattern, and one honest check of whether the decision log ever gets reached for. If the audit holds across those and the log stays unused, the sharpest product is the **profiler + leakage audit as a standalone library**, with the copilot as the delivery vehicle rather than the headline.
