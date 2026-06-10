"""Week 5 demo: plan-then-approve on REAL data (Favorita store sales).

Unlike 01_demo_churn.py — which plants an obvious |corr|=1.0 leak the
profiler's heuristic is built to catch — this notebook runs the planner on
a genuine Kaggle forecasting dataset where:

  - the target `sales` is CONTINUOUS, so target_rate and the numeric-corr
    leakage heuristic stay quiet, and
  - the real leakage risk is TEMPORAL: a random train/test split leaks the
    future into the past, and joins to oil/transactions/holidays on `date`
    can pull lookahead features into a row.

The open question (from MEMO.md): does the LeakageAudit reason about
time-based leakage on its own, when the heuristic can't help? The goal
below is deliberately NEUTRAL — it does NOT hint at time-splitting, so
whether the audit raises it is an honest test.

Run from the project root with:
    .\\.venv\\Scripts\\marimo.exe edit notebooks/02_favorita_realdata.py --no-token
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Plan-then-approve on real data: Favorita store sales

    A real Kaggle time-series forecasting set. The target `sales` is
    continuous and the danger is *temporal*, not correlational — so the
    profiler's automatic hints will be quiet and the burden falls on the
    agent's reasoning. We give it a neutral goal and read the audit.
    """)
    return


@app.cell
def _(pl):
    # 3M rows; loads in a few seconds. profile() samples to 200k internally.
    df = pl.read_csv("datasets/train.csv", try_parse_dates=True)
    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"### {df.shape[0]:,} rows × {df.shape[1]} cols — "
        f"date range {df['date'].min()} → {df['date'].max()}"
    )
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1: profile the schema

    Expect `id` to light up as a unique-per-row ID. `sales` is continuous,
    so there is no target-rate and the |corr| heuristic should stay silent —
    that is the point. The heuristic cannot see temporal leakage.
    """)
    return


@app.cell
def _(df):
    from ds_copilot import profile_widget

    profile_widget(df, target="sales")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2: plan a baseline forecast

    The goal is intentionally neutral — it asks for "a train/validation
    split", not a *time-based* one. A naive planner reaches for a random
    `train_test_split`; a good one flags that this leaks the future. Read
    the LeakageAudit's `transformations_to_audit` to see which we got.

    First call blocks ~40-90s.
    """)
    return


@app.cell
def _(df):
    from ds_copilot import planner_widget

    planner = planner_widget(
        df,
        goal=(
            "Build a baseline model to forecast daily unit sales (`sales`) "
            "for each store and product family. Encode the categorical "
            "columns, create a train/validation split, fit a regression "
            "model, and report error on the validation set."
        ),
        target="sales",
        existing_cells=[
            "import marimo as mo",
            "import polars as pl",
            (
                "df = Favorita store-sales training frame with columns: "
                "id, date, store_nbr, family, sales, onpromotion"
            ),
        ],
        # medium verified to catch temporal + concurrent-aggregate leaks
        # 3/3 here, ~28% faster than high (see scripts/validate_favorita_repeat.py).
        effort="medium",
    )
    return (planner,)


@app.cell
def _(planner):
    planner
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What to watch for (the litmus test)

    - **`id` dropped?** The easy catch — both the heuristic and the agent
      should list it under `columns_to_drop`.
    - **Temporal split flagged?** The hard catch — does
      `transformations_to_audit` say to split by `date` rather than
      randomly? This is the question the synthetic demo could not test.
    - **Lookahead from `onpromotion`/joins mentioned?**
    - **`training_safe` set thoughtfully** given a neutral goal that did
      not pre-solve the time-split problem.

    If it catches the temporal trap, the audit is more than a one-trick
    heuristic. If it only parrots "drop the ID", that is also a finding.
    """)
    return


@app.cell
async def _(planner):
    result = await planner.apply()
    return (result,)


@app.cell
def _(mo, result):
    mo.md(f"""
    ```\n{result}\n```
    """)
    return


if __name__ == "__main__":
    app.run()
