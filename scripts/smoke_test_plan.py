"""One-off smoke test for ds_copilot.agent.plan() against the real Claude API.

Run with:
    .\\.venv\\Scripts\\python.exe scripts\\smoke_test_plan.py

Builds a tiny synthetic churn dataframe, profiles it, asks Claude Opus 4.7
to produce a Plan, and prints the structured result. Not part of the test
suite -- this hits the live API and costs a real (small) amount of money.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from dotenv import load_dotenv

from ds_copilot.agent import plan
from ds_copilot.profiler import profile

# Windows consoles default to cp1252; Claude's prose can include arrows, dashes,
# etc. Reconfigure stdout so we don't crash on display.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _build_churn_df() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 1_000
    return pl.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(20, 120, n).round(2),
            "contract_type": rng.choice(["month-to-month", "1-year", "2-year"], n),
            "internet_service": rng.choice(["dsl", "fiber", "none"], n),
            "payment_method": rng.choice(
                ["bank-transfer", "credit-card", "electronic-check", "mailed-check"], n
            ),
            "churned": rng.choice([0, 1], n, p=[0.74, 0.26]),
        }
    )


def main() -> None:
    load_dotenv()

    df = _build_churn_df()
    p = profile(df, target="churned")
    print(
        f"Profile: rows={p.n_rows}  cols={p.n_cols}  "
        f"target={p.target}  target_rate={p.target_rate:.3f}"
    )
    flagged = [c.name for c in p.columns if c.leakage_hints]
    print(f"Flagged columns: {flagged}")
    print()
    print("Calling plan() against claude-opus-4-7 with adaptive thinking...")

    started = time.monotonic()
    result = plan(
        goal=(
            "Suggest 3 feature engineering ideas that might add signal for "
            "a churn classifier on this data. For each idea, propose a "
            "diagnostic cell that checks whether the new feature actually "
            "correlates with the target before training."
        ),
        profile=p,
        existing_cells=[
            "import marimo as mo",
            "import polars as pl",
            "import numpy as np",
            (
                "df : pl.DataFrame -- synthetic churn data, columns: "
                "customer_id, tenure_months, monthly_charges, contract_type, "
                "internet_service, payment_method, churned"
            ),
        ],
    )
    elapsed = time.monotonic() - started

    print(f"\nGot plan in {elapsed:.1f}s")
    print(f"  Goal:    {result.goal}")
    print(f"  Summary: {result.summary}")
    print()
    print(f"Cells ({len(result.cells)}):")
    for i, c in enumerate(result.cells, 1):
        print(f"  [{i}] {c.title}  ({c.est_cost})")
        print(f"      rationale: {c.rationale}")
        if c.depends_on:
            print(f"      depends_on: {c.depends_on}")
        if c.warnings:
            print(f"      warnings: {c.warnings}")
        print(f"      code preview: {c.code.strip().splitlines()[0][:100]}")
    if result.overall_warnings:
        print()
        print("Overall warnings:")
        for w in result.overall_warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
