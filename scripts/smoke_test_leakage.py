"""Live smoke test: does the agent's LeakageAudit catch a deliberate leak?

Builds a synthetic churn dataset where `churn_score` is essentially the
target plus tiny noise, then asks the planner for a baseline classifier.
We expect:
    - audit.columns_to_drop contains "churn_score" (and probably "customer_id")
    - audit.training_safe reflects whether the plan drops the leak before
      training -- if it does, True; otherwise False.

Run with:
    .\\.venv\\Scripts\\python.exe scripts\\smoke_test_leakage.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import polars as pl
from dotenv import load_dotenv

from ds_copilot.agent import plan
from ds_copilot.profiler import profile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    load_dotenv()

    rng = np.random.default_rng(7)
    n = 1_000
    churned = rng.choice([0, 1], n, p=[0.74, 0.26])
    df = pl.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(20, 120, n).round(2),
            "contract_type": rng.choice(["m2m", "1y", "2y"], n),
            "churn_score": (churned * 100.0 + rng.normal(0, 0.01, n)).round(3),
            "churned": churned,
        }
    )

    p = profile(df, target="churned")
    print(f"Profile flagged: {[c.name for c in p.columns if c.leakage_hints]}")
    print()

    started = time.monotonic()
    print("Calling plan() with target='churned' and a deliberately leaky feature...")
    result = plan(
        goal=(
            "Build a baseline binary classifier predicting `churned`. "
            "Encode categoricals, split train/test, fit a logistic "
            "regression, report accuracy and ROC AUC."
        ),
        profile=p,
        existing_cells=[
            "import marimo as mo",
            "import polars as pl",
            "import numpy as np",
            "df = synthetic churn dataframe (1000 rows, includes a churn_score column derived from churned)",
        ],
        effort="medium",  # faster for smoke testing
    )
    elapsed = time.monotonic() - started
    print(f"\nGot plan in {elapsed:.1f}s")
    print()

    audit = result.leakage_audit
    if audit is None:
        print("FAIL: leakage_audit is None even though target was provided.")
        sys.exit(1)

    print(f"  target:           {audit.target}")
    print(f"  training_safe:    {audit.training_safe}")
    print(f"  columns_to_drop:  {audit.columns_to_drop}")
    print(f"  transforms:       {audit.transformations_to_audit}")
    print(f"  rationale:")
    print(f"    {audit.rationale}")
    print()

    caught_leak = "churn_score" in audit.columns_to_drop
    print(f"  caught churn_score leak? {'YES' if caught_leak else 'NO'}")

    # Check the cells -- does any training cell appear without churn_score being dropped first?
    has_fit = any(".fit(" in c.code for c in result.cells)
    has_drop = any(
        "churn_score" in c.code and ("drop" in c.code.lower() or "select" in c.code.lower())
        for c in result.cells
    )
    print(f"  plan includes training cell? {has_fit}")
    print(f"  plan drops/excludes churn_score before training? {has_drop}")


if __name__ == "__main__":
    main()
