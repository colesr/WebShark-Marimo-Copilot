"""Compare plan() latency + quality across model/effort combinations.

Runs the same leakage-prone churn scenario with multiple (model, effort)
pairs, prints time-to-result and the key LeakageAudit fields so we can
judge quality at a glance.

Run with:
    .\\.venv\\Scripts\\python.exe scripts\\bench_plan_speed.py
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


CONFIGS = [
    ("claude-haiku-4-5", "medium"),
    ("claude-sonnet-4-6", "low"),
    ("claude-sonnet-4-6", "medium"),
    ("claude-opus-4-7", "medium"),  # baseline -- known ~45s
]


def make_df() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    n = 1_000
    churned = rng.choice([0, 1], n, p=[0.74, 0.26])
    return pl.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(20, 120, n).round(2),
            "contract_type": rng.choice(["m2m", "1y", "2y"], n),
            "churn_score": (churned * 100.0 + rng.normal(0, 0.01, n)).round(3),
            "churned": churned,
        }
    )


def main() -> None:
    load_dotenv()
    df = make_df()
    p = profile(df, target="churned")

    goal = (
        "Build a baseline binary classifier predicting `churned`. "
        "Encode categoricals, split train/test, fit a logistic "
        "regression, report accuracy and ROC AUC."
    )
    existing = [
        "import marimo as mo",
        "import polars as pl",
        "import numpy as np",
        "df = synthetic churn data with a churn_score column derived from churned",
    ]

    print(f"{'model':<22} {'effort':<8} {'time':>7}  {'cells':>5}  audit")
    print("-" * 100)

    for model, effort in CONFIGS:
        started = time.monotonic()
        try:
            result = plan(
                goal=goal,
                profile=p,
                existing_cells=existing,
                model=model,
                effort=effort,
            )
            elapsed = time.monotonic() - started
            audit = result.leakage_audit
            drops = audit.columns_to_drop if audit else []
            safe = audit.training_safe if audit else None
            caught = (
                "yes"
                if audit and "churn_score" in audit.columns_to_drop
                else "NO"
            )
            print(
                f"{model:<22} {effort:<8} {elapsed:>6.1f}s  "
                f"{len(result.cells):>5}  caught={caught}  safe={safe}  drops={drops}"
            )
        except Exception as e:
            elapsed = time.monotonic() - started
            print(f"{model:<22} {effort:<8} {elapsed:>6.1f}s  FAIL  {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
