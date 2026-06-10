"""One-off: run plan() on the real Favorita training data and print the
LeakageAudit so we can see whether the agent catches the *temporal* leak
(random split on time-series) without being told to. Not a pytest test.

Run with:
    .\\.venv\\Scripts\\python.exe scripts\\validate_favorita.py
"""

from __future__ import annotations

import sys
import time

import polars as pl

from ds_copilot.agent import plan
from ds_copilot.profiler import profile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    df = pl.read_csv("datasets/train.csv", try_parse_dates=True)
    print(f"Loaded train.csv: {df.shape[0]:,} rows x {df.shape[1]} cols")

    p = profile(df, target="sales")
    flagged = [c.name for c in p.columns if c.leakage_hints]
    print(f"Profiler flagged (heuristic): {flagged}")
    print(f"target_rate (None expected, sales is continuous): {p.target_rate}")

    print("\nCalling plan() with a NEUTRAL goal (no time-split hint)...")
    started = time.monotonic()
    result = plan(
        goal=(
            "Build a baseline model to forecast daily unit sales (`sales`) "
            "for each store and product family. Encode the categorical "
            "columns, create a train/validation split, fit a regression "
            "model, and report error on the validation set."
        ),
        profile=p,
        existing_cells=[
            "import polars as pl",
            (
                "df = Favorita store-sales training frame with columns: "
                "id, date, store_nbr, family, sales, onpromotion"
            ),
        ],
        effort="high",
    )
    print(f"Got plan in {time.monotonic() - started:.1f}s\n")

    audit = result.leakage_audit
    if audit is None:
        print("!! No leakage_audit returned (unexpected — target was named).")
    else:
        print(f"  target:           {audit.target}")
        print(f"  training_safe:    {audit.training_safe}")
        print(f"  columns_to_drop:  {audit.columns_to_drop}")
        print("  transformations_to_audit:")
        for t in audit.transformations_to_audit:
            print(f"    - {t}")
        print(f"\n  rationale:\n    {audit.rationale}")

    # Litmus checks
    text = " ".join(
        [audit.rationale] + list(audit.transformations_to_audit)
        if audit
        else []
    ).lower()
    caught_id = bool(audit and "id" in [c.lower() for c in audit.columns_to_drop])
    caught_time = any(
        kw in text for kw in ("temporal", "time-based", "by date", "chronolog", "future", "look-ahead", "lookahead", "leak the future")
    )
    print("\n--- litmus ---")
    print(f"  dropped `id`?              {caught_id}")
    print(f"  flagged temporal split?   {caught_time}")
    print(f"  plan cell count:          {len(result.cells)}")


if __name__ == "__main__":
    main()
