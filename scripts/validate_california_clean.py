"""One-off: the FALSE-ALARM control. Every prior real-data run
(validate_favorita*, validate_bank_marketing) used a dataset that DOES
contain a leak. This one deliberately does not: California Housing is a
clean regression set where every feature is a census-district attribute
measured contemporaneously with the target (median house value) -- none
is post-outcome, target-derived, or an ID. There is genuinely nothing to
drop.

The question is the inverse of the other scripts: does the audit stay
quiet on safe data, or does it cry wolf? A correct result is
training_safe=True with an empty columns_to_drop. The only legitimate
note is the standard "fit the scaler train-only" hygiene caution, which
is not a false alarm. Dropping any feature as "leaky", or
training_safe=False, IS the false alarm we are testing for.

Run with:
    .\\.venv\\Scripts\\python.exe scripts\\validate_california_clean.py
"""

from __future__ import annotations

import sys
import time

import polars as pl
from sklearn.datasets import fetch_california_housing

from ds_copilot.agent import plan
from ds_copilot.profiler import profile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    bunch = fetch_california_housing(as_frame=True)
    pdf = bunch.frame  # features + MedHouseVal target column
    df = pl.from_pandas(pdf)
    print(f"Loaded California Housing: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"columns: {df.columns}")

    p = profile(df, target="MedHouseVal")
    flagged = [c.name for c in p.columns if c.leakage_hints]
    print(f"Profiler flagged (heuristic, expect none): {flagged}")
    print(f"target_rate (None expected, target is continuous): {p.target_rate}")

    print("\nCalling plan() with a NEUTRAL goal on genuinely clean data...")
    started = time.monotonic()
    result = plan(
        goal=(
            "Build a baseline regression model to predict median house "
            "value (`MedHouseVal`) for California census districts. Use the "
            "available features, do a train/test split, fit a regression "
            "model, and report RMSE and R^2 on the held-out set."
        ),
        profile=p,
        existing_cells=[
            "import polars as pl",
            (
                "df = California Housing frame; one row per census block "
                "group. Columns: MedInc, HouseAge, AveRooms, AveBedrms, "
                "Population, AveOccup, Latitude, Longitude, MedHouseVal "
                "(target: median house value in $100k)"
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

    # Litmus: the INVERSE of the leak scripts -- we want NO false alarm.
    dropped = list(audit.columns_to_drop) if audit else []
    no_false_drop = len(dropped) == 0
    safe = bool(audit and audit.training_safe)
    print("\n--- litmus (false-alarm control) ---")
    print(f"  training_safe == True?         {safe}")
    print(f"  columns_to_drop empty?         {no_false_drop}  ({dropped})")
    verdict = "PASS (no false alarm)" if (safe and no_false_drop) else "FALSE ALARM"
    print(f"  verdict:                       {verdict}")
    print(f"  plan cell count:               {len(result.cells)}")


if __name__ == "__main__":
    main()
