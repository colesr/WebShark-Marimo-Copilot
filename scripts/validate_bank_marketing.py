"""One-off: run plan() on the real UCI Bank Marketing data and print the
LeakageAudit so we can see whether the agent catches a *semantic
post-outcome* target leak without being told to. Not a pytest test.

This is a different leak class from validate_favorita.py (which tests a
temporal leak). Here the danger is the `duration` column: the duration of
the last contact call is only known *after* the call ends -- which is the
same moment the subscription outcome (`y`) is decided. The dataset authors
themselves warn it must be discarded for a realistic predictive model.
It is invisible to the profiler's heuristics: it is not unique-per-row,
and its correlation with the binary target stays well under the 0.9
auto-flag threshold. So the burden is entirely on the agent's reasoning.

The goal below is deliberately NEUTRAL -- it does not mention `duration`
or leakage. Whether the audit raises it is an honest test.

Needs `ucimlrepo` (pip install ucimlrepo) and network. Run with:
    .\\.venv\\Scripts\\python.exe scripts\\validate_bank_marketing.py
"""

from __future__ import annotations

import sys
import time

import polars as pl
from ucimlrepo import fetch_ucirepo

from ds_copilot.agent import plan
from ds_copilot.profiler import profile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    bank = fetch_ucirepo(id=222)
    pdf = bank.data.features.copy()
    pdf["y"] = bank.data.targets.iloc[:, 0]
    df = pl.from_pandas(pdf)
    print(f"Loaded Bank Marketing: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"columns: {df.columns}")

    p = profile(df, target="y")
    flagged = [c.name for c in p.columns if c.leakage_hints]
    print(f"Profiler flagged (heuristic): {flagged}")
    dur = next((c for c in p.columns if c.name == "duration"), None)
    if dur is not None:
        print(f"`duration` heuristic hints (expect none): {dur.leakage_hints}")

    print("\nCalling plan() with a NEUTRAL goal (no leakage / duration hint)...")
    started = time.monotonic()
    result = plan(
        goal=(
            "Build a baseline classifier to predict whether a client will "
            "subscribe to a term deposit (`y`). Encode the categorical "
            "columns, do a stratified train/test split, fit a logistic "
            "regression baseline, and report accuracy, precision, recall, "
            "and a confusion matrix on the held-out set."
        ),
        profile=p,
        existing_cells=[
            "import polars as pl",
            (
                "df = UCI Bank Marketing frame; one row per marketing phone "
                "call. Columns: age, job, marital, education, default, "
                "balance, housing, loan, contact, day_of_week, month, "
                "duration, campaign, pdays, previous, poutcome, y (target: "
                "did the client subscribe, yes/no)"
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

    # Litmus: did it catch the `duration` post-outcome leak?
    text = " ".join(
        [audit.rationale] + list(audit.transformations_to_audit) if audit else []
    ).lower()
    caught_duration_drop = bool(
        audit and "duration" in [c.lower() for c in audit.columns_to_drop]
    )
    mentioned_duration = "duration" in text
    print("\n--- litmus ---")
    print(f"  dropped `duration`?            {caught_duration_drop}")
    print(f"  reasoned about `duration`?     {mentioned_duration}")
    print(f"  training_safe:                 {audit.training_safe if audit else 'n/a'}")
    print(f"  plan cell count:               {len(result.cells)}")


if __name__ == "__main__":
    main()
