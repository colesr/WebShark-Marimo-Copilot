"""Strengthen the W5 Favorita finding past n=1.

Two experiments, all at effort=high (matching the original n=1 run):

  A. Run the SAME base plan 3x. The prompt/profile are identical, so any
     variation is pure LLM nondeterminism. Question: is the temporal-split
     catch stable, or did we get one good roll?

  B. A DIFFERENT leak: join same-day `transactions` (a store/day aggregate)
     onto the training frame and ask to forecast sales. `transactions` is
     not knowable at forecast time, so using it as a feature is lookahead.
     Question: does the audit flag a concurrent-aggregate leak, distinct
     from the temporal-split and id catches?

Run with:
    .\\.venv\\Scripts\\python.exe scripts\\validate_favorita_repeat.py
"""

from __future__ import annotations

import sys
import time

import polars as pl

from ds_copilot.agent import plan
from ds_copilot.profiler import profile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEMPORAL_KW = (
    "temporal", "time-based", "by date", "chronolog", "future",
    "look-ahead", "lookahead", "leak the future", "most recent",
    "earliest", "time order", "time-order",
)

BASE_GOAL = (
    "Build a baseline model to forecast daily unit sales (`sales`) for each "
    "store and product family. Encode the categorical columns, create a "
    "train/validation split, fit a regression model, and report error on "
    "the validation set."
)


def _audit_text(audit) -> str:
    return " ".join([audit.rationale, *audit.transformations_to_audit]).lower()


def run_once(label: str, prof, goal: str, existing_cells: list[str], effort: str) -> dict:
    started = time.monotonic()
    result = plan(goal=goal, profile=prof, existing_cells=existing_cells, effort=effort)
    dur = time.monotonic() - started
    a = result.leakage_audit
    text = _audit_text(a) if a else ""
    row = {
        "label": label,
        "dur": dur,
        "training_safe": a.training_safe if a else None,
        "drops": a.columns_to_drop if a else [],
        "caught_id": bool(a and any(c.lower() == "id" for c in a.columns_to_drop)),
        "caught_temporal": any(k in text for k in TEMPORAL_KW),
        "caught_transactions": bool(
            a
            and (
                any("transaction" in c.lower() for c in a.columns_to_drop)
                or "transaction" in text
            )
        ),
        "n_cells": len(result.cells),
    }
    print(
        f"  [{label}] {dur:.0f}s  safe={row['training_safe']}  "
        f"id={row['caught_id']}  temporal={row['caught_temporal']}  "
        f"transactions={row['caught_transactions']}  drops={row['drops']}"
    )
    return row


def main() -> None:
    efforts = sys.argv[1:] or ["medium", "low"]

    df = pl.read_csv("datasets/train.csv", try_parse_dates=True)
    base_prof = profile(df, target="sales")
    base_cells = [
        "import polars as pl",
        (
            "df = Favorita store-sales training frame with columns: "
            "id, date, store_nbr, family, sales, onpromotion"
        ),
    ]
    tx = pl.read_csv("datasets/transactions.csv", try_parse_dates=True)
    joined = df.join(tx, on=["date", "store_nbr"], how="left")
    join_prof = profile(joined, target="sales")
    join_cells = [
        "import polars as pl",
        (
            "df = Favorita training frame LEFT-JOINED with the transactions "
            "table on (date, store_nbr); columns: id, date, store_nbr, "
            "family, sales, onpromotion, transactions"
        ),
    ]

    summary = []
    for effort in efforts:
        print(f"\n########## EFFORT = {effort} ##########")
        print("=== A. base plan x3 (stability of the temporal catch) ===")
        rows = [
            run_once(f"base#{i + 1}", base_prof, BASE_GOAL, base_cells, effort)
            for i in range(3)
        ]
        print("=== B. transactions-join (concurrent-aggregate leak) ===")
        join_row = run_once("join", join_prof, BASE_GOAL, join_cells, effort)
        summary.append((effort, rows, join_row))

    print("\n================= SUMMARY =================")
    for effort, rows, join_row in summary:
        temporal_hits = sum(r["caught_temporal"] for r in rows)
        id_hits = sum(r["caught_id"] for r in rows)
        durs = [round(r["dur"]) for r in rows] + [round(join_row["dur"])]
        print(
            f"[{effort:6}] base x3: temporal {temporal_hits}/3, id {id_hits}/3 | "
            f"join: transactions={join_row['caught_transactions']} "
            f"temporal={join_row['caught_temporal']} | "
            f"durations={durs}s (avg {round(sum(durs) / len(durs))}s)"
        )


if __name__ == "__main__":
    main()
