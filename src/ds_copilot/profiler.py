"""Schema profiler — first-class structured profiling for data-science notebooks.

Given a Polars or pandas DataFrame (and optionally a target column name),
produces a structured `Profile` describing dtypes, null rates, cardinality,
sample values, distribution summaries, and leakage hints. Designed both as
the "grounded schema" an AI copilot reads before suggesting code and as a
Marimo UI element a human can drop into a notebook.
"""

from __future__ import annotations

from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

FrameLike = Any  # polars.DataFrame | pandas.DataFrame


class NumericStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    min: float
    q25: float
    q50: float
    q75: float
    max: float
    mean: float
    std: float


class CategoricalStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    top_values: list[tuple[str, int]]


class DatetimeStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    min: str
    max: str


class LeakageHint(BaseModel):
    model_config = ConfigDict(frozen=True)

    severity: Literal["high", "medium", "low"]
    reason: str


class ColumnProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    null_count: int
    null_rate: float
    cardinality: int
    card_pct: float
    sample_values: list[Any]
    numeric_stats: NumericStats | None = None
    categorical_stats: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None
    leakage_hints: list[LeakageHint] = []


class Profile(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_rows: int
    n_cols: int
    sampled: bool
    sampled_from: int | None
    target: str | None
    target_rate: float | None
    columns: list[ColumnProfile]
    warnings: list[str] = []


def profile(
    frame: FrameLike,
    target: str | None = None,
    max_rows: int = 200_000,
    seed: int = 42,
) -> Profile:
    """Build a `Profile` for `frame`.

    Args:
        frame: A polars or pandas DataFrame.
        target: Optional column name to enable target-based leakage hints.
            When `target` is binary 0/1 numeric, `Profile.target_rate` is the
            positive-class rate.
        max_rows: If the frame has more rows than this, profile a uniform
            random sample of size `max_rows`. Kept on by default so the
            profiler stays cheap on big tables.
        seed: Seed for the sampling RNG.
    """
    pf = _to_polars(frame)
    n_full = pf.height
    sampled = False
    if n_full > max_rows:
        pf = pf.sample(n=max_rows, seed=seed)
        sampled = True

    n_rows = pf.height
    target_rate: float | None = None
    target_series: pl.Series | None = None
    if target and target in pf.columns:
        t = pf[target]
        target_series = t
        if t.dtype.is_numeric() and t.drop_nulls().len() > 0:
            tmean = float(t.mean())
            if 0.0 <= tmean <= 1.0:
                target_rate = tmean

    column_profiles = [
        _profile_column(
            series=pf[col],
            name=col,
            target_col=target_series if (target and col != target) else None,
            n_rows=n_rows,
        )
        for col in pf.columns
    ]

    warnings: list[str] = []
    if target and target not in pf.columns:
        warnings.append(f"target '{target}' not found in frame; leakage hints disabled")

    return Profile(
        n_rows=n_rows,
        n_cols=len(pf.columns),
        sampled=sampled,
        sampled_from=n_full if sampled else None,
        target=target,
        target_rate=target_rate,
        columns=column_profiles,
        warnings=warnings,
    )


def _to_polars(frame: FrameLike) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    if pd is not None and isinstance(frame, pd.DataFrame):
        try:
            return pl.from_pandas(frame)
        except (ImportError, ModuleNotFoundError):
            # pl.from_pandas requires pyarrow for some paths; column-by-column
            # via numpy works for the basic dtypes we care about in profiling.
            return pl.DataFrame(
                {col: frame[col].to_numpy() for col in frame.columns}
            )
    raise TypeError(
        f"profile() requires a polars or pandas DataFrame, got {type(frame).__name__}"
    )


def _profile_column(
    series: pl.Series,
    name: str,
    target_col: pl.Series | None,
    n_rows: int,
) -> ColumnProfile:
    dtype = series.dtype
    null_count = int(series.null_count())
    null_rate = (null_count / n_rows) if n_rows > 0 else 0.0
    cardinality = int(series.n_unique())
    card_pct = (cardinality / n_rows) if n_rows > 0 else 0.0
    sample = series.drop_nulls().head(3).to_list()

    numeric_stats = _numeric_stats(series) if dtype.is_numeric() else None
    categorical_stats = _categorical_stats(series) if _is_categorical(dtype) else None
    datetime_stats = _datetime_stats(series) if _is_datetime(dtype) else None

    leakage_hints = _leakage_hints(
        series=series,
        target_col=target_col,
        n_rows=n_rows,
        cardinality=cardinality,
        card_pct=card_pct,
    )

    return ColumnProfile(
        name=name,
        dtype=str(dtype),
        null_count=null_count,
        null_rate=round(null_rate, 4),
        cardinality=cardinality,
        card_pct=round(card_pct, 4),
        sample_values=sample,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        datetime_stats=datetime_stats,
        leakage_hints=leakage_hints,
    )


def _is_categorical(dtype: pl.DataType) -> bool:
    return dtype in (pl.Utf8, pl.String) or isinstance(dtype, pl.Categorical)


def _is_datetime(dtype: pl.DataType) -> bool:
    return dtype in (pl.Date, pl.Datetime, pl.Time)


def _numeric_stats(series: pl.Series) -> NumericStats | None:
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return None
    std = non_null.std()
    return NumericStats(
        min=float(non_null.min()),
        q25=float(non_null.quantile(0.25)),
        q50=float(non_null.quantile(0.50)),
        q75=float(non_null.quantile(0.75)),
        max=float(non_null.max()),
        mean=float(non_null.mean()),
        std=float(std) if std is not None else 0.0,
    )


def _categorical_stats(series: pl.Series) -> CategoricalStats | None:
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return None
    vc = non_null.value_counts(sort=True).head(5)
    if vc.height == 0:
        return None
    value_col, count_col = vc.columns[0], vc.columns[1]
    tops = [(str(v), int(c)) for v, c in zip(vc[value_col], vc[count_col])]
    return CategoricalStats(top_values=tops)


def _datetime_stats(series: pl.Series) -> DatetimeStats | None:
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return None
    return DatetimeStats(min=str(non_null.min()), max=str(non_null.max()))


def _leakage_hints(
    series: pl.Series,
    target_col: pl.Series | None,
    n_rows: int,
    cardinality: int,
    card_pct: float,
) -> list[LeakageHint]:
    hints: list[LeakageHint] = []

    # ID heuristic: ~all-unique AND not a continuous float.
    # Floats often have high cardinality without being IDs (prices, latencies).
    if (
        card_pct >= 0.99
        and cardinality >= 10
        and not series.dtype.is_float()
    ):
        hints.append(
            LeakageHint(
                severity="medium",
                reason=(
                    f"unique-per-row ({cardinality}/{n_rows}) — likely an ID, "
                    "drop before training"
                ),
            )
        )

    if (
        target_col is not None
        and series.dtype.is_numeric()
        and target_col.dtype.is_numeric()
    ):
        try:
            import numpy as np

            paired = pl.DataFrame({"x": series, "y": target_col}).drop_nulls()
            if paired.height > 1:
                x = paired["x"].to_numpy().astype(float)
                y = paired["y"].to_numpy().astype(float)
                if float(x.std()) > 0 and float(y.std()) > 0:
                    corr = float(np.corrcoef(x, y)[0, 1])
                    abs_corr = abs(corr)
                    if abs_corr > 0.99:
                        hints.append(
                            LeakageHint(
                                severity="high",
                                reason=(
                                    f"|corr| with target = {abs_corr:.4f} — "
                                    "near-perfect predictor, check for leakage"
                                ),
                            )
                        )
                    elif abs_corr > 0.9:
                        hints.append(
                            LeakageHint(
                                severity="medium",
                                reason=(
                                    f"|corr| with target = {abs_corr:.3f} — "
                                    "unusually strong, audit before training"
                                ),
                            )
                        )
        except Exception:
            pass

    return hints


def profile_widget(
    frame_or_profile: FrameLike | Profile,
    target: str | None = None,
    max_rows: int = 200_000,
):
    """Render a `Profile` (or compute one) as a Marimo display.

    Pass either a dataframe (will profile it) or a pre-built `Profile`.
    Returns an object that Marimo will render in a cell.
    """
    import marimo as mo

    p = (
        frame_or_profile
        if isinstance(frame_or_profile, Profile)
        else profile(frame_or_profile, target=target, max_rows=max_rows)
    )

    rows = []
    for c in p.columns:
        sample = ", ".join(str(v)[:30] for v in c.sample_values[:3])
        leak = "; ".join(h.reason for h in c.leakage_hints) if c.leakage_hints else ""
        rows.append(
            {
                "column": c.name,
                "dtype": c.dtype,
                "null_rate": c.null_rate,
                "cardinality": c.cardinality,
                "card_pct": c.card_pct,
                "samples": sample,
                "leakage": leak,
            }
        )
    table = pl.DataFrame(rows)

    header_bits = [f"**rows:** {p.n_rows:,}", f"**cols:** {p.n_cols}"]
    if p.target:
        header_bits.append(f"**target:** `{p.target}`")
        if p.target_rate is not None:
            header_bits.append(f"**target rate:** {p.target_rate:.3f}")
    if p.sampled and p.sampled_from is not None:
        header_bits.append(f"_(sampled {p.n_rows:,} of {p.sampled_from:,})_")

    elements: list[Any] = [mo.md(" · ".join(header_bits)), table]

    flagged = [c for c in p.columns if c.leakage_hints]
    if flagged:
        lines = [
            f"- `{c.name}` — **{h.severity}**: {h.reason}"
            for c in flagged
            for h in c.leakage_hints
        ]
        elements.append(mo.md("### Leakage / data-quality flags\n" + "\n".join(lines)))

    if p.warnings:
        elements.append(
            mo.md("### Warnings\n" + "\n".join(f"- {w}" for w in p.warnings))
        )

    return mo.vstack(elements)
