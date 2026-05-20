import numpy as np
import pandas as pd
import polars as pl
import pytest

from ds_copilot.profiler import (
    ColumnProfile,
    LeakageHint,
    Profile,
    profile,
)


@pytest.fixture
def churn_frame() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 1_000
    return pl.DataFrame(
        {
            "customer_id": np.arange(n),
            "tenure_months": rng.integers(1, 72, n),
            "monthly_charges": rng.uniform(20, 120, n).round(2),
            "contract_type": rng.choice(["m2m", "1y", "2y"], n),
            "churned": rng.choice([0, 1], n, p=[0.74, 0.26]),
        }
    )


def test_profile_polars_basic(churn_frame: pl.DataFrame) -> None:
    p = profile(churn_frame)
    assert isinstance(p, Profile)
    assert p.n_rows == 1_000
    assert p.n_cols == 5
    assert p.target is None
    assert p.target_rate is None
    assert {c.name for c in p.columns} == set(churn_frame.columns)
    for c in p.columns:
        assert isinstance(c, ColumnProfile)
        assert 0.0 <= c.null_rate <= 1.0


def test_profile_with_binary_target_rate(churn_frame: pl.DataFrame) -> None:
    p = profile(churn_frame, target="churned")
    assert p.target == "churned"
    assert p.target_rate is not None
    assert 0.20 <= p.target_rate <= 0.32  # ~0.26 with sampling noise


def test_numeric_stats_populated(churn_frame: pl.DataFrame) -> None:
    p = profile(churn_frame)
    by_name = {c.name: c for c in p.columns}
    tm = by_name["tenure_months"]
    assert tm.numeric_stats is not None
    assert 1.0 <= tm.numeric_stats.min <= tm.numeric_stats.q25
    assert tm.numeric_stats.q25 <= tm.numeric_stats.q50 <= tm.numeric_stats.q75
    assert tm.numeric_stats.q75 <= tm.numeric_stats.max <= 71.0
    assert tm.numeric_stats.std > 0


def test_categorical_stats_populated(churn_frame: pl.DataFrame) -> None:
    p = profile(churn_frame)
    by_name = {c.name: c for c in p.columns}
    ct = by_name["contract_type"]
    assert ct.categorical_stats is not None
    values = {v for v, _ in ct.categorical_stats.top_values}
    assert values == {"m2m", "1y", "2y"}
    total = sum(c for _, c in ct.categorical_stats.top_values)
    assert total == 1_000


def _build_raw_churn() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 1_000
    return {
        "customer_id": np.arange(n),
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n).round(2),
        "contract_type": rng.choice(["m2m", "1y", "2y"], n),
        "churned": rng.choice([0, 1], n, p=[0.74, 0.26]),
    }


def test_pandas_matches_polars() -> None:
    # Build both frames from the same primitive arrays so pyarrow isn't needed
    # for round-tripping through polars.to_pandas().
    raw = _build_raw_churn()
    p_pl = profile(pl.DataFrame(raw))
    p_pd = profile(pd.DataFrame(raw))

    assert p_pl.n_rows == p_pd.n_rows
    assert p_pl.n_cols == p_pd.n_cols
    assert {c.name for c in p_pl.columns} == {c.name for c in p_pd.columns}

    pl_num = {c.name: c.numeric_stats for c in p_pl.columns if c.numeric_stats}
    pd_num = {c.name: c.numeric_stats for c in p_pd.columns if c.numeric_stats}
    assert pl_num.keys() == pd_num.keys()
    for name in pl_num:
        assert pl_num[name].mean == pytest.approx(pd_num[name].mean, rel=1e-6, abs=1e-9)
        assert pl_num[name].std == pytest.approx(pd_num[name].std, rel=1e-6, abs=1e-9)
        assert pl_num[name].min == pytest.approx(pd_num[name].min)
        assert pl_num[name].max == pytest.approx(pd_num[name].max)


def test_leakage_flags_unique_id(churn_frame: pl.DataFrame) -> None:
    p = profile(churn_frame, target="churned")
    by_name = {c.name: c for c in p.columns}

    cid_hints = by_name["customer_id"].leakage_hints
    assert any("unique-per-row" in h.reason for h in cid_hints), cid_hints

    # No real signal in the synthetic dataset — these should be clean.
    assert by_name["tenure_months"].leakage_hints == []
    assert by_name["monthly_charges"].leakage_hints == []


def test_leakage_flags_near_perfect_correlation() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y = rng.choice([0, 1], n, p=[0.5, 0.5])
    leaker = (y * 100 + rng.normal(0, 0.001, n)).astype(float)
    noise = rng.uniform(0, 1, n)

    df = pl.DataFrame({"leaker": leaker, "noise": noise, "target": y})
    p = profile(df, target="target")
    by_name = {c.name: c for c in p.columns}

    leaker_hints = by_name["leaker"].leakage_hints
    assert any(h.severity == "high" for h in leaker_hints), leaker_hints
    assert any("corr" in h.reason for h in leaker_hints)

    assert by_name["noise"].leakage_hints == []


def test_empty_frame() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series([], dtype=pl.Int64),
            "b": pl.Series([], dtype=pl.Utf8),
        }
    )
    p = profile(df)
    assert p.n_rows == 0
    assert p.n_cols == 2
    for c in p.columns:
        assert c.cardinality == 0
        assert c.null_rate == 0.0  # 0 / 0 is reported as 0 by convention


def test_all_null_column() -> None:
    df = pl.DataFrame({"a": pl.Series([None, None, None], dtype=pl.Float64)})
    p = profile(df)
    only = p.columns[0]
    assert only.null_count == 3
    assert only.null_rate == 1.0
    assert only.numeric_stats is None  # nothing non-null to summarise


def test_sampling_triggers_above_max_rows() -> None:
    n_full = 250_000
    df = pl.DataFrame({"x": pl.int_range(0, n_full, eager=True)})
    p = profile(df, max_rows=10_000)
    assert p.sampled is True
    assert p.sampled_from == n_full
    assert p.n_rows == 10_000


def test_target_missing_warning() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    p = profile(df, target="not_a_column")
    assert p.target == "not_a_column"
    assert p.target_rate is None
    assert any("not found" in w for w in p.warnings)


def test_profile_field_reassignment_blocked() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    p = profile(df)
    with pytest.raises(Exception):
        # pydantic frozen=True blocks field reassignment on the model itself.
        p.n_rows = 999  # type: ignore[misc]
