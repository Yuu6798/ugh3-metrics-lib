from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import math

from tools.stats_advanced import corr_stats, ols_standardized, series_summary
from tools.paper_report import _pick


def test_series_summary_ci() -> None:
    vals = np.arange(10, dtype=float)
    s = pd.Series(vals)
    out = series_summary(s)
    assert out["n"] == 10
    assert out["mean"] == pytest.approx(np.mean(vals))
    ci = out["ci95"]
    assert isinstance(ci, tuple)
    lo, hi = ci
    assert lo is not None and hi is not None
    assert lo < out["mean"] < hi


def test_corr_stats() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(scale=0.5, size=200)
    out = corr_stats(x, y)
    assert out["pearson"]["r"] is not None
    assert out["spearman"]["rho"] is not None
    assert out["pearson"]["r"] > 0.4
    assert out["spearman"]["rho"] > 0.35


def test_corr_ci_contains_r() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(scale=0.6, size=n)
    out = corr_stats(x, y, seed=0)
    pr = out["pearson"]
    lo, hi = pr["ci95"]
    assert lo is not None and hi is not None
    assert lo < pr["r"] < hi
    assert -1.0 <= lo <= hi <= 1.0


def test_spearman_ci_basic() -> None:
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    y = np.tanh(x) + rng.normal(scale=0.2, size=n)
    out = corr_stats(x, y, seed=1)
    sr = out["spearman"]
    lo, hi = sr["ci95"]
    assert lo is not None and hi is not None
    assert hi - lo < 1.0


def test_ols_standardized_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 + 2.0 * x2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    res = ols_standardized(df, "y", ["x1", "x2"])
    assert res["r2"] is not None and res["r2"] > 0.1
    coef = res["coef"]
    assert coef[1] > 0
    assert coef[2] > 0


def test_pick_handles_array_and_series() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    ser = pd.Series(arr)
    assert _pick(arr, 1, "") == pytest.approx(2.0)
    assert _pick(ser, 2, "") == pytest.approx(3.0)


def test_pick_series_label_index() -> None:
    ser = pd.Series([0.1, 0.2, 0.3], index=["x1", "x2", "x3"])
    assert _pick(ser, i=99, key="x2") == pytest.approx(0.2)


def test_pick_non_scalar_from_dict_returns_nan() -> None:
    val = _pick({"x": np.array([1.0, 2.0])}, i=0, key="x")
    assert math.isnan(val)
