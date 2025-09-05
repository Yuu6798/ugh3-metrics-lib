from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.stats_advanced import corr_stats, ols_standardized, series_summary


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
