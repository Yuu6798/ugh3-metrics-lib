from __future__ import annotations

import numpy as np
import pandas as pd

from tools.stats_advanced import series_summary, corr_stats, ols_standardized


def test_series_summary_ci() -> None:
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    summ = series_summary(s)
    assert summ["n"] == 10
    assert 5.0 <= summ["mean"] <= 6.0
    assert summ["ci95"][0] is not None and summ["ci95"][1] is not None


def test_corr_and_ols() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.6 * x + rng.normal(scale=0.4, size=200)
    df = pd.DataFrame({"por": y, "grv": x})
    corr = corr_stats(df["por"], df["grv"])
    assert corr["pearson"]["r"] > 0.4
    reg = ols_standardized(df, "por", ["grv"])
    assert "grv" in reg["coef"]
    assert reg["r2"] > 0.1

