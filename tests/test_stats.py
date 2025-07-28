from __future__ import annotations

import numpy as np
from scipy import stats

from core.stats import cohen_d, cliffs_delta, welch_ttest


def test_cohen_d_against_scipy() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=1.0, size=100)
    b = rng.normal(loc=0.5, scale=1.2, size=80)
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = stats.tstd(a)
    std_b = stats.tstd(b)
    expected = (mean_a - mean_b) / np.sqrt((std_a**2 + std_b**2) / 2)
    assert abs(cohen_d(a, b) - expected) < 1e-6


def test_cliffs_delta_against_scipy() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=50)
    b = rng.normal(loc=0.5, size=60)
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    u = res.statistic
    m, n = len(a), len(b)
    expected = 2 * u / (m * n) - 1
    assert abs(cliffs_delta(a, b) - expected) < 1e-6


def test_welch_ttest_against_scipy() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=30)
    b = rng.normal(loc=0.3, size=40)
    res = stats.ttest_ind(a, b, equal_var=False)
    t, p = welch_ttest(a, b)
    assert abs(t - res.statistic) < 1e-6
    assert abs(p - res.pvalue) < 1e-6
