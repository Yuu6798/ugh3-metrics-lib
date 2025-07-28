"""Statistical effect size and hypothesis test utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def cohen_d(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return Cohen's d for independent samples ``a`` and ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = (var_a + var_b) / 2
    return float((mean_a - mean_b) / np.sqrt(pooled))


def cliffs_delta(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Return Cliff's delta effect size between ``a`` and ``b``."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    u = float(res.statistic)
    m = len(a)
    n = len(b)
    return float(2 * u / (m * n) - 1)


def welch_ttest(a: NDArray[np.float64], b: NDArray[np.float64]) -> Tuple[float, float]:
    """Welch's t-test for two independent samples."""
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)


__all__ = ["cohen_d", "cliffs_delta", "welch_ttest"]
