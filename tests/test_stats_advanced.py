from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.stats_advanced import series_summary


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
