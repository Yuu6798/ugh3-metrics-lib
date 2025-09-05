from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float_]


def _to_numeric(x: Any) -> Optional[FloatArray]:
    """Convert ``x`` to a 1-D float array.

    Non-finite values are removed.  On failure ``None`` is returned
    instead of raising an exception.
    """
    try:
        arr = np.asarray(x, dtype=np.float_).astype(np.float_, copy=False)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        arr = arr[np.isfinite(arr)]
        return cast(FloatArray, arr)
    except Exception:
        return None


def bootstrap_ci(
    vals: Any,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Return bootstrap confidence interval for the mean of ``vals``.

    Parameters are intentionally simple to avoid SciPy dependency.  When
    conversion fails ``(None, None)`` is returned.
    """
    arr = _to_numeric(vals)
    if arr is None or arr.size == 0:
        return (None, None)
    rng = np.random.default_rng(seed)
    n = arr.size
    boots = np.empty(n_boot, dtype=np.float_)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = float(np.mean(sample))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (lo, hi)


def series_summary(s: pd.Series) -> Dict[str, Any]:
    """Return basic statistics for ``s`` including bootstrap CI of the mean."""
    try:
        arr = _to_numeric(s.dropna().to_numpy())
        if arr is None or arr.size == 0:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "std": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "ci95": (None, None),
            }
        n = int(arr.size)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else None
        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        iqr = q3 - q1
        ci95 = bootstrap_ci(arr, alpha=0.05, n_boot=2000, seed=None)
        return {
            "n": n,
            "mean": mean,
            "median": median,
            "std": std,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "ci95": ci95,
        }
    except Exception:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "ci95": (None, None),
        }
