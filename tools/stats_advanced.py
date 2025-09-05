from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypeAlias, cast

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


def corr_stats(
    x: Any,
    y: Any,
    *,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return Pearson and Spearman correlation statistics for *x* and *y*.

    The function operates without a SciPy dependency.  When SciPy is
    available, two-sided p-values are included.  Non-finite values are
    removed pairwise and results fall back to ``None`` on failure.  A
    bootstrap (with ``seed``) re-ranking the data within each resample is
    used to estimate the 95% CI of Spearman's ρ while Pearson's r uses the
    Fisher z approximation.
    """

    def _fail() -> Dict[str, Any]:
        return {
            "pearson": {"r": None, "p": None, "ci95": (None, None), "n": 0},
            "spearman": {"rho": None, "p": None, "ci95": (None, None), "n": 0},
        }

    try:
        x_arr = np.asarray(x, dtype=np.float_).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float_).reshape(-1)
        if x_arr.size != y_arr.size or x_arr.size == 0:
            return _fail()
        mask = np.isfinite(x_arr) & np.isfinite(y_arr)
        xv = x_arr[mask]
        yv = y_arr[mask]
        n = int(xv.size)
        if n < 2:
            return _fail()

        pearson_r = float(np.corrcoef(xv, yv)[0, 1])

        # Pearson CI via Fisher z approximation
        pearson_ci: Tuple[Optional[float], Optional[float]] = (None, None)
        if n > 3 and abs(pearson_r) < 1:
            Z975 = 1.959963984540054  # 97.5th percentile of standard normal
            z = float(np.arctanh(pearson_r))
            se = 1.0 / np.sqrt(n - 3)
            lo = float(np.tanh(z - Z975 * se))
            hi = float(np.tanh(z + Z975 * se))
            pearson_ci = (lo, hi)

        # Spearman rho
        x_rank = pd.Series(xv).rank(method="average").to_numpy()
        y_rank = pd.Series(yv).rank(method="average").to_numpy()
        spearman_rho = float(np.corrcoef(x_rank, y_rank)[0, 1])

        # Spearman CI via bootstrap (re-rank within each resample)
        spearman_ci: Tuple[Optional[float], Optional[float]] = (None, None)
        if n >= 10:
            rng = np.random.default_rng(seed)
            n_boot = 1000
            boots = np.empty(n_boot, dtype=np.float_)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                xr = pd.Series(xv[idx]).rank(method="average").to_numpy()
                yr = pd.Series(yv[idx]).rank(method="average").to_numpy()
                if xr.std(ddof=0) == 0.0 or yr.std(ddof=0) == 0.0:
                    boots[b] = np.nan
                else:
                    boots[b] = float(np.corrcoef(xr, yr)[0, 1])
            valids = boots[np.isfinite(boots)]
            if valids.size >= 30:
                slo = float(np.quantile(valids, 0.025))
                shi = float(np.quantile(valids, 0.975))
                spearman_ci = (slo, shi)

        pearson_p: Optional[float] = None
        spearman_p: Optional[float] = None
        try:  # SciPy is optional
            from scipy import stats

            pearson_p = float(stats.pearsonr(xv, yv)[1])
            spearman_p = float(stats.spearmanr(xv, yv)[1])
        except Exception:
            pass

        return {
            "pearson": {"r": pearson_r, "p": pearson_p, "ci95": pearson_ci, "n": n},
            "spearman": {
                "rho": spearman_rho,
                "p": spearman_p,
                "ci95": spearman_ci,
                "n": n,
            },
        }
    except Exception:
        return _fail()


def ols_standardized(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    *,
    add_intercept: bool = True,
    standardize: bool = True,
) -> Dict[str, Any]:
    """Fit an OLS model optionally standardizing the design matrix.

    Parameters mimic a tiny subset of what StatsModels would provide but
    avoid any dependency on external libraries beyond NumPy/Pandas.
    ``standardize`` Z-scores the columns in ``x_cols``.  When SciPy is
    available, two-sided p-values for the t statistics are reported,
    otherwise ``p`` is ``None``.
    """

    try:
        y = df[y_col].to_numpy(dtype=np.float_)
        X = df[x_cols].to_numpy(dtype=np.float_)
        if standardize:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0, ddof=0)
            std[std == 0] = 1
            X = (X - mean) / std
        colnames: List[str] = list(x_cols)
        if add_intercept:
            X = np.column_stack([np.ones(len(df), dtype=np.float_), X])
            colnames = ["intercept"] + colnames

        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        n = int(y.shape[0])
        p = int(X.shape[1])
        dof = max(n - p, 1)
        rss = float(resid @ resid)
        mse = rss / dof
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(mse * XtX_inv))
        t_stats = beta / se
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - rss / ss_tot if ss_tot > 0 else 0.0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else 0.0

        p_vals: Optional[NDArray[np.float_]] = None
        try:
            from scipy import stats

            p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)
            p_vals = p_vals.astype(float)
        except Exception:
            pass

        return {
            "coef": beta,
            "se": se,
            "t": t_stats,
            "p": p_vals,
            "colnames": colnames,
            "r2": float(r2),
            "adj_r2": float(adj_r2),
            "n": n,
        }
    except Exception:
        return {
            "coef": np.array([], dtype=np.float_),
            "se": np.array([], dtype=np.float_),
            "t": np.array([], dtype=np.float_),
            "p": None,
            "colnames": [],
            "r2": None,
            "adj_r2": None,
            "n": 0,
        }
