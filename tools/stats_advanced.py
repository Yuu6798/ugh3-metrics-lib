from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:  # optional SciPy for p-values
    import scipy.stats as sps  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - degrade gracefully
    sps = None

FloatArray = NDArray[np.float_]


def _to_numeric(x: Any) -> FloatArray:
    """Normalize input to a 1-D float ndarray, dropping NaNs."""
    if isinstance(x, pd.Series):
        arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    else:
        arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    return cast(FloatArray, arr)


def bootstrap_ci(
    vals: Any,
    alpha: float = 0.05,
    n_boot: int = 2000,
    seed: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float]]:
    v = _to_numeric(vals)
    if v.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    samples: FloatArray = v[idx].mean(axis=1)
    low = float(np.percentile(samples, 100 * (alpha / 2)))
    high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return low, high


def series_summary(s: Any) -> Dict[str, Any]:
    v = _to_numeric(s)
    n = int(v.size)
    if n == 0:
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
    mean = float(np.mean(v))
    median = float(np.median(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    q1 = float(np.percentile(v, 25))
    q3 = float(np.percentile(v, 75))
    iqr = float(q3 - q1)
    ci = bootstrap_ci(v, alpha=0.05, n_boot=2000)
    return {"n": n, "mean": mean, "median": median, "std": std, "q1": q1, "q3": q3, "iqr": iqr, "ci95": ci}


def corr_stats(x: Any, y: Any) -> Dict[str, Any]:
    xx = _to_numeric(x)
    yy = _to_numeric(y)
    n = int(min(xx.size, yy.size))
    if n == 0:
        return {
            "pearson": {"r": None, "p": None, "n": 0},
            "spearman": {"rho": None, "p": None, "n": 0},
        }
    # Pearson
    r = float(np.corrcoef(xx, yy)[0, 1])
    p_p: Optional[float] = None
    if sps is not None:
        try:
            rr, pp = sps.pearsonr(xx, yy)
            r, p_p = float(rr), float(pp)
        except Exception:
            pass
    # Spearman
    rho: Optional[float] = None
    p_s: Optional[float] = None
    if sps is not None:
        try:
            rr, pp = sps.spearmanr(xx, yy)
            rho, p_s = float(rr), float(pp)
        except Exception:
            pass
    if rho is None:  # fallback rank-based correlation
        rx = pd.Series(xx).rank().to_numpy(dtype=float)
        ry = pd.Series(yy).rank().to_numpy(dtype=float)
        rho = float(np.corrcoef(rx, ry)[0, 1])
    return {
        "pearson": {"r": r, "p": p_p, "n": n},
        "spearman": {"rho": rho, "p": p_s, "n": n},
    }


def ols_standardized(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    *,
    add_intercept: bool = True,
    standardize: bool = True,
) -> Dict[str, Any]:
    data = df.copy()
    used: List[str] = [c for c in x_cols if c in data.columns]
    if y_col not in data.columns:
        return {"coef": {}, "stderr": {}, "t": {}, "p": {}, "r2": None, "adj_r2": None, "n": 0, "design": []}

    y_raw = pd.to_numeric(data[y_col], errors="coerce")
    X_raw = pd.DataFrame({c: pd.to_numeric(data[c], errors="coerce") for c in used})
    mask = y_raw.notna()
    for c in used:
        mask &= X_raw[c].notna()

    y: FloatArray = cast(FloatArray, y_raw[mask].to_numpy(dtype=float))
    X: FloatArray = cast(FloatArray, X_raw.loc[mask, used].to_numpy(dtype=float))
    n, px = X.shape if X.size else (0, 0)
    p = px + (1 if add_intercept else 0)
    if n == 0 or n <= p:
        return {"coef": {}, "stderr": {}, "t": {}, "p": {}, "r2": None, "adj_r2": None, "n": int(n), "design": []}

    if standardize:
        y = (y - float(y.mean())) / (float(y.std(ddof=1)) or 1.0)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        X = (X - X.mean(axis=0)) / std

    names: List[str] = used.copy()
    if add_intercept:
        X = cast(FloatArray, np.column_stack([np.ones(n, dtype=float), X]))
        names = ["intercept"] + names

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    sse = float((resid ** 2).sum())
    sst = float(((y - float(y.mean())) ** 2).sum())
    p = X.shape[1]
    sigma2 = sse / max(n - p, 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    tvals = np.divide(beta, se, out=np.zeros_like(beta), where=se != 0)
    dfree = max(n - p, 1)
    if sps is not None:
        try:
            pvals: Dict[str, Optional[float]] = {
                nm: float(2 * sps.t.sf(abs(float(tv)), dfree)) for nm, tv in zip(names, tvals)
            }
        except Exception:
            pvals = {nm: None for nm in names}
    else:
        pvals = {nm: None for nm in names}
    r2 = 1.0 - sse / max(sst, 1e-12)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - p, 1)
    return {
        "coef": {nm: float(beta[i]) for i, nm in enumerate(names)},
        "stderr": {nm: float(se[i]) for i, nm in enumerate(names)},
        "t": {nm: float(tvals[i]) for i, nm in enumerate(names)},
        "p": pvals,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "n": int(n),
        "design": names,
    }

