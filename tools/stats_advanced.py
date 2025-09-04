from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_numeric(x: Any) -> np.ndarray:
    s = pd.Series(x, dtype="float64")
    v = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
    return v.astype(float, copy=False)


def n_boot_default(n: int) -> int:
    if n < 50:
        return 2000
    if n < 200:
        return 4000
    return 6000


def bootstrap_ci(
    vals: Any, alpha: float = 0.05, n_boot: Optional[int] = None, seed: Optional[int] = None
) -> Tuple[float, float]:
    v = _to_numeric(vals)
    if v.size == 0:
        return (None, None)  # type: ignore[return-value]
    n_boot = n_boot or n_boot_default(v.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    samples = v[idx].mean(axis=1)
    low = float(np.percentile(samples, 100 * (alpha / 2)))
    high = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return low, high


def series_summary(s: pd.Series) -> Dict[str, Any]:
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
            "ci95": [None, None],
        }
    mean = float(np.mean(v))
    median = float(np.median(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    q1 = float(np.percentile(v, 25))
    q3 = float(np.percentile(v, 75))
    iqr = float(q3 - q1)
    ci_low, ci_high = bootstrap_ci(v)
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "ci95": [ci_low, ci_high],
    }


def corr_stats(x: Any, y: Any) -> Dict[str, Any]:
    xx = _to_numeric(x)
    yy = _to_numeric(y)
    n = int(min(xx.size, yy.size))
    if n == 0:
        return {
            "pearson": {"r": None, "p": None, "n": 0},
            "spearman": {"rho": None, "p": None, "n": 0},
        }
    xx = xx[:n]
    yy = yy[:n]
    r = float(np.corrcoef(xx, yy)[0, 1])
    p_p: Optional[float] = None
    try:
        import scipy.stats as sps  # type: ignore

        _, p_p = sps.pearsonr(xx, yy)
        p_p = float(p_p)
    except Exception:
        p_p = None
    rho = float(
        np.corrcoef(pd.Series(xx).rank().to_numpy(), pd.Series(yy).rank().to_numpy())[0, 1]
    )
    p_s: Optional[float] = None
    try:
        import scipy.stats as sps  # type: ignore

        _, p_s = sps.spearmanr(xx, yy)
        p_s = float(p_s)
    except Exception:
        p_s = None
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
    if y_col not in data.columns or not used:
        return {
            "coef": {},
            "stderr": {},
            "t": {},
            "p": {},
            "r2": None,
            "adj_r2": None,
            "n": 0,
            "design": [],
        }
    y = pd.to_numeric(data[y_col], errors="coerce")
    Xcols = [pd.to_numeric(data[c], errors="coerce") for c in used]
    Xmat = np.column_stack([c.to_numpy() for c in Xcols])
    mask = ~np.isnan(y.to_numpy())
    for i in range(Xmat.shape[1]):
        mask &= ~np.isnan(Xmat[:, i])
    yv = y.to_numpy()[mask].astype(float)
    Xv = Xmat[mask, :].astype(float)
    n = int(Xv.shape[0])
    p = int(Xv.shape[1] + (1 if add_intercept else 0))
    if n == 0 or n <= p:
        return {
            "coef": {},
            "stderr": {},
            "t": {},
            "p": {},
            "r2": None,
            "adj_r2": None,
            "n": n,
            "design": used,
        }
    if standardize:
        y_mean = yv.mean()
        y_std = yv.std(ddof=1) or 1.0
        yv = (yv - y_mean) / y_std
        X_mean = Xv.mean(axis=0)
        X_std = Xv.std(axis=0, ddof=1)
        X_std[X_std == 0] = 1.0
        Xv = (Xv - X_mean) / X_std
    if add_intercept:
        Xv = np.column_stack([np.ones((n, 1)), Xv])
        design = ["intercept"] + used
    else:
        design = used[:]
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    yhat = Xv @ beta
    resid = yv - yhat
    sst = float(((yv - yv.mean()) ** 2).sum())
    sse = float((resid**2).sum())
    r2 = 1.0 - (sse / max(1e-12, sst))
    adj_r2 = 1.0 - ((sse / max(n - p, 1)) / (sst / max(n - 1, 1))) if sst > 0 else 0.0
    XtX = Xv.T @ Xv
    XtX_inv = np.linalg.inv(XtX)
    sigma2 = sse / max(n - p, 1)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    tvals = beta / se
    try:
        import scipy.stats as sps  # type: ignore

        dfree = max(n - p, 1)
        pvec = [float(2 * sps.t.sf(abs(t), dfree)) for t in tvals]
    except Exception:
        pvec = [None for _ in range(len(tvals))]
    coef = {design[i]: float(beta[i]) for i in range(len(design))}
    stderr = {design[i]: float(se[i]) for i in range(len(design))}
    tdict = {design[i]: float(tvals[i]) for i in range(len(design))}
    pvals = {design[i]: pvec[i] for i in range(len(design))}
    return {
        "coef": coef,
        "stderr": stderr,
        "t": tdict,
        "p": pvals,
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "n": n,
        "design": design,
    }

