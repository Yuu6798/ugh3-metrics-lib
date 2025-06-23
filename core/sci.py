"""Semantic Consistency Index."""

from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, Any
from numpy.typing import NDArray

DEFAULT_WEIGHTS: NDArray[Any] = np.array([0.4, 0.45, 0.15])
DEFAULT_ALPHA: float = 0.3
_STATE: float | None = None


def set_weights(weights: Sequence[float], alpha: float | None = None) -> None:
    """Set global weights and EMA parameter."""
    global DEFAULT_WEIGHTS, DEFAULT_ALPHA
    arr = np.asarray(list(weights), dtype=float)
    if arr.size != 3:
        raise ValueError("weights must have length 3")
    DEFAULT_WEIGHTS = arr
    if alpha is not None:
        DEFAULT_ALPHA = float(alpha)


def reset_state() -> None:
    """Reset internal EMA state."""
    global _STATE
    _STATE = None


def sci(
    por: float,
    delta_e: float,
    grv: float,
    *,
    weights: Iterable[float] | None = None,
    alpha: float | None = None,
) -> float:
    """Return EMA of weighted metrics.

    Parameters
    ----------
    por, delta_e, grv : float
        Input metrics in the range 0.0 to 1.0.
    weights : Iterable[float], optional
        Custom weights ``(w1, w2, w3)``.

    Returns
    -------
    float
        Weighted score between 0.0 and 1.0.
    """
    w = (
        np.asarray(list(weights), dtype=float)
        if weights is not None
        else DEFAULT_WEIGHTS
    )
    if w.size != 3:
        raise ValueError("weights must have length 3")
    w = w / w.sum()
    vals = np.array([por, 1.0 - delta_e, grv], dtype=float)
    x = float(np.dot(w, vals))
    a = DEFAULT_ALPHA if alpha is None else float(alpha)
    global _STATE
    _STATE = x if _STATE is None else a * x + (1 - a) * _STATE
    return float(_STATE)


# backward compatibility
score = sci


__all__ = ["sci", "score", "set_weights", "reset_state"]
