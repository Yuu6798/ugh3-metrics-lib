"""Point of Resonance v4 metric."""

from __future__ import annotations

from typing import Optional, Any
from numpy.typing import NDArray

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback for environments without the lib
    class SentenceTransformer:  # type: ignore
        def __init__(self, *args: str, **kwargs: str) -> None:
            pass

        def encode(self, text: str) -> NDArray[Any]:  # pragma: no cover
            return np.zeros(1, dtype=float)

_EMBEDDER: Optional[SentenceTransformer] = None

DEFAULT_ALPHA: float = 13.2
DEFAULT_BETA: float = -10.8


def _get_embedder() -> SentenceTransformer:
    """Return a cached ``SentenceTransformer`` instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def set_params(alpha: float | None = None, beta: float | None = None) -> None:
    """Set logistic parameters."""
    global DEFAULT_ALPHA, DEFAULT_BETA
    if alpha is not None:
        DEFAULT_ALPHA = float(alpha)
    if beta is not None:
        DEFAULT_BETA = float(beta)


def score(prev: str | None, curr: str) -> float:
    """Return PoR probability from text similarity."""
    if prev is None:
        return 0.0

    emb = _get_embedder()
    v1 = emb.encode(prev)
    v2 = emb.encode(curr)
    num = float(np.dot(v1, v2))
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    sim = num / denom if denom else 0.0

    value = 1.0 / (1.0 + np.exp(-(DEFAULT_ALPHA * sim + DEFAULT_BETA)))
    return float(value)


__all__ = ["score", "set_params"]
