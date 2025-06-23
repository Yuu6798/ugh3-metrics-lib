"""ΔE v4 metric using semantic difference."""

from __future__ import annotations

from typing import Optional

import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback if library missing
    class SentenceTransformer:  # type: ignore
        def __init__(self, *args: str, **kwargs: str) -> None:
            pass
        def encode(self, text: str) -> np.ndarray:  # pragma: no cover
            return np.zeros(1, dtype=float)

_EMBEDDER: Optional[SentenceTransformer] = None

DEFAULT_MU: float = 0.35
DEFAULT_SIGMA: float = 0.08


def _get_embedder() -> SentenceTransformer:
    """Return cached ``SentenceTransformer`` instance."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def set_params(mu: float | None = None, sigma: float | None = None) -> None:
    """Set z-score parameters."""
    global DEFAULT_MU, DEFAULT_SIGMA
    if mu is not None:
        DEFAULT_MU = float(mu)
    if sigma is not None:
        DEFAULT_SIGMA = float(sigma)


def delta_e(prev: Optional[str], curr: str) -> float:
    """Return standardized cosine difference clipped to [0, 1]."""
    if prev is None:
        return 0.0

    emb = _get_embedder()
    v1 = emb.encode(prev)
    v2 = emb.encode(curr)
    num = float(np.dot(v1, v2))
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    sim = num / denom if denom else 0.0
    diff = 1.0 - sim

    z = (diff - DEFAULT_MU) / DEFAULT_SIGMA if DEFAULT_SIGMA else diff - DEFAULT_MU
    return float(np.clip(z, 0.0, 1.0))


# backward compatibility
score = delta_e


__all__ = ["score", "set_params"]
