from __future__ import annotations

import numpy as np

from ..models.embedder import EmbedderProtocol
from .base import BaseMetric
from ..utils import cosine_similarity


class PorV4(BaseMetric):
    """Point of Resonance v4 metric."""

    DEFAULT_ALPHA: float = 13.2
    DEFAULT_BETA: float = -10.8

    def __init__(self, *, embedder: EmbedderProtocol | None = None) -> None:
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._embedder = embedder

    def score(self, a: str, b: str) -> float:
        """Return PoR probability from text similarity."""
        if not a:
            return 0.0
        v1 = self._embedder.encode(a)
        v2 = self._embedder.encode(b)
        sim = cosine_similarity(v1, v2)
        value = 1.0 / (1.0 + np.exp(-(self.DEFAULT_ALPHA * sim + self.DEFAULT_BETA)))
        return float(value)

    def set_params(self, **kw: object) -> None:  # pragma: no cover - placeholder
        pass


def calc_por_v4(a: str, b: str) -> float:
    """Legacy wrapper compatible with function-based API."""
    return PorV4().score(a, b)


__all__ = ["PorV4", "calc_por_v4"]
