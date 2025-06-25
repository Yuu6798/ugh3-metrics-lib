from __future__ import annotations

import numpy as np
from typing import Any

from ..models.embedder import EmbedderProtocol
from .base import BaseMetric
from ..utils import cosine_similarity


class DeltaEV4(BaseMetric):
    """Semantic difference metric using cosine distance."""

    DEFAULT_MU: float = 0.35
    DEFAULT_SIGMA: float = 0.08

    def __init__(self, *, embedder: EmbedderProtocol | None = None) -> None:
        if embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:

                class SimpleEmbedder:
                    def encode(self, text: str) -> Any:
                        return [float(len(text.split()))]

                embedder = SimpleEmbedder()
        self._embedder = embedder

    def score(self, a: str, b: str) -> float:
        """Return standardized cosine difference clipped to [0, 1]."""
        if not a:
            return 0.0
        v1 = self._embedder.encode(a)
        v2 = self._embedder.encode(b)
        sim = cosine_similarity(v1, v2)
        diff = 1.0 - sim
        z = (diff - self.DEFAULT_MU) / self.DEFAULT_SIGMA if self.DEFAULT_SIGMA else diff - self.DEFAULT_MU
        return float(np.clip(z, 0.0, 1.0))

    def set_params(self, **kw: object) -> None:  # pragma: no cover - placeholder
        pass


def calc_deltae_v4(a: str, b: str) -> float:
    """Legacy wrapper compatible with function-based API."""
    return DeltaEV4().score(a, b)


__all__ = ["DeltaEV4", "calc_deltae_v4"]
