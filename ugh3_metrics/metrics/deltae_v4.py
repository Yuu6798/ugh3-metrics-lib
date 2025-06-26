from __future__ import annotations

import hashlib
import warnings
import numpy as np
from difflib import SequenceMatcher
from typing import Any

from ..models.embedder import EmbedderProtocol
from .base import BaseMetric



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
                    """Fallback embedder: non-zero 32-D int16 vector via MD5."""

                    _table = np.arange(0, 256, dtype=np.int16)

                    def encode(self, text: str, **_: Any) -> np.ndarray:  # noqa: D401
                        md5 = hashlib.md5(text.encode()).digest()
                        v = self._table[np.frombuffer(md5, dtype=np.uint8)]
                        return np.concatenate([v, v])

                embedder = SimpleEmbedder()
        self._embedder = embedder

    def score(self, prev: str, curr: str) -> float:  # noqa: D401
        v1 = np.asarray(self._embedder.encode(prev), dtype=float)
        v2 = np.asarray(self._embedder.encode(curr), dtype=float)
        if not np.any(v1) or not np.any(v2):
            warnings.warn("zero\u2011vector; diff_sim fallback", RuntimeWarning)
            sim = SequenceMatcher(None, prev, curr).ratio()
            return round(1.0 - sim, 3)
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        cos = float(np.dot(v1, v2)) / denom
        return round(max(0.0, 1.0 - cos), 3)

    def set_params(self, **kw: object) -> None:  # pragma: no cover - placeholder
        pass


def calc_deltae_v4(a: str, b: str) -> float:
    """Legacy wrapper compatible with function-based API."""
    return DeltaEV4().score(a, b)


__all__ = ["DeltaEV4", "calc_deltae_v4"]
