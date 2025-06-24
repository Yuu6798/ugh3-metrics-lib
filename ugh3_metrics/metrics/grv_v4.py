from __future__ import annotations

from typing import Iterable, List, Any

import numpy as np

from .base import BaseMetric
from ..models.embedder import EmbedderProtocol
from ..utils import tokenize, tfidf_topk, entropy, pmi



class GrvV4(BaseMetric):
    """Lexical gravity v4 metric (TF-IDF + PMI + Entropy)."""

    DEFAULT_WEIGHTS: tuple[float, float, float] = (0.42, 0.31, 0.27)
    DEFAULT_WINDOW: int = 50

    def __init__(self, *, embedder: EmbedderProtocol | None = None) -> None:
        self._embedder = embedder
        self._weights = np.asarray(self.DEFAULT_WEIGHTS, dtype=float)
        self._window = self.DEFAULT_WINDOW

    def _tokenize(self, items: str | Iterable[str]) -> list[str]:
        return tokenize(items)

    def _tfidf_score(self, tokens: List[str]) -> float:
        text = " ".join(tokens)
        return tfidf_topk(text, k=30)

    def _entropy_score(self, tokens: List[str]) -> float:
        return entropy(tokens)

    def _pmi_score(self, tokens: List[str]) -> float:
        return pmi(tokens, window=self._window)

    def score(self, a: str, b: str) -> float:
        """Return weighted average score normalized to 0-1."""
        del b  # unused
        tokens = self._tokenize(a)
        if not tokens:
            return 0.0

        w = self._weights
        if w.size != 3:
            raise ValueError("weights must have length 3")
        w = w / w.sum()
        vals = np.array([
            self._tfidf_score(tokens),
            self._pmi_score(tokens),
            self._entropy_score(tokens),
        ])
        x = float(np.dot(w, vals))
        return float(1.0 / (1.0 + np.exp(-x)))

    def set_params(self, **kwargs: Any) -> None:
        weights = kwargs.get("weights")
        window = kwargs.get("window")
        if weights is not None:
            arr = np.asarray(list(weights), dtype=float)
            if arr.size != 3:
                raise ValueError("weights must have length 3")
            self._weights = arr
        if window is not None:
            self._window = int(window)


def calc_grv_v4(text: str) -> float:
    return GrvV4().score(text, "")


__all__ = ["GrvV4", "calc_grv_v4"]
