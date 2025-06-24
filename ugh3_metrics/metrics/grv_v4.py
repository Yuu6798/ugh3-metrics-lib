from __future__ import annotations

from collections import Counter
from math import log
from typing import Iterable, List, Optional, Any

import numpy as np
from numpy.typing import NDArray

from .base import BaseMetric
from ..models.embedder import EmbedderProtocol

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - fallback if sklearn missing
    class _DummyMatrix:
        def __init__(self) -> None:
            self._arr = np.zeros((1, 0))

        def toarray(self) -> NDArray[Any]:  # pragma: no cover - mimic sklearn matrix
            return self._arr

    class TfidfVectorizer:  # type: ignore
        def __init__(self) -> None:
            self.vocabulary_: dict[str, int] = {}

        def fit(self, docs: Iterable[str]) -> "TfidfVectorizer":
            self.vocabulary_ = {}
            return self

        def transform(self, docs: Iterable[str]) -> _DummyMatrix:
            return _DummyMatrix()

        def fit_transform(self, docs: Iterable[str]) -> _DummyMatrix:
            self.fit(docs)
            return _DummyMatrix()


class GrvV4(BaseMetric):
    """Lexical gravity v4 metric (TF-IDF + PMI + Entropy)."""

    DEFAULT_WEIGHTS: tuple[float, float, float] = (0.42, 0.31, 0.27)
    DEFAULT_WINDOW: int = 50

    def __init__(self, *, embedder: EmbedderProtocol | None = None) -> None:
        self._embedder = embedder
        self._weights = np.asarray(self.DEFAULT_WEIGHTS, dtype=float)
        self._window = self.DEFAULT_WINDOW
        self._vectorizer: Optional[TfidfVectorizer] = None

    def _get_vectorizer(self, text: str) -> TfidfVectorizer:
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer()
        if not getattr(self._vectorizer, "vocabulary_", None):
            try:
                self._vectorizer.fit([text])
            except Exception:  # pragma: no cover - fallback for stub
                pass
        return self._vectorizer

    def _tokenize(self, items: str | Iterable[str]) -> list[str]:
        if isinstance(items, str):
            texts = [items]
        else:
            texts = list(items)
        tokens: list[str] = []
        for t in texts:
            tokens.extend(t.split())
        return tokens

    def _tfidf_score(self, tokens: List[str]) -> float:
        text = " ".join(tokens)
        vec = self._get_vectorizer(text)
        try:
            row = vec.transform([text])
        except Exception:  # pragma: no cover - stub fallback
            row = vec.fit_transform([text])
        arr = row.toarray().ravel()
        if arr.size == 0:
            return 0.0
        k = min(30, arr.size)
        top_k = np.sort(arr)[-k:]
        return float(np.sum(top_k) / 30.0)

    def _entropy_score(self, tokens: List[str]) -> float:
        counts = Counter(tokens)
        n = len(tokens)
        probs = [c / n for c in counts.values()]
        ent = -sum(p * log(p) for p in probs if p > 0)
        max_ent = log(len(counts)) if len(counts) > 1 else 1.0
        return float(ent / max_ent)

    def _pmi_score(self, tokens: List[str]) -> float:
        if len(tokens) < 2:
            return 0.0

        pair_freq: Counter[tuple[str, str]] = Counter()
        word_freq = Counter(tokens)
        for i, w1 in enumerate(tokens[:-1]):
            for w2 in tokens[i + 1 : i + 1 + self._window]:
                pair_freq[(w1, w2)] += 1

        total_pairs = sum(pair_freq.values())
        pmi_sum = 0.0
        for (w1, w2), freq in pair_freq.items():
            p_xy = freq / total_pairs
            p_x = word_freq[w1] / len(tokens)
            p_y = word_freq[w2] / len(tokens)
            pmi_sum += np.log2(p_xy / (p_x * p_y))

        return float(pmi_sum / total_pairs) if total_pairs else 0.0

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
