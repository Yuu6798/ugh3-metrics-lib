"""Vocabulary Gravity v4 metric."""

from __future__ import annotations

import numpy as np
from collections import Counter
from math import log
from typing import Iterable, List, Optional, Any
from numpy.typing import NDArray

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

DEFAULT_WEIGHTS: NDArray[Any] = np.array([0.42, 0.31, 0.27])

_VECTORIZER: Optional[TfidfVectorizer] = None


def set_params(weights: Iterable[float]) -> None:
    """Set feature weights."""
    global DEFAULT_WEIGHTS
    arr = np.asarray(list(weights), dtype=float)
    if arr.size != 3:
        raise ValueError("weights must have length 3")
    DEFAULT_WEIGHTS = arr


def _get_vectorizer(text: str) -> TfidfVectorizer:
    """Return cached ``TfidfVectorizer`` fitted on first real text."""
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = TfidfVectorizer()
    if not getattr(_VECTORIZER, "vocabulary_", None):
        try:
            _VECTORIZER.fit([text])
        except Exception:  # pragma: no cover - fallback for stub
            pass
    return _VECTORIZER


def _tokenize(items: str | Iterable[str]) -> list[str]:
    if isinstance(items, str):
        texts = [items]
    else:
        texts = list(items)
    tokens: list[str] = []
    for t in texts:
        tokens.extend(t.split())
    return tokens


def _tfidf_score(tokens: List[str]) -> float:
    """Return TF-IDF density using a cached vectorizer."""
    text = " ".join(tokens)
    vec = _get_vectorizer(text)
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


def _entropy_score(tokens: List[str]) -> float:
    counts = Counter(tokens)
    n = len(tokens)
    probs = [c / n for c in counts.values()]
    ent = -sum(p * log(p) for p in probs if p > 0)
    max_ent = log(len(counts)) if len(counts) > 1 else 1.0
    return float(ent / max_ent)


def _pmi_score(tokens: List[str]) -> float:
    """Return average pairwise PMI within a 50-token window."""
    if len(tokens) < 2:
        return 0.0

    pair_freq: Counter[tuple[str, str]] = Counter()
    word_freq = Counter(tokens)
    for i, w1 in enumerate(tokens[:-1]):
        for w2 in tokens[i + 1 : i + 51]:
            pair_freq[(w1, w2)] += 1

    total_pairs = sum(pair_freq.values())
    pmi_sum = 0.0
    for (w1, w2), freq in pair_freq.items():
        p_xy = freq / total_pairs
        p_x = word_freq[w1] / len(tokens)
        p_y = word_freq[w2] / len(tokens)
        pmi_sum += np.log2(p_xy / (p_x * p_y))

    return float(pmi_sum / total_pairs) if total_pairs else 0.0


def grv(
    text: str | Iterable[str],
    *,
    weights: Iterable[float] | None = None,
) -> float:
    """Return vocabulary gravity using TF-IDF, entropy and PMI."""
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    w = (
        np.asarray(list(weights), dtype=float)
        if weights is not None
        else DEFAULT_WEIGHTS
    )
    if w.size != 3:
        raise ValueError("weights must have length 3")
    w = w / w.sum()
    vals = np.array([
        _tfidf_score(tokens),
        _entropy_score(tokens),
        _pmi_score(tokens),
    ])
    x = float(np.dot(w, vals))
    return float(1.0 / (1.0 + np.exp(-x)))


# backward compatibility
score = grv


__all__ = ["score", "set_params"]
