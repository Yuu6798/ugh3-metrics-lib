from __future__ import annotations

from collections import Counter
from math import log
from typing import Iterable, List

import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - fallback if sklearn missing

    class _DummyMatrix:
        def __init__(self) -> None:
            self._arr = np.zeros((1, 0))

        def toarray(self) -> NDArray:
            return self._arr

    class TfidfVectorizer:  # type: ignore
        def __init__(self, token_pattern: str | None = None) -> None:
            self.token_pattern = token_pattern

        def fit_transform(self, docs: Iterable[str]) -> _DummyMatrix:
            return _DummyMatrix()


__all__ = [
    "tokenize",
    "tfidf_topk",
    "entropy",
    "pmi",
]


def tokenize(text: str | Iterable[str]) -> list[str]:
    """Split text or iterable of texts into whitespace tokens."""
    if isinstance(text, str):
        texts = [text]
    else:
        texts = list(text)
    tokens: list[str] = []
    for t in texts:
        tokens.extend(t.split())
    return tokens


def tfidf_topk(text: str, *, k: int = 30) -> float:
    """Return average of top-k TF-IDF values for the given text."""
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    try:
        row = vec.fit_transform([text])
    except ValueError as exc:  # pragma: no cover - handle empty vocab
        if "empty vocabulary" in str(exc):
            return 0.0
        raise
    arr = row.toarray().ravel()
    if arr.size == 0:
        return 0.0
    k_top = min(k, arr.size)
    top_k = np.sort(arr)[-k_top:]
    return float(np.sum(top_k) / float(k))


def entropy(tokens: List[str]) -> float:
    """Normalized entropy of token distribution."""
    counts = Counter(tokens)
    n = len(tokens)
    if n == 0:
        return 0.0
    probs = [c / n for c in counts.values()]
    ent = -sum(p * log(p) for p in probs if p > 0)
    max_ent = log(len(counts)) if len(counts) > 1 else 1.0
    return float(ent / max_ent)


def pmi(tokens: List[str], *, window: int = 50) -> float:
    """Average pointwise mutual information over token pairs."""
    if len(tokens) < 2:
        return 0.0

    pair_freq: Counter[tuple[str, str]] = Counter()
    word_freq = Counter(tokens)
    for i, w1 in enumerate(tokens[:-1]):
        for w2 in tokens[i + 1 : i + 1 + window]:
            pair_freq[(w1, w2)] += 1

    total_pairs = sum(pair_freq.values())
    pmi_sum = 0.0
    for (w1, w2), freq in pair_freq.items():
        p_xy = freq / total_pairs
        p_x = word_freq[w1] / len(tokens)
        p_y = word_freq[w2] / len(tokens)
        pmi_sum += np.log2(p_xy / (p_x * p_y))

    return float(pmi_sum / total_pairs) if total_pairs else 0.0
