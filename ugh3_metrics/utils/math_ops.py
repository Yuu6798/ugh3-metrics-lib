from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


def cosine(v1: NDArray[np.float32], v2: NDArray[np.float32]) -> float:
    """Return cosine similarity of two vectors."""
    num = float(np.dot(v1, v2))
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return num / denom if denom else 0.0


def tf_idf(tokens: Iterable[str]) -> float:
    """Return naive TF-IDF density."""
    items = list(tokens)
    if not items:
        return 0.0
    counts = Counter(items)
    tf = np.array(list(counts.values()), dtype=np.float32) / float(len(items))
    k = min(30, tf.size)
    top_k = np.sort(tf)[-k:]
    return float(np.sum(top_k) / 30.0)


def pmi(tokens: Iterable[str]) -> float:
    """Return simple pairwise PMI."""
    items = list(tokens)
    if len(items) < 2:
        return 0.0

    pair_freq: Counter[tuple[str, str]] = Counter()
    word_freq = Counter(items)
    for i, w1 in enumerate(items[:-1]):
        for w2 in items[i + 1 : i + 51]:
            pair_freq[(w1, w2)] += 1

    total_pairs = sum(pair_freq.values())
    pmi_sum = 0.0
    for (w1, w2), freq in pair_freq.items():
        p_xy = freq / total_pairs
        p_x = word_freq[w1] / len(items)
        p_y = word_freq[w2] / len(items)
        pmi_sum += np.log2(p_xy / (p_x * p_y))

    return float(pmi_sum / total_pairs) if total_pairs else 0.0
