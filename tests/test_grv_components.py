import numpy as np
from ugh3_metrics.utils import tokenize, tfidf_topk, entropy, pmi

TOKENS = "alpha beta beta gamma".split()


def test_tfidf_topk() -> None:
    assert tfidf_topk(" ".join(TOKENS)) >= 0.0


def test_entropy() -> None:
    assert entropy(TOKENS) >= 0.0


def test_pmi() -> None:
    assert pmi(TOKENS, window=2) >= -1.0
