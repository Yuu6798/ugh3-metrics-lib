"""grv (Vocabulary Gravity) Scoring Module.

This module calculates a lightweight "grv" score representing vocabulary
variety within a text or a list of texts. The score is defined as the
number of unique space-separated words divided by ``vocab_limit`` and
capped at ``1.0``.
"""

from __future__ import annotations

from typing import Iterable


def _collect_vocab(texts: Iterable[str]) -> set[str]:
    """Return a set of unique words from the provided texts."""
    vocab: set[str] = set()
    for item in texts:
        vocab.update(item.split())
    return vocab


def grv_score(text: str | list[str], *, vocab_limit: int = 30) -> float:
    """Calculate vocabulary gravity (grv) for the given text or list.

    Parameters
    ----------
    text : str | list[str]
        Text string or list of strings to analyze.
    vocab_limit : int, optional
        Normalization denominator. Defaults to ``30``.

    Returns
    -------
    float
        ``min(1.0, vocab_size / vocab_limit)`` rounded to three decimals.
    """
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    vocab = _collect_vocab(texts)
    score = min(1.0, len(vocab) / float(vocab_limit))
    return round(score, 3)
