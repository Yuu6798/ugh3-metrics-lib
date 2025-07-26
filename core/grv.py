"""grv (Vocabulary Gravity) Scoring Module.

This module calculates a lightweight "grv" score representing vocabulary
variety within a text or a list of texts. The score is defined as the
number of unique space-separated words divided by ``vocab_limit`` and
capped at ``1.0``.
"""

from __future__ import annotations

from typing import Iterable
from pathlib import Path
import yaml
from math import log2


def load_grv_weights() -> tuple[float, float, float]:
    """Return GRV component weights from YAML, fallback to defaults."""
    path = Path(__file__).resolve().parents[1] / "config" / "grv.yaml"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        data = {}
    return (
        float(data.get("tfidf", 0.42)),
        float(data.get("entropy", 0.31)),
        float(data.get("cooccurrence", 0.27)),
    )


DEFAULT_WEIGHTS: tuple[float, float, float] = load_grv_weights()


def _collect_vocab(texts: Iterable[str]) -> set[str]:
    """Return a set of unique words from the provided texts."""
    vocab: set[str] = set()
    for item in texts:
        vocab.update(item.split())
    return vocab


def grv_score(text: str | list[str], *, vocab_limit: int = 30, mode: str = "simple") -> float:
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
        Score between 0.0 and 1.0 rounded to three decimals.
    """
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    vocab = _collect_vocab(texts)
    if mode == "simple":
        score = min(1.0, len(vocab) / float(vocab_limit))
        return round(score, 3)
    if mode == "entropy":
        tokens: list[str] = []
        for t in texts:
            tokens.extend(t.split())
        if not tokens:
            return 0.0
        ttr = len(vocab) / len(tokens)
        freqs = {w: tokens.count(w) / len(tokens) for w in vocab}
        denom = log2(len(vocab)) if len(vocab) > 1 else 1.0
        ent = -sum(p * log2(p) for p in freqs.values()) / denom
        return round((ttr + ent) / 2.0, 3)
    raise ValueError("invalid mode")


__all__ = ["grv_score", "load_grv_weights"]
