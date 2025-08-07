from __future__ import annotations

"""Adapters for computing UGH metrics using real embeddings.

This module provides thin wrappers around ``sentence-transformers`` to
calculate PoR (Proof of Response), ΔE (delta energy) and grv metrics.
The functions are intentionally lightweight so that higher level modules
can call them without worrying about model initialisation details.
"""

from typing import List, Tuple, Set

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

from core.history_entry import HistoryEntry
from utils.config_loader import CONFIG

LOGGER = logging.getLogger(__name__)

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer | None:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _MODEL
    except Exception as exc:
        LOGGER.warning(
            "SentenceTransformer init failed: %s; falling back to no-op metrics.",
            exc,
        )
        return None


def compute_por(q: str, a: str, theta: float = 0.82) -> float:
    """Return PoR as cosine similarity between question and answer embeddings."""
    model = _get_model()
    if model is None:
        LOGGER.warning("compute_por: no model; returning 0.0")
        return 0.0
    vec_q, vec_a = model.encode([q, a], normalize_embeddings=True)
    sim = float(np.dot(vec_q, vec_a))
    return max(0.0, min(1.0, sim))


def compute_delta_e_embed(prev_q: str, cur_q: str, a: str) -> float:
    """Compute ΔE between previous question and new state (current Q + A)."""
    model = _get_model()
    if model is None:
        LOGGER.warning("compute_delta_e_embed: no model; returning 0.0")
        return 0.0
    prev_vec, new_vec = model.encode([prev_q, f"{cur_q} {a}"], normalize_embeddings=True)
    sim = float(np.dot(prev_vec, new_vec))
    delta_e = 1.0 - sim
    return max(0.0, min(1.0, delta_e))


def compute_grv_window(history_list: List[HistoryEntry]) -> Tuple[float, Set[str]]:
    """Compute grv from recent history using type-token ratio.

    The calculation uses ``GRV_WINDOW`` entries from configuration. grv is a
    weighted combination of ``1 - TTR`` and vocabulary size.
    """
    window = int(CONFIG.get("GRV_WINDOW", 5))
    recent = history_list[-window:] if len(history_list) >= window else history_list
    tokens: List[str] = []
    for entry in recent:
        tokens.extend(entry.question.split())
        tokens.extend(entry.answer_b.split())
    vocab: Set[str] = set(tokens)
    total_tokens = len(tokens)
    ttr = len(vocab) / total_tokens if total_tokens else 1.0
    grv = 0.7 * (1 - ttr) + 0.3 * (len(vocab) / 30.0)
    grv = max(0.0, min(1.0, grv))
    return round(grv, 3), vocab


__all__ = ["compute_por", "compute_delta_e_embed", "compute_grv_window"]
