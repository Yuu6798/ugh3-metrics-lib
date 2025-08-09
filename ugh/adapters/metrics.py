"""Adapters for computing UGH metrics using real embeddings.

This module provides thin wrappers around ``sentence-transformers`` to
calculate PoR (Proof of Response), ΔE (delta energy) and grv metrics.
The functions are intentionally lightweight so that higher level modules
can call them without worrying about model initialisation details.
"""

from __future__ import annotations

from typing import List, Tuple, Set, Optional, TYPE_CHECKING, cast

import logging
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer
from utils.config_loader import CONFIG

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from core.history_entry import HistoryEntry

LOGGER = logging.getLogger(__name__)

_MODEL: Optional[SentenceTransformer] = None


def _get_model() -> Optional[SentenceTransformer]:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _MODEL
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.warning(
            "SentenceTransformer init failed: %s; falling back to no-op metrics.",
            exc,
        )
        return None


def _norm(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def compute_por(q: str, a: str, theta: float = 0.82) -> float:  # noqa: ARG001
    """Return PoR as cosine similarity between question and answer embeddings."""
    model = _get_model()
    if model is None:
        LOGGER.warning("compute_por: no model; returning 0.0")
        return 0.0
    qv = _norm(cast(npt.NDArray[np.float32], model.encode(q)))
    av = _norm(cast(npt.NDArray[np.float32], model.encode(a)))
    sim = float(np.dot(qv, av))
    return max(0.0, min(1.0, sim))


def compute_delta_e_embed(prev_q: str, cur_q: str, a: str) -> float:
    """Compute ΔE between previous question and new state (current Q + A)."""
    model = _get_model()
    if model is None:
        LOGGER.warning("compute_delta_e_embed: no model; returning 0.0")
        return 0.0
    prev_v = _norm(cast(npt.NDArray[np.float32], model.encode(prev_q)))
    now_v = _norm(cast(npt.NDArray[np.float32], model.encode(f"{cur_q} {a}")))
    sim = float(np.dot(prev_v, now_v))
    sim = max(0.0, min(1.0, sim))
    return round(1.0 - sim, 3)


def compute_grv_window(history_list: List["HistoryEntry"]) -> Tuple[float, Set[str]]:
    """Compute grv from recent history using type-token ratio."""
    window = CONFIG.get("GRV_WINDOW", 5)
    recent = history_list[-window:] if len(history_list) >= window else history_list
    vocab: Set[str] = set()
    tokens: List[str] = []
    for entry in recent:
        qt = entry.question.split()
        at = entry.answer_b.split()
        tokens.extend(qt)
        tokens.extend(at)
        vocab |= set(qt) | set(at)
    total = len(tokens)
    if total == 0:
        return 0.0, set()
    ttr = len(vocab) / total
    grv = min(1.0, max(0.0, 0.7 * (1 - ttr) + 0.3 * (len(vocab) / 30.0)))
    return round(grv, 3), vocab


__all__ = ["compute_por", "compute_delta_e_embed", "compute_grv_window"]
