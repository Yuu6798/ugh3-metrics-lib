"""Metric helpers including ΔE simulation using SBERT."""

from __future__ import annotations

import os
import random
from difflib import SequenceMatcher
from typing import Optional, cast, Any

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = cast(Any, None)
    util = None

SBERT_MODEL_ID = os.getenv("SBERT_MODEL_ID", "sentence-transformers/paraphrase-MiniLM-L6-v2")
_ST_MODEL: Optional[SentenceTransformer] = None


def get_sbert() -> SentenceTransformer:
    """Return a singleton SBERT model (lazy-loaded)."""
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer(SBERT_MODEL_ID)
    return _ST_MODEL

# --- ΔE 計算係数 ---
LEN_COEFF = 0.1  # 旧 0.5
COS_COEFF = 0.7
RAND_COEFF = 0.2


def simulate_delta_e(prev_q: str, curr_q: str, answer: str) -> float:
    """Return simulated ΔE based on length gap and semantic distance."""
    q_len_gap = abs(len(prev_q) - len(curr_q)) / 30.0

    sem_gap = 1.0
    try:
        sem_gap = 1.0 - float(util.cos_sim(get_sbert().encode(prev_q), get_sbert().encode(answer)))
    except Exception:
        sem_gap = 1.0 - SequenceMatcher(None, prev_q, answer).ratio()

    rand = random.uniform(0.1, 0.8)
    delta_e = min(1.0, LEN_COEFF * q_len_gap + COS_COEFF * sem_gap + RAND_COEFF * rand)
    return round(delta_e, 3)


__all__ = ["simulate_delta_e", "LEN_COEFF", "COS_COEFF", "RAND_COEFF"]
