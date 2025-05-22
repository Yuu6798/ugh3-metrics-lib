"""Metric helpers including ΔE simulation using SBERT."""

from __future__ import annotations

import random
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    SBERT = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception:  # pragma: no cover - optional dependency
    SBERT = None

# --- ΔE 計算係数 ---
LEN_COEFF = 0.1  # 旧 0.5
COS_COEFF = 0.7
RAND_COEFF = 0.2


def simulate_delta_e(prev_q: str, curr_q: str, answer: str) -> float:
    """Return simulated ΔE based on length gap and semantic distance."""
    q_len_gap = abs(len(prev_q) - len(curr_q)) / 30.0

    sem_gap = 1.0
    if SBERT is not None:
        try:
            sem_gap = 1.0 - float(util.cos_sim(SBERT.encode(prev_q), SBERT.encode(answer)))
        except Exception:
            sem_gap = 1.0 - SequenceMatcher(None, prev_q, answer).ratio()
    else:
        sem_gap = 1.0 - SequenceMatcher(None, prev_q, answer).ratio()

    rand = random.uniform(0.1, 0.8)
    delta_e = min(1.0, LEN_COEFF * q_len_gap + COS_COEFF * sem_gap + RAND_COEFF * rand)
    return round(delta_e, 3)


__all__ = ["simulate_delta_e", "LEN_COEFF", "COS_COEFF", "RAND_COEFF"]
