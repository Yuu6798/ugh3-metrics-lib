"""Utility helpers for metric thresholds and simple metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

POR_FIRE_THRESHOLD: float = 0.82
"""Threshold at which PoR is considered fired."""


def is_por_fire(score: float) -> bool:
    """True if score >= POR_FIRE_THRESHOLD."""
    return score >= POR_FIRE_THRESHOLD


def calc_delta_e_internal(prev_vec: NDArray[np.float64], new_vec: NDArray[np.float64]) -> float:
    """ΔE (internal) = 1 - cosine_similarity(prev_vec, new_vec), clipped to [0,1]."""
    sim = float(cosine_similarity(prev_vec.reshape(1, -1), new_vec.reshape(1, -1)))
    delta = 1.0 - sim
    return max(0.0, min(1.0, delta))


__all__ = ["POR_FIRE_THRESHOLD", "is_por_fire", "calc_delta_e_internal"]
