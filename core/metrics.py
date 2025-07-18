"""Utility helpers for metric thresholds."""

from __future__ import annotations

POR_FIRE_THRESHOLD: float = 0.82
"""Threshold at which PoR is considered fired."""


def is_por_fire(score: float) -> bool:
    """True if score >= POR_FIRE_THRESHOLD."""
    return score >= POR_FIRE_THRESHOLD

__all__ = ["POR_FIRE_THRESHOLD", "is_por_fire"]
