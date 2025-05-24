from __future__ import annotations

from typing import TypedDict


class PorTriggerResult(TypedDict):
    """Return type for :func:`por_trigger`."""

    E_prime: float
    score: float
    triggered: bool


__all__ = ["PorTriggerResult"]
