"""Legacy wrapper for PoR v4 metric."""

from __future__ import annotations

from ugh3_metrics.metrics.por_v4 import PorV4


def calc_por_v4(a: str, b: str) -> float:
    """Backward-compatible entry point."""
    return PorV4().score(a, b)


# Old API aliases
score = calc_por_v4


def set_params(**_: object) -> None:  # pragma: no cover - retained for API
    pass


__all__ = ["calc_por_v4", "score", "set_params"]
