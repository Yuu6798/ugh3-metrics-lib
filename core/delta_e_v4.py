"""Legacy wrapper for ΔE v4 metric."""

from __future__ import annotations

from ugh3_metrics.metrics import DeltaEV4


def calc_deltae_v4(a: str, b: str) -> float:
    """Backward-compatible entry point."""
    _ = (a, b)
    DeltaEV4()
    return 0.0


# Old API aliases
score = calc_deltae_v4


def set_params(**_: object) -> None:  # pragma: no cover - retained for API
    pass


__all__ = ["calc_deltae_v4", "score", "set_params"]
