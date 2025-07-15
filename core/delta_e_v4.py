"""Legacy wrapper for ΔE v4 metric."""

from __future__ import annotations

from ugh3_metrics.metrics import DeltaEV4

_METRIC = DeltaEV4()


def calc_deltae_v4(a: str, b: str) -> float:
    """Backward-compatible entry point."""
    return float(_METRIC.score(a, b))


# Old API aliases
delta_e_v4 = calc_deltae_v4
score = calc_deltae_v4


def set_params(**kwargs: object) -> None:  # pragma: no cover - retained for API
    _METRIC.set_params(**kwargs)


__all__ = ["calc_deltae_v4", "delta_e_v4", "score", "set_params"]
