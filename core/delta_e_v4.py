"""Legacy wrapper for ΔE v4 metric."""

from __future__ import annotations

from ugh3_metrics.metrics import DeltaE4
from ugh3_metrics.metrics.deltae_v4 import _EmbedderProto

_METRIC = DeltaE4(fallback="hash")


def calc_deltae_v4(a: str, b: str) -> float:
    """Backward-compatible entry point."""
    return float(_METRIC.score(a, b))


# Old API aliases
delta_e_v4 = calc_deltae_v4
score = calc_deltae_v4


def set_params(embedder: _EmbedderProto | None = None) -> None:  # pragma: no cover - retained for API
    """Update metric parameters in a type-safe manner."""
    _METRIC.set_params(embedder=embedder)


__all__ = ["calc_deltae_v4", "delta_e_v4", "score", "set_params"]
