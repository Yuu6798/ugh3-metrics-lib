"""Legacy wrapper for grv v4 metric."""

from __future__ import annotations

from ugh3_metrics.metrics import GrvV4, calc_grv_v4

# Old API aliases
score = calc_grv_v4


def set_params(**_: object) -> None:  # pragma: no cover - retained for API
    pass

__all__ = ["GrvV4", "calc_grv_v4", "score", "set_params"]
