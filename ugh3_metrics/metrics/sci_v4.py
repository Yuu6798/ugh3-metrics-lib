from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ugh3_metrics.metrics.base import BaseMetric, Dataclass
from ugh3_metrics.metrics.delta_e_v4 import DeltaEV4
from ugh3_metrics.metrics.grv_v4 import GrvV4
from ugh3_metrics.metrics.por_v4 import PorV4

DEFAULT_WEIGHTS: NDArray[np.float32] = np.array([0.4, 0.45, 0.15], dtype=np.float32)
DEFAULT_ALPHA: float = 0.3


@dataclass
class SciParams:
    """Parameters for :class:`SciV4`."""

    alpha: float = DEFAULT_ALPHA


class SciV4(BaseMetric):
    """Semantic Consistency Index v4."""

    _state: float | None = None

    def __init__(
        self,
        por_metric: Optional[PorV4] = None,
        delta_metric: Optional[DeltaEV4] = None,
        grv_metric: Optional[GrvV4] = None,
        *,
        params: SciParams = SciParams(),
    ) -> None:
        self.por_metric = por_metric or PorV4()
        self.delta_metric = delta_metric or DeltaEV4()
        self.grv_metric = grv_metric or GrvV4()
        self.params = params

    def score(self, *, x: str, y: str) -> float:  # noqa: D401 - see BaseMetric
        por = self.por_metric.score(x=x, y=y)
        delta_e = self.delta_metric.score(x=x, y=y)
        grv = self.grv_metric.score(x=x, y=y)
        w = DEFAULT_WEIGHTS / float(np.sum(DEFAULT_WEIGHTS))
        vals = np.array([por, 1.0 - delta_e, grv], dtype=np.float32)
        x_val = float(np.dot(w, vals))
        alpha = self.params.alpha
        if SciV4._state is None:
            SciV4._state = x_val
        else:
            SciV4._state = alpha * x_val + (1 - alpha) * SciV4._state
        return float(SciV4._state)

    def set_params(self, params: Dataclass | dict[str, Any]) -> None:
        if not params:
            return
        if isinstance(params, SciParams):
            self.params = params
            return
        if isinstance(params, dict):
            if "alpha" in params:
                self.params.alpha = float(params["alpha"])
            return
        if is_dataclass(params):
            data = asdict(params)
            if "alpha" in data:
                self.params.alpha = float(data["alpha"])

    @classmethod
    def reset_state(cls) -> None:
        """Reset internal EMA state."""
        cls._state = None


__all__ = ["SciV4", "SciParams"]
