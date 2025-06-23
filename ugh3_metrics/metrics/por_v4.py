from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ugh3_metrics.metrics.base import BaseMetric, Dataclass
from ugh3_metrics.models.embedder import DefaultEmbedder, EmbedderProtocol
from ugh3_metrics.utils import math_ops

DEFAULT_ALPHA: float = 13.2
DEFAULT_BETA: float = -10.8


@dataclass
class PorParams:
    """Parameters for :class:`PorV4`."""

    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA


class PorV4(BaseMetric):
    """Point of Resonance metric v4."""

    def __init__(
        self,
        embedder: Optional[EmbedderProtocol] = None,
        *,
        params: PorParams = PorParams(),
    ) -> None:
        self.embedder = embedder or DefaultEmbedder()
        self.params = params

    def score(self, *, x: str, y: str) -> float:  # noqa: D401 - see BaseMetric
        v1 = self.embedder.encode(x)
        v2 = self.embedder.encode(y)
        sim = math_ops.cosine(v1.astype(np.float32), v2.astype(np.float32))
        value = self.params.alpha * sim + self.params.beta
        return float(1.0 / (1.0 + np.exp(-value)))

    def set_params(self, params: Dataclass | dict[str, Any]) -> None:
        if not params:
            return
        if isinstance(params, PorParams):
            self.params = params
            return
        if isinstance(params, dict):
            if "alpha" in params:
                self.params.alpha = float(params["alpha"])
            if "beta" in params:
                self.params.beta = float(params["beta"])
            return
        if is_dataclass(params):
            data = asdict(params)
            if "alpha" in data:
                self.params.alpha = float(data["alpha"])
            if "beta" in data:
                self.params.beta = float(data["beta"])


__all__ = ["PorV4", "PorParams"]
