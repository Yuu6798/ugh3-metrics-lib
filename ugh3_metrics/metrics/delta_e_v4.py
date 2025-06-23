from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ugh3_metrics.metrics.base import BaseMetric, Dataclass
from ugh3_metrics.models.embedder import DefaultEmbedder, EmbedderProtocol
from ugh3_metrics.utils import math_ops

DEFAULT_MU: float = 0.35
DEFAULT_SIGMA: float = 0.08


@dataclass
class DeltaEParams:
    """Parameters for :class:`DeltaEV4`."""

    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA


class DeltaEV4(BaseMetric):
    """Semantic difference metric v4."""

    def __init__(
        self,
        embedder: Optional[EmbedderProtocol] = None,
        *,
        params: DeltaEParams = DeltaEParams(),
    ) -> None:
        self.embedder = embedder or DefaultEmbedder()
        self.params = params

    def score(self, *, x: str, y: str) -> float:  # noqa: D401 - see BaseMetric
        v1 = self.embedder.encode(x)
        v2 = self.embedder.encode(y)
        sim = math_ops.cosine(v1.astype(np.float32), v2.astype(np.float32))
        diff = 1.0 - sim
        mu = self.params.mu
        sigma = self.params.sigma
        z = (diff - mu) / sigma if sigma else diff - mu
        return float(np.clip(z, 0.0, 1.0))

    def set_params(self, params: Dataclass | dict[str, Any]) -> None:
        if not params:
            return
        if isinstance(params, DeltaEParams):
            self.params = params
            return
        if isinstance(params, dict):
            if "mu" in params:
                self.params.mu = float(params["mu"])
            if "sigma" in params:
                self.params.sigma = float(params["sigma"])
            return
        if is_dataclass(params):
            data = asdict(params)
            if "mu" in data:
                self.params.mu = float(data["mu"])
            if "sigma" in data:
                self.params.sigma = float(data["sigma"])


__all__ = ["DeltaEV4", "DeltaEParams"]
