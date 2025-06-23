from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Iterable, List

import numpy as np
from numpy.typing import NDArray

from ugh3_metrics.metrics.base import BaseMetric, Dataclass
from ugh3_metrics.utils import math_ops

DEFAULT_WEIGHTS: NDArray[np.float32] = np.array([0.42, 0.31, 0.27], dtype=np.float32)


@dataclass
class GrvParams:
    """Parameters for :class:`GrvV4`."""

    weights: NDArray[np.float32] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    window: int = 50


def _tokenize(items: str | Iterable[str]) -> list[str]:
    if isinstance(items, str):
        texts = [items]
    else:
        texts = list(items)
    tokens: list[str] = []
    for t in texts:
        tokens.extend(t.split())
    return tokens


def _entropy_score(tokens: List[str]) -> float:
    counts = Counter(tokens)
    n = len(tokens)
    probs = [c / n for c in counts.values()]
    ent = -sum(p * np.log(p) for p in probs if p > 0)
    max_ent = np.log(len(counts)) if len(counts) > 1 else 1.0
    return float(ent / max_ent)


class GrvV4(BaseMetric):
    """Vocabulary gravity metric v4."""

    def __init__(self, params: GrvParams = GrvParams()) -> None:
        self.params = params

    def score(self, *, x: str, y: str) -> float:  # noqa: D401 - see BaseMetric
        tokens = _tokenize(y)
        if not tokens:
            return 0.0
        tfidf_val = math_ops.tf_idf(tokens)
        entropy_val = _entropy_score(tokens)
        pmi_val = math_ops.pmi(tokens)
        w = self.params.weights / self.params.weights.sum()
        vals = np.array([tfidf_val, entropy_val, pmi_val], dtype=np.float32)
        x_val = float(np.dot(w, vals))
        return float(1.0 / (1.0 + np.exp(-x_val)))

    def set_params(self, params: Dataclass | dict[str, Any]) -> None:
        if not params:
            return
        if isinstance(params, GrvParams):
            self.params = params
            return
        if isinstance(params, dict):
            if "weights" in params:
                arr = np.asarray(list(params["weights"]), dtype=np.float32)
                if arr.size != 3:
                    raise ValueError("weights must have length 3")
                self.params.weights = arr
            if "window" in params:
                self.params.window = int(params["window"])
            return
        if is_dataclass(params):
            data = asdict(params)
            if "weights" in data:
                arr = np.asarray(list(data["weights"]), dtype=np.float32)
                if arr.size != 3:
                    raise ValueError("weights must have length 3")
                self.params.weights = arr
            if "window" in data:
                self.params.window = int(data["window"])


__all__ = ["GrvV4", "GrvParams"]
