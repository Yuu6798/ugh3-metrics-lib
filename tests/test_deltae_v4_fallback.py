from ugh3_metrics.metrics.deltae_v4 import DeltaE4
import numpy as np
from numpy.typing import NDArray
import pytest


class ZeroEmb:
    def encode(self, _: str) -> NDArray[np.float64]:  # noqa: D401
        """Return a 2-D zero vector for error check."""
        return np.zeros(2, dtype=float)


def test_zero_vector_error() -> None:
    m = DeltaE4(embedder=ZeroEmb())
    with pytest.raises(RuntimeError):
        m.score("abc", "abd")
    with pytest.raises(RuntimeError):
        m.score("same", "same")
