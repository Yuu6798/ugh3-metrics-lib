from ugh3_metrics.metrics.deltae_v4 import DeltaE4
import numpy as np
from numpy.typing import NDArray


class ZeroEmb:
    def encode(self, _: str) -> NDArray[np.float64]:  # noqa: D401
        """Return a 2-D zero vector for fallback check."""
        return np.zeros(2, dtype=float)


def test_zero_vector_fallback() -> None:
    m = DeltaE4(embedder=ZeroEmb())
    assert m.score("abc", "abd") > 0.0 and m.score("same", "same") == 0.0
