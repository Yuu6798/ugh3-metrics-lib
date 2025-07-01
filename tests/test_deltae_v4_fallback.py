from ugh3_metrics.metrics.deltae_v4 import DeltaEV4
import numpy as np


class ZeroEmb:
    def encode(self, _: str):  # noqa: D401
        return np.zeros(2)


def test_zero_vector_fallback() -> None:
    m = DeltaEV4()
    m.set_params(embedder=ZeroEmb())
    assert m.score("abc", "abd") > 0.0 and m.score("same", "same") == 0.0
