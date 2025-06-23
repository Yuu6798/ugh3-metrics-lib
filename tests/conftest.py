from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


class DummyEmbedder:
    """Simple deterministic embedder for testing."""

    def encode(self, text: str) -> NDArray[np.float32]:
        if text == "hello":
            return np.array([1.0, 0.0], dtype=np.float32)
        if text == "hello world":
            return np.array([0.62, np.sqrt(1 - 0.62**2)], dtype=np.float32)
        n = float(len(text.split()))
        return np.array([n, 0.0], dtype=np.float32)


@pytest.fixture()
def dummy_embedder() -> DummyEmbedder:
    return DummyEmbedder()


def metric_classes() -> list[type]:
    from ugh3_metrics import DeltaEV4, GrvV4, PorV4, SciV4

    return [GrvV4, PorV4, DeltaEV4, SciV4]
