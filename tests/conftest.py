import sys
from pathlib import Path

import pytest
import numpy as np
from typing import Any
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import cast

from ugh3_metrics.metrics import PorV4, DeltaEV4, GrvV4


class DummyEmbedder:
    def encode(self, text: str) -> NDArray[np.float_]:
        """Return simple embedding vector based on token count."""
        n = float(len(text.split()))
        return np.array([n, 0.0], dtype=float)


@pytest.fixture(scope="session")  # type: ignore[misc]
def dummy_emb() -> DummyEmbedder:
    return DummyEmbedder()


@pytest.fixture(params=[PorV4, DeltaEV4, GrvV4])  # type: ignore[misc]
def metric_cls(request: pytest.FixtureRequest) -> type:
    return cast(type, request.param)
