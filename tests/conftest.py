import sys
from pathlib import Path

import pytest
import numpy as np
from numpy.typing import NDArray

# mypy: disable-error-code=unused-ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import cast

from ugh3_metrics.metrics import PorV4, GrvV4


class DummyEmbedder:
    def encode(self, text: str) -> NDArray[np.float_]:
        """Return simple embedding vector based on token count."""
        n = float(len(text.split()))
        return np.array([n, 0.0], dtype=float)


@pytest.fixture(scope="session")  # type: ignore[misc,untyped-decorator]  # pytest decorator lacks typing
def dummy_emb() -> DummyEmbedder:
    return DummyEmbedder()


@pytest.fixture(params=[PorV4, GrvV4])  # type: ignore[misc,untyped-decorator]  # pytest decorator lacks typing
def metric_cls(request: pytest.FixtureRequest) -> type:
    return cast(type, request.param)
