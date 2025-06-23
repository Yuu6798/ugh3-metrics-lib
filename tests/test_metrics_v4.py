from __future__ import annotations

import pytest

from ugh3_metrics import DeltaEV4, GrvV4, PorV4, SciV4
from tests.conftest import metric_classes


@pytest.mark.parametrize(
    "x,y",
    [
        ("hello", "hello world"),
        ("a", "b"),
        ("quick brown", "fox jumps"),
        ("one", "two"),
        ("alpha beta", "beta gamma"),
        ("foo", "bar"),
    ],
)
@pytest.mark.parametrize("cls", metric_classes())
def test_metric_range(cls: type, x: str, y: str, dummy_embedder: object) -> None:
    if cls is GrvV4:
        metric = cls()
    elif cls is SciV4:
        metric = cls(
            por_metric=PorV4(embedder=dummy_embedder),
            delta_metric=DeltaEV4(embedder=dummy_embedder),
            grv_metric=GrvV4(),
        )
    else:
        metric = cls(embedder=dummy_embedder)  # type: ignore[arg-type]
    val = metric.score(x=x, y=y)
    assert 0.0 <= val <= 1.0
