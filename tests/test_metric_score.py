from typing import Any

import pytest


@pytest.mark.parametrize(  # type: ignore[misc]
    "text_a,text_b",
    [
        ("alpha beta", "gamma"),
        ("a b", ""),
        ("word " * 30, "word"),
    ],
)
def test_score_smoke(metric_cls: type, dummy_emb: Any, text_a: str, text_b: str) -> None:
    metric = metric_cls(embedder=dummy_emb)
    val = metric.score(text_a, text_b)
    assert 0.0 <= val <= 1.0
