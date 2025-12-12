from typing import Any

# mypy: disable-error-code=unused-ignore

import pytest


@pytest.mark.parametrize(  # type: ignore[misc,untyped-decorator]  # pytest decorator lacks typing
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
