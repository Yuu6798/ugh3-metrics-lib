from __future__ import annotations

import pytest

import ugh3_metrics.metrics.deltae_v4 as dmod
from ugh3_metrics.metrics.deltae_v4 import DeltaE4


class StubEmbedder:
    def encode(self, text: str) -> list[float]:
        # simple deterministic embedding: [length, vowel count]
        vowels = sum(c in 'aeiouAEIOU' for c in text)
        return [float(len(text)), float(vowels)]


def test_real_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """DeltaE4 should load a real embedder by default."""
    monkeypatch.setattr(dmod, "_load_embedder", lambda: StubEmbedder())
    metric = DeltaE4()
    assert abs(metric.score("A", "B")) > 0


@pytest.mark.xfail(reason="no network", strict=True)  # type: ignore[misc]
def test_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    from ugh3_metrics.metrics.deltae_v4 import DeltaE4 as _DE

    m = _DE()
    with pytest.raises(RuntimeError):
        m.score("a", "b")
