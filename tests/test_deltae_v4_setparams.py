from ugh3_metrics.metrics.deltae_v4 import DeltaE4


class DummyEmb:
    def encode(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_set_params_embedder() -> None:
    metric = DeltaE4(fallback="hash")
    metric.set_params(embedder=DummyEmb())
    assert metric.score("a", "b") == 0.0
