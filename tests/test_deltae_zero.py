from ugh3_metrics.metrics import DeltaEV4

def test_deltae_nonzero() -> None:
    m = DeltaEV4()
    assert m.score("abc", "abd") > 0.0
