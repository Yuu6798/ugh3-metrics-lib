from ugh3_metrics.metrics import DeltaEV4

def test_deltae_hash_fallback() -> None:
    m = DeltaEV4()
    assert m.score("abc", "abd") > 0.0
    assert m.score("", "") == 0.0
