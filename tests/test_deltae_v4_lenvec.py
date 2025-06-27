from ugh3_metrics.metrics.deltae_v4 import DeltaEV4


def test_score_lenvec() -> None:
    m = DeltaEV4()
    assert m.score("hello", "world!") > 0 and m.score("same", "same") == 0
