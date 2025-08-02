from ugh3_metrics.metrics.deltae_v4 import DeltaE4, DeltaEV4


def test_import() -> None:
    assert DeltaE4.__name__ == "DeltaE4"
    assert DeltaEV4 is DeltaE4
