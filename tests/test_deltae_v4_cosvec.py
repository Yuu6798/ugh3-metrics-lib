from ugh3_metrics.metrics.deltae_v4 import DeltaEV4
import numpy as np


def test_score_cosvec() -> None:
    a = "hello"
    b = "world"
    metric = DeltaEV4()
    val = metric.score(a, b)
    v1 = np.asarray([len(a), hash(a) % 13], dtype=float)
    v2 = np.asarray([len(b), hash(b) % 13], dtype=float)
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    expected = round(1.0 - cos, 3)
    assert val == expected
