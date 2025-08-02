import hashlib
import numpy as np

from ugh3_metrics.metrics.deltae_v4 import DeltaE4


def test_score_cosvec() -> None:
    a = "hello"
    b = "world"
    metric = DeltaE4(fallback="hash")
    val = metric.score(a, b)
    h1 = int.from_bytes(hashlib.md5(a.encode()).digest()[:4], "big", signed=True) / 2**31
    h2 = int.from_bytes(hashlib.md5(b.encode()).digest()[:4], "big", signed=True) / 2**31
    v1 = np.asarray([len(a), h1], dtype=float)
    v2 = np.asarray([len(b), h2], dtype=float)
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    expected = round(1.0 - cos, 3)
    assert val == expected
