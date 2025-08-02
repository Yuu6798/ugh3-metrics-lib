import hashlib
import numpy as np

from ugh3_metrics.metrics.deltae_v4 import DeltaE4


def test_score_lenvec() -> None:
    m = DeltaE4(fallback="hash")
    h1 = int.from_bytes(hashlib.md5("hello".encode()).digest()[:4], "big", signed=True) / 2**31
    h2 = int.from_bytes(hashlib.md5("world!".encode()).digest()[:4], "big", signed=True) / 2**31
    v1 = np.asarray([5, h1], dtype=float)
    v2 = np.asarray([6, h2], dtype=float)
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    expected = round(1.0 - cos, 3)
    assert m.score("hello", "world!") == expected
    assert m.score("same", "same") == 0.0
