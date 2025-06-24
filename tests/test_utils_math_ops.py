import numpy as np
from ugh3_metrics.utils import cosine_similarity

def test_cosine_similarity() -> None:
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 0.0])
    assert cosine_similarity(v1, v2) == 1.0
