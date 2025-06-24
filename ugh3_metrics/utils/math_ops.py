from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["cosine_similarity"]


def cosine_similarity(v1: NDArray, v2: NDArray) -> float:
    """Return cosine similarity between two vectors."""
    num = float(np.dot(v1, v2))
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    return num / denom if denom else 0.0
