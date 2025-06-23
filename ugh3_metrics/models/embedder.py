from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback if library unavailable

    class SentenceTransformer:  # type: ignore
        def __init__(self, *args: str, **kwargs: str) -> None:
            pass

        def encode(self, text: str) -> NDArray[np.float32]:  # pragma: no cover
            return np.zeros(1, dtype=np.float32)


class EmbedderProtocol(Protocol):
    """Minimal protocol for text embedding."""

    def encode(self, text: str) -> NDArray[np.float32]:
        """Return embedding vector."""


class DefaultEmbedder:
    """Wrapper around ``SentenceTransformer`` with lazy model load."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._impl = SentenceTransformer(model)

    def encode(self, text: str) -> NDArray[np.float32]:
        arr = self._impl.encode(text)
        return np.asarray(arr, dtype=np.float32)


__all__ = ["EmbedderProtocol", "DefaultEmbedder"]
