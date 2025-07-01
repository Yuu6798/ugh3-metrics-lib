from typing import Protocol

import hashlib
import numpy as np


class _EmbedderProto(Protocol):
    def encode(self, text: str) -> list[float]:
        """Return an embedding for the given text."""


class DeltaEV4:  # noqa: D101
    """Stub metric: length+hash 2‐D コサイン距離を返す暫定実装."""

    _embedder: _EmbedderProto | None = None  # set_params で動的に上書き

    def score(self, a: str, b: str) -> float:  # noqa: D401
        """Return |len(a)-len(b)| 正規化値 (0-1)。"""
        if self._embedder is not None:
            v1 = np.asarray(self._embedder.encode(a), dtype=float)
            v2 = np.asarray(self._embedder.encode(b), dtype=float)
        else:
            h1 = int.from_bytes(hashlib.md5(a.encode()).digest()[:4], "big")
            h2 = int.from_bytes(hashlib.md5(b.encode()).digest()[:4], "big")
            v1 = np.asarray([len(a), h1], dtype=float)
            v2 = np.asarray([len(b), h2], dtype=float)
        # どちらかがゼロベクトルの場合は距離を 1.0 とする
        if not np.linalg.norm(v1) or not np.linalg.norm(v2):
            return 1.0
        if np.allclose(v1, v2):
            return 0.0
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return round(1.0 - cos, 3)

    def set_params(self, *, embedder: _EmbedderProto | None = None) -> None:
        if embedder is not None:
            self._embedder = embedder  # 動的差し替え

__all__: list[str] = ["DeltaEV4"]
