from typing import Protocol, runtime_checkable, Optional, Any

import hashlib
import numpy as np
import warnings
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer: Optional[Any]
    SentenceTransformer = None


@runtime_checkable
class _EmbedderProto(Protocol):
    def encode(self, text: str) -> list[float]:
        """Return an embedding for the given text."""


class DeltaEV4:  # noqa: D101
    """Return a simple length+hash cosine distance.

    Falls back to :class:`difflib.SequenceMatcher` when either embedding is
    the zero vector.
    """

    _embedder: _EmbedderProto | None = None  # lazy-loaded or set_params

    def score(self, a: str, b: str) -> float:  # noqa: D401
        """Return |len(a)-len(b)| 正規化値 (0-1)。"""
        # --- lazy-load 実エンベッダー --------------------------
        if self._embedder is None and SentenceTransformer is not None:
            try:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:  # pragma: no cover – network-less CI
                pass

        # --- ベクトルを取得 ------------------------------------
        if self._embedder is not None:
            v1 = np.asarray(self._embedder.encode(a), dtype=float)
            v2 = np.asarray(self._embedder.encode(b), dtype=float)
        else:
            h1 = int.from_bytes(hashlib.md5(a.encode()).digest()[:4], "big")
            h2 = int.from_bytes(hashlib.md5(b.encode()).digest()[:4], "big")
            v1 = np.asarray([len(a), h1], dtype=float)
            v2 = np.asarray([len(b), h2], dtype=float)
        # どちらかがゼロベクトルなら difflib.SequenceMatcher による
        # 文字列類似度へフォールバックする
        if not np.linalg.norm(v1) or not np.linalg.norm(v2):
            warnings.warn(
                "zero\u2011vector detected; using diff.sim fallback",
                RuntimeWarning,
            )
            return round(1.0 - SequenceMatcher(None, a, b).ratio(), 3)
        if np.allclose(v1, v2):
            return 0.0
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return round(1.0 - cos, 3)

    def set_params(self, *, embedder: _EmbedderProto | None = None) -> None:
        if embedder is not None:
            self._embedder = embedder  # 動的差し替え

__all__: list[str] = ["DeltaEV4"]
