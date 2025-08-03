"""ΔE v4 metric based on sentence embeddings."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING, cast, Literal
import os

import hashlib
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer
else:  # pragma: no cover - resolved at runtime
    SentenceTransformer = cast(Optional[Any], None)  # type: ignore[assignment]


@runtime_checkable
class _EmbedderProto(Protocol):
    """Minimal interface required for embedding models."""

    def encode(self, text: str) -> list[float]:  # noqa: D401
        """Return an embedding for ``text``."""


_EMBEDDER_CACHE: _EmbedderProto | None = None


def _load_embedder() -> _EmbedderProto:
    """Load and cache the default SentenceTransformer model.

    Raises
    ------
    RuntimeError
        If ``sentence-transformers`` or the model cannot be loaded.
    """

    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is not None:
        return _EMBEDDER_CACHE

    try:
        from sentence_transformers import SentenceTransformer as _ST
    except Exception as err:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "sentence-transformers is required for DeltaE4"
        ) from err

    try:
        _EMBEDDER_CACHE = cast(_EmbedderProto, _ST("all-MiniLM-L6-v2"))
    except Exception as err:  # pragma: no cover - network failure etc.
        raise RuntimeError(
            "failed to load sentence-transformers model 'all-MiniLM-L6-v2'"
        ) from err
    return _EMBEDDER_CACHE


class DeltaE4:  # noqa: D101
    """ΔE v4 metric using cosine distance between sentence embeddings."""

    _embedder: _EmbedderProto | None

    def __init__(
        self,
        *,
        embedder: _EmbedderProto | None = None,
        fallback: Literal["hash"] | None = None,
    ) -> None:
        if fallback is None:
            env_fb = os.getenv("DELTAE4_FALLBACK")
            if env_fb == "hash":
                fallback = "hash"
        if embedder is not None:
            self._embedder = embedder
        elif fallback == "hash":
            self._embedder = None
        else:
            self._embedder = _load_embedder()
        self._fallback = fallback

    def score(self, a: str, b: str) -> float:  # noqa: D401
        """Return the cosine distance between embeddings of ``a`` and ``b``."""

        if self._embedder is not None:
            v1 = np.asarray(self._embedder.encode(a), dtype=float)
            v2 = np.asarray(self._embedder.encode(b), dtype=float)
        elif self._fallback == "hash":
            h1 = int.from_bytes(hashlib.md5(a.encode()).digest()[:4], "big", signed=True)
            h2 = int.from_bytes(hashlib.md5(b.encode()).digest()[:4], "big", signed=True)
            # normalize to [-1, 1]
            v1 = np.asarray([len(a), h1 / 2**31], dtype=float)
            v2 = np.asarray([len(b), h2 / 2**31], dtype=float)
        else:
            raise RuntimeError(
                "embedder not available; DeltaE4 requires sentence-transformers"
            )

        if not np.linalg.norm(v1) or not np.linalg.norm(v2):
            raise RuntimeError("embedding zero")
        if np.allclose(v1, v2):
            return 0.0
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return round(1.0 - cos, 3)

    def set_params(self, *, embedder: _EmbedderProto | None = None) -> None:
        if embedder is not None:
            self._embedder = embedder


DeltaEV4 = DeltaE4

__all__ = ["DeltaE4", "DeltaEV4"]

