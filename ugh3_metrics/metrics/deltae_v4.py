from __future__ import annotations
# ------------------------------------------------------------
# 型チェック専用インポート
# ------------------------------------------------------------
from typing import (
    Protocol,
    runtime_checkable,
    Optional,
    Any,
    TYPE_CHECKING,
    cast,
)

import hashlib
import numpy as np
import warnings
from difflib import SequenceMatcher

# ------------------------------------------------------------
# SentenceTransformer ― 実行時に import 失敗しても型だけ残す
# ------------------------------------------------------------
if TYPE_CHECKING:                 # typing 時のみ解決
    from sentence_transformers import SentenceTransformer  # pragma: no cover
else:                             # 実行時
    SentenceTransformer = cast(Optional[Any], None)        # type: ignore[assignment]


@runtime_checkable
class _EmbedderProto(Protocol):
    def encode(self, text: str) -> list[float]:
        """Return an embedding for the given text."""


# --------------------------------------------------------------------------------
# ΔE v4  ― MiniLM ベース (fallback 付き) コサイン距離
# --------------------------------------------------------------------------------

class DeltaEV4:  # noqa: D101
    """ΔE v4 metric.

    * 既定は MiniLM-L6 ベースの文ベクトル同士のコサイン距離 (0‑1)。
    * MiniLM が import 不可／ネットワーク不可の場合は
      - 長さ＋ハッシュ 2‑D コサイン
      - さらにゼロベクトル時は ``difflib.SequenceMatcher`` 比
      へ段階的にフォールバックする。
    """

    # runtime default to avoid union expression TypeError on Python 3.9
    _embedder: _EmbedderProto | None = None  # overridden via set_params/lazy-load

    # ------------------------------------------------------------------
    # コンストラクタ
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        embedder: _EmbedderProto | None = None,
        auto_load: bool = False,
    ) -> None:  # noqa: D401
        """Create a metric instance.

        Parameters
        ----------
        embedder :
            外部から埋め込み器を差し込む場合に指定する。
        auto_load :
            *True* なら可能な場合は `SentenceTransformer` を即座にロードする。
            ネットワーク制限などで取得失敗しても例外は握りつぶす。
        """
        self._auto_load = auto_load
        if embedder is not None:
            self._embedder = embedder
        elif auto_load and SentenceTransformer is not None:
            try:
                self._embedder = cast(
                    _EmbedderProto,
                    SentenceTransformer("all-MiniLM-L6-v2"),  # pragma: no mutate
                )
            except Exception:  # pragma: no cover – network-less CI
                pass

    def score(self, a: str, b: str) -> float:  # noqa: D401
        """Return |len(a)-len(b)| 正規化値 (0-1)。"""
        # --- lazy-load エンベッダー(__init__ で失敗した場合のみ) ----
        if (
            self._embedder is None
            and self._auto_load
            and SentenceTransformer is not None
        ):
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
