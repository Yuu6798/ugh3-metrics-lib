import numpy as np


class DeltaEV4:  # noqa: D101
    """Stub metric that always returns 0.0 (safety fence)."""

    def score(self, a: str, b: str) -> float:  # noqa: D401
        """Return |len(a)-len(b)| 正規化値 (0-1)。"""
        v1 = np.asarray([len(a), hash(a) % 13], dtype=float)
        v2 = np.asarray([len(b), hash(b) % 13], dtype=float)
        # どちらかがゼロベクトルの場合は距離を 1.0 とする
        if not np.linalg.norm(v1) or not np.linalg.norm(v2):
            return 1.0
        if np.allclose(v1, v2):
            return 0.0
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return round(1.0 - cos, 3)

__all__: list[str] = ["DeltaEV4"]
