class DeltaEV4:  # noqa: D101
    """Stub metric that always returns 0.0 (safety fence)."""

    def score(self, a: str, b: str) -> float:  # noqa: D401
        """Return |len(a)-len(b)| 正規化値 (0-1)。"""
        import math

        if a == b:
            return 0.0
        diff = abs(len(a) - len(b))
        denom = max(len(a), len(b))
        return round(diff / denom, 3)

__all__: list[str] = ["DeltaEV4"]
