from __future__ import annotations

from ..metrics.base import BaseMetric
from typing import cast
import numpy as np


class SciV4(BaseMetric):
    """Self-Coherence Index v4 using exponential moving average."""

    DEFAULT_ALPHA: float = 0.6

    def __init__(self, *, alpha: float | None = None) -> None:
        self._alpha = float(alpha) if alpha is not None else self.DEFAULT_ALPHA
        self._state: float | None = None

    # --- public API ---
    def reset_state(self) -> None:
        """Reset internal EMA state."""
        self._state = None

    def score(self, a: str | float, b: str) -> float:
        """Update EMA with value ``a`` and return normalized coherence."""
        del b
        x = float(a)
        if self._state is None:
            self._state = x
        else:
            self._state = self._alpha * x + (1 - self._alpha) * self._state
        return float(np.clip(self._state, 0.0, 1.0))

    def set_params(self, **kw: object) -> None:
        if "alpha" in kw:
            self._alpha = float(cast(float, kw["alpha"]))


_SCI = SciV4()

def sci(val: float) -> float:
    """Module-level wrapper for backward compatibility."""
    return _SCI.score(val, "")


def reset_state() -> None:
    """Reset global metric state."""
    _SCI.reset_state()


__all__ = ["SciV4", "sci", "reset_state"]
