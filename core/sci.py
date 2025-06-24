"""Legacy wrapper for Self-Coherence Index v4."""

from __future__ import annotations

from ugh3_metrics.metrics import SciV4

_SCI = SciV4()


def sci(val: float) -> float:
    """Backward-compatible entry point."""
    return _SCI.score(val, "")


def reset_state() -> None:
    """Reset internal EMA state."""
    _SCI.reset_state()


__all__ = ["SciV4", "sci", "reset_state"]
