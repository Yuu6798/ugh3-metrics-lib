"""ΔE v2 Module.

Provides ``delta_e`` which computes a normalized difference between two
strings.  This is a lightweight implementation intended for tests.
"""

from __future__ import annotations


def delta_e(prev: str | None, curr: str) -> float:
    """Return normalized ΔE value between two answers.

    Parameters
    ----------
    prev : str | None
        Previous answer text. ``None`` yields ``0.0``.
    curr : str
        Current answer text.

    Returns
    -------
    float
        Normalized difference in the range ``0.0`` to ``1.0``.
    """
    if prev is None:
        return 0.0
    diff = min(1.0, abs(len(prev) - len(curr)) / 50.0)
    return round(diff, 3)


__all__ = ["delta_e"]
