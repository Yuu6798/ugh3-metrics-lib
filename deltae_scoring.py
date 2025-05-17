"""ΔE Scoring Module.

This module provides a simple utility to compute ``\u0394E`` (delta E), which
is the difference between two energy values. The functionality mirrors the
style of other modules in this repository.
"""

from __future__ import annotations


def deltae_score(E1: float, E2: float) -> float:
    """Calculate the difference ``\u0394E`` between ``E2`` and ``E1``.

    Parameters
    ----------
    E1 : float
        The first energy value.
    E2 : float
        The second energy value.

    Returns
    -------
    float
        The difference ``E2 - E1``.
    """
    # Calculate delta E by subtracting the first value from the second
    return E2 - E1
