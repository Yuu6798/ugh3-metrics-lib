"""Simple utilities for evaluating metric records.

This module provides helper constants and a basic ``evaluate_record`` function
used by tests. It categorises a metrics tuple of PoR score, ΔE and grv into one
of three text labels.
"""

GOOD = "GOOD"
OKAY = "OKAY"
BAD = "BAD"


def evaluate_record(por: float, delta_e: float, grv: float) -> str:
    """Return a qualitative label for a metric record.

    Parameters
    ----------
    por : float
        Proof-of-Resonance score.
    delta_e : float
        ΔE metric value (unused in the simple heuristic).
    grv : float
        grv metric value (unused in the simple heuristic).

    Returns
    -------
    str
        One of ``GOOD``, ``OKAY`` or ``BAD`` depending on ``por``.
    """
    if por >= 0.7:
        return GOOD
    if por >= 0.5:
        return OKAY
    return BAD


__all__ = ["evaluate_record", "GOOD", "OKAY", "BAD"]
