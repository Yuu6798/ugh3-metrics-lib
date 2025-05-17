"""Design Prototypes for PoR-ΔE Library.

This module contains simple prototype implementations for the two core
functions used throughout the library. They are kept intentionally
minimal to highlight the structure and intended usage of the PoR trigger
and ΔE scoring logic.
"""

from __future__ import annotations


def por_trigger(q: float, s: float, t: float, phi_C: float, D: float, *, theta: float = 0.6) -> dict:
    """Prototype PoR trigger calculation.

    Parameters
    ----------
    q : float
        Quantity factor for reserves.
    s : float
        Sensitivity factor for reserves.
    t : float
        Time factor applied to the calculation.
    phi_C : float
        Scoring coefficient representing additional adjustments.
    D : float
        Distortion factor reducing the effective score.
    theta : float, optional
        Threshold for triggering the PoR check. Defaults to ``0.6``.

    Returns
    -------
    dict
        Dictionary with ``"E_prime"``, ``"score"``, and ``"triggered"`` keys.
    """
    # Compute the primary metric E'
    E_prime = q * s * t

    # Calculate score adjusted by the coefficient
    score = E_prime * phi_C

    # Determine whether the final score, adjusted for distortion, exceeds the threshold
    triggered = (score * (1 - D)) > theta

    return {
        "E_prime": E_prime,
        "score": score,
        "triggered": triggered,
    }


def deltae_score(E1: float, E2: float) -> float:
    """Prototype calculation of ``\u0394E``.

    Parameters
    ----------
    E1 : float
        The first energy value.
    E2 : float
        The second energy value.

    Returns
    -------
    float
        Difference ``E2 - E1``.
    """
    # Simply return the difference between the two energy values
    return E2 - E1
