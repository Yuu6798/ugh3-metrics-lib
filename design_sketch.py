"""Design prototypes for core scoring functions.

This module contains minimal implementations that outline the intended
behaviour of the Proof-of-Reserves trigger calculation and the
``\u0394E`` scoring utility. These functions are simplified prototypes
meant to demonstrate the structure and usage of the future
implementation.
"""

from __future__ import annotations


def por_trigger(q: float, s: float, t: float, phi_C: float, D: float, *, theta: float = 0.6) -> dict:
    """Compute whether a PoR event should be triggered.

    The calculation follows a basic energy scoring approach:
    ``E_prime = q * s * t`` and ``score = E_prime * phi_C``.  The result
    is considered triggered if ``(score * (1 - D)) > theta``.

    Parameters
    ----------
    q : float
        Quantity factor.
    s : float
        Sensitivity factor.
    t : float
        Time factor applied to the energy metric.
    phi_C : float
        Coefficient adjusting the score.
    D : float
        Distortion factor reducing the effective score.
    theta : float, optional
        Threshold for triggering. Defaults to ``0.6``.

    Returns
    -------
    dict
        Dictionary with ``"E_prime"``, ``"score"``, and ``"triggered"``
        values.
    """
    # Calculate the intermediate energy value E'
    E_prime = q * s * t

    # Apply the coefficient to obtain the raw score
    score = E_prime * phi_C

    # Check if the adjusted score exceeds the threshold
    triggered = (score * (1 - D)) > theta

    return {
        "E_prime": E_prime,
        "score": score,
        "triggered": triggered,
    }


def deltae_score(E1: float, E2: float) -> float:
    """Return the difference ``\u0394E`` between ``E2`` and ``E1``.

    This simple helper is used to compare two energy values.

    Parameters
    ----------
    E1 : float
        First energy measurement.
    E2 : float
        Second energy measurement.

    Returns
    -------
    float
        The difference ``E2 - E1``.
    """
    # Compute the delta by subtracting the first value from the second
    return E2 - E1
