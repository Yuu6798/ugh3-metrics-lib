"""PoR Trigger Calculation Module.

This module provides functionality to compute whether a Proof-of-Reserves (PoR)
check should be triggered based on various parameters. The implementation is
based on the descriptions provided in design_sketch.py and spec.md.
"""

from __future__ import annotations

from secl.qa_cycle import main_qa_cycle
from design_sketch import PorTriggerResult



def por_trigger(q: float, s: float, t: float, phi_C: float, D: float, *, theta: float = 0.6) -> PorTriggerResult:
    """Calculate whether a PoR event should be triggered.

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
        Dictionary containing ``"E_prime"``, ``"score"``, and ``"triggered"``
        keys representing intermediate values and the final boolean result.
    """
    # Compute the primary energy metric E'
    E_prime = q * s * t

    # Calculate the raw score using the provided coefficient
    score = E_prime * phi_C

    # Determine if the final score, adjusted for distortion, exceeds the threshold
    triggered = (score * (1 - D)) > theta

    return {
        "E_prime": E_prime,
        "score": score,
        "triggered": triggered,
    }


__all__ = ["por_trigger", "main_qa_cycle"]
