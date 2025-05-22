from __future__ import annotations

from utils.typing import PorTriggerResult

def por_trigger(q: float, s: float, t: float, phi_C: float, D: float, *, theta: float = 0.6) -> PorTriggerResult:
    """Return PoR trigger metrics for the given parameters."""
    # Compute the intermediate energy metric
    E_prime = q * s * t
    # Apply coefficient to get the score
    score = E_prime * phi_C
    # Check if the distorted score exceeds the threshold
    triggered = (score * (1 - D)) > theta
    return {"E_prime": E_prime, "score": score, "triggered": triggered}


def deltae_score(E1: float, E2: float) -> float:
    """Return the ΔE value between two energy readings."""
    return E2 - E1
