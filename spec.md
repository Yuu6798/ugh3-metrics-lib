# PoR ΔE Library Specification

This project provides a minimal open source reference for Proof-of-Reserves (PoR) calculations and ΔE energy difference scoring.

## Core Functions

### `por_trigger`

* **Purpose**: Determine whether a PoR event should trigger.
* **Inputs**:
  * `q` – quantity factor
  * `s` – sensitivity factor
  * `t` – time factor
  * `phi_C` – scoring coefficient
  * `D` – distortion factor
  * `theta` – trigger threshold (default `0.6`)
* **Returns**: `dict` with keys `"E_prime"`, `"score"`, and `"triggered"`.
* **Formulas**:
  * `E_prime = q * s * t`
  * `score = E_prime * phi_C`
  * `triggered = (score * (1 - D)) > theta`

### `deltae_score`

* **Purpose**: Compute the energy difference ΔE between two values.
* **Inputs**: `E1`, `E2`
* **Returns**: `float` equal to `E2 - E1`.

