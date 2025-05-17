# PoR ΔE Library

A lightweight Python library that demonstrates Proof-of-Reserves (PoR) trigger computation and ΔE (delta E) energy scoring. The project offers straightforward functions for experimentation with reserve verification models.

## Features

- **PoR trigger calculation** via `por_trigger`
- **ΔE scoring** with `deltae_score`
- Minimal dependencies and easy to integrate
- Python 3.8+ compatible

## Requirements

- Python 3.8 or higher

## Installation

Install directly from GitHub using `pip`:

```bash
pip install git+https://github.com/Yuu6798/por-deltae-lib.git
```

Alternatively, clone the repository:

```bash
git clone https://github.com/Yuu6798/por-deltae-lib.git
```

## Quick Start

```python
from por_trigger import por_trigger
from deltae_scoring import deltae_score

# Quick PoR check
result = por_trigger(q=1.0, s=0.9, t=1.2, phi_C=1.05, D=0.1)
print(result)

# Simple ΔE calculation
print(deltae_score(E1=10.0, E2=12.5))
```

## Usage

The `por_trigger` function returns a dictionary with the intermediate score and a boolean indicating whether the PoR threshold was exceeded. `deltae_score` simply returns the difference between two energy values.

```python
from por_trigger import por_trigger
from deltae_scoring import deltae_score

metrics = por_trigger(q=0.8, s=0.95, t=1.1, phi_C=1.02, D=0.05, theta=0.6)
if metrics["triggered"]:
    print("PoR event triggered!", metrics)

energy_gap = deltae_score(E1=8.0, E2=9.5)
print("ΔE:", energy_gap)
```

## API Reference

### `por_trigger(q: float, s: float, t: float, phi_C: float, D: float, *, theta: float = 0.6) -> dict`

Calculates whether a PoR event should be triggered.

**Parameters**
- `q` – quantity factor for reserves
- `s` – sensitivity factor
- `t` – time factor
- `phi_C` – scoring coefficient
- `D` – distortion factor
- `theta` – trigger threshold (default `0.6`)

**Returns**
A dictionary containing:
- `"E_prime"` – intermediate energy metric
- `"score"` – calculated score
- `"triggered"` – `True` if the threshold is exceeded

### `deltae_score(E1: float, E2: float) -> float`

Computes `E2 - E1`.

**Parameters**
- `E1` – first energy value
- `E2` – second energy value

**Returns**
- The difference `E2 - E1`

## Project Structure

- `por_trigger.py` – PoR trigger implementation
- `deltae_scoring.py` – ΔE scoring helper
- `design_sketch.py` – simplified reference code
- `spec.md` – specification outline (currently empty)
- `pyproject.toml` – packaging metadata

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Contact

Questions or suggestions? Reach out to [Yuu6798](https://github.com/Yuu6798).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

