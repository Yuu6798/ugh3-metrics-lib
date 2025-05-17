# PoR ΔE Library

This repository provides a reference implementation of the PoR trigger and ΔE scoring models from **UGHer** theory. It is intended for researchers and developers who want a simple, open source base for experimenting with Proof‑of‑Reserves (PoR) calculations.

## Features

- **`por_trigger`** – Calculate whether a PoR check should be triggered based on reserve and distortion parameters.
- **`deltae_score`** – Compute the energy difference ΔE between two values.

## Installation

Clone this repository or install directly via `pip`:

```bash
pip install git+https://github.com/Yuu6798/por-deltae-lib.git
```

Python 3.8 or higher is recommended.

## Usage

```python
from por_trigger import por_trigger
from deltae_scoring import deltae_score

# Determine if a PoR event should trigger
res = por_trigger(q=1.0, s=0.9, t=1.2, phi_C=1.05, D=0.1)
print(res["triggered"], res)

# Calculate ΔE between two energy values
print(deltae_score(E1=10.0, E2=12.5))
```

## Project Structure

- `por_trigger.py` – implementation of the PoR trigger calculation.
- `deltae_scoring.py` – implementation of the ΔE scoring helper.

## Contributing

Contributions are welcome. Feel free to open issues or submit pull requests.

## Contact

For questions or suggestions, please contact [Yuu6798](https://github.com/Yuu6798).
