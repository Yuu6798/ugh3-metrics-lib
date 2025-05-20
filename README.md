# PoR ΔE Library

A lightweight Python library that demonstrates Proof-of-Reserves (PoR) trigger computation, ΔE (delta E) energy scoring, and grv (語彙重力) calculations. These three metrics together offer a simple toolkit for experimenting with reserve verification and vocabulary dynamics.

## Features
- **deltae_score** from `deltae_scoring.py` - Calculate the difference ``ΔE`` between ``E2`` and ``E1``.
- **por_trigger** from `design_sketch.py` - Return PoR trigger metrics for the given parameters.
- **deltae_score** from `design_sketch.py` - Return the ΔE value between two energy readings.
- **grv_score** from `grv_scoring.py` - Calculate vocabulary gravity (grv) for the given text or list.
- **load_config** from `por_deltae_grv_collector.py` - Load configuration if present; return empty defaults otherwise.
- **novelty_score** from `por_deltae_grv_collector.py` - Return novelty score based on maximum similarity to history.
- **simulate_delta_e** from `por_deltae_grv_collector.py` - Compute ΔE following secl_qa_cycle implementation.
- ... (9 more functions in `por_deltae_grv_collector.py`)
- **por_trigger** from `por_trigger.py` - Calculate whether a PoR event should be triggered.
- **load_config** from `secl_qa_cycle.py` - Load configuration parameters from a JSON file.
- **novelty_score** from `secl_qa_cycle.py`
- **is_duplicate_question** from `secl_qa_cycle.py`
- ... (21 more functions in `secl_qa_cycle.py`)

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
from deltae_scoring import deltae_score
from design_sketch import por_trigger
from design_sketch import deltae_score
from grv_scoring import grv_score
from por_deltae_grv_collector import load_config
from por_deltae_grv_collector import novelty_score
from por_deltae_grv_collector import simulate_delta_e
from por_trigger import por_trigger
from secl_qa_cycle import load_config
from secl_qa_cycle import novelty_score
from secl_qa_cycle import is_duplicate_question
```

## Usage

The `por_trigger` function returns a dictionary with the intermediate score and a boolean indicating whether the PoR threshold was exceeded. `deltae_score` simply returns the difference between two energy values. `grv_score` counts unique vocabulary items and normalizes the result.

```python
from por_trigger import por_trigger
from grv_scoring import grv_score
from deltae_scoring import deltae_score

metrics = por_trigger(q=0.8, s=0.95, t=1.1, phi_C=1.02, D=0.05, theta=0.6)
if metrics["triggered"]:
    print("PoR event triggered!", metrics)

energy_gap = deltae_score(E1=10.0, E2=12.5)
print("ΔE:", energy_gap)
qa_history = ["質問1 応答1", "質問2 応答2"]
print(grv_score(qa_history))
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

### `grv_score(text: str | list[str], *, vocab_limit: int = 30) -> float`

Counts unique vocabulary items in the provided text(s) and returns `min(1.0, vocab_size / vocab_limit)`.

**Parameters**
- `text` – single string or list of strings to analyze
- `vocab_limit` – normalization denominator (default `30`)

**Returns**
- Normalized vocabulary gravity value

## Project Structure

- `por_trigger.py` – PoR trigger implementation
- `deltae_scoring.py` – ΔE scoring helper
- `grv_scoring.py` – grv score helper
- `design_sketch.py` – simplified reference code
- `spec.md` – specification outline (currently empty)
- `pyproject.toml` – packaging metadata

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Contact

Questions or suggestions? Reach out to [Yuu6798](https://github.com/Yuu6798).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

