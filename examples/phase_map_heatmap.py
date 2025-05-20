from __future__ import annotations

"""Generate a simple phase map heatmap for PoR scores.

This script creates a dummy history of timestamped PoR scores and
plots them using ``matplotlib.imshow``. The resulting image is saved to
``images/phase_map.png`` for inclusion in the project README.
"""

from dataclasses import dataclass
import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PorRecord:
    """Container for a single PoR measurement."""

    timestamp: float
    por: float


def generate_dummy_history(n: int = 20) -> List[PorRecord]:
    """Return ``n`` dummy ``PorRecord`` entries."""
    base_ts = time.time()
    history: List[PorRecord] = []
    for i in range(n):
        history.append(PorRecord(base_ts + i * 60, random.uniform(0.0, 1.0)))
    return history


def plot_heatmap(history: List[PorRecord]) -> None:
    """Create a phase vs PoR heatmap using ``imshow``."""
    por_values = [r.por for r in history]
    # repeat values vertically so ``imshow`` renders a band
    heatmap = np.tile(por_values, (50, 1))
    plt.figure(figsize=(8, 2))
    plt.imshow(
        heatmap,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[0, len(por_values), 0.0, 1.0],
    )
    plt.xlabel("Phase")
    plt.ylabel("PoR score")
    plt.colorbar(label="PoR")
    plt.tight_layout()


def main() -> None:
    history = generate_dummy_history()
    plot_heatmap(history)
    output_path = "images/phase_map.png"
    plt.savefig(output_path)
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
