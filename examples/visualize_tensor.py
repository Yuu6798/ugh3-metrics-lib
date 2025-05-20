"""Visualize relationships between UGH3 metric tensors.

This simplified example no longer depends on ``pandas`` or ``seaborn`` so it
can run in minimal environments.  It generates a small random demo tensor of
metrics and saves a correlation heatmap to ``images/tensor.png`` for inclusion
in the project README.

Run directly for a demo dataset::

    python examples/visualize_tensor.py --demo
"""

from __future__ import annotations

import argparse
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def metrics_matrix(
    q: Iterable[float],
    s: Iterable[float],
    t: Iterable[float],
    por: Iterable[float],
    delta_e: Iterable[float],
    grv: Iterable[float],
) -> np.ndarray:
    """Return metrics stacked as a ``numpy`` array of shape ``(6, N)``."""
    return np.vstack([list(q), list(s), list(t), list(por), list(delta_e), list(grv)])


def plot_heatmap(data: np.ndarray, labels: list[str]) -> None:
    """Display a simple correlation heatmap using ``matplotlib``."""
    corr = np.corrcoef(data)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(im, ax=ax, label="corr")
    fig.tight_layout()


def demo_matrix(n: int = 20) -> tuple[np.ndarray, list[str]]:
    """Return a demo tensor and associated labels."""
    rng = np.random.default_rng(0)
    q = rng.random(n)
    s = rng.random(n)
    t = rng.random(n)
    por = 0.5 * q + 0.4 * s + 0.1 * t
    delta_e = 1.0 - s
    grv = (q + t) / 2
    labels = ["Q", "S", "t", "PoR", "ΔE", "grv"]
    matrix = metrics_matrix(q, s, t, por, delta_e, grv)
    return matrix, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize UGH3 metric tensor")
    parser.add_argument(
        "--demo", action="store_true", help="Use a randomly generated dataset"
    )
    args = parser.parse_args()

    if not args.demo:
        raise SystemExit("Only --demo mode is supported in this example.")

    data, labels = demo_matrix()
    for row in data.T[:5]:
        print(", ".join(f"{v:.3f}" for v in row))

    plot_heatmap(data, labels)
    output_path = "images/tensor.png"
    plt.savefig(output_path)
    print(f"Saved tensor visualisation to {output_path}")


if __name__ == "__main__":
    main()
