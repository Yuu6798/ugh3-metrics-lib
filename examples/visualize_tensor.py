"""Visualize relationships between UGH3 metric tensors.

This example converts lists of metrics (Q, S, t, PoR, ΔE, grv) into a
``pandas.DataFrame`` and shows a pairplot and correlation heatmap using
``seaborn``.

Run directly for a demo dataset::

    python examples/visualize_tensor.py --demo
"""

from __future__ import annotations

import argparse
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def metrics_to_df(
    q: Iterable[float],
    s: Iterable[float],
    t: Iterable[float],
    por: Iterable[float],
    delta_e: Iterable[float],
    grv: Iterable[float],
) -> pd.DataFrame:
    """Return metrics in a ``DataFrame`` for analysis."""
    return pd.DataFrame(
        {
            "Q": list(q),
            "S": list(s),
            "t": list(t),
            "PoR": list(por),
            "ΔE": list(delta_e),
            "grv": list(grv),
        }
    )


def plot_relationships(df: pd.DataFrame, save_path: str = "images/tensor.png") -> None:
    """Show seaborn pairplot and correlation heatmap and save the result."""
    # pairplot returns a ``FacetGrid`` so we can save the figure directly
    g = sns.pairplot(df)
    g.figure.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.tight_layout()
    # Save the last active figure which contains the heatmap
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.show()
    plt.close("all")


def demo_df(n: int = 20) -> pd.DataFrame:
    """Generate a small random demo ``DataFrame``."""
    import numpy as np

    rng = np.random.default_rng(0)
    q = rng.random(n)
    s = rng.random(n)
    t = rng.random(n)
    por = 0.5 * q + 0.4 * s + 0.1 * t
    delta_e = 1.0 - s
    grv = (q + t) / 2
    return metrics_to_df(q, s, t, por, delta_e, grv)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize UGH3 metric tensor")
    parser.add_argument(
        "--demo", action="store_true", help="Use a randomly generated dataset"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="images/tensor.png",
        help="Path to save the generated figure",
    )
    args = parser.parse_args()

    if args.demo:
        df = demo_df()
    else:
        raise SystemExit("Only --demo mode is supported in this example.")

    print(df.head())
    plot_relationships(df, args.out)


if __name__ == "__main__":
    main()
