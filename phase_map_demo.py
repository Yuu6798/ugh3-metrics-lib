"""Simple demo generating and plotting PoR, ΔE, and grv metrics.

This script creates a small in-memory history of metric values and plots them
using ``matplotlib``. It can be run directly for a quick visualisation::

    python phase_map_demo.py

The output shows timestamps with corresponding metric values followed by a
simple line plot. Horizontal dashed lines mark example good/poor thresholds for
PoR, ΔE, and grv.
"""

from __future__ import annotations

import datetime
import random
import time
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


@dataclass
class MetricRecord:
    """Container for a single metric snapshot."""

    timestamp: float
    por: float
    delta_e: float
    grv: float


def generate_dummy_records(n: int = 10) -> List[MetricRecord]:
    """Return ``n`` metric records with randomised values."""
    records: List[MetricRecord] = []
    for i in range(n):
        delta_e = random.uniform(0.0, 1.0)
        novelty = random.uniform(0.5, 1.0)
        por = round(0.6 * novelty + 0.4 * delta_e, 3)
        grv = random.uniform(0.0, 1.0)
        records.append(MetricRecord(time.time() + i, por, delta_e, grv))
    return records


def plot_records(records: List[MetricRecord]) -> None:
    """Plot PoR, ΔE and grv metrics on an indexed timeline."""
    indices = list(range(1, len(records) + 1))
    por_values = [r.por for r in records]
    delta_e_values = [r.delta_e for r in records]
    grv_values = [r.grv for r in records]

    plt.figure(figsize=(10, 6))
    plt.plot(indices, por_values, label="PoR")
    plt.plot(indices, delta_e_values, label="ΔE")
    plt.plot(indices, grv_values, label="grv")

    # Hard-coded thresholds for demonstration
    POR_THRESHOLD = 0.7
    DELTA_E_THRESHOLD = 0.5
    GRV_THRESHOLD = 0.6

    plt.axhline(POR_THRESHOLD, color="C0", linestyle="--", alpha=0.5)
    plt.axhline(DELTA_E_THRESHOLD, color="C1", linestyle="--", alpha=0.5)
    plt.axhline(GRV_THRESHOLD, color="C2", linestyle="--", alpha=0.5)

    plt.xlabel("Record index")
    plt.ylabel("Metric value")
    plt.title("Phase Map Demo")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Generate demo records and display them."""
    records = generate_dummy_records()
    for r in records:
        ts = datetime.datetime.fromtimestamp(r.timestamp).isoformat(timespec="seconds")
        comment_parts = []
        if r.por > 0.6:
            comment_parts.append("照合成功")
        elif r.por >= 0.4:
            comment_parts.append("PoRやや低め")
        else:
            comment_parts.append("照合失敗")

        if r.delta_e < 0.2:
            comment_parts.append("ΔE安定")
        elif r.delta_e <= 0.5:
            comment_parts.append("ΔE変化あり")
        else:
            comment_parts.append("ΔE大きな変化")

        if r.grv > 0.7:
            comment_parts.append("高難度")
        elif r.grv >= 0.4:
            comment_parts.append("平均難度")
        else:
            comment_parts.append("低難度")

        comment = " / ".join(comment_parts)
        print(f"{ts} | PoR: {r.por:.3f} | ΔE: {r.delta_e:.3f} | grv: {r.grv:.3f} -> {comment}")
    plot_records(records)


if __name__ == "__main__":
    main()
