"""Simple SECL history evaluator and graphing utility.

This script loads PoR, \u0394E, and grv metrics from a CSV file or generates
sample data, assigns a quality label to each record, and plots the metrics
with color-coded labels using ``matplotlib``.

Evaluation thresholds (theta values) for each metric can be adjusted via
command line arguments. The default thresholds are PoR 0.7, \u0394E 0.4,
and grv 0.5.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# configuration and data structures
# ---------------------------------------------------------------------------

@dataclass
class Record:
    por: float
    delta_e: float
    grv: float
    label: str | None = None


# evaluation thresholds (theta) for each metric
POR_THETA = 0.7
DELTA_E_THETA = 0.4
GRV_THETA = 0.5

# label names
GOOD = "良"
OKAY = "可"
BAD = "不可"


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def evaluate_record(por: float, delta_e: float, grv: float) -> str:
    """Return quality label for a single record.

    A record is **Good** if all metrics meet or exceed their thresholds.
    If metrics are within 80% of the thresholds it is rated **Okay**.
    Otherwise the record is **Bad**.
    """
    if por >= POR_THETA and delta_e >= DELTA_E_THETA and grv >= GRV_THETA:
        return GOOD
    if (
        por >= 0.8 * POR_THETA
        and delta_e >= 0.8 * DELTA_E_THETA
        and grv >= 0.8 * GRV_THETA
    ):
        return OKAY
    return BAD


def load_history(path: str | None, n_samples: int = 20) -> List[Record]:
    """Load records from CSV or create sample data if no path is given."""
    records: List[Record] = []
    if path:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                records.append(
                    Record(
                        por=float(row["por"]),
                        delta_e=float(row["delta_e"]),
                        grv=float(row["grv"]),
                    )
                )
    else:
        for _ in range(n_samples):
            records.append(
                Record(
                    por=round(random.uniform(0, 1), 3),
                    delta_e=round(random.uniform(0, 1), 3),
                    grv=round(random.uniform(0, 1), 3),
                )
            )
    return records


def assign_labels(records: List[Record]) -> None:
    """Assign a quality label to each record in-place."""
    for rec in records:
        rec.label = evaluate_record(rec.por, rec.delta_e, rec.grv)


def plot_records(records: List[Record]) -> None:
    """Plot metrics with color-coded labels."""
    x = list(range(1, len(records) + 1))
    por_vals = [r.por for r in records]
    delta_vals = [r.delta_e for r in records]
    grv_vals = [r.grv for r in records]
    labels = [r.label for r in records]

    color_map = {GOOD: "green", OKAY: "orange", BAD: "red"}
    colors = [color_map.get(lbl, "black") for lbl in labels]

    plt.figure(figsize=(8, 4))
    plt.plot(x, por_vals, label="PoR", color="blue")
    plt.plot(x, delta_vals, label="ΔE", color="purple")
    plt.plot(x, grv_vals, label="grv", color="brown")
    plt.scatter(x, por_vals, c=colors, zorder=5)
    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1)
    plt.legend()
    plt.title("SECL Metrics History with Labels")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    global POR_THETA, DELTA_E_THETA, GRV_THETA

    parser = argparse.ArgumentParser(description="SECL history evaluator")
    parser.add_argument("csv", nargs="?", help="CSV path with por,delta_e,grv")
    parser.add_argument("--por", type=float, default=POR_THETA, help="PoR theta")
    parser.add_argument("--deltae", type=float, default=DELTA_E_THETA, help="ΔE theta")
    parser.add_argument("--grv", type=float, default=GRV_THETA, help="grv theta")
    args = parser.parse_args()
    POR_THETA = args.por
    DELTA_E_THETA = args.deltae
    GRV_THETA = args.grv

    records = load_history(args.csv)
    assign_labels(records)
    for idx, r in enumerate(records, 1):
        print(f"{idx:02d}: PoR {r.por:.3f} ΔE {r.delta_e:.3f} grv {r.grv:.3f} -> {r.label}")
    plot_records(records)


if __name__ == "__main__":
    main()
