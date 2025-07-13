#!/usr/bin/env python3
"""Batch recompute ΔE v2 and update CSV.

This script reads a CSV file produced by the QA cycle and
recomputes the ``delta_e_norm`` column using :func:`core.deltae_v2.delta_e`.
The original column layout is preserved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from core.deltae_v2 import delta_e


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalculate ΔE v2 values in CSV")
    parser.add_argument("--in", dest="in_file", required=True, help="input CSV path")
    parser.add_argument("--out", dest="out_file", required=True, help="output CSV path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    in_path = Path(args.in_file)
    out_path = Path(args.out_file)

    with in_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "delta_e_norm" not in fieldnames:
        fieldnames.append("delta_e_norm")

    prev_answer: str | None = None
    for row in rows:
        answer_b = row.get("answer_b", "")
        row["delta_e_norm"] = f"{delta_e(prev_answer, answer_b):.3f}"
        prev_answer = answer_b

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully wrote recomputed CSV to {out_path}")


if __name__ == "__main__":
    main()
