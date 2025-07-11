#!/usr/bin/env python3
"""Recalculate PoR and ΔE metrics for paired answers.

This utility loads a CSV or Parquet file containing columns ``question``,
``answer_a`` and ``answer_b``. It computes the following metrics for each row:

* ``por_a``: PoR score between ``question`` and ``answer_a``.
* ``por_b``: PoR score between ``question`` and ``answer_b``.
* ``delta_e``: ΔE score between ``answer_a`` and ``answer_b``.

The resulting dataframe is written to the specified output path as Parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed

from ugh3_metrics.metrics.deltae_v4 import DeltaEV4
from ugh3_metrics.metrics.por_v4 import PorV4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalculate PoR/ΔE v4 scores")
    parser.add_argument("--infile", required=True, help="input CSV or Parquet")
    parser.add_argument("--outfile", required=True, help="output Parquet path")
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq", ".parq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    in_path = Path(args.infile)
    out_path = Path(args.outfile)

    df = load_dataframe(in_path)

    required = {"question", "answer_a", "answer_b"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise SystemExit(f"missing required columns: {missing}")

    de = DeltaEV4(auto_load=True)
    pv = PorV4(auto_load=True)

    def _score_row(q: str, a1: str, a2: str) -> tuple[float, float, float]:
        """Return PoR for both answers and ΔE between them."""
        return pv.score(q, a1), pv.score(q, a2), de.score(a1, a2)

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_score_row)(q, a1, a2)
        for q, a1, a2 in tqdm(
            zip(df["question"], df["answer_a"], df["answer_b"]),
            total=len(df),
            desc="Scoring",
        )
    )

    df["por_a"], df["por_b"], df["delta_e"] = map(list, zip(*results))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
