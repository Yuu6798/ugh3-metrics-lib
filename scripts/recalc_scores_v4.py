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
import logging
import pandas as pd
import os
import warnings

from tqdm import tqdm

from ugh3_metrics.metrics.deltae_v4 import DeltaEV4
from ugh3_metrics.metrics.por_v4 import PorV4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalculate PoR/ΔE v4 scores")
    parser.add_argument("--infile", required=True, help="input CSV or Parquet")
    parser.add_argument("--outfile", required=True, help="output Parquet path")
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load CSV/Parquet and sanitize text columns."""
    if path.suffix.lower() in {".parquet", ".pq", ".parq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, keep_default_na=False)

    return df.assign(
        question=df["question"].astype(str).fillna(""),
        answer_a=df["answer_a"].astype(str).fillna(""),
        answer_b=df["answer_b"].astype(str).fillna(""),
    )


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

    por_a: list[float] = []
    por_b: list[float] = []
    delta_e: list[float] = []
    valid_indices: list[int] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        q, a1, a2 = row["question"], row["answer_a"], row["answer_b"]
        try:
            pa = pv.score(q, a1)
            pb = pv.score(q, a2)
            de_val = de.score(a1, a2)
        except TypeError as e:
            logging.warning("Skip row %s: %s", idx, e)
            continue
        por_a.append(pa)
        por_b.append(pb)
        delta_e.append(de_val)
        valid_indices.append(idx)

    df = df.iloc[valid_indices].copy()
    df["por_a"] = por_a
    df["por_b"] = por_b
    df["delta_e"] = delta_e

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)          # 通常は Parquet で保存
        print(f"Wrote results to {out_path}")
    except (ImportError, ValueError) as e:
        # pyarrow / fastparquet 不在の CI で失敗した場合は CSV にフォールバック
        warnings.warn(f"Parquet export failed ({e}); falling back to CSV.")
        csv_fallback = out_path.with_suffix(".csv")
        df.to_csv(csv_fallback, index=False)
        print(f"Wrote CSV fallback to {csv_fallback}")


if __name__ == "__main__":
    main()
