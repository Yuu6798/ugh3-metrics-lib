#!/usr/bin/env python3
"""Build dataset from raw CSV files into a single table."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover - missing pandas
    print("pandas is required to build the dataset", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine raw CSV files")
    p.add_argument("--raw-dir", type=Path, default=Path("raw"), help="directory with raw CSV files")
    p.add_argument("--out-csv", type=Path, default=None, help="optional CSV output path")
    p.add_argument("--out-parquet", type=Path, default=None, help="optional Parquet output path")
    return p.parse_args()


def load_raw_csvs(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("no CSV files found")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    rename_map = {"Q": "question", "A1": "answer_a", "A2": "answer_b"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    for col in ["question", "answer_a", "answer_b"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "por_fire" not in df.columns:
        if "por" in df.columns:
            por_val = pd.to_numeric(df["por"], errors="coerce").fillna(0.0)
            df["por_fire"] = por_val >= 0.1
        else:
            df["por_fire"] = False
    return df


def main() -> int:
    args = parse_args()

    try:
        df = load_raw_csvs(args.raw_dir)
    except FileNotFoundError:
        print("[ERROR] no raw CSV found", file=sys.stderr)
        return 3
    except Exception as e:  # pragma: no cover - defensive
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if not args.out_csv and not args.out_parquet:
        print("[ERROR] specify --out-csv or --out-parquet", file=sys.stderr)
        return 2

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        if (gh := os.getenv("GITHUB_OUTPUT")):
            with open(gh, "a", encoding="utf-8") as fh:
                fh.write(f"csv={args.out_csv}\n")

    if args.out_parquet:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out_parquet, index=False)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

