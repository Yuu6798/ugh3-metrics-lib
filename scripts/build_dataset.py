#!/usr/bin/env python3
"""Build dataset from raw CSV files into a single Parquet table."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import tempfile

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dep
    print("pandas is required to build the dataset", file=sys.stderr)
    raise SystemExit(1)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine raw CSVs and recalc metrics")
    p.add_argument("--raw-dir", type=Path, default=Path("raw"), help="directory of raw CSV files")
    p.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("datasets/current_recalc.parquet"),
        help="output Parquet file",
    )
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

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = Path(tmp.name)
        df.to_csv(temp_path, index=False)

    recalc = Path(__file__).resolve().parent / "recalc_scores_v4.py"
    try:
        subprocess.run(
            [sys.executable, str(recalc), "--infile", str(temp_path), "--outfile", str(args.out_parquet)],
            check=True,
        )
    except subprocess.CalledProcessError:
        return 1
    finally:
        temp_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
