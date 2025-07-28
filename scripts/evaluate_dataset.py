from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from core import stats as core_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate dataset for public release")
    p.add_argument(
        "--infile",
        type=Path,
        default=Path("datasets/current_recalc.parquet"),
        help="input dataset file (Parquet/CSV/JSONL)",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("evaluation_output"),
        help="output directory",
    )
    p.add_argument(
        "--group-col",
        type=str,
        default=None,
        help="boolean column used to split rows into two groups",
    )
    p.add_argument(
        "--metric-cols",
        nargs="+",
        default=["delta_e_internal", "grv_score"],
        help="metric columns to compute effect size and Welch t-test",
    )
    return p.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".parquet", ".pq", ".parq"}:
        return pd.read_parquet(path)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def main() -> int:
    args = parse_args()
    infile = args.infile
    if not infile.exists():
        print(f"WARNING: {infile} not found", file=sys.stderr)
        return 3

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_table(infile)

    count_ok = len(df) >= 1000

    if "delta_e_internal" in df.columns:
        frac_valid = (
            df["delta_e_internal"].dropna().between(0, 1, inclusive="both").mean()
        )
    else:
        frac_valid = 0.0
    delta_ok = frac_valid >= 0.99

    if "por_fire" in df.columns:
        rate = float(df["por_fire"].mean())
    else:
        rate = 0.0
    por_ok = 0.1 <= rate <= 0.4

    metrics_ok = True
    results: list[tuple[str, float, float, float]] = []
    group_col = args.group_col
    if group_col and group_col in df.columns:
        for col in args.metric_cols:
            if col not in df.columns:
                continue
            a_group = df[df[group_col]].get(col).dropna()
            b_group = df[~df[group_col]].get(col).dropna()
            if len(a_group) == 0 or len(b_group) == 0:
                continue
            d = core_stats.cohen_d(a_group.to_numpy(dtype=float), b_group.to_numpy(dtype=float))
            t, p = core_stats.welch_ttest(a_group.to_numpy(dtype=float), b_group.to_numpy(dtype=float))
            results.append((col, d, t, p))
            if not (abs(d) >= 0.5 and p < 0.05):
                metrics_ok = False

    all_ok = count_ok and delta_ok and por_ok and metrics_ok

    report = outdir / "report.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# Dataset Evaluation\n\n")
        fh.write(f"Total records: {len(df)}\n\n")
        fh.write(f"Valid delta_e_internal: {frac_valid*100:.2f}%\n\n")
        fh.write(f"por_fire rate: {rate:.3f}\n\n")
        if results:
            fh.write("## Metric Comparison\n\n")
            for col, d, t, p in results:
                fh.write(f"### {col}\n")
                fh.write(f"- Cohen's d: {d:.3f}\n")
                fh.write(f"- Welch t-stat: {t:.3f}\n")
                fh.write(f"- p-value: {p:.4f}\n\n")
        fh.write("Status: **PASS**\n" if all_ok else "Status: **FAIL**\n")

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
