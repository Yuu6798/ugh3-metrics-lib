from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # for CI/non-GUI environments
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze dataset and generate report")
    p.add_argument(
        "--infile",
        type=Path,
        default=Path("datasets/current_recalc.parquet"),
        help="input Parquet file",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_output"),
        help="output directory",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.infile)
    columns = [
        "delta_e_internal",
        "por_fire",
        "tfidf",
        "entropy",
        "cooccurrence",
        "grv_score",
    ]
    present = [c for c in columns if c in df.columns]
    if not present:
        raise SystemExit("no required columns found")

    summary_df = df[present].describe()

    if "delta_e_internal" in df.columns:
        plt.figure()
        df["delta_e_internal"].dropna().hist(bins=30)
        plt.title("delta_e_internal distribution")
        plt.savefig(outdir / "hist_delta_e.png")
        plt.close()

    if "por_fire" in df.columns:
        plt.figure()
        df["por_fire"].value_counts().sort_index().plot.bar()
        plt.title("por_fire rate")
        plt.savefig(outdir / "bar_por_fire.png")
        plt.close()

    if "delta_e_internal" in df.columns and "grv_score" in df.columns:
        plt.figure()
        plt.scatter(df["delta_e_internal"], df["grv_score"])
        plt.title("delta_e_internal vs grv_score")
        plt.xlabel("delta_e_internal")
        plt.ylabel("grv_score")
        plt.savefig(outdir / "scatter_delta_vs_grv.png")
        plt.close()

    report_path = outdir / "report.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Dataset Analysis\n\n")
        fh.write(f"Dataset size: {len(df)}\n\n")
        fh.write(summary_df.to_markdown())
        fh.write("\n\n")
        fh.write("![](hist_delta_e.png)\n")
        fh.write("![](bar_por_fire.png)\n")
        if (outdir / "scatter_delta_vs_grv.png").exists():
            fh.write("![](scatter_delta_vs_grv.png)\n")


if __name__ == "__main__":
    main()
