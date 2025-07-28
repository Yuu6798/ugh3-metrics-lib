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

    numeric_cols = [
        c
        for c in present
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool
    ]
    missing_df = df[present].isna().mean().to_frame("missing_rate")
    missing_df.to_csv(outdir / "missing_rate.csv")
    missing_df.to_markdown(outdir / "missing_rate.md")

    outlier_counts = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[f"{col}_outlier"] = (df[col] < lower) | (df[col] > upper)
        outlier_counts[col] = int(df[f"{col}_outlier"].sum())

        plt.figure()
        df[col].dropna().plot.box()
        plt.title(f"{col} boxplot")
        plt.savefig(outdir / f"box_{col}.png")
        plt.close()

    outlier_df = pd.DataFrame.from_dict(outlier_counts, orient="index", columns=["outlier_count"])
    outlier_df.to_csv(outdir / "outlier_summary.csv")

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
