import argparse
import os
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # CI 環境向け
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return diff.mean() / diff.std(ddof=1)


def bootstrap_ci(data: np.ndarray, func, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    stats_ = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        stats_.append(func(sample))
    lower = np.percentile(stats_, 100 * alpha / 2)
    upper = np.percentile(stats_, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scored QA dataset")
    parser.add_argument("--input", required=True, help="scored CSV path")
    parser.add_argument("--report-dir", required=True, help="output directory")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    df = pd.read_csv(args.input)

    metrics = ["por", "delta_e", "grv", "bertscore", "bleurt", "rougeL"]

    for m in metrics:
        plt.figure()
        sns.histplot(df[m].dropna(), kde=True)
        plt.title(f"{m} distribution")
        plt.savefig(os.path.join(args.report_dir, f"{m}_hist.png"))
        plt.close()

        plt.figure()
        sns.boxplot(x=df[m].dropna())
        plt.title(f"{m} boxplot")
        plt.savefig(os.path.join(args.report_dir, f"{m}_box.png"))
        plt.close()

    corr = df[metrics].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(args.report_dir, "correlation_heatmap.png"))
    plt.close()

    results = []
    baseline = ["bertscore", "bleurt", "rougeL"]
    bonf_factor = len(baseline)
    for m in baseline:
        paired = df[["por", m]].dropna()
        x = paired["por"].to_numpy()
        y = paired[m].to_numpy()
        t_stat, p_val = stats.ttest_rel(x, y)
        diff = x - y
        sem = diff.std(ddof=1) / np.sqrt(len(diff))
        ci_low = diff.mean() - 1.96 * sem
        ci_high = diff.mean() + 1.96 * sem
        bs_low, bs_high = bootstrap_ci(diff, np.mean)
        d = cohens_d(x, y)
        results.append(
            {
                "metric": m,
                "t_stat": t_stat,
                "p_val": min(1.0, p_val * bonf_factor),
                "cohens_d": d,
                "95CI_low": ci_low,
                "95CI_high": ci_high,
                "boot_low": bs_low,
                "boot_high": bs_high,
            }
        )

    summary_path = os.path.join(args.report_dir, "ttest_summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)

    if "adopted" in df.columns:
        adoption_rate = df[df["adopted"] == 1].shape[0] / max(1, df.shape[0])
        pd.DataFrame({"adoption_rate": [adoption_rate]}).to_csv(
            os.path.join(args.report_dir, "adoption_summary.csv"), index=False
        )


if __name__ == "__main__":
    main()
