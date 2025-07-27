from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_analyze_dataset_smoke(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "delta_e_internal": np.random.rand(100),
            "por_fire": np.random.choice([True, False], size=100),
            "tfidf": np.random.rand(100),
            "entropy": np.random.rand(100),
            "cooccurrence": np.random.rand(100),
            "grv_score": np.random.rand(100),
        }
    )
    infile = tmp_path / "sample.parquet"
    df.to_parquet(infile)

    outdir = tmp_path / "analysis_output"
    subprocess.run(
        [sys.executable, "scripts/analyze_dataset.py", "--infile", str(infile), "--outdir", str(outdir)],
        check=True,
    )

    assert (outdir / "report.md").exists()
    assert (outdir / "hist_delta_e.png").exists()
    assert (outdir / "bar_por_fire.png").exists()
    assert (outdir / "scatter_delta_vs_grv.png").exists()

    report_text = (outdir / "report.md").read_text(encoding="utf-8")
    assert "count" in report_text
    assert str(len(df)) in report_text
