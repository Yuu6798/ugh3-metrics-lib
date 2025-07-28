from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def make_df(kind: str, rows: int) -> pd.DataFrame:
    groups = np.random.choice([True, False], size=rows, p=[0.25, 0.75])
    if kind == "good":
        delta = np.where(groups, np.random.normal(0.2, 0.05, size=rows), np.random.normal(0.8, 0.05, size=rows))
    else:
        delta = np.where(groups, np.random.normal(0.5, 0.05, size=rows), np.random.normal(0.51, 0.05, size=rows))
    return pd.DataFrame({
        "delta_e_internal": np.clip(delta, 0, 1),
        "por_fire": groups,
    })


@pytest.mark.parametrize("kind,rows,expected", [("good", 1200, 0), ("bad", 1200, 2)])  # type: ignore[misc]
def test_evaluate_dataset_smoke(tmp_path: Path, kind: str, rows: int, expected: int) -> None:
    df = make_df(kind, rows)
    infile = tmp_path / "in.parquet"
    df.to_parquet(infile)
    outdir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_dataset.py",
            "--infile",
            str(infile),
            "--outdir",
            str(outdir),
            "--group-col",
            "por_fire",
            "--metric-cols",
            "delta_e_internal",
        ]
    )
    assert result.returncode == expected
    assert (outdir / "report.md").exists()
