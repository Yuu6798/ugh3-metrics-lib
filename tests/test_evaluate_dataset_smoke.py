from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def make_df(kind: str, rows: int) -> pd.DataFrame:
    if kind == "good":
        return pd.DataFrame({
            "delta_e_internal": np.clip(np.random.rand(rows), 0, 1),
            "por_fire": np.random.choice([True, False], size=rows, p=[0.25, 0.75]),
        })
    return pd.DataFrame({
        "delta_e_internal": np.concatenate([np.ones(rows - 1), np.array([1.5])]),
        "por_fire": np.random.choice([True, False], size=rows),
    })


@pytest.mark.parametrize("kind,rows,expected", [("good", 1200, 0), ("bad", 50, 2)])  # type: ignore[misc]
def test_evaluate_dataset_smoke(tmp_path: Path, kind: str, rows: int, expected: int) -> None:
    df = make_df(kind, rows)
    infile = tmp_path / "in.parquet"
    df.to_parquet(infile)
    outdir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_dataset.py", "--infile", str(infile), "--outdir", str(outdir)]
    )
    assert result.returncode == expected
    assert (outdir / "report.md").exists()
