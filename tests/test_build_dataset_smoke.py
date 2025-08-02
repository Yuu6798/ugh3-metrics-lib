from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_build_dataset_smoke(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    df1 = pd.DataFrame({"question": ["q1"], "answer_a": ["a1"], "answer_b": ["b1"], "por": [0.2]})
    df2 = pd.DataFrame({"question": ["q2"], "answer_a": ["a2"], "answer_b": ["b2"], "por": [0.3]})
    df1.to_csv(raw / "a.csv", index=False)
    df2.to_csv(raw / "b.csv", index=False)

    out = tmp_path / "out.parquet"
    env = {**os.environ, "DELTAE4_FALLBACK": "hash"}
    proc = subprocess.run(
        [sys.executable, "scripts/build_dataset.py", "--raw-dir", str(raw), "--out-parquet", str(out)],
        check=False,
        env=env,
    )
    assert proc.returncode == 0
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == len(df1) + len(df2)
