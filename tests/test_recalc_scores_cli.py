import os
import subprocess
import sys
from pathlib import Path


def test_recalc_cli_runs(tmp_path: Path) -> None:
    csv_path = tmp_path / "current.csv"
    csv_path.write_text("question,answer_a,answer_b\nq,a,b\n", encoding="utf-8")
    out_path = tmp_path / "out.parquet"
    env = {**os.environ, "DELTAE4_FALLBACK": "hash"}
    subprocess.run(
        [sys.executable, "scripts/recalc_scores_v4.py", "--infile", str(csv_path), "--outfile", str(out_path)],
        check=True,
        env=env,
    )
    assert out_path.exists()
