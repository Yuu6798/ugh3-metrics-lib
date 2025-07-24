import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.metrics import POR_FIRE_THRESHOLD


def test_recalc_scores_v4_internal(tmp_path: Path) -> None:
    infile = tmp_path / "in.jsonl"
    outfile = tmp_path / "out.jsonl"
    records = [
        {"question": "q1", "answer_a": "a1", "answer_b": "b1", "por": "0.81", "hidden_state": [0.0, 1.0]},
        {"question": "q2", "answer_a": "a2", "answer_b": "b2", "por": "0.83", "hidden_state": [1.0, 0.0]},
    ]
    with infile.open("w", encoding="utf-8") as fh:
        for rec in records:
            json.dump(rec, fh)
            fh.write("\n")

    subprocess.run(
        [sys.executable, "scripts/recalc_scores_v4.py", "--infile", str(infile), "--outfile", str(outfile)],
        check=True,
    )
    outs = [json.loads(line) for line in outfile.read_text(encoding="utf-8").splitlines()]

    assert outs[0]["delta_e_internal"] is None
    assert pytest.approx(outs[1]["delta_e_internal"], abs=0.01) == 1.0
    assert not outs[0]["por_fire"]
    assert outs[1]["por_fire"]
