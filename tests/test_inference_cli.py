import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers", reason="HF not installed")


def test_run_inference_cli(tmp_path: Path) -> None:
    txt_path = tmp_path / "input.txt"
    txt_path.write_text("hello\nworld\n", encoding="utf-8")
    jsonl_path = tmp_path / "out.jsonl"
    subprocess.run(
        [
            sys.executable,
            "models/run_inference.py",
            "--model",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--infile",
            str(txt_path),
            "--dump-hidden",
            str(jsonl_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["turn"] == i
        assert isinstance(record["hidden_state"], list)
