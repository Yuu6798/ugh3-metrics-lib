from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_missing_raw_csv(tmp_path: Path) -> None:
    script = Path("scripts/build_dataset.py").resolve()
    proc = subprocess.run([
        sys.executable,
        str(script),
        "--raw-dir",
        str(tmp_path / "raw"),
    ], capture_output=True)
    assert proc.returncode == 3
    assert b"[ERROR] no raw CSV found" in proc.stderr
