from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_build_dataset_missing_pandas() -> None:
    script = Path("scripts/build_dataset.py").resolve()
    proc = subprocess.run([
        sys.executable,
        "-S",
        str(script),
    ], capture_output=True)
    assert proc.returncode == 1
    assert b"pandas is required" in proc.stderr
