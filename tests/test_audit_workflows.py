from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def test_audit_workflows(tmp_path: Path) -> None:
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True)
    (wf_dir / "workflow1.yml").write_text(
        """
name: wf1
on: [push]
jobs:
  build:
    steps:
      - name: step1
        run: echo 1
      - name: step2
        run: echo 2
"""
    )
    (wf_dir / "workflow2.yml").write_text(
        """
name: wf2
on: [push]
jobs:
  build:
    steps:
      - name: step1
        run: echo 1
      - name: step2
        run: echo 2
"""
    )
    out = tmp_path / "report.md"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_workflows.py"
    subprocess.run([
        sys.executable,
        str(script),
        "--out",
        str(out),
    ], check=True, cwd=tmp_path)
    assert out.exists()
    assert out.read_text().count("\n") > 5
