# mypy: warn-unused-ignores=false
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from scripts.detect_and_purge_zero_deltae import main


def _create_files(root: Path) -> tuple[Path, Path, Path]:
    zero_csv = root / "zero.csv"
    pd.DataFrame({"deltae": [0, 0, 0, 0, 0]}).to_csv(zero_csv, index=False)
    half_parquet = root / "half.parquet"
    pd.DataFrame({"deltae": [0, 1, 0, 1]}).to_parquet(half_parquet, index=False)
    good_csv = root / "good.csv"
    pd.DataFrame({"deltae": [1, 2, 3, 4]}).to_csv(good_csv, index=False)
    return zero_csv, half_parquet, good_csv


def _latest_report() -> Path:
    reports = sorted(Path("reports").glob("zero_deltae_scan_*.json"))
    return reports[-1]


def test_scan_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    zero_csv, half_parquet, good_csv = _create_files(tmp_path)
    main([], tmp_path)
    out = capsys.readouterr().out
    assert "Zero ΔE Scan Report" in out
    assert str(zero_csv) in out
    data = json.loads(_latest_report().read_text())
    assert data["total_scanned"] == 3
    assert data["total_flagged"] == 1
    flagged = [r["path"] for r in data["results"] if r["flagged"]]
    assert str(zero_csv) in flagged
    assert str(half_parquet) not in flagged
    assert str(good_csv) not in flagged


def test_thresholds_and_purge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zero_csv, half_parquet, _ = _create_files(tmp_path)
    main(["--max-zero-percent", "50"], tmp_path)
    data = json.loads(_latest_report().read_text())
    assert data["total_flagged"] == 2
    main(["--min-rows", "10"], tmp_path)
    data = json.loads(_latest_report().read_text())
    assert data["total_flagged"] == 0

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:2] == ["git", "rm"]:
            for f in cmd[2:]:
                Path(f).unlink()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    main(["--purge", "--force"], tmp_path)
    assert not zero_csv.exists()
    assert half_parquet.exists()
    assert calls[0][:2] == ["git", "rm"]
    assert calls[1][:2] == ["git", "commit"]
