#!/usr/bin/env python3
"""Prune dataset files with ≥95% zero ``deltae`` values and open a draft PR."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from github import Github


def _zero_ratio(path: Path) -> float:
    """Return fraction of rows where ``deltae`` equals zero."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=["deltae"])
    else:
        df = pd.read_parquet(path, columns=["deltae"])
    if "deltae" not in df.columns:
        return 0.0
    return float((df["deltae"] == 0).mean())


def _collect_targets(root: Path) -> list[Path]:
    targets: list[Path] = []
    for pattern in ("*.parquet", "*.csv"):
        for path in root.rglob(pattern):
            try:
                if _zero_ratio(path) >= 0.95:
                    targets.append(path)
            except Exception as exc:  # pragma: no cover - best effort
                print(f"failed to inspect {path}: {exc}", file=sys.stderr)
    return targets


def _git(*args: Sequence[str]) -> None:
    subprocess.run(["git", *args], check=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "datasets"
    targets = _collect_targets(data_dir)
    if not targets:
        print("no datasets require pruning")
        return 0

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    branch = f"auto/prune-zero-deltae_{date}"

    _git("switch", "-c", branch)
    _git("rm", "--", *[str(p) for p in targets])
    _git("commit", "-m", "chore: prune datasets with ≥95% zero deltae")
    _git("push", "-u", "origin", branch)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 1
    origin = (
        subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    if origin.endswith(".git"):
        origin = origin[:-4]
    full_name = origin.split("github.com/")[-1]

    gh = Github(token)
    repo = gh.get_repo(full_name)
    body = "Deleted datasets:\n\n" + "\n".join(f"- {p}" for p in targets)
    repo.create_pull(
        title="Prune zero-deltae datasets",
        body=body,
        head=branch,
        base="main",
        draft=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
