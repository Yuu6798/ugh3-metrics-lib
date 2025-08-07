# mypy: warn-unused-ignores=false
"""Scan datasets for zero or NaN ΔE columns and optionally purge files.

This utility searches recursively under ``datasets`` for ``.csv`` and ``.parquet``
files containing a ``deltae`` or ``delta_e`` column. It streams the column to
avoid loading entire files into memory, reports the fraction of zero/NaN values
and can remove offending files via ``git rm``.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class ScanResult:
    path: Path
    rows: int
    zero_percent: float
    delta_column: str | None
    flagged: bool


DELTA_COLS = ["deltae", "delta_e"]


def _detect_delta_column(path: Path) -> str | None:
    if path.suffix.lower() == ".csv":
        columns = pd.read_csv(path, nrows=0).columns
    else:
        columns = pq.ParquetFile(path).schema.names
    for col in DELTA_COLS:
        if col in columns:
            return col
    return None


def _scan_csv(path: Path, column: str) -> tuple[int, int]:
    rows = 0
    zeros = 0
    for chunk in pd.read_csv(path, usecols=[column], chunksize=65_536):
        arr = chunk[column].to_numpy()
        zeros += int(np.count_nonzero((arr == 0) | np.isnan(arr)))
        rows += arr.size
    return rows, zeros


def _scan_parquet(path: Path, column: str) -> tuple[int, int]:
    rows = 0
    zeros = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=[column], batch_size=65_536):
        arr = batch.column(0).to_numpy(zero_copy_only=False)
        zeros += int(np.count_nonzero((arr == 0) | np.isnan(arr)))
        rows += arr.size
    return rows, zeros


def _scan_file(path: Path, min_rows: int, max_zero_percent: float) -> ScanResult:
    delta_col = _detect_delta_column(path)
    if delta_col is None:
        return ScanResult(path, 0, 0.0, None, False)
    if path.suffix.lower() == ".csv":
        rows, zeros = _scan_csv(path, delta_col)
    else:
        rows, zeros = _scan_parquet(path, delta_col)
    zero_percent = (zeros / rows * 100) if rows else 0.0
    flagged = rows >= min_rows and zero_percent >= max_zero_percent
    return ScanResult(path, rows, zero_percent, delta_col, flagged)


def _iter_files(root: Path) -> Iterator[Path]:
    yield from root.rglob("*.csv")
    yield from root.rglob("*.parquet")


def _print_markdown(results: Sequence[ScanResult]) -> None:
    print("# Zero ΔE Scan Report\n")
    print("| File | Rows | Zero % | Column | Flagged |")
    print("| --- | ---: | ---: | --- | --- |")
    for r in results:
        flagged = "yes" if r.flagged else ""
        column = r.delta_column or "-"
        print(
            f"| {r.path} | {r.rows} | {r.zero_percent:.2f} | {column} | {flagged} |"
        )


def _write_json(results: Sequence[ScanResult]) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = reports_dir / f"zero_deltae_scan_{timestamp}.json"
    payload = {
        "generated_at": timestamp,
        "results": [
            {
                "path": str(r.path),
                "rows": r.rows,
                "zero_percent": r.zero_percent,
                "delta_column": r.delta_column,
                "flagged": r.flagged,
            }
            for r in results
        ],
        "total_scanned": len(results),
        "total_flagged": sum(r.flagged for r in results),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def _purge(files: Sequence[Path], force: bool) -> None:
    if not files:
        return
    if not force:
        ans = input("Remove flagged files? [y/N]: ").strip().lower()
        if ans != "y":
            print("Purge aborted.")
            return
    rm_cmd = ["git", "rm", *[str(p) for p in files]]
    subprocess.run(rm_cmd, check=True)
    subprocess.run([
        "git",
        "commit",
        "-m",
        "remove zero-ΔE datasets (auto)",
    ], check=True)


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--min-rows", type=int, default=0)
    parser.add_argument("--max-zero-percent", type=float, default=100.0)
    parser.add_argument("--purge", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    datasets_dir: Path | None = None,
) -> int:
    args = parse_args(argv)
    root = datasets_dir or Path("datasets")
    results = [
        _scan_file(path, args.min_rows, args.max_zero_percent)
        for path in _iter_files(root)
    ]
    _print_markdown(results)
    _write_json(results)
    flagged_paths = [r.path for r in results if r.flagged]
    if args.purge:
        _purge(flagged_paths, args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
