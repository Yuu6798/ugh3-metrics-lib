"""Add ``por_fire`` field to JSONL metrics files.

Usage examples::

    python extract_metrics.py --infile data.jsonl --outfile out.jsonl
    python extract_metrics.py datasets/*.jsonl > merged.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, TextIO
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.metrics import is_por_fire


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add por_fire flag to JSONL records")
    p.add_argument("paths", nargs="*", type=Path, help="Input JSONL paths")
    p.add_argument("--infile", type=Path)
    p.add_argument("--outfile", type=Path)
    return p.parse_args()


def _process_stream(fh_in: Iterable[str], fh_out: TextIO) -> None:
    for line in fh_in:
        line = line.strip()
        if not line:
            continue
        j: Dict[str, Any] = json.loads(line)
        if "por" in j:
            por_val = j.get("por")
            if por_val is None or (
                isinstance(por_val, str) and por_val.strip() == ""
            ):
                j["por_fire"] = False
            else:
                try:
                    j["por_fire"] = is_por_fire(float(por_val))
                except (TypeError, ValueError):
                    j["por_fire"] = False
        json.dump(j, fh_out, ensure_ascii=False)
        fh_out.write("\n")


def main() -> None:
    args = parse_args()
    if args.infile:
        out_fh = args.outfile.open("w", encoding="utf-8") if args.outfile else sys.stdout
        with args.infile.open("r", encoding="utf-8") as fh_in:
            _process_stream(fh_in, out_fh)
        if out_fh is not sys.stdout:
            out_fh.close()
    else:
        if not args.paths:
            raise SystemExit("no input files provided")
        for path in args.paths:
            with path.open("r", encoding="utf-8") as fh_in:
                _process_stream(fh_in, sys.stdout)


if __name__ == "__main__":
    main()
