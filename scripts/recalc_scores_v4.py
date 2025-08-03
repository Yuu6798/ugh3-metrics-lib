#!/usr/bin/env python3
"""Recalculate PoR/ΔE metrics and derive extra flags.

This script supports CSV, Parquet and JSONL input files.  For tabular
formats (CSV/Parquet) the records are loaded into a :class:`pandas.DataFrame`.
For JSONL files the lines are streamed into a ``list[dict]``.

The following columns are (re)computed:

* ``por_a`` / ``por_b`` – PoR score for each answer.
* ``delta_e`` – ΔE score between ``answer_a`` and ``answer_b``.
* ``delta_e_internal`` – internal ΔE of successive ``hidden_state`` vectors.
* ``por_fire`` – boolean flag derived from the ``por`` column.

Output format is inferred from ``--outfile`` extension.
"""
from __future__ import annotations

import argparse
import json
import csv
import logging
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dep
    pd = None
import numpy as np
from typing import Any, Dict, List, cast

from tqdm import tqdm

from ugh3_metrics.metrics.deltae_v4 import DeltaE4
from ugh3_metrics.metrics.por_v4 import PorV4
from core.metrics import POR_FIRE_THRESHOLD, calc_delta_e_internal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recalculate PoR/ΔE v4 scores")
    parser.add_argument("--infile", type=Path, required=True, help="input file")
    parser.add_argument("--outfile", type=Path, required=True, help="output file")
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load CSV/Parquet/JSONL into a list of dicts."""
    ext = path.suffix.lower()
    if ext == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    if pd is None:
        if ext != ".csv":
            raise SystemExit("pandas is required for non-CSV formats")
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]
    if ext in {".parquet", ".pq", ".parq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, keep_default_na=False)
    df = df.assign(
        question=df["question"].astype(str).fillna(""),
        answer_a=df["answer_a"].astype(str).fillna(""),
        answer_b=df["answer_b"].astype(str).fillna(""),
    )
    return cast(List[Dict[str, Any]], df.to_dict(orient="records"))


def save_records(path: Path, recs: List[Dict[str, Any]]) -> None:
    """Save records in CSV/Parquet/JSONL format."""
    ext = path.suffix.lower()
    if ext == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for r in recs:
                json.dump(r, fh, ensure_ascii=False)
                fh.write("\n")
        return
    if pd is None:
        if ext != ".csv":
            raise SystemExit("pandas is required for non-CSV output")
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            writer.writeheader()
            writer.writerows(recs)
        print(f"Wrote results to {path}")
        return
    df = pd.DataFrame.from_records(recs)
    if ext in {".parquet", ".pq", ".parq"}:
        try:
            df.to_parquet(path, index=False)
            print(f"Wrote results to {path}")
        except (ImportError, ValueError) as e:
            warnings.warn(f"Parquet export failed ({e}); falling back to CSV.")
            csv_fallback = path.with_suffix(".csv")
            df.to_csv(csv_fallback, index=False)
            print(f"Wrote CSV fallback to {csv_fallback}")
    else:
        df.to_csv(path, index=False)
        print(f"Wrote results to {path}")


def main() -> int:
    args = parse_args()
    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    in_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        candidates = list(in_path.parent.glob("*.parquet"))
        fallback = Path("datasets/current_recalc.parquet")
        if candidates:
            in_path = candidates[0]
        elif fallback.exists():
            in_path = fallback
        else:
            print("[ERROR] no input file", file=sys.stderr)
            return 3

    recs = load_records(in_path)
    if not recs:
        print("[ERROR] no records found", file=sys.stderr)
        return 3

    required = {"question", "answer_a", "answer_b"}
    if not required.issubset(recs[0]):
        missing = ", ".join(sorted(required - set(recs[0].keys())))
        raise SystemExit(f"missing required columns: {missing}")

    try:
        de = DeltaE4()
    except RuntimeError:
        print("[ERROR] embedding failed", file=sys.stderr)
        sys.exit(1)
    pv = PorV4(auto_load=True)

    for i, rec in enumerate(tqdm(recs, desc="Scoring")):
        q = str(rec.get("question", ""))
        a1 = str(rec.get("answer_a", ""))
        a2 = str(rec.get("answer_b", ""))
        try:
            rec["por_a"] = pv.score(q, a1)
            rec["por_b"] = pv.score(q, a2)
            rec["delta_e"] = de.score(a1, a2)
        except TypeError as e:
            logging.warning("Skip record %s: %s", i, e)
            continue

        if i > 0 and "hidden_state" in rec and "hidden_state" in recs[i - 1]:
            try:
                prev_v = np.asarray(recs[i - 1]["hidden_state"], dtype=np.float64)
                cur_v = np.asarray(rec["hidden_state"], dtype=np.float64)
                rec["delta_e_internal"] = calc_delta_e_internal(prev_v, cur_v)
            except Exception:
                rec["delta_e_internal"] = None
        else:
            rec["delta_e_internal"] = None

        try:
            por_val = float(rec.get("por", 0))
            rec["por_fire"] = por_val >= POR_FIRE_THRESHOLD
        except (TypeError, ValueError):
            rec["por_fire"] = False

    save_records(out_path, recs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
