from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def pick_col(df: pd.DataFrame, cands: Tuple[str, ...]) -> Optional[str]:
    """Return first column name that exists in df from candidates."""
    for c in cands:
        if c in df.columns:
            return c
    return None


def float_(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:  # pragma: no cover - defensive
        return default


def now_iso() -> str:
    """Return current UTC time in ISO format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def attach_meta(
    payload: Dict[str, Any], *, csv: str, meta_path: Optional[str]
) -> Dict[str, Any]:
    """Attach normalized meta to payload.

    Parameters
    ----------
    payload:
        Original payload to be extended.
    csv:
        Path to the CSV used to generate statistics.
    meta_path:
        Path to optional ``meta.json`` containing dataset build information.
    """

    out = dict(payload)
    meta: Dict[str, Any] = {
        "date": os.environ.get("DATE") or now_iso(),
        "csv": csv,
    }
    if meta_path:
        p = Path(meta_path)
        meta["meta_path"] = str(p)
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                counts = {}
                for k in ("records_total", "zeros_removed", "kept"):
                    if k in raw:
                        counts[k] = int(raw[k])
                if counts:
                    meta["counts"] = counts
                for k in ("date", "commit", "run_id"):
                    if k in raw:
                        meta[k] = raw[k]
            except Exception:  # pragma: no cover - best effort
                meta["counts"] = {"error": "failed_to_parse_meta"}
    out["meta"] = meta
    return out

