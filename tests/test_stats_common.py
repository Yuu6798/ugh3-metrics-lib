from __future__ import annotations

import json
from pathlib import Path

from tools.stats_common import attach_meta


def test_attach_meta(tmp_path: Path) -> None:
    meta = {"records_total": 10, "zeros_removed": 3, "kept": 7, "date": "T"}
    mp = tmp_path / "meta.json"
    mp.write_text(json.dumps(meta), encoding="utf-8")
    payload = attach_meta({"ok": True}, csv="x.csv", meta_path=str(mp))
    assert payload["meta"]["csv"] == "x.csv"
    assert payload["meta"]["counts"]["kept"] == 7

