from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration parameters from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return cast(Dict[str, Any], json.load(fh))


CONFIG: Dict[str, Any] = load_config()

__all__ = ["load_config", "CONFIG"]
