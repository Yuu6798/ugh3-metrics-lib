from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

CONFIG_PATH = Path(__file__).resolve().parents[3] / "config.json"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration parameters from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        result = json.load(fh)
    config_data = cast(Dict[str, Any], result)
    return config_data


CONFIG: Dict[str, Any] = load_config()
MAX_VOCAB_CAP: int = int(CONFIG.get("MAX_VOCAB_CAP", 50))

__all__ = ["load_config", "CONFIG", "MAX_VOCAB_CAP"]
