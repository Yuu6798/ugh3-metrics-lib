from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class HistoryEntry:
    """Common history record for metric collection."""

    question: str
    answer_a: str
    answer_b: str
    por: float
    delta_e: float
    grv: float
    domain: str
    difficulty: int
    score: float = 0.0
    spike: bool = False
    external: bool = False
    anomaly_por: bool = False
    anomaly_delta_e: bool = False
    anomaly_grv: bool = False
    por_null: bool = False
    score_threshold: float | None = None
    timestamp: float = field(default_factory=time.time)

__all__ = ["HistoryEntry"]
