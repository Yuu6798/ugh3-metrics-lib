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
    timestamp: float = field(default_factory=time.time)

__all__ = ["HistoryEntry"]
