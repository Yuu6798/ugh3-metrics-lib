from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.history_entry import HistoryEntry
from secl.api import StepInputs, StepResult, evaluate_step


def _external_knowledge_stub() -> None:
    """Placeholder for the external knowledge pathway."""
    return None


def maybe_apply_secl(
    question: str, history: List[HistoryEntry], config: Dict[str, Any]
) -> Optional[StepResult]:
    """Run SECL evaluation if enabled."""

    if not config.get("SECL_ENABLED", True):
        return None
    res = evaluate_step(StepInputs(question=question, history_list=history, config=config))
    if res.decision == "jump":
        _external_knowledge_stub()
    return res


__all__ = ["maybe_apply_secl"]
