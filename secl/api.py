from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from core.history_entry import HistoryEntry
from ugh.adapters.metrics import (
    compute_por,
    compute_delta_e_embed,
    compute_grv_window,
    prefetch_embed_model,
)


@dataclass
class StepInputs:
    """Inputs for :func:`evaluate_step`.

    Attributes
    ----------
    question:
        The current question being processed.
    history_list:
        Existing history of steps. The latest step (if any) should be at the end.
    config:
        Configuration values controlling SECL behaviour.
    """

    question: str
    history_list: List[HistoryEntry]
    config: Dict[str, Any]


@dataclass
class StepResult:
    """Result from :func:`evaluate_step`."""

    updated_history: List[HistoryEntry]
    decision: Literal["none", "jump"]
    debug: Dict[str, Any]


_jump_cooldown: int = 0


def evaluate_step(inputs: StepInputs) -> StepResult:
    """Evaluate one step and decide whether to trigger a jump.

    This function computes PoR, ΔE and grv metrics for the provided question
    and history.  It returns an updated history, a decision (``"jump"`` or
    ``"none"``), and a dictionary of debug information.  No logging or
    printing is performed inside this function.
    """

    try:
        prefetch_embed_model()
    except Exception:
        # ここでは握りつぶす。実際の可否は _require_model() 側で判断する。
        pass

    global _jump_cooldown

    cfg = inputs.config
    low_por_th: float = float(cfg.get("LOW_POR_TH", 0.25))
    high_delta_th: float = float(cfg.get("HIGH_DELTA_TH", 0.85))
    cooldown_val: int = int(cfg.get("JUMP_COOLDOWN", 0))

    question = inputs.question
    history = inputs.history_list

    # Placeholder answer; external pipeline may replace metrics by patching
    # ``compute_por`` or ``compute_delta_e_embed`` during tests.
    answer_stub = question

    if history:
        prev_q = history[-1].question
        delta_e = compute_delta_e_embed(prev_q, question, answer_stub)
    else:
        delta_e = 0.0

    por = compute_por(question, answer_stub)

    temp_entry = HistoryEntry(
        question=question,
        answer_a=history[-1].answer_b if history else "",
        answer_b=answer_stub,
        por=por,
        delta_e=delta_e,
        grv=0.0,
        domain="general",
        difficulty=1,
    )
    grv, _ = compute_grv_window(history + [temp_entry])
    temp_entry.grv = grv

    low_por = por < low_por_th
    high_delta = delta_e >= high_delta_th

    decision: Literal["none", "jump"] = "none"
    if _jump_cooldown > 0:
        _jump_cooldown = max(0, _jump_cooldown - 1)
    if low_por and high_delta and _jump_cooldown == 0:
        decision = "jump"
        _jump_cooldown = max(cooldown_val, 0)

    # 重要: jump の有無に関係なく、そのターンのレコードは必ず履歴へ追加する。
    updated_history = history + [temp_entry]

    debug = {
        "por": por,
        "delta_e": delta_e,
        "grv": grv,
        "low_por": low_por,
        "high_delta": high_delta,
        "low_por_th": low_por_th,
        "high_delta_th": high_delta_th,
        "jump_cooldown": _jump_cooldown,
    }

    return StepResult(updated_history=updated_history, decision=decision, debug=debug)


__all__ = ["StepInputs", "StepResult", "evaluate_step"]
