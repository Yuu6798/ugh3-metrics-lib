from __future__ import annotations

from typing import Dict, List

import pytest

from core.history_entry import HistoryEntry
from secl.api import StepInputs, StepResult, evaluate_step
from facade.secl_hook import maybe_apply_secl


def _patch_metrics(monkeypatch: pytest.MonkeyPatch, por: float, delta_e: float) -> None:
    monkeypatch.setattr("secl.api.compute_por", lambda q, a, theta=0.82: por)
    monkeypatch.setattr("secl.api.compute_delta_e_embed", lambda pq, cq, a: delta_e)
    monkeypatch.setattr("secl.api.compute_grv_window", lambda hist: (0.1, set()))


def _make_history() -> List[HistoryEntry]:
    return [
        HistoryEntry(
            question="prev",
            answer_a="",
            answer_b="ans",
            por=0.5,
            delta_e=0.2,
            grv=0.1,
            domain="d",
            difficulty=1,
        )
    ]


def test_low_por_high_delta_triggers_jump(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metrics(monkeypatch, por=0.1, delta_e=0.9)
    import secl.api as api
    api._jump_cooldown = 0
    history = _make_history()
    cfg: Dict[str, float | int] = {"LOW_POR_TH": 0.25, "HIGH_DELTA_TH": 0.85, "JUMP_COOLDOWN": 1}
    res = evaluate_step(StepInputs(question="cur", history_list=history, config=cfg))
    assert res.decision == "jump"


def test_cooldown_suppresses_repeated_jumps(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_metrics(monkeypatch, por=0.1, delta_e=0.9)
    import secl.api as api
    api._jump_cooldown = 0
    history = _make_history()
    cfg: Dict[str, float | int] = {"LOW_POR_TH": 0.25, "HIGH_DELTA_TH": 0.85, "JUMP_COOLDOWN": 2}
    res1 = evaluate_step(StepInputs(question="q1", history_list=history, config=cfg))
    assert res1.decision == "jump"
    res2 = evaluate_step(StepInputs(question="q2", history_list=res1.updated_history, config=cfg))
    assert res2.decision == "none"


def test_disabled_flag_bypasses_secl(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_eval(inputs: StepInputs) -> StepResult:
        nonlocal called
        called = True
        return StepResult(updated_history=inputs.history_list, decision="none", debug={})

    monkeypatch.setattr("facade.secl_hook.evaluate_step", fake_eval)
    history: List[HistoryEntry] = []
    res = maybe_apply_secl("q", history, {"SECL_ENABLED": False})
    assert res is None
    assert not called
    res = maybe_apply_secl("q", history, {"SECL_ENABLED": True})
    assert called
    assert isinstance(res, StepResult)
