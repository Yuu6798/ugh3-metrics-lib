from __future__ import annotations

from pathlib import Path

from secl import qa_cycle


def _patch_metrics(monkeypatch, por: float = 0.9):
    monkeypatch.setattr(qa_cycle, "compute_por", lambda q, a, theta=0.82: por)
    monkeypatch.setattr(qa_cycle, "compute_delta_e_embed", lambda pq, cq, a: 0.1)
    monkeypatch.setattr(qa_cycle, "compute_grv_window", lambda hist: (0.1, set()))


def test_high_por_adopt(monkeypatch, tmp_path):
    _patch_metrics(monkeypatch, por=0.9)
    history = qa_cycle.main_qa_cycle(1, tmp_path / "hist.csv")
    assert len(history) == 1
    assert history[0].por >= 0.9


def test_low_por_reject(monkeypatch):
    _patch_metrics(monkeypatch, por=0.1)
    history = qa_cycle.main_qa_cycle(1)
    assert history == []


def test_grv_stagnation_triggers_jump(monkeypatch):
    _patch_metrics(monkeypatch, por=0.9)
    monkeypatch.setattr(
        qa_cycle,
        "is_grv_stagnation",
        lambda grv_hist, window=qa_cycle.GRV_WINDOW, threshold=qa_cycle.GRV_STAGNATION_TH: True,
    )
    monkeypatch.setattr(qa_cycle, "select_action_for_jump", lambda state: "jump")
    history = qa_cycle.main_qa_cycle(2)
    assert any(entry.question.startswith("(ジャンプ)") for entry in history)
