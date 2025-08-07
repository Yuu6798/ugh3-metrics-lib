from __future__ import annotations

from secl import qa_cycle


def _patch_metrics(monkeypatch, por: float):
    monkeypatch.setattr(qa_cycle, "compute_por", lambda q, a, theta=0.82: por)
    monkeypatch.setattr(qa_cycle, "compute_delta_e_embed", lambda pq, cq, a: 0.1)
    monkeypatch.setattr(qa_cycle, "compute_grv_window", lambda hist: (0.1, set()))


def test_low_por_threshold_blocks_adoption(monkeypatch, tmp_path):
    _patch_metrics(monkeypatch, por=0.88)
    qa_cycle.CONFIG["LOW_POR_TH"] = 0.90
    history = qa_cycle.main_qa_cycle(1, tmp_path / "hist.csv")
    assert history == []


def test_high_por_threshold_allows_adoption(monkeypatch, tmp_path):
    _patch_metrics(monkeypatch, por=0.93)
    qa_cycle.CONFIG["LOW_POR_TH"] = 0.90
    history = qa_cycle.main_qa_cycle(1, tmp_path / "hist.csv")
    assert len(history) == 1
