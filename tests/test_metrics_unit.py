import pytest
from ugh3_metrics_lib.secl.qa_cycle import (
    novelty_score,
    is_duplicate_question,
    simulate_delta_e,
    HistoryEntry,
)


def test_novelty_score_basic() -> None:
    assert novelty_score("foo", []) == 1.0


def test_is_duplicate_question() -> None:
    hist = [HistoryEntry("hello", "ans", 0, 0, 0, False, False)]
    assert is_duplicate_question("hello", hist)
    assert not is_duplicate_question("different", hist)


def test_simulate_delta_e_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("secl.qa_cycle.util", None, raising=False)
    monkeypatch.setattr("secl.qa_cycle._ST_MODEL", None, raising=False)
    result = simulate_delta_e("prev", "curr", "ans")
    assert 0.0 <= result <= 1.0
