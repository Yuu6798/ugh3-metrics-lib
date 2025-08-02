import pytest
import ugh3_metrics.metrics.deltae_v4 as dmod
from ugh3_metrics.metrics.deltae_v4 import DeltaE4


def test_error_when_no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELTAE4_FALLBACK", raising=False)
    monkeypatch.setattr(dmod, "_load_embedder", lambda: (_ for _ in ()).throw(RuntimeError("no model")))
    with pytest.raises(RuntimeError):
        DeltaE4()


def test_hash_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dmod, "_load_embedder", lambda: (_ for _ in ()).throw(RuntimeError("no model")))
    m = DeltaE4(fallback="hash")
    assert isinstance(m.score("foo", "bar"), float)
