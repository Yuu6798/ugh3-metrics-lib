import types
from types import SimpleNamespace
import numpy as np
import pytest
import ugh3_metrics.metrics.deltae_v4 as dmod
from ugh3_metrics.metrics.deltae_v4 import DeltaEV4


def test_lazyload_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """STS が import 出来ない環境でも動くか確認."""
    monkeypatch.setitem(dmod.__dict__, "SentenceTransformer", None)
    m = DeltaEV4()
    assert m._embedder is None
    assert isinstance(m.score("foo", "bar"), float)

    dummy = SimpleNamespace(encode=lambda _: np.ones(2))
    monkeypatch.setitem(dmod.__dict__, "SentenceTransformer", lambda *_: dummy)
    m2 = DeltaEV4()
    assert m2.score("x", "x") == 0.0
