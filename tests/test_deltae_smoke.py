from ugh.adapters.metrics import compute_delta_e_embed, prefetch_embed_model
import pytest


def test_deltae_nonzero_on_different_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DELTAE4_FALLBACK", raising=False)
    try:
        prefetch_embed_model()
    except Exception as exc:  # pragma: no cover - optional path
        pytest.skip(f"model unavailable: {exc}")
    d = compute_delta_e_embed("犬が好き", "猫が好き", "答え")
    assert d > 0.0
