from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import pytest
import torch
from scripts import check_deltae_zero


class DummyModel:
    """Minimal stand-in for SentenceTransformer."""

    def encode(self, texts: Sequence[str], convert_to_tensor: bool = True) -> torch.Tensor:
        mapping: dict[str, list[float]] = {
            "q1": [1.0, 0.0, 0.0],
            "b1": [0.0, 1.0, 0.0],
            "q2": [0.0, 0.0, 0.0],
            "a2": [0.0, 0.0, 0.0],
            "b2": [0.0, 0.0, 0.0],
        }
        return torch.tensor([mapping[t] for t in texts], dtype=torch.float32)


def run_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext: str) -> None:
    df = pd.DataFrame(
        [
            {"question": "q1", "answer_a": "q1", "answer_b": "b1", "deltae": 0.0},
            {"question": "q2", "answer_a": "a2", "answer_b": "b2", "deltae": 0.0},
            {"question": "q3", "answer_a": "a3", "answer_b": "b3", "deltae": 0.5},
        ]
    )
    path = tmp_path / f"data{ext}"
    if ext == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)

    monkeypatch.setattr(check_deltae_zero, "SentenceTransformer", lambda *a, **k: DummyModel())
    rc = check_deltae_zero.main([str(path)])
    assert rc == 0

    if ext == ".csv":
        out_df = pd.read_csv(path, keep_default_na=False)
    else:
        out_df = pd.read_parquet(path)

    assert list(out_df["deltae_zero_flag"]) == [True, True, False]
    assert list(out_df["deltae_zero_reason"]) == ["IDENTICAL_TEXT", "ZERO_VECTOR", ""]


def test_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_case(tmp_path, monkeypatch, ".csv")


def test_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_case(tmp_path, monkeypatch, ".parquet")
