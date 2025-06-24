import unittest
from pathlib import Path
import sys
from typing import Any
from numpy.typing import NDArray
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ugh3_metrics.metrics import DeltaEV4, calc_deltae_v4


class DummyEmbedder:
    def encode(self, text: str) -> NDArray[Any]:
        if text == "hello":
            return np.array([1.0, 0.0])
        if text == "hello world":
            return np.array([0.62, np.sqrt(1 - 0.62 ** 2)])
        n = float(len(text.split()))
        return np.array([n, 0.0])


class TestDeltaEV4(unittest.TestCase):
    def test_score(self) -> None:
        metric = DeltaEV4(embedder=DummyEmbedder())
        val = metric.score("hello", "hello world")
        self.assertAlmostEqual(val, 0.375, places=3)

    def test_calc_deltae_v4(self) -> None:
        fake_module = SimpleNamespace(SentenceTransformer=lambda *a, **k: DummyEmbedder())
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            val = calc_deltae_v4("hello", "hello world")
        self.assertAlmostEqual(val, 0.375, places=3)


if __name__ == "__main__":
    unittest.main()
