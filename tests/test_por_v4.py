import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from typing import Any
from numpy.typing import NDArray
from unittest.mock import patch
import types

from ugh3_metrics.metrics import PorV4, calc_por_v4


class DummyEmbedder:
    def encode(self, text: str) -> NDArray[Any]:
        return np.ones(2)


class TestPorV4(unittest.TestCase):
    def test_score(self) -> None:
        metric = PorV4(embedder=DummyEmbedder())
        val = metric.score("a", "b")
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_calc_por_v4(self) -> None:
        fake_module = types.SimpleNamespace(SentenceTransformer=lambda *a, **k: DummyEmbedder())
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            val = calc_por_v4("a", "b")
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)


if __name__ == "__main__":
    unittest.main()
