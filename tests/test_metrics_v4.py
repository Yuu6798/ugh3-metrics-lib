import unittest
from unittest.mock import patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from typing import Any
from numpy.typing import NDArray

import core.delta_e_v4 as delta_e_v4  # noqa: F401
import core.grv_v4 as grv_v4  # noqa: F401
import core.sci as sci  # noqa: F401
import importlib

delta_e_v4 = importlib.import_module("core.delta_e_v4")
grv_v4 = importlib.import_module("core.grv_v4")
sci = importlib.import_module("core.sci")


class DummyEmbedder:
    def encode(self, text: str) -> NDArray[Any]:
        if text == "hello":
            return np.array([1.0, 0.0])
        if text == "hello world":
            return np.array([0.62, np.sqrt(1 - 0.62 ** 2)])
        n = float(len(text.split()))
        return np.array([n, 0.0])


class TestMetricsV4(unittest.TestCase):

    @patch("core.delta_e_v4._get_embedder", return_value=DummyEmbedder())
    def test_delta_e_v4(self, mock_emb: Any) -> None:
        val = delta_e_v4.delta_e("hello", "hello world")
        self.assertAlmostEqual(val, 0.375, places=3)

    def test_grv_v4(self) -> None:
        score_val = grv_v4.grv("alpha beta beta gamma")
        self.assertGreater(score_val, 0.0)
        self.assertLessEqual(score_val, 1.0)
        long_text = " ".join(["word"] * 60)
        long_val = grv_v4.grv(long_text)
        self.assertGreater(long_val, 0.0)
        self.assertLessEqual(long_val, 1.0)
        rep_text = ("a " * 30 + "b " * 30).strip()
        self.assertLess(grv_v4.grv(rep_text), grv_v4.grv("a b"))
        self.assertLess(
            grv_v4._pmi_score(rep_text.split()),
            grv_v4._pmi_score("a b".split()),
        )

    def test_sci(self) -> None:
        sci.reset_state()
        v1 = sci.sci(0.0, 0.0, 0.0)
        v2 = sci.sci(1.0, 1.0, 1.0)
        self.assertNotEqual(v1, v2)


if __name__ == "__main__":
    unittest.main()
