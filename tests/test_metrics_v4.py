import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from typing import Any
from numpy.typing import NDArray

import core.sci as sci  # noqa: F401
import importlib

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

    def test_sci(self) -> None:
        sci.reset_state()
        v1 = sci.sci(0.0, 0.0, 0.0)
        v2 = sci.sci(1.0, 1.0, 1.0)
        self.assertNotEqual(v1, v2)


if __name__ == "__main__":
    unittest.main()
