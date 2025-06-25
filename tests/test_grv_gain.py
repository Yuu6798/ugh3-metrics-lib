import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from secl.qa_cycle import simulate_grv_gain_with_jump, simulate_grv_gain_with_external_info

class TestGrvGain(unittest.TestCase):
    def test_gain_returns_float(self) -> None:
        state: dict[str, float | set[str]] = {"vocab_set": {"a", "b"}, "grv": 0.2}
        gain_jump = simulate_grv_gain_with_jump(state)
        gain_ext = simulate_grv_gain_with_external_info(state)
        self.assertIsInstance(gain_jump, float)
        self.assertIsInstance(gain_ext, float)

if __name__ == '__main__':
    unittest.main()
