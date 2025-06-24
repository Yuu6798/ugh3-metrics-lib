import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ugh3_metrics.metrics import SciV4, sci, reset_state


class TestSciV4(unittest.TestCase):
    def test_stateful_transition_and_wrapper(self) -> None:
        reset_state()
        v1 = sci(0.0)
        v2 = sci(1.0)
        self.assertNotEqual(v1, v2)

        metric = SciV4()
        metric.reset_state()
        w1 = metric.score(0.0, "")
        w2 = metric.score(1.0, "")
        self.assertEqual(v1, w1)
        self.assertEqual(v2, w2)


if __name__ == "__main__":
    unittest.main()
