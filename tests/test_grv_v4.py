import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ugh3_metrics.metrics import GrvV4, calc_grv_v4


class TestGrvV4(unittest.TestCase):
    def test_components_and_wrapper(self) -> None:
        metric = GrvV4()
        tokens = "alpha beta beta gamma".split()
        self.assertGreaterEqual(metric._tfidf_score(tokens), 0.0)
        self.assertGreaterEqual(metric._pmi_score(tokens), -1.0)
        self.assertGreaterEqual(metric._entropy_score(tokens), 0.0)

        val1 = metric.score("alpha beta beta gamma", "")
        val2 = calc_grv_v4("alpha beta beta gamma")
        self.assertAlmostEqual(val1, val2)

        rep_text = ("a " * 30 + "b " * 30).strip()
        self.assertLess(metric.score(rep_text, ""), metric.score("a b", ""))
        self.assertLess(
            metric._pmi_score(rep_text.split()),
            metric._pmi_score("a b".split()),
        )


if __name__ == "__main__":
    unittest.main()
