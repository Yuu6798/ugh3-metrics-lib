import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

from history_evaluator import evaluate_record, GOOD, OKAY, BAD


class TestHistoryEvaluator(unittest.TestCase):
    def test_evaluate_good(self):
        self.assertEqual(evaluate_record(0.8, 0.5, 0.6), GOOD)

    def test_evaluate_okay(self):
        self.assertEqual(evaluate_record(0.6, 0.35, 0.45), OKAY)

    def test_evaluate_bad(self):
        self.assertEqual(evaluate_record(0.2, 0.1, 0.1), BAD)


if __name__ == '__main__':
    unittest.main()
