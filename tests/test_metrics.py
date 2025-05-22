import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

from secl.qa_cycle import novelty_score, is_duplicate_question, HistoryEntry


class TestMetrics(unittest.TestCase):
    def test_novelty_empty_history(self) -> None:
        self.assertEqual(novelty_score('q', []), 1.0)

    def test_novelty_similarity_penalty(self) -> None:
        history = [HistoryEntry('hello world', 'ans', 0, 0, 0, False, False)]
        result = novelty_score('hello world', history)
        self.assertLess(result, 1.0)

    def test_duplicate_detection(self) -> None:
        history = [HistoryEntry('what is life?', 'ans', 0, 0, 0, False, False)]
        self.assertTrue(is_duplicate_question('what is life?', history))
        self.assertFalse(is_duplicate_question('another', history))

    def test_delta_e_range(self) -> None:
        import random
        from secl.qa_cycle import main_qa_cycle

        random.seed(42)
        hist = main_qa_cycle(50)
        values = [h.delta_e for h in hist]
        avg = sum(values) / len(values)
        self.assertGreaterEqual(avg, 0.35)
        self.assertLessEqual(avg, 0.65)


if __name__ == '__main__':
    unittest.main()
