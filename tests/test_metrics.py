import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest

from secl_qa_cycle import novelty_score, is_duplicate_question, HistoryEntry


class TestMetrics(unittest.TestCase):
    def test_novelty_empty_history(self):
        self.assertEqual(novelty_score('q', []), 1.0)

    def test_novelty_similarity_penalty(self):
        history = [HistoryEntry('hello world', 'ans', 0, 0, 0, False, False)]
        result = novelty_score('hello world', history)
        self.assertLess(result, 1.0)

    def test_duplicate_detection(self):
        history = [HistoryEntry('what is life?', 'ans', 0, 0, 0, False, False)]
        self.assertTrue(is_duplicate_question('what is life?', history))
        self.assertFalse(is_duplicate_question('another', history))


if __name__ == '__main__':
    unittest.main()
