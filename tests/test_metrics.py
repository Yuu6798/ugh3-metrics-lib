import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from secl.qa_cycle import novelty_score, is_duplicate_question  # noqa: E402
from core.history_entry import HistoryEntry


class TestMetrics(unittest.TestCase):
    def test_novelty_empty_history(self) -> None:
        self.assertEqual(novelty_score("q", []), 1.0)

    def test_novelty_similarity_penalty(self) -> None:
        history = [
            HistoryEntry(
                question="hello world",
                answer_a="",
                answer_b="ans",
                por=0.0,
                delta_e=0.0,
                grv=0.0,
                domain="test",
                difficulty=1,
            )
        ]
        result = novelty_score("hello world", history)
        self.assertLess(result, 1.0)

    def test_duplicate_detection(self) -> None:
        history = [
            HistoryEntry(
                question="what is life?",
                answer_a="",
                answer_b="ans",
                por=0.0,
                delta_e=0.0,
                grv=0.0,
                domain="test",
                difficulty=1,
            )
        ]
        self.assertTrue(is_duplicate_question("what is life?", history))
        self.assertFalse(is_duplicate_question("another", history))


if __name__ == "__main__":
    unittest.main()
