import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from pathlib import Path
from secl.qa_cycle import (
    check_metric_anomalies,
    detect_por_null,
    HistoryEntry,
    backup_history,
)
import json


class TestAnomalies(unittest.TestCase):
    def test_check_metric_anomalies(self) -> None:
        por, de, grv = check_metric_anomalies(0.95, 0.95, 0.96)
        self.assertTrue(por)
        self.assertTrue(de)
        self.assertTrue(grv)

    def test_detect_por_null(self) -> None:
        self.assertTrue(detect_por_null("", "ans", 0, 0))
        self.assertTrue(detect_por_null("q", "", 0, 0))
        self.assertFalse(detect_por_null("q", "ans", 1.0, 0.5))

    def test_backup_history(self) -> None:
        hist = [
            HistoryEntry(
                question="q",
                answer_a="",
                answer_b="a",
                por=0.1,
                delta_e=0.2,
                grv=0.3,
                domain="test",
                difficulty=1,
            )
        ]
        tmp_dir = Path("tests/tmp_bk")
        backup_history(tmp_dir, hist, "test")
        files = list(tmp_dir.glob("test_*.json"))
        self.assertTrue(files)
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            self.assertEqual(data[0]["question"], "q")
            f.unlink()
        tmp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
