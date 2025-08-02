from __future__ import annotations

import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

os.environ.setdefault("DELTAE4_FALLBACK", "hash")

import phase_map_demo  # noqa: E402
import facade.collector  # noqa: E402
import secl.qa_cycle  # noqa: E402


def test_import_example_modules() -> None:
    pass


def test_scripts_run(tmp_path: Path) -> None:
    (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
    phase_map_demo.main()
    facade.collector.main([
        "--auto",
        "-n",
        "1",
        "-o",
        str(tmp_path / "out.csv"),
    ])
    secl.qa_cycle.main_qa_cycle(1, tmp_path / "hist.csv")


def test_run_cycle_generates_csv(tmp_path: Path) -> None:
    """run_cycle should create a CSV with expected columns and rows."""
    os.environ.setdefault("DELTAE4_FALLBACK", "hash")
    from facade.collector import run_cycle

    (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
    out_file = tmp_path / "cycle.csv"
    steps = 2
    run_cycle(steps, out_file, interactive=False)

    import csv

    with open(out_file, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = reader.fieldnames

    assert header == [
        "question",
        "answer_a",
        "answer_b",
        "por",
        "delta_e",
        "grv",
        "domain",
        "difficulty",
        "timestamp",
        "score",
        "spike",
        "external",
        "anomaly_por",
        "anomaly_delta_e",
        "anomaly_grv",
        "por_null",
        "score_threshold",
    ]
    assert 1 <= len(rows) <= steps
