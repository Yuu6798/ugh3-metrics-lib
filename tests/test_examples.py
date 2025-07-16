import matplotlib
from pathlib import Path

matplotlib.use("Agg")


def test_import_example_modules() -> None:
    pass


def test_scripts_run(tmp_path: Path) -> None:
    (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
    import phase_map_demo
    import facade.collector
    import secl.qa_cycle

    # phase_map_demo main should run without errors
    phase_map_demo.main()

    # run a short cycle in the collector using the CLI in auto mode
    facade.collector.main(
        [
            "--auto",
            "-n",
            "1",
            "-o",
            str(tmp_path / "out.csv"),
        ]
    )

    # run a single step of the QA cycle
    secl.qa_cycle.main_qa_cycle(1, tmp_path / "hist.csv")


def test_run_cycle_generates_csv(tmp_path: Path) -> None:
    """run_cycle should create a CSV with expected columns and rows."""
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
