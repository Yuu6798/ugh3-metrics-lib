import matplotlib
matplotlib.use('Agg')

def test_import_example_modules() -> None:
    from ugh3_metrics_lib import phase_map_demo, facade, secl, core


from pathlib import Path


def test_scripts_run(tmp_path: Path) -> None:
    from ugh3_metrics_lib import phase_map_demo, facade, secl

    # phase_map_demo main should run without errors
    phase_map_demo.main()

    # run a short cycle in the collector using the CLI in auto mode
    facade.collector.main([
        "--auto",
        "-n",
        "1",
        "-o",
        str(tmp_path / "out.csv"),
    ])

    # run a single step of the QA cycle
    secl.qa_cycle.main_qa_cycle(1, tmp_path / "hist.csv")


def test_run_cycle_generates_csv(tmp_path: Path) -> None:
    """run_cycle should create a CSV with expected columns and rows."""
    from ugh3_metrics_lib.facade.collector import run_cycle
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
        "answer",
        "por",
        "delta_e",
        "grv",
        "timestamp",
    ]
    assert 1 <= len(rows) <= steps

