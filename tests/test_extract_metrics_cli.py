import subprocess
import json
import sys
import pathlib

SAMPLE = pathlib.Path("tests/fixtures/sample.jsonl")


def test_cli_smoke() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/extract_metrics.py", str(SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    outs = []
    for line in proc.stdout.strip().splitlines():
        record = json.loads(line)
        outs.append(record)

    fired = {record["por_fire"] for record in outs}
    assert fired == {False, True}

