import subprocess, sys, tempfile, pathlib

def test_cli_single_run() -> None:
    out = pathlib.Path(tempfile.gettempdir()) / "cycle_test.csv"
    if out.exists():
        out.unlink()
    cp = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning:runpy", "-m", "facade.collector",
         "--auto", "-n", "1", "--summary", "-o", str(out)],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr or cp.stdout
    assert out.exists()
    assert cp.stdout.count("== Summary ==") <= 1
