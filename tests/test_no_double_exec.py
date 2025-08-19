import subprocess, sys, tempfile, pathlib

def test_cli_single_run():
    out = pathlib.Path(tempfile.gettempdir()) / "cycle_test.csv"
    if out.exists():
        out.unlink()
    # runpy 由来の RuntimeWarning が出たら失敗
    cp = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning:runpy",
            "-m",
            "facade.collector",
            "--auto",
            "-n",
            "1",
            "--summary",
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr or cp.stdout
    assert out.exists()
    # 終了メッセージの重複が無いこと（Summary が1回）
    assert cp.stdout.count("== Summary ==") <= 1
