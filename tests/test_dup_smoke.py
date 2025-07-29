from pathlib import Path
import subprocess


def test_dup_smoke(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "detect_duplicates.py"
    subprocess.run(["python", str(script), "--outdir", str(out)], check=True)
    dup_csv = out / "duplicate_files.csv"
    assert dup_csv.exists()
    assert sum(1 for _ in dup_csv.open()) > 1
