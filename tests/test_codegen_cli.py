import subprocess
import sys


def test_codegen_help() -> None:
    out = subprocess.check_output([sys.executable, "scripts/ai_issue_codegen.py", "-h"])
    assert b"usage" in out.lower()
