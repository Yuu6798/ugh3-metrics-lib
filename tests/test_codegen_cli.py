import sys
import os
from unittest.mock import patch
from io import StringIO

# Add the scripts directory to the Python path so we can import the CLI directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ai_issue_codegen import main


def test_codegen_help() -> None:
    """Verify that the CLI prints usage information."""
    with patch("sys.argv", ["ai_issue_codegen.py", "-h"]):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            try:
                main()
            except SystemExit:
                # argparse exits after printing help
                pass
            output = mock_stdout.getvalue()
            assert "usage" in output.lower()


def test_codegen_version() -> None:
    """Verify that the CLI prints its version."""
    with patch("sys.argv", ["ai_issue_codegen.py", "--version"]):
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            try:
                main()
            except SystemExit:
                # argparse exits after printing version
                pass
            output = mock_stdout.getvalue()
            assert "0.0.2" in output
