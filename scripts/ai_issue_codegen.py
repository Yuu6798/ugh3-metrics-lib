#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mypy: disable-error-code=unused-ignore
"""Generate and apply patches from GitHub Issues.

Usage:
    python scripts/ai_issue_codegen.py "<issue body>"

Exit codes:
    0: success
    1: patch failed
"""

from __future__ import annotations

__version__ = "0.0.1"

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

GEN_DIR = Path("ai_generated")


def llm(issue_body: str) -> str:
    import openai

    api_key = os.getenv("OPENAI_API_KEY", "")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert at generating unified diff patches for GitHub repositories. "
                    "When given a GitHub issue description, generate ONLY a properly formatted unified diff. "
                    "CRITICAL REQUIREMENTS:\n"
                    "1. CAREFULLY READ the issue to identify the EXACT filename to modify\n"
                    "2. If the issue mentions 'README.md', use 'README.md' as the filename\n"
                    "3. Always start with 'diff --git a/FILENAME b/FILENAME' using the correct filename\n"
                    "4. Include proper file headers: '--- a/FILENAME' and '+++ b/FILENAME'\n"
                    "5. Include hunk headers with line numbers: '@@ -start,count +start,count @@'\n"
                    "6. Use '+' for added lines, '-' for removed lines, ' ' for context\n"
                    "7. Include 3 lines of context before and after changes\n"
                    "8. Output ONLY the diff, no explanations or code blocks\n\n"
                    "Example for adding content to README.md:\n"
                    "diff --git a/README.md b/README.md\n"
                    "index 1234567..abcdefg 100644\n"
                    "--- a/README.md\n"
                    "+++ b/README.md\n"
                    "@@ -10,3 +10,6 @@\n"
                    " existing line\n"
                    " existing line\n"
                    " \n"
                    "+## New Section\n"
                    "+New content here\n"
                    "+\n"
                ),
            },
            {
                "role": "user",
                "content": issue_body,
            },
        ],
        temperature=0.1,
    )
    content: str
    if hasattr(response, "choices"):
        content = response.choices[0].message.content  # type: ignore[attr-defined]
    else:
        choice = cast(dict[str, Any], response["choices"][0])  # type: ignore[index]
        message = cast(dict[str, Any], choice["message"])
        content = message["content"]
    return str(content)


def apply_patch(diff_text: str | bytes) -> None:
    """Apply the unified diff text using multiple strategies with verbose output."""
    print("=== GPT-4 GENERATED DIFF DEBUG ===")
    print("Raw input:")
    print(repr(diff_text))
    is_bytes = isinstance(diff_text, bytes)
    text = diff_text.decode() if is_bytes else diff_text
    print("\nFormatted input:")
    print(text)
    print("=== END DEBUG ===")
    # --- normalize incoming diff ---------------------------------
    cleaned = []
    for raw in text.splitlines():
        line = raw.lstrip("| ")  # remove leading quote mark
        if line.startswith("```"):  # drop code-fence lines
            continue
        cleaned.append(line)
    # add missing  diff --git  header
    if cleaned and cleaned[0].startswith("--- a/"):
        first = cleaned[0][4:]  # "a/README.md"
        second = first.replace("a/", "b/", 1)
        cleaned.insert(0, f"diff --git {first} {second}")
    processed_text = "\n".join(cleaned) + "\n"
    # --------------------------------------------------------------
    GEN_DIR.mkdir(exist_ok=True)
    patch_path = GEN_DIR / "auto.patch"
    patch_path.write_text(processed_text)

    print(f"[debug] cwd: {os.getcwd()}")
    print(f"[debug] patch written to: {patch_path}")
    print("[debug] processed patch:\n" + processed_text)

    targets = []
    for line in processed_text.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            path = line[4:].split()[0]
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if path not in targets:
                targets.append(path)
    for t in targets:
        print(f"[debug] check exists {t}: {Path(t).exists()}")

    cmds = [
        ["patch", "-p1", "-d", "."],
        ["patch", "-p1"],
        ["patch", "-p0", "-d", "."],
        ["git", "apply", "--verbose"],
    ]
    for cmd in cmds:
        try:
            print(f"[debug] running: {' '.join(cmd)}")
            if is_bytes:
                proc = subprocess.run(
                    cmd,
                    input=processed_text.encode(),
                    capture_output=True,
                    text=False,
                    timeout=30,
                )
            else:
                proc = subprocess.run(
                    cmd,
                    input=processed_text,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            print(proc.stdout)
            print(proc.stderr)
            if proc.returncode == 0:
                print("[debug] patch applied successfully")
                return
            else:
                print(f"[warn] command failed with code {proc.returncode}")
        except FileNotFoundError:
            print(f"[error] command not found: {cmd[0]}")
        except subprocess.TimeoutExpired:
            print(f"[error] command timed out: {' '.join(cmd)}")
    sys.exit("❌ patch failed")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate patch from issue body")
    parser.add_argument("issue_body", nargs="?")
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Print program version and exit",
    )
    args = parser.parse_args()
    if args.version:
        print(__version__)
        sys.exit(0)

    if args.issue_body is None:
        parser.error("issue_body required")

    diff_text = llm(args.issue_body)
    print(diff_text)
    apply_patch(diff_text)


if __name__ == "__main__":
    main()
# CI-touch: no functional change
