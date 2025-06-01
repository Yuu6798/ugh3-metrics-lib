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
                    "You are an experienced Python engineer tasked with generating diffs."
                    " Read the following GitHub Issue and return only unified diff text."
                ),
            },
            {
                "role": "user",
                "content": issue_body,
            },
        ],
        temperature=0.0,
    )
    content: str
    if hasattr(response, "choices"):
        content = response.choices[0].message.content  # type: ignore[attr-defined]
    else:
        choice = cast(dict[str, Any], response["choices"][0])  # type: ignore[index]
        message = cast(dict[str, Any], choice["message"])
        content = message["content"]
    return str(content)


def apply_patch(diff_text: str) -> None:
    # --- Strip code-fence lines (` ``` `) that break `patch`
    diff_text = "\n".join(l for l in diff_text.splitlines() if not l.startswith("```")) + "\n"
    # ------------------------------------------------------
    GEN_DIR.mkdir(exist_ok=True)
    patch_path = GEN_DIR / "auto.patch"
    patch_path.write_text(diff_text)
    try:
        subprocess.run(
            ["patch", "-p1", "-d", str(GEN_DIR)],
            input=diff_text.encode(),
            check=True,
        )
    except subprocess.CalledProcessError:
        sys.exit("\u274c patch failed")


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
