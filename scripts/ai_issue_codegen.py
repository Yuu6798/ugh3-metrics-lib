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
import sys
from pathlib import Path
from typing import Any, Union, cast, Dict, List, Optional, Tuple

print(f"[DEBUG] Running script: {__file__}")
print(f"[DEBUG] Script last modified: {os.path.getmtime(__file__)}")

GEN_DIR = Path("ai_generated")


DEFAULT_DIFF = """diff --git a/README.md b/README.md
index 1234567..abcdefg 100644
--- a/README.md
+++ b/README.md
@@ -10,3 +10,6 @@
 existing line
 existing line

+## New Section
+New content here
"""


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


def parse_unified_diff(diff_text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse unified diff into file operations (Claude Code style)"""
    operations: Dict[str, List[Dict[str, Any]]] = {}
    current_file: Optional[str] = None
    current_hunks: List[Dict[str, Any]] = []

    lines = diff_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith("--- a/"):
            i += 1
            continue
        elif line.startswith("+++ b/"):
            current_file = line[6:]
            operations[current_file] = []
            i += 1
            continue
        elif line.startswith("@@"):
            hunk_info = line.split()[1:3]
            old_info = hunk_info[0][1:].split(',')
            new_info = hunk_info[1][1:].split(',')

            old_start = int(old_info[0])
            new_start = int(new_info[0])

            i += 1
            hunk_lines = []
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("---"):
                if lines[i].startswith(" ") or lines[i].startswith("+") or lines[i].startswith("-"):
                    hunk_lines.append(lines[i])
                i += 1

            if current_file:
                operations[current_file].append({
                    'old_start': old_start,
                    'new_start': new_start,
                    'lines': hunk_lines
                })
            continue

        i += 1

    return operations

def apply_file_operations(file_path: str, operations: List[Dict[str, Any]]) -> None:
    """Apply operations directly to file (Claude Code style)"""
    from pathlib import Path

    file_obj = Path(file_path)

    if not file_obj.exists():
        print(f"[debug] creating new file: {file_path}")
        content_lines: List[str] = []
        for op in operations:
            for line in op['lines']:
                if line.startswith('+'):
                    content_lines.append(line[1:])
        file_obj.write_text('\n'.join(content_lines), encoding='utf-8')
        return

    try:
        original_content: str = file_obj.read_text(encoding='utf-8')
        lines: List[str] = original_content.splitlines()
    except Exception as e:
        print(f"[error] could not read {file_path}: {e}")
        raise

    for op in reversed(operations):
        old_start: int = op['old_start'] - 1
        remove_lines: List[str] = []
        add_lines: List[str] = []
        for line in op['lines']:
            if line.startswith('-'):
                remove_lines.append(line[1:])
            elif line.startswith('+'):
                add_lines.append(line[1:])
        for remove_line in remove_lines:
            try:
                line_index: Optional[int] = None
                for idx in range(old_start, min(old_start + 10, len(lines))):
                    if idx < len(lines) and lines[idx].strip() == remove_line.strip():
                        line_index = idx
                        break
                if line_index is not None:
                    lines.pop(line_index)
            except Exception:
                pass
        insert_point: int = min(old_start, len(lines))
        for add_line in add_lines:
            lines.insert(insert_point, add_line)
            insert_point += 1

    try:
        file_obj.write_text('\n'.join(lines), encoding='utf-8')
        print(f"[debug] successfully wrote {len(lines)} lines to {file_path}")
    except Exception as e:
        print(f"[error] could not write {file_path}: {e}")
        raise
def main() -> None:
    # Force verification that subprocess is not imported
    import sys
    print(f"[DEBUG] File: {__file__}")
    print(f"[DEBUG] Modules loaded: {[m for m in sys.modules.keys() if 'subprocess' in m]}")
    print(f"[DEBUG] Python path: {sys.path[0]}")

    print("[DEBUG] Starting ai_issue_codegen.py")
    print("[DEBUG] Python version:", sys.version)
    print("[DEBUG] Current working directory:", os.getcwd())
    print("[DEBUG] Script arguments:", sys.argv)

    if 'subprocess' in sys.modules:
        print("[WARNING] subprocess module is loaded!")
        print("[WARNING] This should not happen in Claude Code method")
    else:
        print("[DEBUG] subprocess module not loaded - good!")

    parser = argparse.ArgumentParser(description="Generate patch from issue body")
    parser.add_argument(
        "issue_body",
        nargs="?",
        help="The issue body content (optional)",
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-V", "--version", action="store_true", help="Print program version and exit")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    if args.test:
        diff_text = """diff --git a/README.md b/README.md
index 1234567..abcdefg 100644
--- a/README.md
+++ b/README.md
@@ -10,3 +10,6 @@
 existing line
 existing line
 
+## New Section
+New content here
+"""
        print(f"[debug] test diff:")
        print(diff_text)
        print("[debug] Using Claude Code method: direct file editing")
        try:
            file_operations = parse_unified_diff(diff_text)
            print(f"[debug] parsed {len(file_operations)} file operations")
            for file_path, operations in file_operations.items():
                try:
                    apply_file_operations(file_path, operations)
                    print(f"[debug] applied changes to: {file_path}")
                except Exception as e:
                    print(f"[error] failed to apply changes to {file_path}: {e}")
                    sys.exit("❌ patch failed")
            print("[success] All files updated successfully!")
            print("[info] Changes applied using Claude Code method (no subprocess)")
            return
        except Exception as e:
            print(f"[error] Claude Code method failed: {e}")
            sys.exit("❌ patch failed")

    if not args.issue_body and not args.test:
        args.issue_body = DEFAULT_DIFF
        print(f"[INFO] Using default issue body: {args.issue_body}")

    try:
        file_operations = parse_unified_diff(args.issue_body)
        print(f"[debug] parsed {len(file_operations)} file operations")
        for file_path, operations in file_operations.items():
            try:
                apply_file_operations(file_path, operations)
                print(f"[debug] applied changes to: {file_path}")
            except Exception as e:
                print(f"[error] failed to apply changes to {file_path}: {e}")
                sys.exit("❌ patch failed")
        print("[success] All files updated successfully!")
        return
    except Exception as e:
        print(f"[error] failed to process patch: {e}")
        sys.exit("❌ patch failed")


if __name__ == "__main__":
    main()
# CI-touch: no functional change
# Force rebuild Mon Jun  2 17:22:47 UTC 2025
