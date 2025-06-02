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
from typing import Any, Union, cast, Dict, List, Optional, Tuple

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


def apply_patch(diff_text: Union[str, bytes]) -> None:
    """Apply the unified diff text using multiple strategies with verbose output."""
    # Type validation
    if not isinstance(diff_text, (str, bytes)):
        raise TypeError(f"diff_text must be str or bytes, got {type(diff_text)}")

    print("=== GPT-4 GENERATED DIFF DEBUG ===")
    print("Raw input:")
    print(repr(diff_text))

    # Convert to string for processing
    if isinstance(diff_text, bytes):
        text_content = diff_text.decode("utf-8")
    else:
        text_content = diff_text

    print("\nFormatted input:")
    print(text_content)
    print("=== END DEBUG ===")

    # --- normalize incoming diff ---------------------------------
    cleaned = []
    for raw in text_content.splitlines():
        line = raw.lstrip("| ")  # remove leading quote mark
        if line.startswith("```"):
            continue
        cleaned.append(line)
    # add missing  diff --git  header
    if cleaned and cleaned[0].startswith("--- a/"):
        first = cleaned[0][4:]
        second = first.replace("a/", "b/", 1)
        cleaned.insert(0, f"diff --git {first} {second}")
    processed_text = "\n".join(cleaned) + "\n"
    # --------------------------------------------------------------
    GEN_DIR.mkdir(exist_ok=True)
    patch_path = GEN_DIR / "auto.patch"
    patch_path.write_text(processed_text, encoding="utf-8")

    print(f"[debug] cwd: {os.getcwd()}")
    print(f"[debug] patch written to: {patch_path}")
    print(f"[debug] processed patch:\n{processed_text}")

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



    # Claude Code style: Direct file editing (complete subprocess avoidance)
    print("[debug] Using Claude Code method: direct file editing")
    try:
        # Parse unified diff into file operations
        file_operations = parse_unified_diff(processed_text)
        print(f"[debug] parsed {len(file_operations)} file operations")
    
        # Apply changes directly to files
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
