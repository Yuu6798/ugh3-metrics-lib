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

__version__ = "0.0.2"

import argparse
import os
import sys
from pathlib import Path
from typing import Any, cast, Dict, List, Optional
import re

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

DEFAULT_MODEL = "gpt-4o"

REQUIRED_FILES = ["scripts/ai_issue_codegen.py"]


def ensure_directory_exists(file_path: str) -> None:
    """Create parent directory for *file_path* if it does not exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"[INFO] Created directory: {directory}")
        except Exception as e:
            print(f"[ERROR] Failed to create directory {directory}: {e}")
            raise


def detect_file_type(issue_body: str, file_path: str | None = None) -> str:
    """Guess file type from issue body or file path."""
    if file_path:
        if file_path.endswith((".yml", ".yaml")):
            return "yaml"
        if file_path.endswith(".json"):
            return "json"
        if file_path.endswith(".py"):
            return "python"
        if file_path.endswith(".md"):
            return "markdown"

    body = issue_body.lower()
    if "workflow" in body:
        return "yaml"
    if "json" in body or "config" in body:
        return "json"
    if "python" in body:
        return "python"
    return "text"


def extract_code_from_response(response: str, file_type: str) -> str:
    """Extract code block matching *file_type* from LLM response."""
    patterns = {
        "yaml": r"```ya?ml\n(.*?)\n```",
        "json": r"```json\n(.*?)\n```",
        "python": r"```python\n(.*?)\n```",
    }
    regex = patterns.get(file_type)
    if regex:
        m = re.search(regex, response, re.DOTALL)
        if m:
            return m.group(1).strip()
    generic = re.search(r"```[^\n]*\n(.*?)\n```", response, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return response.strip()


def llm(issue_body: str, model: str = DEFAULT_MODEL) -> str:
    import openai

    api_key = os.getenv("OPENAI_API_KEY", "")
    actual_model: str = os.getenv("AI_MODEL", model)
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=actual_model,
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
                    "8. Output ONLY the diff, no explanations or code blocks\n"
                    "9. ONLY modify file(s) explicitly mentioned in the Issue body\n"
                    "10. If creating or updating a GitHub Actions workflow, add a step"
                    " to install dependencies with 'pip install -e .[dev]'"
                    " immediately after the Python setup step and include any"
                    " required environment variable secrets such as OPENAI_API_KEY\n"
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
            old_info = hunk_info[0][1:].split(",")
            new_info = hunk_info[1][1:].split(",")

            old_start = int(old_info[0])
            new_start = int(new_info[0])

            i += 1
            hunk_lines = []
            while i < len(lines) and not lines[i].startswith("@@") and not lines[i].startswith("---"):
                if lines[i].startswith(" ") or lines[i].startswith("+") or lines[i].startswith("-"):
                    hunk_lines.append(lines[i])
                i += 1

            if current_file:
                operations[current_file].append({"old_start": old_start, "new_start": new_start, "lines": hunk_lines})
            continue

        i += 1

    return operations


def apply_file_operations(file_path: str, operations: List[Dict[str, Any]]) -> None:
    """Apply operations directly to file (Claude Code style)"""
    from pathlib import Path

    ensure_directory_exists(file_path)
    file_obj = Path(file_path)

    if not file_obj.exists():
        print(f"[debug] creating new file: {file_path}")
        content_lines: List[str] = []
        for op in operations:
            for line in op["lines"]:
                if line.startswith("+"):
                    content_lines.append(line[1:])
        file_obj.write_text("\n".join(content_lines), encoding="utf-8")
        return

    try:
        original_content: str = file_obj.read_text(encoding="utf-8")
        lines: List[str] = original_content.splitlines()
    except Exception as e:
        print(f"[error] could not read {file_path}: {e}")
        raise

    for op in reversed(operations):
        old_start: int = op["old_start"] - 1
        remove_lines: List[str] = []
        add_lines: List[str] = []
        for line in op["lines"]:
            if line.startswith("-"):
                remove_lines.append(line[1:])
            elif line.startswith("+"):
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
        file_obj.write_text("\n".join(lines), encoding="utf-8")
        print(f"[debug] successfully wrote {len(lines)} lines to {file_path}")
    except Exception as e:
        print(f"[error] could not write {file_path}: {e}")
        raise


def main() -> None:
    """Fixed main function that properly uses LLM to generate diff from issue body"""

    print("[DEBUG] Starting ai_issue_codegen.py")
    print("[DEBUG] Python version:", sys.version)
    print("[DEBUG] Current working directory:", os.getcwd())
    print("[DEBUG] Script arguments:", sys.argv)

    parser = argparse.ArgumentParser(description="Generate patch from issue body")
    parser.add_argument(
        "issue_body_pos",
        nargs="?",
        help="The issue body content (optional positional)",
    )
    parser.add_argument(
        "--issue-body",
        dest="issue_body_opt",
        help="Issue body content",
    )
    parser.add_argument(
        "--file-path",
        help="Target file path for creation",
    )
    parser.add_argument(
        "--file-type",
        help="File type (yaml, json, py, md)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name (overrides AI_MODEL env var)",
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-V", "--version", action="store_true", help="Print program version and exit")
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    # Test mode with predefined diff
    if args.test:
        diff_text = DEFAULT_DIFF
        print("[debug] test mode: using predefined diff")
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
            print("Code generation completed")
            return
        except Exception as e:
            print(f"[error] Claude Code method failed: {e}")
            sys.exit("❌ patch failed")

    # Get issue body from arguments or environment
    issue_body = args.issue_body_opt or args.issue_body_pos
    if not issue_body:
        issue_body = os.environ.get("ISSUE_BODY", "")

    if not issue_body:
        print("[ERROR] No issue body provided")
        sys.exit(1)

    print(f"[DEBUG] Issue body length: {len(issue_body)}")
    print(f"[DEBUG] Issue body preview: {issue_body[:200]}...")

    # **THIS IS THE CRITICAL FIX: Actually use LLM to generate diff**
    try:
        print("[DEBUG] Calling LLM to generate diff from issue body...")

        # Use the model from args or environment
        model = args.model or os.getenv("AI_MODEL", DEFAULT_MODEL)
        print(f"[DEBUG] Using model: {model}")

        # **CALL THE LLM FUNCTION TO GENERATE DIFF**
        generated_output = llm(issue_body, model)

        print(f"[DEBUG] LLM response length: {len(generated_output)}")
        print("[DEBUG] LLM response preview:")
        print(generated_output[:500] + "..." if len(generated_output) > 500 else generated_output)

        if not generated_output or not generated_output.strip():
            print("[ERROR] LLM returned empty response")
            sys.exit(1)

        # If a direct file path is provided, write the response as code
        if args.file_path:
            file_type = args.file_type or detect_file_type(issue_body, args.file_path)
            print(f"[DEBUG] Detected file type: {file_type}")
            try:
                code_text = extract_code_from_response(generated_output, file_type)
                ensure_directory_exists(args.file_path)
                Path(args.file_path).write_text(code_text, encoding="utf-8")
                print(f"[success] Wrote file: {args.file_path}")
                return
            except Exception as e:
                print(f"[ERROR] Failed to write file {args.file_path}: {e}")
                sys.exit(1)

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        sys.exit(1)

    # Parse and apply the LLM-generated diff
    try:
        print("[DEBUG] Parsing generated diff...")
        file_operations = parse_unified_diff(generated_output)
        print(f"[debug] parsed {len(file_operations)} file operations")

        if not file_operations:
            print("[ERROR] No file operations found in generated diff")
            print("[DEBUG] Generated diff was:")
            print(generated_output)
            sys.exit(1)

        # Apply changes to each file
        for file_path, operations in file_operations.items():
            print(f"[DEBUG] Applying operations to: {file_path}")
            try:
                apply_file_operations(file_path, operations)
                print(f"[debug] applied changes to: {file_path}")
            except Exception as e:
                print(f"[error] failed to apply changes to {file_path}: {e}")
                sys.exit("❌ patch failed")

        print("[success] All files updated successfully!")
        print("Code generation completed")

    except Exception as e:
        print(f"[error] failed to process generated diff: {e}")
        print(f"[debug] Generated diff was: {generated_output}")
        sys.exit("❌ patch failed")


if __name__ == "__main__":
    main()
# CI-touch: no functional change
# Force rebuild Mon Jun  2 17:22:47 UTC 2025
