#!/usr/bin/env python3
"""Process review comments and apply AI-generated patches."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

PATTERNS = {
    "fix_request": r"(?i)(fix|correct|bug|error|issue)",
    "add_request": r"(?i)(add|implement|include|create)",
    "modify_request": r"(?i)(change|modify|update|refactor)",
    "test_request": r"(?i)(test|coverage|unit test|spec)",
    "doc_request": r"(?i)(document|comment|explain|readme)",
}


def github_api_request(url: str, token: str) -> Optional[Dict[str, Any]]:
    """Send a GET request to the GitHub API and return JSON data."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ai-review-response",
    }
    try:
        req = subprocess.run(
            ["curl", "-s", "-H", f"Authorization: token {token}", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(req.stdout)
    except Exception as exc:  # pragma: no cover - debug safety
        print(f"API request failed: {exc}")
    return None


@dataclass
class ReviewRequest:
    body: str
    path: Optional[str] = None
    line: Optional[int] = None


@dataclass
class ChangeInstruction:
    request_type: str
    file_path: str
    line: Optional[int]
    content: str


def parse_review_comments(pr_number: int) -> List[ReviewRequest]:
    """Fetch review comments from the GitHub API."""
    owner = os.getenv("REPO_OWNER")
    repo = os.getenv("REPO_NAME")
    token = os.getenv("GITHUB_TOKEN")
    if not (owner and repo and token):
        print("Missing environment for GitHub API access")
        return []
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    data = github_api_request(url, token)
    requests: List[ReviewRequest] = []
    if isinstance(data, list):
        for item in data:
            body = item.get("body", "")
            path = item.get("path")
            line = item.get("line")
            requests.append(ReviewRequest(body=body, path=path, line=line))
    return requests


def analyze_change_request(comment: str, file_path: str | None = None) -> ChangeInstruction:
    """Analyze a comment and generate a change instruction."""
    request_type = "modify"
    for key, pattern in PATTERNS.items():
        if re.search(pattern, comment):
            request_type = key
            break
    return ChangeInstruction(request_type, file_path or "", None, comment)


def apply_code_modifications(instructions: List[ChangeInstruction]) -> bool:
    """Apply modifications to files based on instructions."""
    success = True
    for inst in instructions:
        if not inst.file_path:
            continue
        try:
            with open(inst.file_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            # naive example: append comment at target line or end
            insert_at = inst.line - 1 if inst.line else len(lines)
            lines.insert(insert_at, f"# TODO: {inst.content}\n")
            with open(inst.file_path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Modification failed for {inst.file_path}: {exc}")
            success = False
    return success


def commit_and_push_changes(branch: str, message: str) -> bool:
    """Commit and push changes back to the repository."""
    try:
        subprocess.run(["git", "checkout", "-b", branch], check=True)
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", message], check=True)
        subprocess.run(["git", "push", "-u", "origin", branch], check=True)
        return True
    except subprocess.CalledProcessError as exc:  # pragma: no cover - network
        print(f"Git error: {exc}")
    return False


def main() -> None:  # pragma: no cover - CLI utility
    pr_number = os.getenv("PR_NUMBER")
    if not pr_number or not pr_number.isdigit():
        print("PR_NUMBER environment variable missing")
        sys.exit(1)
    requests = parse_review_comments(int(pr_number))
    if not requests and os.getenv("COMMENT_BODY"):
        requests = [ReviewRequest(body=os.getenv("COMMENT_BODY", ""))]
    instructions = [analyze_change_request(r.body, r.path) for r in requests]
    if apply_code_modifications(instructions):
        branch = f"ai-review-{pr_number}"
        commit_and_push_changes(branch, "AI review updates")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
