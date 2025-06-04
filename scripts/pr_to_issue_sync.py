#!/usr/bin/env python3
"""Synchronize pull request state with its referenced GitHub issue.

This script is designed to run in a GitHub Actions environment. It expects the
following environment variables:
- ``GITHUB_TOKEN``: Personal access token for API authentication.
- ``GITHUB_REPOSITORY``: Repository in ``owner/repo`` format.
- ``PR_MERGED``: ``true`` if the pull request was merged, otherwise ``false``.
- ``PR_NUMBER``: The pull request number.
- ``ISSUE_NUMBER``: The raw body of the pull request which should contain an
  issue reference (e.g. ``Fixes #123``).

If the issue number cannot be determined, the script exits gracefully.
The script will comment on the referenced issue and close it if the PR was
merged. It also updates the progress tracker status for task index 7.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Optional, Dict, Any, cast


def extract_issue_number(text: Optional[str]) -> Optional[int]:
    """Extract an issue number from text like "Fixes #123"."""
    if not text:
        return None
    # direct number
    if text.strip().isdigit():
        return int(text.strip())
    match = re.search(r"#(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def github_request(url: str, method: str = "POST", data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set")
        return None

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
        "User-Agent": "pr-to-issue-sync",
    }

    encoded = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=encoded, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            return cast(Dict[str, Any], json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        try:
            print(e.read().decode())
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Request failed: {exc}")
    return None


def main() -> None:
    repo = os.getenv("GITHUB_REPOSITORY")
    pr_number = os.getenv("PR_NUMBER")
    pr_merged = os.getenv("PR_MERGED", "false").lower() == "true"
    issue_raw = os.getenv("ISSUE_NUMBER")

    issue_number = extract_issue_number(issue_raw)
    if not repo or not pr_number:
        print("Repository or PR number not provided")
        sys.exit(1)

    if issue_number is None:
        print("Issue number could not be determined. Nothing to do.")
        return

    base_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    comment_body = (
        f"PR #{pr_number} merged. Closing linked issue." if pr_merged else
        f"PR #{pr_number} closed without merge. Issue remains open."
    )

    # Post comment to issue
    github_request(base_url + "/comments", "POST", {"body": comment_body})

    # Close issue if PR merged
    if pr_merged:
        github_request(base_url, "PATCH", {"state": "closed"})

    # Update progress tracker (task index 7)
    try:
        subprocess.run([
            sys.executable,
            "scripts/progress_tracker.py",
            str(issue_number),
            "complete",
            "7",
            str(pr_merged).lower(),
        ], check=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Progress tracker update failed: {exc}")


if __name__ == "__main__":
    main()
