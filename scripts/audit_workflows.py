#!/usr/bin/env python3
"""Audit GitHub workflow files and detect duplicates."""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_workflow(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    name = data.get("name", path.stem)
    on = data.get("on", {})
    if isinstance(on, dict):
        triggers = list(on.keys())
    elif isinstance(on, list):
        triggers = [str(t) for t in on]
    else:
        triggers = [str(on)]
    jobs = data.get("jobs", {}) or {}
    job_names = list(jobs.keys())
    steps: List[str] = []
    for job in jobs.values():
        for step in job.get("steps", [])[:2]:
            if isinstance(step, dict):
                name = step.get("name")
                if name:
                    steps.append(str(name))
    tokens = job_names + steps
    return {
        "path": path,
        "name": name,
        "triggers": triggers,
        "jobs": jobs,
        "job_names": job_names,
        "tokens": tokens,
    }


def diff_lines(a: Path, b: Path) -> List[str]:
    a_lines = a.read_text().splitlines()
    b_lines = b.read_text().splitlines()
    diff = difflib.unified_diff(a_lines, b_lines, fromfile=a.name, tofile=b.name, lineterm="")
    lines = [
        d
        for d in diff
        if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))
    ]
    return list(lines)


def group_similar(wfs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    used: set[str] = set()
    for i, w1 in enumerate(wfs):
        if str(w1["path"]) in used:
            continue
        group = [w1]
        t1 = " ".join(sorted([str(t) for t in w1["tokens"]]))
        for w2 in wfs[i + 1 :]:
            t2 = " ".join(sorted([str(t) for t in w2["tokens"]]))
            ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
            if ratio >= 0.8:
                group.append(w2)
                used.add(str(w2["path"]))
        if len(group) > 1:
            for g in group:
                used.add(str(g["path"]))
            groups.append(group)
    return groups


def generate_report(wfs: List[Dict[str, Any]], groups: List[List[Dict[str, Any]]]) -> str:
    lines = ["# Workflow Audit", "", "## Summary", "| File | Workflow | Triggers | Jobs |", "| --- | --- | --- | --- |"]
    for wf in wfs:
        triggers = ", ".join(wf["triggers"])
        lines.append(
            f"| {Path(str(wf['path'])).name} | {wf['name']} | {triggers} | {len(wf['job_names'])} |"
        )
    for idx, group in enumerate(groups, 1):
        lines.extend(["", f"## Duplicate group {idx}", "| File | Workflow | Jobs |", "| --- | --- | --- |"])
        for wf in group:
            lines.append(
                f"| {Path(str(wf['path'])).name} | {wf['name']} | {', '.join(wf['job_names'])} |"
            )
        base = group[0]
        for other in group[1:]:
            diff = "\n".join(diff_lines(base["path"], other["path"]))
            lines.extend([
                "",
                f"### Diff with {Path(str(other['path'])).name}",
                "```diff",
                diff,
                "```",
            ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("report/workflow_audit.md"))
    args = parser.parse_args()

    wf_dir = Path(".github/workflows")
    wfs = [load_workflow(p) for p in list(wf_dir.rglob("*.yml"))]
    groups = group_similar(wfs)
    report = generate_report(wfs, groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
