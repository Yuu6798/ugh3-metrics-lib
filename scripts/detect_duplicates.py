#!/usr/bin/env python3
"""Utility to detect duplicate files, similar code, and dead code."""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicate_files(root: Path, ignore: Path) -> List[Tuple[str, List[str]]]:
    hashes: dict[str, List[str]] = {}
    for p in root.rglob("*"):
        if not p.is_file() or ignore in p.parents:
            continue
        digest = file_hash(p)
        hashes.setdefault(digest, []).append(p.as_posix())
    return [(h, files) for h, files in hashes.items() if len(files) > 1]


def extract_functions(py: Path) -> Iterable[Tuple[str, str]]:
    text = py.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            src = ast.get_source_segment(text, node) or ""
            yield f"{py}:{node.name}", src


def find_similar_functions(files: Iterable[Path]) -> List[Tuple[str, str, str]]:
    funcs = list(
        (name, src)
        for f in files
        for name, src in extract_functions(f)
    )
    sims: List[Tuple[str, str, str]] = []
    for i in range(len(funcs)):
        for j in range(i + 1, len(funcs)):
            (n1, s1), (n2, s2) = funcs[i], funcs[j]
            ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
            if ratio >= 0.9:
                sims.append((n1, n2, f"{ratio:.2f}"))
    return sims


def run_vulture(root: Path, outfile: Path) -> None:
    try:
        with outfile.open("w") as f:
            subprocess.run(["vulture", str(root)], stdout=f, check=False)
    except FileNotFoundError:
        outfile.write_text("vulture not installed\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("dup_report"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        dup = find_duplicate_files(Path("."), outdir)
        dup_csv = outdir / "duplicate_files.csv"
        with dup_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["hash", "files"])
            for h, files in dup:
                w.writerow([h, ";".join(files)])

        py_files = [p for p in Path(".").rglob("*.py") if outdir not in p.parents]
        sim = find_similar_functions(py_files)
        sim_csv = outdir / "code_similarity.csv"
        with sim_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["func1", "func2", "similarity"])
            w.writerows(sim)

        dead_code = outdir / "dead_code.txt"
        run_vulture(Path("."), dead_code)

        report = outdir / "report_dup.md"
        report.write_text(
            f"# Duplicate Report\n\n"
            f"- [duplicate_files.csv]({dup_csv.name})\n"
            f"- [code_similarity.csv]({sim_csv.name})\n"
            f"- [dead_code.txt]({dead_code.name})\n"
        )
    except Exception as e:  # pragma: no cover
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
