import argparse
import ast
import json
from pathlib import Path
import shutil


def parse_functions(py_path: Path):
    """Return a list of function metadata for a Python file."""
    with open(py_path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            args = [arg.arg for arg in node.args.args if arg.arg != "self"]
            doc = ast.get_docstring(node)
            summary = doc.splitlines()[0].strip() if doc else ""
            functions.append({
                "name": node.name,
                "args": args,
                "summary": summary,
            })
    return functions


def scan_modules(root: Path):
    modules = {}
    for py_file in sorted(root.glob("*.py")):
        if py_file.name == Path(__file__).name:
            continue
        funcs = parse_functions(py_file)
        if funcs:
            modules[py_file.stem] = funcs
    return modules


def load_config(root: Path):
    cfg_path = root / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def build_features(modules, limit=3):
    lines = []
    for mod, funcs in modules.items():
        for f in funcs[:limit]:
            desc = f["summary"]
            line = f"- **{f['name']}** from `{mod}.py`"
            if desc:
                line += f" - {desc}"
            lines.append(line)
        if len(funcs) > limit:
            lines.append(f"- ... ({len(funcs) - limit} more functions in `{mod}.py`)")
    return lines


def build_quickstart(modules, limit=3):
    imports = []
    for mod, funcs in modules.items():
        for f in funcs[:limit]:
            imports.append(f"from {mod} import {f['name']}")
    lines = ["```python"] + imports + ["```"]
    return lines


def replace_section(content_lines, heading, new_lines):
    start = None
    end = None
    for i, line in enumerate(content_lines):
        if line.strip() == heading:
            start = i
            break
    if start is None:
        return content_lines
    for j in range(start + 1, len(content_lines)):
        if content_lines[j].startswith("## "):
            end = j
            break
    if end is None:
        end = len(content_lines)
    return content_lines[:start + 1] + new_lines + [""] + content_lines[end:]


def update_readme(root: Path, modules):
    readme = root / "README.md"
    backup = root / "README.md.bak"
    if not backup.exists():
        shutil.copy(readme, backup)

    content = readme.read_text(encoding="utf-8").splitlines()

    features_lines = build_features(modules)
    quickstart_lines = build_quickstart(modules)

    content = replace_section(content, "## Features", features_lines)
    content = replace_section(content, "## Quick Start", quickstart_lines)

    readme.write_text("\n".join(content) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Auto-update README")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root")
    args = parser.parse_args()

    modules = scan_modules(args.root)
    update_readme(args.root, modules)


if __name__ == "__main__":
    main()
