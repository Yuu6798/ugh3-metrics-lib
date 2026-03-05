#!/usr/bin/env bash
set -euo pipefail

# Auto-fix loop for Codex-driven workflows.
# - Re-runs pytest up to MAX_TRIES.
# - Parses traceback frames and chooses actionable repo-local location.
# - Applies a minimal, line-scoped fix via FIX_CMD if provided.
# - Uses temp logs outside repo root and cleans them on exit.
# - Aborts without push/PR when tests remain failing.

MAX_TRIES=${MAX_TRIES:-5}
BRANCH_PREFIX=${BRANCH_PREFIX:-codex-auto-fix}
COMMIT_MSG_PREFIX=${COMMIT_MSG_PREFIX:-"AUTO: fix until tests pass"}
FIX_CMD=${FIX_CMD:-}
ALLOW_TEST_FIXES=${ALLOW_TEST_FIXES:-0}

REPO_ROOT=$(git rev-parse --show-toplevel)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

run_pytest() {
    local log_file
    log_file="$1"
    set +e
    python -m pytest -q 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

extract_failure_location() {
    local log_file
    log_file="$1"

    python - "$REPO_ROOT" "$ALLOW_TEST_FIXES" "$log_file" <<'PY'
import os
import re
import sys

repo_root = os.path.realpath(sys.argv[1])
allow_test_fixes = sys.argv[2] == "1"
log_file = sys.argv[3]

blocked_parts = ("/.venv/", "/venv/", "/site-packages/", "/dist-packages/", "/.tox/", "/.nox/")
frame_re = re.compile(r'\s*File "([^"]+)", line (\d+)')

frames = []
with open(log_file, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        m = frame_re.search(line)
        if not m:
            continue
        raw_path, line_no = m.group(1), m.group(2)
        abs_path = os.path.realpath(raw_path if os.path.isabs(raw_path) else os.path.join(repo_root, raw_path))

        if not abs_path.startswith(repo_root + os.sep):
            continue
        if any(part in abs_path for part in blocked_parts):
            continue

        rel_path = os.path.relpath(abs_path, repo_root)
        rel_norm = rel_path.replace("\\", "/")
        if rel_norm.startswith(".pytest_cache/"):
            continue

        is_test = rel_norm.startswith("tests/") or "/tests/" in f"/{rel_norm}"
        frames.append((rel_path, line_no, is_test))

non_test = [f for f in frames if not f[2]]
test_frames = [f for f in frames if f[2]]

chosen = non_test[-1] if non_test else (test_frames[-1] if allow_test_fixes and test_frames else None)
if chosen:
    print(f"{chosen[0]}:{chosen[1]}")
PY
}

apply_fix() {
    local file line normalized
    file="$1"
    line="$2"

    normalized=$(python - "$REPO_ROOT" "$file" <<'PY'
import os
import sys
repo_root = os.path.realpath(sys.argv[1])
candidate = os.path.realpath(sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(repo_root, sys.argv[2]))
if candidate.startswith(repo_root + os.sep):
    print(candidate)
PY
)

    if [[ -z "$normalized" ]]; then
        echo "🛑 Refusing to edit path outside repo root: ${file}" >&2
        exit 1
    fi
    if [[ ! -e "$normalized" ]]; then
        echo "🛑 Target does not exist: ${normalized}" >&2
        exit 1
    fi
    if [[ ! -f "$normalized" ]]; then
        echo "🛑 Target is not a regular file: ${normalized}" >&2
        exit 1
    fi
    if [[ ! -w "$normalized" ]]; then
        echo "🛑 Target is not writable: ${normalized}" >&2
        exit 1
    fi

    if [[ -n "$FIX_CMD" ]]; then
        echo "🛠️  invoking FIX_CMD on ${normalized}:${line}"
        "$FIX_CMD" "$normalized" "$line"
        return
    fi

    echo "[INFO] FIX_CMD is not configured."
    echo "[INFO] Target for localized edit: ${normalized}:${line}"
    echo "[INFO] Set FIX_CMD to an executable that edits only the indicated range."
    exit 1
}

for n in $(seq 1 "$MAX_TRIES"); do
    echo "🔄 pytest run #$n"
    log_file="$TMP_DIR/pytest-run-${n}.log"

    if run_pytest "$log_file"; then
        echo "✅ tests passed on attempt #$n"
        break
    fi

    echo "❌ tests failed – attempting auto-fix #$n"
    fail_info=$(extract_failure_location "$log_file")
    file=${fail_info%:*}
    line=${fail_info##*:}

    if [[ -z "$fail_info" || -z "$file" || -z "$line" || "$file" == "$line" ]]; then
        echo "🛑 No eligible repo-local traceback frame found in ${log_file}; aborting." >&2
        exit 1
    fi

    apply_fix "$file" "$line"
done

if ! python -m pytest -q; then
    echo "🛑 tests still failing after ${MAX_TRIES} attempts; abort." >&2
    exit 1
fi

branch="${BRANCH_PREFIX}-$(date +%Y%m%d%H%M%S)"
git switch -c "$branch"
git add -A

git commit -m "${COMMIT_MSG_PREFIX} (all tests pass)"
git push -u origin "$branch"
gh pr create --fill
