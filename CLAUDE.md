# CLAUDE.md — AI Assistant Guide for ugh3-metrics-lib

This file provides context for AI assistants working in this repository. Read it fully before making changes.

---

## Project Overview

**Package name**: `por-deltae-lib` (v0.1.2)
**Purpose**: Reference OSS implementation of three NLP evaluation metrics — PoR (Point of Resonance), ΔE (Delta E), and grv (Gravity of Lexicon) — used in the UGHer conversational AI evaluation framework.

The library ships a clean v4 API in `ugh3_metrics/`, while `core/` and `facade/` contain legacy and integration code that must remain stable.

---

## Repository Layout

```
ugh3-metrics-lib/
├── ugh3_metrics/          # Primary library (v4 API — main focus)
│   ├── metrics/           # PorV4, DeltaE4, GrvV4, SciV4 classes
│   ├── utils/             # text_proc.py, math_ops.py
│   └── models/            # EmbedderProtocol (duck-typed interface)
├── core/                  # Legacy metric implementations (keep stable)
├── facade/                # CLI + integration layer (collector.py, trigger.py)
├── secl/                  # State Evaluation Conversation Logic (qa_cycle.py)
├── tests/                 # 50+ pytest test files
├── scripts/               # Utility scripts (dataset ops, AI codegen, analysis)
├── tools/                 # Reporting and audit utilities
├── examples/              # Notebooks and demo scripts
├── config/                # YAML weight configs (grv.yaml)
├── data/                  # CSV datasets and stopword lists
├── datasets/              # Generated Parquet/CSV datasets
├── stubs/                 # Type stubs for external packages
├── .github/               # CI/CD workflows and shared actions
├── pyproject.toml         # Project metadata and dependencies
├── mypy.ini               # Type checking configuration
├── .ruff.toml             # Linter configuration
├── Makefile               # Local dev shortcuts
└── config.json            # Runtime thresholds and parameters
```

---

## Core Metrics — API Reference

All metrics live in `ugh3_metrics/metrics/` and are exported from `ugh3_metrics/metrics/__init__.py`.

### PorV4 — Point of Resonance
```python
from ugh3_metrics.metrics import PorV4
por = PorV4(auto_load=True)
score = por.score("query text", "reference text")  # returns float in [0, 1]
```
- Uses `all-MiniLM-L6-v2` via sentence-transformers by default.
- Falls back to a simple token-count embedder (`SimpleEmbedder`) if the model is unavailable.
- Normalizes via sigmoid: `1 / (1 + exp(-(13.2 * sim - 10.8)))`.
- Fire threshold (from `config.json`): `POR_FIRE_THRESHOLD = 0.82`.

### DeltaE4 — Delta E
```python
from ugh3_metrics.metrics import DeltaE4
de = DeltaE4()
score = de.score("previous answer", "current answer")  # returns float in [0, 1]
```
- Computes `round(1.0 - cosine_similarity(embed(a), embed(b)), 3)`; output range is [0.0, 2.0] (not clipped — identical inputs return 0.0, opposite vectors return 2.0).
- Lazy-loads sentence-transformers on first call.
- Set `DELTAE4_FALLBACK=hash` environment variable to use hash-based approximation (no model needed).
- Raises `RuntimeError` if embedding fails outside fallback mode.

### GrvV4 — Gravity of Lexicon
```python
from ugh3_metrics.metrics import GrvV4
grv = GrvV4()
score = grv.score("text", "")  # second arg unused; returns float in [0, 1]
```
- Weighted composite: TF-IDF top-k (0.50) + entropy (0.30) + PMI co-occurrence (0.20).
- Sigmoid-normalized output.
- Weights are configurable via `config/grv.yaml` or `set_params(weights=[0.50, 0.30, 0.20])`.
- `set_params` expects an **iterable of floats** (list/tuple in `[tfidf, pmi, entropy]` order); passing a dict will raise a `ValueError` because dict keys are strings and cannot be cast to `float`.

### SciV4 — Science Score
```python
from ugh3_metrics.metrics import SciV4, sci, reset_state
```
- Stateful metric; call `reset_state()` between independent evaluations.

---

## Development Commands

### Install (development mode)
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest -q                          # all tests, quiet
pytest tests/test_metrics.py       # single file
pytest -k "deltae"                 # filter by name
```

### Type checking
```bash
mypy .
```

### Linting
```bash
ruff check .
```

### Build package
```bash
python -m build --sdist --wheel
```

### Dataset operations (Makefile)
```bash
make recalc IN=data/old.csv OUT=data/new.parquet
```

### CLI usage (facade)
```bash
# Collect metrics from CSV input
python facade/collector.py --input prompts.csv --output out.json --por --delta_e --grv

# Auto mode with OpenAI provider
python facade/collector.py --auto -n 10 --q-provider openai --ai-provider openai \
  --quiet --summary --output runs/metrics.csv --por --delta_e --grv
```

---

## CI/CD Workflows

All workflows are in `.github/workflows/`.

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push/PR to `main` | mypy → build → pytest → upload artifacts |
| `typecheck.yml` | push/PR | mypy only (Python 3.11) |
| `nightly-collect-build-dataset.yml` | cron 00:30 JST | Collect Q&A, compute metrics, generate datasets |
| `unified-ai-issue-to-pr.yml` | issue events | AI-powered issue → code → PR automation |
| `secret-smoke.yml` | push | Security validation for secrets |

CI runs on Python 3.12. The shared `.github/actions/setup-deps/action.yml` installs all dependencies.

**Required secrets for AI workflows**: `OPENAI_API_KEY`, `PAT_TOKEN`.

**Environment variables used in CI**:
- `PYTHONPATH`: set to workspace root (so top-level packages resolve)
- `ST_CACHE`: HuggingFace model cache path (`.cache/st` in CI)

---

## Code Conventions

### Naming
- Classes: `PascalCase` — `PorV4`, `DeltaE4`, `GrvV4`
- Functions/methods: `snake_case` — `calc_por_v4`, `tfidf_topk`
- Constants: `UPPER_CASE` — `POR_FIRE_THRESHOLD`, `ADOPT_TH`
- Private attributes: `_underscore_prefix` — `_embedder`, `_safe_encode`

### Type hints
- Core modules (`ugh3_metrics/`) use `from __future__ import annotations`.
- `mypy` is strict for `ugh3_metrics_lib.core.*`, `facade.trigger`, `facade.collector`, `secl.qa_cycle`.
- Use `# type: ignore[<code>]` with a comment explaining why when suppressing mypy errors.
- Protocol classes (e.g. `EmbedderProtocol`) are preferred over concrete base classes for embedder typing.

### Error handling
- Raise `RuntimeError` for embedding or model load failures.
- Use environment variables for toggling fallback behavior (e.g., `DELTAE4_FALLBACK=hash`).
- Lazy-load heavy dependencies (sentence-transformers) inside methods, not at module import time.

### Testing
- Use the `DummyEmbedder` fixture from `conftest.py` to avoid loading real models in unit tests.
- Parametrize with `@pytest.mark.parametrize` and shared `metric_cls` fixture.
- Add `# type: ignore[misc,untyped-decorator]` on pytest decorator lines that mypy cannot type-check.
- Smoke tests (just checking it runs without error) go in `test_*_smoke.py` files.

### Linting (ruff)
- Line length: **120 characters**.
- Target Python: 3.9.
- `F401` (unused imports) is ignored globally.
- Jupyter notebooks in `examples/` are excluded from linting.

---

## Key Configuration Files

### `config.json` — Runtime thresholds
```json
{
  "POR_FIRE_THRESHOLD": 0.82,
  "DELTA_E_HIGH": 0.65,
  "ANOMALY_POR_THRESHOLD": 0.95,
  "ANOMALY_DELTA_E_THRESHOLD": 0.85,
  "DUPLICATE_THRESHOLD": 0.9
}
```

### `config/grv.yaml` — GRV weights
```yaml
tfidf: 0.50
entropy: 0.30
cooccurrence: 0.20
```
Keys are at the **top level** (no nesting). `load_grv_weights()` reads `data.get("tfidf", ...)` directly; a nested `weights:` key would be silently ignored and defaults used.

### `facade/collector.py` — Integration constants
- `W_POR = 0.4`, `W_DE = 0.4`, `W_GRV = 0.2` — composite score weights
- `ADOPT_TH = 0.45` — adoption threshold

---

## Module Dependency Map

```
ugh3_metrics/metrics/   ← primary public API
    ↓ uses
ugh3_metrics/utils/     ← text_proc.py (tokenize, tfidf, entropy, pmi)
                           math_ops.py  (cosine_similarity)
ugh3_metrics/models/    ← EmbedderProtocol (duck-typed)

facade/                 ← calls ugh3_metrics/ + core/ (legacy compat)
    ↓ uses
core/                   ← legacy metric functions (keep stable, don't refactor)

secl/                   ← standalone state evaluation, used by facade/
```

---

## Important Patterns

1. **Lazy loading**: Import `sentence_transformers` only inside `_load_embedder()` or similar methods, never at module top-level. This keeps import times fast and allows fallback modes.

2. **Fallback modes**: Both PorV4 and DeltaE4 have fallback implementations (token-count and hash-based respectively). Tests should cover both paths.

3. **Sigmoid normalization**: All v4 metrics output values in [0, 1] via sigmoid. Do not clip raw cosine similarity values directly.

4. **Backwards compatibility**: `core/` and the `calc_*` function aliases in `ugh3_metrics/metrics/__init__.py` exist for legacy callers. Do not remove them.

5. **PYTHONPATH**: The repo root must be in `sys.path`. Tests set this via `conftest.py`; scripts assume it via `PYTHONPATH` env var.

---

## What to Avoid

- Do not add new top-level packages without updating `[tool.setuptools.packages.find]` in `pyproject.toml`.
- Do not import `sentence_transformers` or `torch` at module top-level — use lazy loading.
- Do not modify `core/` behavior; it is a legacy compatibility layer.
- Do not change scoring thresholds in `config.json` without updating corresponding tests.
- Do not skip `# type: ignore` comments on pytest decorators — mypy will error on untyped decorators.
- Do not use `torch` on macOS (excluded in `pyproject.toml` via platform marker).
