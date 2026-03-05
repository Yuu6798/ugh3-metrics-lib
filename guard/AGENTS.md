## Project Overview

`ughp-guard` is a subproject of `ugh3-metrics-lib`.
It is an anti-deception audit tool for LLM outputs — not a generic quality scorer.
It detects operationally suspicious response patterns: unsupported claims, completion bluffing, goal substitution, scope drift, and concealment.

It reuses the UGHP protocol already defined in the parent project:

- **PoR** (Point of Resolution): the user's real requested outcome, including success criteria, required evidence, and falsification conditions.
- **grv** (gravity): the stable task objective that must not be silently substituted.
- **ΔE** (delta-E): measurable deviation — scope drift, unsupported claims, hidden assumptions, unverifiable completion, evidence mismatch.
- **Mesh**: cross-view consistency checks that expose coherent-looking but evasive answers.

## Relationship to parent project

- `ugh3-metrics-lib` provides the core metric definitions (PoR, ΔE, grv) and embedding-based computation.
- `ughp-guard` applies those concepts as an operational audit protocol for LLM response verification.
- `ughp-guard` may import from `core/` or `ugh3_metrics/` where appropriate, but must also work standalone with its own mock adapter.

## Tech Stack

- Python 3.12
- Pydantic v2 for schemas
- Typer for CLI
- pytest for tests
- No mandatory external web dependencies
- Provider-agnostic with pluggable adapter interface

## Subproject Structure

```
guard/
├── AGENTS.md
├── src/ughp_guard/
│   ├── __init__.py
│   ├── models.py
│   ├── contract_parser.py
│   ├── claim_extractor.py
│   ├── auditors/
│   │   ├── __init__.py
│   │   ├── evidence.py
│   │   ├── completion.py
│   │   ├── scope.py
│   │   ├── goal_shift.py
│   │   ├── consistency.py
│   │   ├── concealment.py
│   │   └── mesh.py
│   ├── scoring.py
│   ├── recommendation.py
│   ├── report.py
│   ├── cli.py
│   └── adapters/
│       ├── __init__.py
│       ├── base.py
│       └── mock.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   ├── test_contract_parser.py
│   ├── test_claim_extractor.py
│   ├── test_completion_bluff.py
│   ├── test_goal_shift.py
│   ├── test_mesh.py
│   └── test_end_to_end.py
├── pyproject.toml
└── README.md
```

## Commands

All commands run from the `guard/` directory:

- Install: `cd guard && pip install -e ".[dev]"`
- Test: `cd guard && pytest tests/ -v`
- CLI: `cd guard && python -m ughp_guard.cli audit --task task.txt --response response.txt`

## Coding Conventions

- Pydantic v2 `BaseModel` with strict typing for all data models.
- `Enum` or `Literal` for categorical fields, never bare strings.
- Every auditor exposes one public function: `audit_*(contract, claims, evidence) -> list[Finding]`.
- Auditors never raise on malformed input; return findings instead.
- All risk scores are `float` in `[0.0, 1.0]`.
- All outputs separate: observed facts / inferred risks / unverified assumptions.
- Docstrings on all public functions and classes.
- Readability over abstraction.

## Testing Standards

- pytest with JSON fixtures in `tests/fixtures/`.
- Each fixture: `task`, `response`, `expected_disposition` at minimum.
- Tests verify specific findings, not just "no crash".
- End-to-end tests produce both JSON and Markdown reports.
- Use `pytest.approx` for float comparisons.

## Design Principles

- Detect operationally suspicious patterns, not intent or consciousness.
- Conservative: prefer false negatives over false positives.
- Reject unsupported completion claims based on evidence.
- Mock adapter uses heuristics only — no LLM calls in MVP.
- Audit reports must not commit the sins they audit.

## Anti-Patterns

- Do not treat confident wording as evidence.
- Do not conflate "sounds good" with "verified".
- Do not over-abstract.
- Do not add external API dependencies in MVP.
- Do not modify files outside `guard/` without explicit instruction.

## Commit Prefix Convention

`guard/models:`, `guard/auditors/completion:`, `guard/cli:`, `guard/tests:`, etc.
