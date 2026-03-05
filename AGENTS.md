# AGENTS.md for ughp-guard

## Project Overview

`ughp-guard` is an anti-deception audit tool for LLM outputs.
It is **not** a generic quality scorer.
It detects operationally suspicious response patterns such as unsupported claims, completion bluffing, goal substitution, scope drift, and concealment.

UGHP protocol terms:

- **PoR** (Point of Resolution): the user's real requested outcome, including success criteria, required evidence, and falsification conditions.
- **grv** (gravity): the stable task objective that must not be silently substituted.
- **ΔE** (delta-E): measurable deviation, including scope drift, unsupported claims, hidden assumptions, unverifiable completion, and evidence mismatch.
- **Mesh**: cross-view consistency checks that expose coherent-looking but evasive answers.

## Tech Stack

- Python 3.12
- Pydantic v2 for schemas
- Typer for CLI
- pytest for tests
- No mandatory external web dependencies
- Provider-agnostic with pluggable adapter interface

## Project Structure

```text
ughp-guard/
├── pyproject.toml
├── README.md
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
```

## Commands

- Install: `pip install -e ".[dev]"`
- Test: `pytest tests/ -v`
- CLI: `ughp-guard audit --task task.txt --response response.txt`

## Coding Conventions

- Use Pydantic v2 `BaseModel` with strict typing for all data models.
- Use `Enum` or `Literal` for categorical fields, never bare strings.
- Every auditor exposes one public function: `audit_*(contract, claims, evidence) -> list[Finding]`.
- Auditors never raise on malformed input; return findings instead.
- All risk scores are `float` in `[0.0, 1.0]`.
- All outputs separate: observed facts / inferred risks / unverified assumptions.
- Add docstrings on all public functions and classes.
- Prefer readability over abstraction.

## Testing Standards

- Use pytest with JSON fixtures in `tests/fixtures/`.
- Each fixture includes at minimum: `task`, `response`, and `expected_findings`.
- Tests verify specific findings, not just “no crash”.
- End-to-end tests produce both JSON and Markdown reports.
- Use `pytest.approx` for float comparisons.

## Design Principles

- Detect operationally suspicious patterns, not intent or consciousness.
- Conservative by default: prefer false negatives over false positives.
- Reject unsupported completion claims using evidence.
- Mock adapter uses heuristics only — no LLM calls in MVP.
- Audit reports must not commit the sins they audit.

## Anti-Patterns

- Do **not** treat confident wording as evidence.
- Do **not** conflate “sounds good” with “verified”.
- Do **not** over-abstract.
- Do **not** add external API dependencies in MVP.

## Commit Prefix Convention

Use commit prefixes such as:

- `models:`
- `auditors/completion:`
- `cli:`
- `tests:`
- etc.
