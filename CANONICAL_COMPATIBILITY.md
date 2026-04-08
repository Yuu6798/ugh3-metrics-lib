# Canonical Compatibility

This repository preserves PoR/ΔE/grv as theoretical concepts.

For production-compatible scoring and verdict behavior, the canonical
operational contract is defined in:

- `Yuu6798/ugh-audit-core/docs/canonical_metrics_contract.md`

Rules for this repository:

- Non-operational formulas must be treated as `research_variant`.
- Do not expose incompatible formulas as plain `delta_e` without a mode tag.
- Emit explicit method metadata when needed:
  - `metric_mode`
  - `delta_e_method`
  - `por_method`
  - `grv_method`
