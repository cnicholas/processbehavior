# Release Gate: 0.1.0

This checklist defines the quality bar for publishing `processbehavior` to PyPI.
It focuses on public API consistency, long-term maintainability, and user ergonomics.

## Gate Policy

1. `0.1.0` blockers: items 1-5 must pass.
2. Items 6-8 should pass, or have a tracked issue with a target milestone.
3. Items 9-10 may defer if called out in release notes with compatibility guidance.

## Project Decisions (Current)

- Item 6 (`Strict vs Convenience Data-Cleaning Mode`): downgraded to `CAN-DEFER`.
  Current action is to document behavior clearly; no strict-mode feature required for `0.1.0`.
- Item 7 (`Mutable Public Internals Exposure`): marked `PASS`.
  Existing `.copy()` protections are accepted as sufficient for `0.1.0`.

## Status Legend

- `MUST-FIX`: blocker for `0.1.0`.
- `SHOULD-FIX`: strongly recommended for `0.1.0`.
- `CAN-DEFER`: can ship with explicit documentation and follow-up.

## Checklist

### 1) Normalize Exception Model (`MUST-FIX`)

Goal:
- All public contract/input failures raise `ProcessBehaviorError` subclasses.

Acceptance Criteria:
- Invalid `execute(chart=...)` raises `ChartNotAvailableError` or `ValidationError`, not raw `ValueError`.
- Invalid `by` / `value` / `recentered` combinations raise typed custom exceptions.
- Catching `ProcessBehaviorError` reliably covers user-facing API failures.

### 2) Unify Stratified API Semantics (`MUST-FIX`)

Goal:
- `strata`, `focus`, and stratified helper methods share one canonical contract.

Acceptance Criteria:
- `result.strata` values are exactly valid inputs to `result.focus(...)`.
- `get_stratified_chart()` resolves exact strata keys (no ambiguous substring matching).
- `list_strata()` returns canonical values consistent with `result.strata`.
- Multi-factor strata round-trip correctly (tuple strata supported and documented).

### 3) Resolve Docs vs Runtime Mismatches (`MUST-FIX`)

Goal:
- Public docs/examples match actual runtime behavior.

Acceptance Criteria:
- API examples are executable in CI smoke/doctest checks.
- `plan` docs reflect current required keys (`factors`, `T`, `N`) or runtime is updated to match docs.
- Naming is consistent (`pb.cols` vs `pb.columns`, etc.).

### 4) Correct Public Type Hints and Docstrings for Strata (`MUST-FIX`)

Goal:
- Type signatures accurately reflect real strata types (`str`, tuple, etc.).

Acceptance Criteria:
- `AnalysisResult.strata` and `focus(...)` annotations match runtime values.
- Examples include multi-factor tuple strata usage.
- Static-checking examples pass.

### 5) Harden `ColumnAccessor` Edge Cases (`MUST-FIX`)

Goal:
- Accessor remains safe and predictable with messy/real-world schemas.

Acceptance Criteria:
- Empty column names do not crash sanitization.
- Sanitization collisions are detectable and non-lossy for user access.
- Reserved/accessor attribute collisions do not break core behavior.

### 6) Define Strict vs Convenience Data-Cleaning Mode (`CAN-DEFER`)

Goal:
- Make constructor cleaning/coercion behavior explicit and user-controlled.

Acceptance Criteria:
- Default behavior is clearly documented (NA replacement + numeric coercion rules).
- Release notes include current cleaning/coercion policy and intended future direction (if any).

### 7) Reduce Mutable Public Internals Exposure (`PASS`)

Goal:
- Avoid accidental mutation of internal result structures through public API.

Acceptance Criteria:
- Returned chart/stat containers are defensive copies or read-only interfaces.
- User mutations do not corrupt later operations.
- Spot tests confirm `get_chart()`, `get_statistics()`, and `summary` are mutation-safe.

### 8) Stabilize `chart_table()` Join Logic (`SHOULD-FIX`)

Goal:
- Robust subgroup-size joins across key naming/config variants.

Acceptance Criteria:
- `chart_table()` works with default and non-default subgroup key naming.
- No row duplication/loss from join-key mismatch.

### 9) Narrow Top-Level Export Surface (`CAN-DEFER`)

Goal:
- Keep root package namespace focused on stable, supported API.

Acceptance Criteria:
- Explicit public API contract documented.
- Internal/advanced exports reduced or marked provisional.

### 10) Clarify ODS vs ADS Messaging (`CAN-DEFER`)

Goal:
- Prevent misinterpretation of which SDS governs chart validity.

Acceptance Criteria:
- User-facing summaries explicitly label ODS vs ADS roles.
- `design()` and `summary` terminology cannot be read as contradictory.

## Recommended Release Notes Addendum

If any non-`PASS` deferred items remain open at release time:

1. List deferred item IDs.
2. Note user impact.
3. Provide migration guidance or workaround.
4. State the target version for resolution.
