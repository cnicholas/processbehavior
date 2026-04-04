# Claude Code Instructions

## Output format for plans
- Goal
- Non-Goals
- Files to Touch (with why) - always show the coding changes
- Behavioral Changes
- Invariants to Preserve
- Test Strategy
- Risk Areas / Edge Cases

## Git Commits

- Do not add Co-Authored-By lines to commit messages
- Do not add "Generated with Claude Code" footer to commit messages
- The user should be the sole author of all commits

## What This System Is

processbehavior is a differentiated SPC library — nothing like it exists in Python or R. It faithfully implements Thomas A. Bishop's Variance Analysis System (VAS) methodology, equation-by-equation.

The primary user is the analyst. The API must be simple, mirror how analysts think and work, do the basics excellently, and enable the most experienced analyst to extract deep value.

## Engineering Philosophy

- **Analyst-first API**: The API mirrors the analyst's workflow, not the programmer's. `formulate()` is how analysts think: understand your data structure before computing. `execute()` is the computation. This two-step pattern is non-negotiable — it's not an architecture choice, it's a reflection of how analysis works.
- **Do the basics excellently**: Simple things should be simple. `pb.formulate(response=..., factors=..., time=...)` → `study.execute()` → done. No configuration ceremony. The system auto-detects SDS, selects valid charts, cleans garbage characters, and produces correct results.
- **Enable the expert**: Progressive disclosure. The casual analyst gets correct charts in two calls. The experienced analyst can drill into VAS residuals (R1-R5), variance decomposition, effects, interactions, `DesignReport` plan-vs-observed comparison, and `why_not()` explanations.
- **Methodology fidelity**: This library implements Bishop's VAS — not "inspired by," but equation-by-equation. When convenience conflicts with the methodology, methodology wins. Don't suggest shortcuts that diverge from the reference. If you don't know what Bishop says, say so rather than guess.
- **Pit of success (Pythonic Hadley)**: The easy path is the correct path. Self-diagnostic errors that say what's available and how to fix it. Constrained APIs that prevent misuse. `ColumnRef` for IDE auto-completion. Garbage cleaned automatically.
- **Correct before complete**: Validate against Bishop's Minitab reference data (TABVASTESTDATABASE.csv). Fewer features done correctly beats more features done approximately.
- **SDS drives everything**: Detected once on raw data, passed through the system. No class re-detects SDS. It determines valid charts, R2 method, and variance decomposition.
- **Composition, single responsibility, immutability**: AnalysisDataSet orchestrates but delegates. Study is a frozen dataclass. Calculation functions are pure where possible.

## Domain & Architecture

### Critical Invariants
- **SDS detection runs on RAW data** (before dropping NA response rows). Cells with all-NA responses count as "attempted" cells — required to detect SDS 4-6 (incomplete designs).
- **R2 is structure-dependent** (exact/ma2/hybrid based on cell sizes). R1, R3, R4, R5 are pure algebra.
- **`rsg_vars` dual semantics**: variance decomposition groups for Xbar/S; stratification (separate charts) for IMR/R.
- **obs_id assigned BEFORE sort**, cell_key = (factor × time) tuple. Canonical sort: (cell_key, obs_id).

### Pipeline
- `formulate()` is expensive: SDS detection on raw data, builds AnalysisDataSet (residuals, effects)
- `execute()` is cheap: runs chart strategy on pre-computed data
- Multiple charts from same Study without re-computation

### Key Classes
- `ProcessBehavior` → `.formulate()` → `Study` → `.execute()` → `AnalysisResult`
- `AnalysisDataSet` orchestrates: `DataPreparation` → `SDSRegistry` → `ResidualCalculator` / `EffectsCalculator`
- `DataPrepConfig` (base config) → `AnalysisSpecification` (adds analysis_type)

### Validation & Testing
- `validation/TABVASTESTDATABASE.csv`: Bishop's reference data. PM SDS 1–6 columns, `*` = NA
- pytest with `.venv/bin/python -m pytest tests/`
- Synthetic data: `from processbehavior.datasets.synthetic import make_sds`

### Wheeler Terminology
- Process Behavior Chart = Control Chart
- Natural Process Limits ≠ Specification Limits
- XmR = IMR (Individual Moving Range)

## Commit Message Template
## Summary
- **What:** <!-- one sentence: feature/bugfix -->
- **Why:** <!-- user value / problem -->
- **Scope:** <!-- what changed at a high level -->

## Contract / Invariants (must remain true)
- [ ] **Default behavior unchanged:** `phased=False` (and other defaults) produce byte-for-byte equivalent results
- [ ] **Residuals unaffected:** no residual/SDS recomputation; charting-only change
- [ ] **Row/index alignment preserved:** output row count + ordering unchanged (unless explicitly documented)
- [ ] **Lane boundaries preserved:** boundary positions/semantics unchanged (or explicitly documented)
- [ ] **Output schema compatible:** column names/types and metadata contract remain compatible with existing consumers

## Behavior Changes (explicit)
- **New API:** <!-- e.g., study.execute(chart='XmR', by=[], phased=True) -->
- **New semantics:** <!-- bullet list of intended behavior -->

## Key Design Decisions
- **Phase definition:** <!-- contiguous runs of rsg_key, etc. -->
- **NaN policy:** <!-- e.g., NaN points not flagged; plot skips NaNs -->
- **Run rules policy:** <!-- e.g., disabled when phased -->
- **Edge cases:** <!-- single-point phases, etc. -->

## Risks & Mitigations
- **P0 risks:** <!-- silent-wrongness risks -->
- **Mitigations:** <!-- gating, assertions, metadata flags -->

## Tests
### Added / Updated
- [ ] <!-- test name --> — <!-- what it proves -->
- [ ] <!-- test name --> — <!-- what it proves -->

### Regression Focus (must fail if broken)
- [ ] <!-- e.g., R lane boundary alignment under phased -->
- [ ] <!-- e.g., phased=False identical output -->
- [ ] <!-- e.g., plot_col asserted, correct series used -->

## Manual Verification
- [ ] `pytest tests/ -x`
- [ ] `pytest <new_test_file> -v`
- [ ] Visual check / notebook snippet: <!-- paste minimal repro -->

## Notes
- **Docs:** <!-- docstring updates / user-facing notes -->
- **Follow-ups (not in this PR):** <!-- explicitly defer gold plating -->

##Testing
- Always use the validation dataset for testing.  It is ground truth.
- File: [text](validation/TABVASTESTDATABASE.csv)