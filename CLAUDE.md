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

## STOP — do not accept the VAS analysis recipe into this repository

**This repo is Apache 2.0. The ordered VAS recipe is Dr. Bishop's intellectual property and the material his training is built on. It must not be published under this licence.**

The recipe — *which* analyses to run, gated how, and in what narrative order — lives in `analysis_pack.py` in the **processbehavior-app** repository, deliberately. This library provides the building blocks (`study.execute` / `study.capability` / `study.loss_function`); the selection, gating and ordering are not ours to publish.

The failure mode is one-way. A maintainer — or an assistant asked to reduce duplication — sees the app hand-rolling an ordered sequence against library primitives, correctly identifies it as logic that "belongs" in the library, and upstreams it. **An Apache-2.0 release cannot be un-published.** Every other mistake in this repo is recoverable; this one is not.

So: if a change would move fold names, fold ordering, gating rules, or the Analysis Pack recipe into this repository, **stop and ask** — regardless of how much duplication it would remove. The duplication is the price, and it is the right price.

This is not hypothetical. A design review of this codebase read the `analysis_pack.py` docstring, understood it as a product-architecture choice, and recommended re-deciding it — because the actual reason was written down nowhere. That is what this section fixes.

**The public statement of this boundary lives in `CONTRIBUTING.md` → "What this library is, and what it is not."** This file is not part of the published repository, so it cannot be the only place the rule is written; CONTRIBUTING.md is what an outside contributor (or a future maintainer) actually reads. Keep the two in agreement — if you change the boundary here, change it there.

## Engineering Philosophy

- **Analyst-first API**: The API mirrors the analyst's workflow, not the programmer's. `formulate()` is how analysts think: understand your data structure before computing. `execute()` is the computation. This two-step pattern is non-negotiable — it's not an architecture choice, it's a reflection of how analysis works.
- **Do the basics excellently**: Simple things should be simple. `pb.formulate(response=..., factors=..., time=...)` → `study.execute()` → done. No configuration ceremony. The system auto-detects the PDS / ODS / ADS lineage, selects valid charts, cleans garbage characters, and produces correct results.
- **Enable the expert**: Progressive disclosure. The casual analyst gets correct charts in two calls. The experienced analyst can drill into VAS residuals (R1-R5), variance decomposition, effects, interactions, `DesignReport` plan-vs-observed comparison, and `why_not()` explanations.
- **Methodology fidelity**: This library implements Bishop's VAS — not "inspired by," but equation-by-equation. When convenience conflicts with the methodology, methodology wins. Don't suggest shortcuts that diverge from the reference. If you don't know what Bishop says, say so rather than guess.
- **Pit of success (Pythonic Hadley)**: The easy path is the correct path. Self-diagnostic errors that say what's available and how to fix it. Constrained APIs that prevent misuse. `ColumnRef` for IDE auto-completion. Garbage cleaned automatically.
- **Correct before complete**: Validate against Bishop's Minitab reference data (PBTESTDATABASE_T100.csv). Fewer features done correctly beats more features done approximately.
- **ADS drives everything**: The library reports three design states — PDS (planned), ODS (observed-on-raw-data, before NA drop), ADS (analytical, after tidying). **ADS** determines valid charts, R2 method, and variance decomposition. Detected once during `formulate()` and passed through the system; no class re-detects state. PDS and ODS are exposed for diagnostics via `study.plan_design_state`, `study.observed_design_state`, `DesignReport`. The legacy term "SDS" survives in internal class names (`SDSResult`, `SDSRegistry`, `sds_detector.py`) and in the integer 1-6 code that each state carries on its `.sds` field (Bishop's reference scale); the conceptual model is three-state.
- **Composition, single responsibility, immutability**: AnalysisDataSet orchestrates but delegates. Study is a frozen dataclass. Calculation functions are pure where possible.
- **Exception convention**: Input/parameter validation raises `ValidationError` or one of its subclasses (`ColumnNotFoundError`, `FactorNotFoundError`, `ChartNotAvailableError`) from `processbehavior/exceptions.py`. Methodology invariants can stay `RuntimeError`. Never raise raw `ValueError` for user-facing input errors. (As of v0.1.0 there are ~71 raw `raise ValueError` calls left from earlier code; the hierarchy is the target — don't add new ones.)

## Releasing

- Single-source version: `processbehavior/__init__.py:__version__`. `pyproject.toml` reads it dynamically via `[tool.hatch.version]`.
- Move content from `## [Unreleased]` in root `CHANGELOG.md` into a new `## [X.Y.Z] - YYYY-MM-DD` section; update compare links at the bottom.
- Tag `vX.Y.Z` triggers `.github/workflows/publish.yml` which builds, runs `twine check`, and uploads to PyPI via OIDC trusted publishing (no API tokens). The trusted publisher must be configured once at pypi.org/manage/project/processbehavior/settings/publishing/ before the first tag.
- Full process is in CONTRIBUTING.md → Releasing.

## Domain & Architecture

### Critical Invariants
- **ODS detection runs on RAW data** (before dropping NA response rows). A cell whose responses are all NA counts as *attempted but empty* — it is an empty cell in the grid, not an occupied one, which is what makes ODS 4-6 (incomplete designs) detectable. ADS is then computed on the tidied data; ODS {4,5,6} collapse to ADS {1,2,3}.
- **R2 is structure-dependent** (exact/ma2/hybrid based on cell sizes). R1, R3, R4, R5 are pure algebra.
- **`rsg_vars` dual semantics**: variance decomposition groups for Xbar/S; stratification (separate charts) for IMR/R.
- **obs_id assigned BEFORE sort**, cell_key = (factor × time) tuple. Canonical sort: (cell_key, obs_id).
- **Stats-dict shape**: `result.get_statistics(name)` returns a dict with keys `{N, center, lpl, upl}`. Never `mean`, never `Mean`. Doc examples, README snippets, and tests must use `'center'`. (See the bug fixed at `analysis_result.py:603` in Phase 2.)
- **Xbar center line is Bishop VAS unweighted**: mean of (factor × time) cell means on `value_col`, equal weight per experimental condition regardless of cell N_kt. Holds for both response and residual Xbar charts; balanced designs collapse to the observation-weighted mean, unbalanced designs differ by a methodology-required (and per Bishop, practically negligible) amount. Computed in `_calculate_xbar` at `analysis.py:~699`. Don't reintroduce `df[value_col].mean()` as the center — that's the dead-branch bug fixed in this release.

### Pipeline
- `formulate()` is expensive: ODS detection on raw data, ADS on tidy data, builds AnalysisDataSet (residuals, effects)
- `execute()` is cheap: runs chart strategy on pre-computed data
- Multiple charts from same Study without re-computation

### Key Classes
- `ProcessBehavior` → `.formulate()` → `Study` → `.execute()` → `AnalysisResult`
- `AnalysisDataSet` orchestrates: `DataPreparation` → `SDSRegistry` → `ResidualCalculator` / `EffectsCalculator`
- `FormulationSpec` (frozen, structural, built once in `formulate()`) → `ChartRequest` (frozen, ephemeral, one per `execute()`)

### Dependency boundaries
- Runtime deps (in `[project.dependencies]`): `numpy>=1.23`, `pandas>=2.0,<3`, `natsort>=8.0`, `plotly>=5.18,<7`. Upper bounds on pandas/plotly are intentional — the next major bumps both have documented breaking changes.
- numpy is declared explicitly even though pandas pulls it. Don't rely on transitive resolution.
- `openpyxl` and `kaleido` are NOT runtime deps. They live in `[excel]` and `[images]` extras, respectively, and are lazy-imported inside the export paths. Don't move them back to the runtime list.
- `[dev]` is a recursive aggregate — `[test, lint, images, excel, docs]`.

### Validation & Testing
- `validation/PBTESTDATABASE_T100.csv`: Bishop's reference data. PM SDS 1–6 columns, `*` = NA
- pytest with `.venv/bin/python -m pytest tests/`
- Synthetic data: top-level `pb.make_design(state=1..6, seed=...)` (re-exported); module path `from processbehavior.datasets.synthetic import make_design`
- Use pytest's `tmp_path` for any file the test writes; never `tempfile.NamedTemporaryFile(delete=False)` + manual `os.remove`. The hand-rolled pattern races with openpyxl/pandas readers on Windows (WinError 32). The `temp_excel_file` fixture in `tests/test_excel_export.py` is the reference pattern.
- CI matrix is `[3.10, 3.11, 3.13]` × `[ubuntu, macos, windows]`. Lint + mypy run only on `ubuntu/3.11` to avoid platform-specific drift.
- Python floor is 3.10 (3.9 went EOL Oct 2025 and was dropped pre-publish). `typing_extensions` is still a conditional dep below 3.11 — `NotRequired` only entered `typing` in 3.11.
- Run a fast local slice with `.venv/bin/python -m pytest tests/ -m "not slow"`.

### Wheeler Terminology
- Process Behavior Chart = Control Chart
- Natural Process Limits ≠ Specification Limits
- XmR = IMR (Individual Moving Range)

### Vocabulary (canonical internal names — reject new aliases at review)

The codebase had eight overlapping terms for two underlying concepts. Pick from these names; do not introduce new spellings.

**Rational subgrouping (the "what factors compose the subgroup" question):**
- `rsg_vars : tuple[str, ...] | None` — the canonical attribute on `FormulationSpec`. Tuple of factor column names that compose the rational subgroup.
- `spec.rsg_vars_list : list[str]` — property that converts to a list (empty when None). Use this when you need to pass to pandas as a list. Don't recreate `list(spec.rsg_vars or ())` in callers.
- `rsg_var_name : str` — the **column name** of the composite rsg column (default `'rsg'`). A NAMING concept, not a content one. Used as a `groupby` key. Don't shorten to `rsg_var` — that conflates with `rsg_vars`.
- `rsg_key : str` — the **value** of the rsg column for a row (e.g., `'1_2'`). Not a name, not a list.

**Stratification (the "separate into charts" question):**
- `stratify_by : list[str] | tuple[str, ...]` — the list of column names to stratify on. User-facing parameter shape.
- `stratify_col : str` — the **single composite output column** that holds the stratification key (often `'rsg'`, sometimes `'_stratify_key'` for multi-factor strata). Different concept from `stratify_by`.
- `stratum : str` — one stratum identifier value.
- `strata : list[str]` — collection of stratum identifier values.
- Bare `stratify` is fine only in prose/docstrings as a verb; do not use it as an identifier.

**Residual scope (which residuals exist where):**

Two genuinely different kinds share the word "residual". An enumeration that omits R6 is
correct *iff* it means the stored set — say which you mean. Defined in `spc_constants.py`.
- `STORED_RESIDUALS` — R1–R5 (and the recentered RCR1–RCR5). Computed once during
  `formulate()`, stored as columns on the analysis dataset. **Independent of `by=`**: R5 is
  the same series whatever grouping you execute with.
- `REQUEST_RESIDUALS` — R6 (and RCR6). Computed during `execute()` from that request's `by=`
  and materialised onto that result only. `R6(by=['A'])` ≠ `R6(by=['B'])`, so there is **no
  canonical study-level R6** — this is why `result.residuals` excludes it, deliberately.
- `ALL_RESIDUALS` = stored + request. `study.RESIDUAL_CODES` is this.
- Prose: "stored residual" / "request residual". **Do not use "derived"** — `derivations.py`
  owns that word for transform/binning specs.
- Request residuals are computed in `Analysis.calculate()` onto a per-request frame
  (`residual_calculator.calculate_r6_residuals`); the study-level frame is **never mutated
  after `formulate()`** — `execute()` and the probes (`why_not`, `supports_calibration`) are
  pure. A result's `_dataset` carries R6/RCR6 iff that request computed them, so
  `_locate_residual`'s column check on the result's own frame is the correct and sufficient
  test. Don't reintroduce writes to `AnalysisDataSet.analysis_dataset` to "share" R6 — a
  canonical study-level R6 does not exist, and the shared-frame version of this design
  required a metadata gate to contain the stale-column leak it created.
- One lookup: `AnalysisResult._locate_residual`. `get_residual` and `plot_residuals` both use
  it. Nothing else should read `self._residuals` directly to answer "give me residual X" —
  three sites diverging is what produced the original bug.

**Grain markers (factor × time):**
- `cell_key : tuple` — the (factor, time) tuple identifying one cell in the analysis grid.
- `kt_cols : list[str]` — list of column names that together form the factor × time grain. (`k` = factor index, `t` = time index — Bishop notation.)

**Plot visuals:**
- `lane` — an X-chart vertical-divider concept (visual only); see `processbehavior/plotting/lane_boundaries.py` and `x_axis_layout.py`. Not a grouping concept.

When you see existing code using a non-canonical alias and you're touching nearby lines, rename in passing. Don't introduce new aliases.

### Docs
- Jupyter Book / MyST is the only doc system. Config: `docs/myst.yml`. The previous `mkdocs.yml` was deleted in Phase 2; do not restore it.
- Root `CHANGELOG.md` is the source of truth — `docs/appendix/changelog.md` is a `{include}` of it.
- The `Docs` workflow at `.github/workflows/docs.yml` builds with the MyST CLI and deploys to GitHub Pages on push to `main` (gated on Pages being enabled in repo Settings).
- Docs versioning policy (single live version tracking `main`; no version strings in prose; hand-pinned mystmd) is in `CONTRIBUTING.md` → "Docs versioning".

### Audit history
- Phase 3 release-prep follow-ups (mypy re-enable, broad-except cleanup, lockfile, etc.) are tracked in GitHub Issue #77.
- The original four-agent pre-release audit synthesis lives in the commit messages around `88477a5` (Phase 1) → `5b8ce17` (Phase 2) → `a97e079` (Phase 3). The aggregated `COMPREHENSIVE_PRE_RELEASE_EVALUATION.md` was untracked in Phase 1.

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

## Testing
- Always use the validation dataset for testing.  It is ground truth.
- File: [text](validation/PBTESTDATABASE_T100.csv)

## Anti-patterns (don't add these back)
- Don't track planning docs at repo root. Use `.claude/notes/` (already gitignored).
- Don't bundle CSVs in `processbehavior/datasets/data/` without a `load_<name>()` function in `processbehavior/datasets/__init__.py`. Test-only CSVs belong in `tests/fixtures/data/`.
- Don't track generated artifacts: `*.html` reports, `*.docx`, `simulation/output/*`. They are gitignored — keep them so.
- Don't pin GitHub Actions by major tag. Use full SHA + a `# vX.Y.Z` comment; Dependabot keeps them current.
- Don't use `pip-audit --strict` while the project is editable and not yet on PyPI — `--strict` errors on any skipped package and the editable project is always skipped. Re-enable after first publish.
- Don't suggest restoring `mkdocs.yml` — see "Docs" above.
- Don't use `tempfile.NamedTemporaryFile(delete=False) + os.remove` in tests — see "Validation & Testing" above.