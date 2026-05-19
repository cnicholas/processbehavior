# Changelog

All notable changes to **processbehavior** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Synced validation dataset to Tom Bishop's corrected golden copy
  (`validation/PBTESTDATABASE_T100.csv`, replacing `TABVASTESTDATABASE.csv`).
  PM SDS 4/5/6 columns are now labeled so PM SDS N data classifies as ODS N
  (the prior file had a cyclic offset that the old `TestSDSRenumberDrift`
  test class documented). `PM KNOWN` column removed. PM SDS 1/2/3 + factor,
  time, and `PM INERT` columns are byte-identical. The e2e Bishop validator
  remains all-green; `TestSDSRenumberDrift` is replaced by
  `TestSDSColumnLabelAlignment` with two cleaner parametrized invariants.

### Fixed
- Xbar center line now correctly computes Bishop VAS's mean of (factor × time)
  cell means on the charted column for both response and residual charts.
  Previously a dead-code branch in `_calculate_xbar` caused the center to fall
  through to an observation-weighted mean (`df[value_col].mean()`), producing a
  small (<0.05) divergence from Bishop's reference on unbalanced designs.
  Surfaced on SDS 3 R3 (recentered) Xbar by PRODUCTION_TIME, where the center
  shifts from 237.81 to the methodologically-correct 237.83. Balanced designs
  (SDS 1, SDS 2's degenerate single-obs cells) and stratified Xbar charts were
  unaffected.
- `AnalysisResult.strata` now returns the order-preserving intersection of
  every chart's strata list. Previously it returned the first chart's list,
  which for X+mR companion results inherited the X chart's full strata even
  when mR had dropped first-row-per-stratum on single-observation cells.
  `result.focus(stratum)` then raised on those strata, breaking any caller
  that drove `.focus()` from `.strata`. The mR chart's published `strata`
  list is also now filtered to exclude insufficient strata.
- `AnalysisResult.chart_table()` previously raised
  `ValueError: You are trying to merge on object and int64 columns` when a
  chart's by-column had been stringified during construction (e.g. PRODUCTION
  TIME → object) while the analysis dataset kept it numeric. The n-join now
  coerces both sides to `str` when the join-key dtypes differ, and falls
  back to skipping the n-join entirely if the merge still fails — the
  table renders either way. Surfaced on `chart='Xbar', by=[time_var],
  value='R4', companion=True, recentered=True`.

## [0.1.0] - 2026-05-03

Initial public release. The 0.1.x line is **alpha** — the public API is settling
and may change between minor versions until 1.0.

### Added

#### Core API
- `ProcessBehavior` wrapper for pandas DataFrames with IDE auto-completion
  for column references via `pb.cols`
- Two-step analyst workflow: `pb.formulate(...) → study.execute(...) → AnalysisResult`
- Automatic detection of Bishop's six **Design States (DS 1–6)** on raw data
  before NA handling, including incomplete designs (DS 4–6)
- Self-diagnostic exceptions (`ColumnNotFoundError`, `FactorNotFoundError`,
  `ChartNotAvailableError`, `ValidationError`) reporting what is available
  and how to fix the call

#### Charts
- Xbar, S, X (Individual), mR (Moving Range), Histogram with correct limit
  calculations per Bishop's VAS methodology
- Stratified X/mR analysis: a single combined chart with per-stratum limits
  via `study.execute(chart='X', by=[...])` and `result.focus(stratum)`
- Phased limits (`phased=True`) for run-segmented X/mR charts

#### Variance Analysis (VAS)
- Residuals R1–R5 (and the rounded R6) for factorial designs in DS 1–3
- Variance decomposition for Xbar/S; rsg-based stratification for X/mR
- Effects analysis: factor effects, time effects, and factor × time interactions
- `DesignReport` for plan-vs-observed comparison
- `why_not()` explanations for unavailable chart types

#### Capability and Information
- Capability indices (Cp, Cpk, Pp, Ppk) via `CapabilityResult` with `SpecLimits`
- Loss-function analysis (`LossResult`)
- Maximum-information analysis (`MaximumInformationResult`)

#### Signal Detection
- Western Electric rules (1–8) configurable via `RuleSet`
- `SignalDetector` and `SignalResult` for programmatic access

#### Visualization
- Plotly-based interactive charts with zone shading and signal markers
- Themed plots (`processbehavior`, `minimal`, `dark`, `ggplot`) via
  `ChartTheme`, `register_theme`, `get_theme`, `list_themes`
- `result.plot()`, faceted plots for stratified analysis

#### Export
- Excel export with `result.to_excel(path)` (requires `[excel]` extra)
- HTML export of interactive figures
- Static image export (requires `[images]` extra)

#### Datasets
- `make_sds(sds=N, seed=...)` synthetic dataset generators for DS 1–6,
  re-exported at the top level for `pb.make_sds(...)`
- Edge-case generators (`make_edge_cases`)
- Bishop's `PBTESTDATABASE_T100.csv` reference is shipped in the source
  distribution for validation

#### Packaging
- PEP 561 typed package (`py.typed` marker)
- PEP 639 license metadata (`License-Expression: Apache-2.0`)
- Optional dependencies: `[excel]`, `[images]`, `[test]`, `[lint]`, `[docs]`
- Supports Python 3.9–3.13

### Dependencies
- `numpy >= 1.23`
- `pandas >= 2.0, < 3`
- `natsort >= 8.0`
- `plotly >= 5.18, < 7`

[Unreleased]: https://github.com/cnicholas/processbehavior/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cnicholas/processbehavior/releases/tag/v0.1.0
