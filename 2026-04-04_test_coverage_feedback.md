# Test Coverage Feedback

Review date: April 4, 2026. Scope: suite structure, coverage gaps, confidence risks, and cleanup needs. Methods: test collection review, coverage run, marker and skip inspection, fixture-usage review, and spot checks of large and high-risk test modules.

## Overall Health

- The suite is broad and healthy at a high level: `1297 passed, 4 skipped`.
- Collection spans more than 40 test modules and roughly 1300 tests.
- Overall coverage is good at `82%`, but it is uneven in a few important public-facing modules.
- The main weakness is not lack of tests everywhere; it is concentration of debt in a few public contract surfaces and several oversized test files.
- Strengths:
  - Broad subsystem reach across analysis, SDS, plotting, capability, exports, signals, and data cleaning.
  - Strong domain-specific fixture strategy using synthetic SDS generators in [`tests/conftest.py`](/Users/nicholas/Documents/projects/processbehavior/tests/conftest.py).
  - Good regression preservation in `by` parameter, phased limits, recentered MR, release-gate, and xbar stratification areas.
  - Good use of golden masters and reference datasets for end-to-end stability.

## Coverage Gaps

- `processbehavior/analysis_result.py` is the highest-value gap at `59%` coverage.
  - Why it matters: this is a public API surface, so low coverage here is more dangerous than low coverage in internal helpers.
  - The missing areas cluster around stratified helper behavior, `chart_table`, focused-result paths, and convenience accessors.
  - This aligns with recent API review findings: helper semantics are easy to regress because the surface is branchy and partly convenience-driven.
  - Minimal fix: add deterministic contract tests for `strata`, `focus`, `get_statistics`, `get_stratified_chart`, `get_stratified_charts`, `list_strata`, and `chart_table` using real `AnalysisResult` objects rather than mocks.

- `processbehavior/signals/result.py` is only `53%` covered.
  - Why it matters: result/reporting objects often look simple but hide user-visible formatting, export, and summary behaviors.
  - Current coverage likely exercises the core signal detection flow more than the result-surface contract.
  - Minimal fix: add explicit tests for result summary fields, filtering helpers, export-facing behavior, and failure-mode messaging in [`tests/test_signals.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_signals.py) or a dedicated result-surface module.

- `processbehavior/signals/detectors.py` is only `38%` covered.
  - Why it matters: if signals are part of the supported surface, under-testing the detector family is a real behavioral risk.
  - The low number suggests there are branch families and detector variants that are not being hit at all.
  - Minimal fix: enumerate supported detector types and add behavior-specific tests per detector branch, including negative and edge conditions, rather than relying on a few integration-style checks.

- `processbehavior/datasets/synthetic.py` is only `49%` covered.
  - Why it matters: the suite depends heavily on synthetic data generation, so weak coverage here undercuts confidence in the very fixture generator used across tests and docs.
  - Minimal fix: add direct tests for the generator’s branch families, parameter validation, and SDS-shape guarantees instead of only consuming it indirectly through downstream tests.

- `processbehavior/exporters/excel_exporter.py` is `73%` covered.
  - Why it matters: Excel export is user-visible and branch-heavy; moderate coverage still leaves meaningful release risk.
  - The uncovered areas are likely formatting, optional chart embedding, and fallback/error paths.
  - Minimal fix: expand [`tests/test_excel_export.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_excel_export.py) with real workbook assertions across chart/no-chart, residual/effects, and dependency/failure paths.

- Plotting orchestration is only moderately covered despite being branch-heavy.
  - `processbehavior/plotting/plotter.py` is `76%`.
  - `processbehavior/plotting/renderers.py` is `77%`.
  - `processbehavior/plotting/control_chart.py` is `78%`.
  - Why it matters: visual/output orchestration bugs often survive broad smoke tests because many branches are metadata-driven rather than numerically obvious.
  - Minimal fix: add focused integration tests around subplot selection, stratified rendering, renderer fallbacks, and save/export behavior using real results and small deterministic datasets.

## Confidence Risks

- Some regression tests are skip-based rather than deterministic.
  - [`tests/test_analysis_bugs.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_analysis_bugs.py) contains several `pytest.skip(...)` branches when a chart is not valid for the runtime data shape.
  - Why it matters: a regression test is strongest when it pins a known-valid setup and asserts one expected behavior, not when it conditionally exits.
  - Minimal fix: refactor those cases to use fixtures that deterministically satisfy the intended SDS/chart preconditions.

- Parquet coverage is environment-conditional.
  - [`tests/test_process_behavior.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_process_behavior.py) uses `pytest.importorskip("pyarrow")`.
  - Why it matters: `read_parquet()` is public, so coverage should follow a deliberate environment strategy rather than opportunistic dependency presence.
  - Minimal fix: either make parquet coverage mandatory in CI or isolate it behind a clearly labeled optional-I/O test job.

- `test_plotting_hardening.py` leans heavily on `MagicMock`.
  - [`tests/test_plotting_hardening.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_plotting_hardening.py) is useful for defensive-path testing, but many cases stub out the real result contract.
  - Why it matters: mock-heavy tests can validate control flow while missing integration breakage between `AnalysisResult`, plot metadata, and renderer expectations.
  - Minimal fix: keep the mocks for exception hardening, but pair them with a small number of real-object integration tests for each major plotting contract.

- Golden master tests provide stability, but there is risk of snapshot sprawl.
  - [`tests/test_analysis_golden_master.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_analysis_golden_master.py) and the fixture tree under [`tests/fixtures/golden_masters`](/Users/nicholas/Documents/projects/processbehavior/tests/fixtures/golden_masters) are valuable.
  - Why it matters: snapshot-style tests are best when paired with semantic assertions; otherwise they can become hard to maintain and easy to bless without understanding.
  - Minimal fix: keep the golden masters, but ensure each scenario also has a few behavior-level assertions on meaning, not just fixture equality.

- Release-gate tests are useful but narrow.
  - [`tests/test_release_gate_0_1_0_checklist.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_release_gate_0_1_0_checklist.py) gives strong coverage for selected public-contract issues.
  - Why it matters: these tests improve confidence on known failure modes, but they do not substitute for broad behavioral correctness across the API.
  - Minimal fix: treat release-gate tests as a contract layer, not the main correctness layer.

- Reference-data tests have external fragility.
  - [`tests/test_bishop_reference.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_bishop_reference.py) skips if reference files are absent.
  - Why it matters: reference tests are high-value only when reliably present in the environments that matter.
  - Minimal fix: either ensure those files are always available in CI or move these tests into a clearly separated optional/reference job.

## Cleanup Required

- Several test files are too large and mix too many concerns.
  - Highest-priority split candidates:
    - [`tests/test_sampling_plan.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_sampling_plan.py)
    - [`tests/test_process_behavior.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_process_behavior.py)
    - [`tests/test_sds_detector.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_sds_detector.py)
    - [`tests/test_data_preparation.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_data_preparation.py)
    - [`tests/test_plotting.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_plotting.py)
    - [`tests/test_capability.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_capability.py)
  - Why it matters: giant files slow navigation, encourage copy-paste growth, and make targeted refactors harder.
  - Minimal fix: split by behavior family, not by arbitrary size. For example, separate contract tests, edge cases, integrations, and regressions.

- There is repeated inline dataset construction despite strong shared fixtures.
  - Many files still build many `pd.DataFrame(...)` values inline even though [`tests/conftest.py`](/Users/nicholas/Documents/projects/processbehavior/tests/conftest.py) already provides good canonical SDS fixtures.
  - Why it matters: repeated inline data shapes drift, duplicate intent, and make updates expensive.
  - Minimal fix: centralize more canonical builders for repeated shapes and reserve inline frames for truly one-off edge cases.

- Test organization style is inconsistent.
  - Some modules are long top-level function lists, while others use many tiny `Test*` classes.
  - Why it matters: inconsistency makes the suite harder to scan and target quickly.
  - Minimal fix: standardize on behavior-oriented grouping, using classes only when shared setup or conceptual grouping adds clarity.

- Some modules mix multiple concerns in one place.
  - Example pattern: contract tests, edge cases, regressions, and integration scenarios living in the same file.
  - Why it matters: mixed-purpose files become junk drawers and make future failures harder to triage.
  - Minimal fix: separate files or sections by intent, especially for high-churn surfaces like plotting, process behavior entrypoints, and sampling-plan logic.

- Marker taxonomy is too thin.
  - [`pyproject.toml`](/Users/nicholas/Documents/projects/processbehavior/pyproject.toml#L37) only defines `slow` and `benchmark`.
  - Why it matters: targeted runs become less useful as the suite grows.
  - Minimal fix: add markers such as `integration`, `golden`, `io`, `plotting`, and `contract`.

- Some test names are historical rather than behavior-specific.
  - Files like [`tests/test_analysis_bugs.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_analysis_bugs.py) and [`tests/test_p0_p2_regressions.py`](/Users/nicholas/Documents/projects/processbehavior/tests/test_p0_p2_regressions.py) preserve history, but they are not ideal entrypoints for maintenance.
  - Why it matters: tests are easier to maintain when names communicate the supported behavior, not the story of how the bug was found.
  - Minimal fix: keep regressions, but prefer naming by current behavior or contract and note original issue IDs in comments/docstrings.

## Recommended Priorities

- Immediate release-confidence work:
  - Raise confidence in [`processbehavior/analysis_result.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_result.py), [`processbehavior/signals/result.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/signals/result.py), and [`processbehavior/signals/detectors.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/signals/detectors.py).
  - Add deterministic tests for stratified helper semantics and focused-result behavior.
  - Reduce skip-based regressions where practical.

- Next tranche:
  - Strengthen Excel, plotting, and export-path tests with real objects rather than mocks alone.
  - Add targeted direct coverage for [`processbehavior/datasets/synthetic.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/datasets/synthetic.py) branches used by tests and docs.

- Cleanup tranche:
  - Split oversized files.
  - Normalize fixture strategy around reusable canonical builders.
  - Add markers and naming cleanup so targeted test execution becomes easier.

- Optional polish:
  - Review whether golden masters can be trimmed or consolidated without losing semantic coverage.
  - Add a small documented testing taxonomy so future tests land in the right layer on first write.
