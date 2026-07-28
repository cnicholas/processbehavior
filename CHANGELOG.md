# Changelog

All notable changes to **processbehavior** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **``get_residual('R6')`` no longer reports "not found" about data the result is holding.**
  A single result object gave three answers: ``result.dataset['R6']`` held 4,000 values,
  ``result.residuals`` omitted it, and ``get_residual('R6')`` logged a false "not found" and
  returned an **empty Series** — so a caller who did not check ``len()`` computed on nothing.
  The lookup consulted a snapshot taken at construction and never looked at the result's own
  dataset. ``plot_residuals('R6')`` had the same defect and is fixed by the same change;
  both now route through one lookup.

  ``result.residuals`` still excludes R6, deliberately — see below.

### Changed
- **The two kinds of VAS residual now have names.** ``spc_constants`` defines
  ``STORED_RESIDUALS`` (R1-R5, RCR1-RCR5 — computed at ``formulate()``, independent of
  ``by=``) and ``REQUEST_RESIDUALS`` (R6, RCR6 — computed at ``execute()`` from that
  request's ``by=``). ``R6(by=['A'])`` and ``R6(by=['B'])`` are different series, so there is
  no canonical study-level R6 and ``result.residuals`` correctly omits it. Previously both
  kinds were just "residuals", so an enumeration that omitted R6 was indistinguishable from
  a bug. ("derived" is deliberately not used — ``derivations.py`` owns it.)
- **BREAKING (return contract): ``get_residual`` raises instead of returning an empty
  Series.** Unknown codes, and residuals genuinely absent from a result, now raise
  ``ValidationError`` with a message saying which case applies and what to do. The empty
  return was the only silent-wrongness path in the public API.
- **``get_residual`` will not hand back another request's numbers.** Computing R6 writes its
  column onto the study-level dataset, so a result created afterwards inherits an R6 it never
  asked for. The lookup gates on the result's own chart metadata, so that case raises rather
  than returning values from a different ``by=``.

- **``why_not()`` no longer contradicts the error that sends users to it.** Asking for an
  unavailable chart x residual pair raised a correct ``ChartNotAvailableError`` ending "Use
  study.why_not('Xbar', value='R2') for details" — and that call replied "'Xbar' (R2) is not a
  recognized chart type", which is false, then referred on to ``study.support``, which had no
  row for the pair. ``why_not`` answered only from ``support``, whose residual rows enumerate
  just the *potentially valid* pairs, so "no row" was conflated with "unknown chart".

  Both ``why_not`` and ``execute`` now answer from one predicate
  (``Study._residual_pair_problem``), so they cannot disagree. ``why_not`` also distinguishes
  an unrecognised chart name, an unrecognised residual code (``R9``), and a recognised chart in
  an invalid pairing — three cases that previously produced the same wrong message — and
  accepts recentered (``RCR5``) and lowercase forms, matching what ``execute`` already
  normalised.

### Changed
- **BREAKING (validation only): ``execute(chart='S', value='R1')`` now raises.** R1 was exempt
  from the chart x residual check, so ``residual_charts`` claimed R1 was Xbar/X-only while
  ``execute`` accepted any chart — the two authorities genuinely disagreed, and ``why_not``
  could not be made correct while both existed. The exemption is deleted;
  ``residual_charts`` now governs outright. Nothing informative is lost: R1 is the response
  shifted by a constant, so its S chart is byte-identical to the response's S chart (same N,
  center, lpl, upl — verified). Use ``chart='Xbar'`` or ``chart='X'`` for R1, or the plain
  response S chart. ``Histogram`` and ``mR`` residual behaviour is unchanged; both remain
  structurally exempt by design.

### Performance
- **``ProcessBehavior(df)`` is ~19x faster** — 4.09s → 0.21s at 1M rows × 50 columns, same
  machine. It was the single most expensive step in the pipeline, costing more than
  ``formulate()``, which is not where any of the documentation points.

  Two changes, both output-identical (same values *and* same dtypes — client code reads
  post-init dtypes to decide what a column can be used for):

  - The garbage-token scan ran on every column. A token is a string, so numeric, datetime
    and boolean columns cannot contain one; they are now skipped. The test is "is text-like",
    **not** "is not numeric" — a *categorical of strings* can hold a token and must still be
    scanned.
  - Numeric-formatting cleanup ran per row. It now cleans the *distinct* values and maps
    back, on plain ``object`` columns whose cardinality is under 5% of their length.
    Subtlety: the 80% acceptance threshold is frequency-weighted, so it is re-scored against
    the full column rather than the distinct values — a column of 9,900 copies of ``'1'``
    plus 100 labels converts by row (99%) but not by distinct value (1%). The shortcut is
    restricted to ``object`` because ``Series.map`` preserves extension dtypes: mapping a
    categorical returns a categorical of ints where the direct path returns ``int64``.

  Design-state detection is unaffected, and provably so: ``_build_structure_view``
  normalises missing tokens and canonicalises the kt columns itself on a minimal
  projection, so N_kt is identical whether or not ``__init__`` cleaned first. That property
  now has its own test. Cleaning also stays **eager and total** — deferring it to
  ``formulate()`` would leave a ``['235.5', '*', '237.2']`` column non-numeric at the point a
  client builds its column pickers, hiding the very column cleaning exists to rescue.

- **``formulate()`` is 1.4-3.8x faster.** Observed-design-state detection computed N_kt (valid
  responses per cell) with ``groupby(...).apply(lambda s: s.notna().sum())`` — one Python
  round-trip per cell. ``GroupBy.count()`` computes exactly the same thing in Cython. Cells
  whose responses are all NA still yield 0, so the "attempted but empty" semantics that ODS
  4-6 detection depends on are unchanged, as are all N_kt values, design-state
  classifications, residuals, limits, and the 280 Bishop reference assertions.

  The win scales with cell count, so it is largest where cells are singletons:

  | 1M rows | before | after | |
  |---|---|---|---|
  | SDS-2 (no replication) formulate | 25.17s | **6.56s** | 40K → 152K rows/sec |
  | SDS-1 (full replication) formulate | 3.13s | **2.25s** | 319K → 444K rows/sec |

  Profiling also corrected a long-standing misattribution: the MA2 sort, previously blamed
  for SDS-2's cost, is 0.09s at 1M — 0.4% of formulate. The perf-suite docstrings said
  otherwise and have been fixed. Peak memory is ~5% higher (3.9x input, against a 6x cap).

### Fixed
- **R6 chart titles rendered as ``R6 (R6)``.** ``_RESIDUAL_LABELS`` had no entry for R6, and
  ``_generate_title`` resolves labels with ``.get(code, code)``, so the code was printed in
  place of its name. R6's computation was never affected — it is validated against Bishop's
  Minitab reference in ``validation/e2e_bishop_report.py`` — this was display only.
- **R1 chart titles carried R2's concept.** R1 was labelled ``Within-Subgroup``, but Bishop
  §13.1 (*Centering the Original PM Data at 0*) defines ``R1 = Y_ktn − Ȳ..`` — the response
  re-expressed as ± about zero, a location shift, not a within-subgroup quantity. Now
  ``Response Centered at 0``.
- **R1 was advertised but not chartable.** ``study.residuals`` listed R1 while
  ``study.residual_charts`` omitted it entirely, so ``support`` / ``why_not()`` silently had
  nothing to say about it. R1 is now in the allowlist for **Xbar and X only** — it differs
  from the response by a constant, so an S or mR chart of R1 would duplicate the response's
  dispersion charts exactly.

### Changed
- **Residual labels adopt Bishop's design-condition vocabulary.** Chart titles, the user
  guide, the README, and the tutorials now read ``Design Condition Main Effects`` (R5) and
  ``Design Factor Main Effects`` (R6) in place of "Factor Effects" / "Factor Main Effect",
  matching the "process design conditions (PDC)" terminology already used by
  ``loss_function.py``. R2 is ``Within-Cell`` (was ``Within-Subgroup Variation``) and R4 is
  ``Time Main Effects``. Display-only: residual codes, ``value=`` arguments, dataset column
  names, and every computed value are unchanged, and all 280 Bishop reference assertions
  pass untouched. Callers passing an explicit ``title=`` are unaffected.
- **The user guide documents R1 as a chartable diagnostic.** ``docs/user-guide/residuals.md``
  gains an R1 section (it previously appeared only in the formulas appendix), and the
  availability table gains an R6 column. The README documents R6 for the first time.
- **Demo dataset ``load_coffee_shop()`` rebuilt into a fuller process-behavior story.**
  New schema — ``date``, ``year_week``, ``week``, ``day_of_week``, ``hour``,
  ``daypart`` (Peak/Off-peak, the process-design factor), ``wait_sec`` — dropping the
  old ``subgroup_id`` and ``reading_no`` columns (data-contract change). Formulate with
  ``factors=['daypart'], time='date'`` for a complete (SDS 1) design. The series now
  carries three teachable events: a week-8 espresso-machine improvement (Xbar shift), a
  ~one-week week-12 POS learning curve (a run, not a lone outlier), and a weeks 14–16
  new-hire spell where the mean holds but within-subgroup spread doubles (an S-chart
  signal). The before/after eras (~240s → ~205s) are preserved. Regenerated
  deterministically by ``scripts/generate_coffee_shop_demo.py``. New tutorial
  ``docs/tutorials/coffee-shop.ipynb`` walks the whole story; the standalone
  ``calibration.ipynb`` tutorial was retired (calibration is now shown in-context there).
- **Design Report** now labels the observed/sampling lineage row **``SDS (Sampling)``**
  instead of ``ODS (Observed)``, aligning with Bishop's "Sampling Design State"
  terminology (the lineage reads Planned → Sampling → Analytical). Display-only: internal
  attributes (``study.observed_design_state``) and all values are unchanged.
- **Observed Design State (ODS) now reports Bishop's full R = K×T design state** even
  without an explicit sampling plan. The no-plan path previously classified only over
  cells that contained data, so sparse *observational* data (many absent
  condition×time combinations, no ``*``/NA markers) was reported as complete/
  semi-complete (SDS 1–3). It now classifies over the full grid — the cross of the
  distinct observed process-design conditions with the distinct observed production
  times — so genuinely absent ``(condition, time)`` cells count as ``N_kt=0`` and the
  design is correctly detected as incomplete (SDS 4/5/6), matching Bishop's definition.
  The grid uses the **composite** observed condition tuple (not the independent cross of
  each factor's levels), so nested / non-crossed designs are not over-flagged.
  This is a **diagnostic-only** change: ODS is lineage/reporting only. The Analytical
  Design State (ADS) — which drives valid charts, R2 method, residual availability, and
  variance decomposition — is computed independently on tidy data and is unchanged, as
  are all chart/residual/variance outputs. Datasets that declare empties via ``*``/NA
  rows or via a plan (e.g. the ``PBTESTDATABASE_T100`` reference) are unaffected.
- ``chart_table`` (the per-chart display table) now uses a **1-based** position index
  (matching the chart's plotted x-axis, which starts at 1) instead of a 0-based counter,
  and adds the source-row id as an **"Obs"** column on Individuals (X/mR) tables for
  traceability. Xbar/S tables are unchanged (their rows are subgroup aggregates).
- ``obs_id`` is now the **1-based source-row id** (raw→analytic lineage), replacing the
  previous 0-based post-cleaning counter. It is stamped on the raw frame *before*
  cleaning, so every analytic row traces back to its source-file row; rows dropped in
  cleaning leave gaps in ``obs_id`` (e.g. ``1, 3, 4, 5``). Ordering is unchanged —
  ``obs_id`` remains the ``(cell_key, obs_id)`` sort tie-breaker and the SDS-4
  implicit-time key — and the "Obs N" shown in signal summaries and plot hover is now the
  source-row number. Data-contract change: consumers relying on ``obs_id`` starting at 0
  or being contiguous should update.

### Fixed
- Individuals (**X**) and Moving-Range (**mR**) charts now label the y-axis
  **"Individual Value"** / **"Moving Range"** instead of the response variable name.
  Fixed in ``Plotter._get_yaxis_label`` (keyed on chart type), so it applies to residual
  X/mR charts too (e.g. the R2 "Unexplained Effects" chart). Xbar/S labels
  ("Sample Average" / "Sample Standard Deviation") are unchanged, and an explicit
  ``.plot(yaxis_title=...)`` override still takes precedence.
- Signal detection no longer aborts on small groups. ``detect_signals`` previously
  raised ``ValueError("Insufficient observations…")`` whenever a group had fewer
  points than the most-demanding *applicable* run-rule (Rule 7 needs 15) — which
  broke detection on small stratified subgroups. It now evaluates the rules the
  group supports and reports the rest via ``SignalResult.rules_skipped`` (plus
  ``is_partial`` / ``evaluation_status`` accessors), so "no signals" is
  distinguishable from "not fully evaluated". The genuine input guards (empty data,
  missing limit statistics) now raise ``ValidationError`` instead of raw
  ``ValueError``; ``ValidationError`` additionally subclasses ``ValueError`` so
  existing ``except ValueError`` handling keeps working.
- Input data cleaning no longer coerces a pre-parsed ``datetime64`` column to
  int64. ``_try_clean_numeric_strings`` guarded only numeric dtypes, so a
  ``datetime64`` column passed to ``ProcessBehavior(df)`` (e.g. from
  ``pd.read_csv(..., parse_dates=...)``) was silently converted to nanosecond
  integers (``pd.to_numeric`` succeeds on datetimes). It now skips datetime and
  Period dtypes, mirroring ``DataPreparation._detect_and_convert_type``. String
  date columns are unaffected (still parsed to ``datetime64`` at formulation).

### Added
- **Derived variables** — create new columns from existing ones with fluent,
  pipeable verbs on ``ProcessBehavior`` that return a *new* (immutable) instance:
  ``pb.transform(column, function, ...)`` (``log``/``ln``, ``log10``, ``sqrt``,
  ``arcsin`` = arcsin√x for proportions, ``inverse``, ``square``, ``power``,
  ``zscore``) and ``pb.bin(column, method=..., n=..., breaks=..., bin_labels=...)``
  (``equal_freq``/``equal_width``/``breaks``/``sd`` → an ordered categorical).
  Derived columns become referenceable by ``response=/factors=/time=`` after
  ``formulate()``, which is the immutability boundary that freezes data-dependent
  fits (bin edges, z-score μ/σ) onto the new ``Study.derivations``. A binned
  ``bin_labels='ordinal'`` factor now charts in bin order (Low → High).
  Derivations are serializable specs (``Derivation`` with a stable ``id``,
  ``to_dict``/``from_dict`` round-trip) built by the same ``Derivation.transform``
  / ``Derivation.bin`` factories the application's attach-free live preview uses
  via ``evaluate(spec, column)``; ``validate(spec, dataset)`` returns structured
  pass/fail (label collisions, breakpoint/label-count checks) without raising.
  Domain violations are reported as structured data (count + row index), resolved
  at formulation per ``on_invalid`` (``'error'`` raises with a count; ``'na'``
  coerces) or a ``shift=`` constant. Inspect/edit pending specs with
  ``pb.derivations`` / ``remove_derived`` / ``replace_derived`` (keyed on ``id``).
  Box–Cox is intentionally deferred (no scipy dependency); derived-on-derived
  chaining and custom expressions are out of scope for v1.
- Named **Calibrations** — standards-given control limits. ``Calibration(label,
  mean, sigma)`` (top-level export) is a frozen value object pinning a chart's
  limits to a known mean and within-subgroup *individual* sigma instead of
  data-derived estimates. Attach by label with ``study.with_calibration(cal)``
  (immutable; returns a new ``Study``) and apply per call via
  ``study.execute(..., calibration=cal_or_label)``. The sigma is applied
  **forward** and used as-is — never run back through c4/d2/b3/b4 to "recover" a
  process sigma. Location charts place limits constant-free in sigma
  (X/residual: ``center ± n_sigma·σ``; Xbar: ``center ± n_sigma·σ/√N``);
  dispersion-statistic charts carry their sampling-distribution constants
  (S: center ``c4(N)·σ``, ``B5(N)·σ … B6(N)·σ``; mR: center ``d2·σ``,
  ``0 … D4·d2·σ``). Plain residuals center at 0 (mean ignored); raw response and
  recentered residuals center at ``calibration.mean``. ``n_sigma`` composes on
  Xbar/S; calibrated X/mR stay 3-sigma and reject a non-default ``n_sigma``.
  Result metadata carries a ``limits_source='calibration'`` badge.
  ``calibration=None`` (the default) is byte-for-byte unchanged. Stratified and
  phased calibration are rejected with a clear error (follow-up).
- ``load_coffee_shop()`` (top-level + ``processbehavior.datasets``): a bundled demo
  dataset of coffee-shop wait times with a deliberate process change at week 8,
  returned as a ready-to-formulate ``ProcessBehavior``. Backs the new
  "Capability — Before vs. After" tutorial (``docs/tutorials/process-capability-before-after.ipynb``),
  which demonstrates time-windowed capability on the dataset.
- Time-windowed capability views: ``study.capability(..., window=(start, end))``
  (and ``assess_capability(..., window=...)``) render *current* capability on a
  before/after subset of the study's declared time axis — an integer sequence or
  a date, half-open ``start <= t < end`` with either bound ``None`` for open. This
  is a view over the immutable analytic dataset: the windowed observed values
  drive current stats and the potential *centering*, while the potential noise
  floor (``σ̂_R2``) stays the full-study pooled R2 — never re-estimated on the
  subset. Thin windows refuse (``n < 8``) or warn (``8 ≤ n < 30``); the chart
  self-describes the window and labels the pooled potential basis. ``window=None``
  (the default) is byte-for-byte unchanged.

### CI / Infrastructure
- Docs notebooks are now executed as part of CI via ``pytest --nbmake``
  on the ubuntu / Python 3.13 cell. The allowlist covers
  ``docs/getting-started/quickstart.ipynb``,
  ``docs/tutorials/process-capability.ipynb``, and
  ``docs/tutorials/loss-function.ipynb`` — the three notebooks known to
  execute cleanly against the current API. Catches future versions of
  the ``chart='Imr'``-style typos that lived in the quickstart for
  months. ``nbmake`` is added to the ``[test]`` extra.
- ``publish.yml`` now runs the full test matrix (3 Python versions × 2
  primary OSes + macOS / 3.13) before building and uploading to PyPI.
  Previously a tag fired ``build → twine check → publish`` with no test
  step, so a regression on ``main`` between the last CI run and the tag
  push could publish a broken wheel.
- ``.pre-commit-config.yaml`` bumped ruff from v0.6.4 to v0.14.9 so
  contributors' pre-commit hooks align with editor-bundled ruff
  versions. Dependabot now tracks ``pre-commit`` revs alongside
  ``pip`` and ``github-actions``. The bump applied ``ruff-format`` to
  87 files (pure whitespace + import-sort changes; no behavior change).
- Six tutorial notebooks in ``docs/tutorials/`` are **not yet** in the
  nbmake gate — ``basic-imr``, ``signal-detection``,
  ``stratified-analysis``, ``xbar-s-analysis``, ``sds-validation``,
  ``sds1-complete-analysis`` — because they reference stale API
  (Study.sds attribute removed, chart names changed, missing
  ``companion=True``). Tracked for rewrite in a follow-up; the CI gate
  catches new bugs while these are getting addressed.

### Added
- Two tutorial notebooks added to the published docs site:
  ``docs/tutorials/process-capability.ipynb`` and
  ``docs/tutorials/loss-function.ipynb``. Both existed on disk but
  weren't in the MyST TOC, so they didn't deploy. Re-executed against
  the current API.
- README **Validation** section explicitly bounding the Bishop-reference
  claim: 280 numerical assertions pass for ADS 1, ADS 2, ADS 3 (the
  three complete-design states). ODS 4–6 detection works; Bishop end-to-
  end coverage of incomplete-grid scenarios is on the 0.2.0 roadmap.
- PyPI keyword expansion: ``variance-analysis``, ``bishop-vas``,
  ``wheeler``, ``design-of-experiments`` — surfaces the library to
  searches for the differentiated methodology.
- GitHub repo description, homepage, and topics set via ``gh repo edit``
  so the repo landing page reflects the differentiator at a glance.
- ``processbehavior.types`` module formalizing the chart-payload contract
  as TypedDicts (``ChartPayload``, ``ChartStatistics``, ``HistogramExtras``,
  ``ChartMetadata``, ``Charts`` type alias). Producer return types in
  ``analysis.py`` and consumer signatures in ``plotting/plotter.py`` and
  ``plotting/renderers.py`` now reference the typed contract so mypy
  catches key drift at edit time. The strata-list and Xbar dead-branch
  bug classes (commits ``f938fdf`` and ``1444b63``) were producer/consumer
  disagreements about this shape; the TypedDicts close that gap.
- ``make_design(state=N, ...)`` is now the canonical synthetic-data
  generator, using the three-state vocabulary (PDS / ODS / ADS). The
  former ``make_sds(sds=N, ...)`` dispatcher has been **removed**
  (it was never released) — all internal call sites migrated. The
  ignored ``n`` parameter is gone from the canonical signature. The
  per-state generators ``make_sds1``..``make_sds6`` remain available.
- ``SDSResult`` is re-exported at the top level so users can
  ``from processbehavior import SDSResult`` for type hints on
  ``study.observed_design_state`` and ``study.analytical_design_state``.
- ``DesignReport.factors_table`` — preferred name for the factor-level
  summary DataFrame. The legacy ``DesignReport.factors`` continues to
  work as an alias; rename avoids confusion with ``Study.factors``
  (which is a ``list[str]`` of column names, not a DataFrame).
- ``Literal[...]`` type hints on stringly-typed enum parameters:
  ``study.execute(chart=, value=, n_mode=)``,
  ``result.plot(theme=)`` and the three sibling plot methods,
  ``MaximumInformationResult.plot(view=)``, and
  ``CapabilityResult.plot(view=)``. IDEs now offer autocomplete for
  the valid values; arbitrary strings still work for custom themes /
  forward-compat.

### Removed
- 5 stale planning docs at the root of ``docs/`` removed from the
  source tree: ``release_gate_0_1_0.md``, ``pre_release_audit.md``,
  ``api_ux_review.md``, ``chart_statistics.md``,
  ``sampling_plan_design.md``. They were never linked from the MyST
  TOC but would have deployed as orphan pages.

### Changed
- ``docs/myst.yml`` navigation label "Sampling Design States" renamed
  to "Design-State Lineage (PDS / ODS / ADS)" to match the
  three-state vocabulary the page now teaches. Copyright bumped to
  ``2025-2026``.
- **`AnalysisResult.get_statistics()` no longer uses the string literal
  `'Varies'` as a sentinel for variable limits.** The affected stats
  fields (`N`, `lpl`, `upl`) are now `None` when limits vary, and a new
  optional `'limits_vary': True` flag is added to the dict so callers
  can detect variable limits without comparing against magic strings.
  Fixes the type-pollution where `stats['upl'] > x` would raise
  `TypeError` on charts with variable cell sizes or phased limits.
  Pre-release breaking change: code that does `stats['upl'] == 'Varies'`
  must migrate to `stats.get('limits_vary')` or `stats['upl'] is None`.
- ``AnalysisResult.get_statistics()`` returns a unified shape across
  every chart type: ``{N, center, lpl, upl}``. The Histogram chart
  additionally carries ``{mean, std, n}`` as extras (``mean`` is an
  alias for ``center``; ``n`` is an alias for ``N``); its ``lpl`` and
  ``upl`` are ``None`` because a histogram has no control limits.
  Previously the Histogram path returned ``{mean, std, n}`` only,
  contradicting the documented contract.
- ``Study.observed_design_state`` now raises ``RuntimeError`` with a
  clear message when accessed on a ``Study`` instance that was not
  built via ``ProcessBehavior.formulate()``. Previously it silently
  returned ``None`` while typed as ``SDSResult``.

### Changed
- **Design-state terminology rolled out across the user surface.** The
  library has always reported three states (Planned / Observed / Analytical
  Design State), but documentation, error messages, and the module
  docstring previously talked about "SDS" as if it were a single concept.
  The user-facing surfaces now consistently use **PDS / ODS / ADS** for
  the states and reserve the integer 1–6 codes for Bishop's reference
  scale (which each state carries on its ``.sds`` field). Touched:
  ``__init__.py`` module docstring, ``README.md``, ``CLAUDE.md``,
  ``docs/intro.md``, ``docs/getting-started/quickstart.ipynb``,
  ``docs/getting-started/installation.md``, ``docs/sds-detection.md``,
  ``docs/user-guide/sds-detection.md``, ``docs/reference/sds_definitions.md``,
  and the public docstrings of ``Study``, ``ProcessBehavior``, ``AnalysisResult``.
- ``Study.__repr__`` lineage block now reads ``PDS (Planned)`` /
  ``ODS (Observed)`` / ``ADS (Analytical)`` and drops the redundant
  ``SDS`` prefix on the integer code.
- User-facing error messages from ``study.execute()`` ("Chart type X is
  not valid for ADS N" instead of "for SDS N") now name the analytical
  state explicitly.
- Internal infrastructure (``SDSResult``, ``SDSRegistry``,
  ``SDSAnalysisPlan``, ``sds_detector.py``, the ``.sds`` field on each
  state, ``test_sds_detector.py``, ``test_sds.py``) is unchanged — a
  deeper internal rename is deferred to a future breaking-change pass.
- Yesterday's [0.1.1] entry that claimed "Sampling Design State (SDS)"
  as canonical terminology is superseded by this pass.

### Fixed
- ``make_sds(sds=4)``, ``make_sds(sds=5)``, ``make_sds(sds=6)`` now
  produce data that classifies as ODS 4 / 5 / 6 respectively. Prior
  implementations were scenario archetypes (single-stream time series,
  nested hierarchy, regime-change sparse sampling) that all classified
  as ODS 2 (complete grid, no replication) regardless of the requested
  state. The new generators produce Bishop's structural incomplete-grid
  shapes via empty (NaN-y) cells. After tidying, ODS 4/5/6 collapse to
  ADS 1/2/3 as Bishop's table prescribes.
- New ``tests/test_design_state_lineage.py`` (21 tests) pins the
  ``make_sds(sds=N) → ODS N`` contract so this drift cannot recur.

## [0.1.1] - 2026-05-18

### Changed
- Synced validation dataset to Tom Bishop's corrected golden copy
  (`validation/PBTESTDATABASE_T100.csv`, replacing `TABVASTESTDATABASE.csv`).
  PM SDS 4/5/6 columns are now labeled so PM SDS N data classifies as ODS N
  (the prior file had a cyclic offset that the old `TestSDSRenumberDrift`
  test class documented). `PM KNOWN` column removed. PM SDS 1/2/3 + factor,
  time, and `PM INERT` columns are byte-identical. The e2e Bishop validator
  remains all-green; `TestSDSRenumberDrift` is replaced by
  `TestSDSColumnLabelAlignment` with two cleaner parametrized invariants.
- Documentation, docstrings, and the quickstart notebook now use the
  canonical "Sampling Design State (SDS)" terminology consistently. The
  prior "Design State (DS)" phrasing in `__init__.py`, the installation
  guide, and the quickstart was non-canonical.
- `study.execute()`, `AnalysisResult.get_chart()`, and `get_statistics()`
  docstring `Raises` blocks now name the actual exception subclasses
  (`ValidationError`, `ChartNotAvailableError`) instead of bare
  `ValueError` / `KeyError`.

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
- Automatic detection of Bishop's six **Sampling Design States (SSDS 1–6)** on raw data
  before NA handling, including incomplete designs (SDS 4–6)
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
- `make_sds(sds=N, seed=...)` synthetic dataset generators for SDS 1–6,
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

[Unreleased]: https://github.com/cnicholas/processbehavior/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/cnicholas/processbehavior/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cnicholas/processbehavior/releases/tag/v0.1.0
