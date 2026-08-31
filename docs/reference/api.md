# API Reference

Complete API documentation for ProcessBehavior.

**Stability (pre-1.0):** everything importable from the top-level `processbehavior`
package — the 33 names in `processbehavior.__all__` — is the public API and is
documented here. Symbols imported from submodules (e.g. `processbehavior.signals`)
are a secondary surface: real and documented below where useful, but more likely to
change before 1.0.

## Entry Points

### formulate() — one-call entry point

```python
from processbehavior import formulate

study = formulate(
    data,                       # DataFrame (or anything ProcessBehavior accepts)
    response,                   # str | ColumnRef
    factors=None,               # list[str | ColumnRef] | None
    time=None,                  # str | ColumnRef | None
    *,
    plan=None,                  # dict | None — the sampling plan, if you have one
    precision=3,                # int — display rounding
    unit_of_analysis=None,      # str | None — what one row represents
)  # -> Study
```

Wraps the data and formulates in one call — equivalent to
`ProcessBehavior(data).formulate(...)`. This is the form the quickstart uses.

### ProcessBehavior

The data-wrapping entry point: cleaning, column access, and derived variables
before formulation.

```python
from processbehavior import ProcessBehavior

pb = ProcessBehavior(df, na_values=None)
```

**Parameters:**
- `df`: pandas DataFrame containing your data
- `na_values`: extra strings to treat as missing, on top of the built-in garbage
  list (`'*'`, `'?'`, `'--'`, `'ND'`, `'BDL'`, …)

**Constructors:**

```python
ProcessBehavior.read_csv(path, na_values=None, **kwargs)
ProcessBehavior.read_excel(path, sheet_name=0, na_values=None, **kwargs)
ProcessBehavior.read_parquet(path, na_values=None, **kwargs)
ProcessBehavior.read_clipboard(na_values=None, **kwargs)
```

Each returns a `ProcessBehavior`; `**kwargs` pass through to the matching pandas
reader.

**Attributes:**
- `.data` — the underlying (cleaned) DataFrame
- `.cols` — column accessor with auto-completion; `pb.cols.weight` is a
  `ColumnRef` usable anywhere a column name is expected
- `.derivations` — tuple of `Derivation` specs attached so far

**Formulation:**

```python
pb.formulate(
    response,                   # str | ColumnRef
    factors=None,               # list[str | ColumnRef] | None
    time=None,                  # str | ColumnRef | None
    plan=None,                  # dict | None
    precision=3,
    unit_of_analysis=None,
)  # -> Study
```

**Derived-variable verbs** (each returns a *new* `ProcessBehavior`; the original
is unchanged):

```python
pb.transform(column, function, *, label=None, shift=None, exponent=None,
             on_invalid='error')                       # -> ProcessBehavior
pb.bin(column, *, method='equal_freq', n=4, breaks=None, bin_labels='range',
       label=None, right=False)                        # -> ProcessBehavior
pb.add_derived(*specs)                                 # -> ProcessBehavior
pb.remove_derived(id)                                  # -> ProcessBehavior
pb.replace_derived(id, spec)                           # -> ProcessBehavior
```

See [Derived Variables](#derived-variables) for the spec types and functions.

### ColumnRef

The object `pb.cols.<name>` returns. Behaves as the column name string wherever
the API takes one; exists so column names auto-complete and typos fail at
attribute access rather than deep inside an analysis.

## Study

Produced by `formulate()`. A frozen description of *what you are analyzing*;
`execute()` computes charts from it.

### Study.execute()

```python
result = study.execute(
    chart=None,          # 'Histogram' | 'Xbar' | 'S' | 'X' | 'mR' | None (recommended)
    by=None,             # list[str] | None — grouping / stratification factors
    value=None,          # 'response' | 'R1'..'R6' | None — what to chart
    recentered=False,    # bool — recentered residual (RCR*) form
    bins=None,           # int | None — Histogram bins
    companion=False,     # bool — also compute the companion chart (see below)
    phased=False,        # bool — phase-wise limits
    n_sigma=3.0,         # float — limit width
    n_mode='actual',     # 'actual' | 'average' — subgroup-size handling
    calibration=None,    # Calibration | str | None — standards-given limits
    stratify_by=None,    # list[str] | None — explicit stratification
)  # -> AnalysisResult
```

**`execute()` returns the requested (or recommended) chart only.** The paired
chart — S for Xbar, mR for X — is opt-in via `companion=True`:

```python
result = study.execute(chart='Xbar', companion=True)
result.all_charts       # ['Xbar', 'S']
```

Chart names are exactly `'Histogram'`, `'Xbar'`, `'S'`, `'X'`, `'mR'`
(case-insensitive). Legacy spellings (`'Imr'`, `'R'`, `'xbar-s'`, …) raise
`ValidationError` with a redirect to the current name.

Residual values `'R1'`–`'R5'` are stored on the study at `formulate()` time.
`'R6'` (and its recentered form `'RCR6'`) is a **request residual**: computed per
`execute()` call from that call's `by=`, and available only on the result that
asked for it. Not every (chart, residual) pair is valid — `study.residual_charts`
lists the valid ones, and e.g. `execute(chart='S', value='R1')` raises.

### Study.why_not()

```python
study.why_not(chart, value=None)   # -> str
```

Human-readable explanation of why a chart (optionally with a residual `value=`)
is or is not available. Answers from the same predicate `execute()` validates
against, so the two cannot disagree.

### Study.design()

```python
report = study.design()   # -> DesignReport
```

### Design-state accessors

```python
study.observed_design_state     # SDSResult — ODS: what the raw data supports
study.analytical_design_state   # SDSResult — ADS: what the analysis runs at
study.plan_design_state         # SDSResult | None — PDS, when plan= was given
study.ads_reason                # str — machine-readable reason ('full_replication', ...)
study.ads_description           # str — human-readable description
```

`SDSResult.sds` is the design state on Bishop's 1–6 reference scale — see the
[Design-State Reference Scale](sds_definitions.md).

### Other Study surface

```python
study.valid_charts              # list[str]
study.recommended_chart         # str
study.residual_charts           # list[(chart, residual)] pairs valid here
study.residuals                 # list[str] — residual codes available
study.support                   # DataFrame: chart × value availability with reasons
study.charts                    # chart availability summary
study.dataset                   # DataFrame copy — immutable after formulate();
                                # execute() never adds columns to it
study.factors / study.time / study.response / study.unit_of_analysis
study.precision
study.derivations               # tuple[Derivation, ...]
study.available_analysis_methods

study.capability(specs=None, *, usl=None, lsl=None, target=None, window=None)
                                # -> CapabilityResult
study.loss_function(target=None)          # -> LossResult
study.maximum_information()               # -> MaximumInformationResult
study.supports_calibration(chart=None, by=None, value=None, recentered=False,
                           bins=None, companion=False, phased=False,
                           n_sigma=3.0, n_mode='actual', stratify_by=None)
                                # -> bool — pure probe; mutates nothing
study.with_calibration(calibration)       # -> Study (new study, original unchanged)
```

### DesignReport

Returned by `study.design()`. Read-only properties describing the sampling plan
versus what was observed:

- Plan/observed counts: `K`, `K_observed`, `K_missing`, `T`, `T_observed`,
  `T_missing`, `N`, `N_observed`, `R`, `R_observed`, `R_missing`
- Structure: `factors`, `factors_table`, `structure_summary`, `min_cell_size`,
  `n_empty_cells`, `unit_of_analysis`, `has_plan`
- Coverage: `coverage`, `plan_adherence`, `missing_combos`, `missing_levels`,
  `extra_combos`, `extra_levels`, `extra_count`
- Classification: `sds_reason`, `sds_reason_detail`, `remediation`

### SDSResult

Dataclass describing one design-state classification:

```python
SDSResult(sds: int, min_cell_size: int, reason=None, n_empty_cells=0)
```

- `.sds` — the state on Bishop's 1–6 reference scale
- `.min_cell_size` — smallest observed (factor × time) cell
- `.reason` — why this classification (an `SDSReasonType`)
- `.n_empty_cells` — empty cells in the expected grid

## AnalysisResult

Returned by `execute()`.

### Charts

```python
result.all_charts              # list[str] — chart names in this result
result.charts                  # dict[str, dict] — raw payloads (see shape below)
result.get_chart(name)         # -> DataFrame — plotted points for one chart
result.iter_charts()           # yields (name, data, statistics) tuples
result.chart_table(chart=None, include_signal_col=True, signal_symbols=True)
                               # -> DataFrame — publication-style table
```

`result.charts` is a nested dict, one entry per chart:

```python
result.charts['Xbar']['data']         # DataFrame
result.charts['Xbar']['statistics']   # dict (or {stratum: dict} when stratified)
result.charts['Xbar']['metadata']     # dict
result.charts['Xbar']['strata']       # per-stratum payloads, when stratified
```

### Statistics — the four-key contract

```python
result.get_statistics(name)        # -> dict
result.statistics(chart, stratum=None)   # -> dict, always the flat four-key form
```

Every chart's statistics dict carries exactly these keys:

| key | meaning |
|-----|---------|
| `N` | subgroup size the limits assume |
| `center` | center line |
| `lpl` | lower process limit |
| `upl` | upper process limit |

When limits vary point-to-point (e.g. varying subgroup sizes with
`n_mode='actual'`), `N`, `lpl` and `upl` are `None` and the dict additionally has
`'limits_vary': True` — read per-point limits from `get_chart()`. Histogram adds
`{'mean', 'std', 'n'}` and has `lpl`/`upl` of `None`. For a stratified result,
`get_statistics()` returns `{stratum: {...}}`; `statistics(chart, stratum=...)`
picks one.

### Residuals

```python
result.has_residuals           # bool
result.residuals               # DataFrame of the *stored* residuals R1–R5
result.get_residual(residual_type)   # -> Series; raises rather than returning empty
result.plot_residuals(residual_type='R1', plot_type='all',
                      theme='processbehavior', width=1200, height=400)
```

`result.residuals` deliberately **excludes R6/RCR6**: R6 depends on the request's
`by=`, so there is no canonical study-level R6. A result executed with
`value='R6'` carries it on `result.dataset` and via `get_residual('R6')`; any
other result raises with instructions on how to get one.

### Signals

```python
result.detect_signals(chart=None, rules=None, config=None, **kwargs)
    # -> SignalResult, or {chart_name: SignalResult} when chart=None
result.get_signals(chart_name=None)    # -> DataFrame of detected signals
```

`rules` accepts `'standard'` (rules 1–4, the default), `'extended'` / `'all'`
(rules 1–8), an explicit list like `['rule_1', 'rule_4']`, or a
[`RuleSet`](#signal-detection-processbehaviorsignals). Rules are filtered by
chart type: **Xbar and S charts evaluate rule 1 only; X and mR charts evaluate
all eight** (aggregation across subgroups invalidates the run-based rules'
independence assumptions).

### Stratification

```python
result.is_stratified           # bool
result.strata                  # list[str]
result.list_strata()           # list[str] (same values; valid inputs to focus())
result.focus(stratum)          # -> FocusedAnalysisResult
```

`focus()` returns a **`FocusedAnalysisResult`** — the same interface as
`AnalysisResult`, filtered to one stratum. It adds `.focused_stratum`, its
`.dataset`/`.residuals` are masked to the stratum, and `.is_stratified` is
`False`. (`FocusedAnalysisResult` is created by `focus()`, not constructed
directly, and is not a top-level export.)

### Plotting, reporting, export

```python
result.plot(
    chart=None, facet=False, ncols=2,
    highlight_signals=True, show_limits=True, show_limit_values=True,
    show_zones=False, show_rules=False, show_stats=False,
    theme='processbehavior', width=1000, height=None, aspect_ratio=None,
    title=None, xaxis_title=None, yaxis_title=None,
    shared_yaxis=True, yaxis_padding=0.05, vertical_spacing=0.15,
)   # -> ControlChartFigure

result.plot_effects(effect_type='factor', theme='processbehavior',
                    width=800, height=500)
result.report(filepath, include_charts=True, include_residuals=True,
              include_effects=True, include_summary=True,
              theme='processbehavior', width=1200, title=None)
result.to_excel(filepath, **kwargs)    # -> list[Path] — files written
                                       # (export_html=False by default)
```

`ControlChartFigure` wraps a plotly figure; call `.show()` in a notebook or use
it anywhere a plotly figure works.

### Effects

```python
result.has_effects / result.has_interactions   # bool
result.effects / result.interactions           # dict | None
result.summary                                 # dict — comprehensive metadata
result.dataset                                 # DataFrame copy; for a request-
                                               # residual result it carries that
                                               # request's R6/RCR6 columns
```

## Analysis Result Types

### SpecLimits

```python
from processbehavior import SpecLimits
SpecLimits(usl=None, lsl=None, target=None)
```

Specification limits for capability analysis; pass to `study.capability(specs=...)`
or use the keyword shortcuts `capability(usl=..., lsl=..., target=...)`.

### Calibration

```python
from processbehavior import Calibration
Calibration(label: str, mean: float, sigma: float)
```

Named standards-given control limits. Use via
`execute(..., calibration=...)` or `study.with_calibration(...)`; probe
compatibility first with `study.supports_calibration(...)`.

### CapabilityResult

From `study.capability()`. Fields include `pp`, `ppk` (with `ppk_lower`/`ppk_upper`),
`cp`, `cpk` (with bounds), `sigma_hat`, `sigma_hat_r2`, `y_bar`, `s`, `n`,
`z_lower`/`z_upper`, out-of-spec counts and percentages
(`n_below_lsl`, `n_above_usl`, `n_outside`, `pct_below_lsl`, `pct_above_usl`,
`pct_outside`), potential-performance counterparts, `stability_warning`, and the
capability `window` when one was requested.

```python
cap.as_dict(round_to=None)   # -> dict
cap.plot(values=None, *, theme=None, show_potential=True,
         view='current',            # 'current' | 'potential'
         paired=False, x_label=None, nbins=None, histnorm='',
         width=900, height=500, title=None)
```

### LossResult

From `study.loss_function()`. Taguchi loss decomposition: `centering`,
`unexplained`, `pdc`, `time`, `interaction`, `total`, with `pct_*`
counterparts, `pdc_by_factor`, and the `target` used (`target_is_default` says
whether it was derived).

```python
loss.as_dict(round_to=None)
loss.plot(*, structured=False, orientation='vertical', theme=None,
          width=700, height=400, title=None)
```

### MaximumInformationResult

From `study.maximum_information()`. Fields: `n`, `r2_mean`, `r2_mR`,
`sigma_hat`, `upl`, `lpl`, `n_signals`.

```python
mi.as_dict(round_to=None)
mi.plot(*, view='combined',        # 'combined' | 'xmr' | 'histogram'
        bins=10, theme=None, width=900, height=700, title=None)
```

## Derived Variables

Specs and functions behind `pb.transform()` / `pb.bin()`:

```python
from processbehavior import (Derivation, EvalResult, ValidationResult,
                             evaluate, validate,
                             derivations, remove_derived, replace_derived)

Derivation(family, column, function, label=None, params={}, fitted={}, id=...)
    # family: 'transform' | 'bin'; function: e.g. 'log', 'sqrt', 'equal_freq'

evaluate(spec, column)            # -> EvalResult(values, n_invalid, invalid_index,
                                  #               fitted, message)
validate(spec, dataset, existing_names=None)   # -> ValidationResult(ok, issues)

derivations(pb)                   # -> tuple[Derivation, ...]
remove_derived(pb, id)            # functional forms of the ProcessBehavior verbs
replace_derived(pb, id, spec)
```

The word *derived* belongs to this module: transform/binning specs. (VAS
residuals are "stored" or "request" residuals, never "derived".)

## Signal Detection (`processbehavior.signals`)

Secondary surface — import from the subpackage:

```python
from processbehavior.signals import SignalConfig, RuleSet, SignalResult
```

### SignalResult

Returned by `result.detect_signals()`.

- Status: `.count`, `.has_signals`, `.is_partial`, `.evaluation_status`,
  `.rules_skipped`
- Views: `.violations` (list), `.by_rule` (dict), `.by_observation` (dict),
  `.flagged_observations`, `.summary`
- Filters: `.get_rule_violations(rule_name)`, `.get_observation_violations(obs_id)`
- Export: `.to_dataframe()`, `.to_excel(filepath)`, `.to_json(filepath)`

### RuleSet

Fluent builder for a custom rule collection:

```python
rules = (RuleSet()
         .beyond_limits()               # rule 1
         .run(length=8)                 # rule 4
         .trend(length=6, direction='both'))
result.detect_signals(chart='X', rules=rules)
```

Builders: `beyond_limits()`, `zone_a(consecutive=2, window=3)`,
`zone_b(consecutive=4, window=5)`, `run(length=8)`, `trend(length=6,
direction='both')`, `oscillation(length=14)`, `avoiding_center(length=8)`,
`reduced_variation(length=15)`, `custom(name, detector, min_observations=1)`.
Also `get_rules()` and `to_config()`.

### SignalConfig

```python
SignalConfig(enabled_rules='default',   # 'default'|'standard'|'extended'|'all'|list
             min_observations=..., ignore_first_n=..., ignore_last_n=...,
             use_vectorized=...)
config.get_rules_for_chart(chart_type)  # applies the per-chart filtering
```

Per-chart applicability: Xbar and S evaluate `rule_1` only; X and mR evaluate
rules 1–8.

## Plotting & Themes

```python
from processbehavior import ChartTheme, get_theme, list_themes, register_theme

list_themes()    # ['processbehavior', 'ggplot', 'minimal', 'dark', 'publication']
get_theme(name)                  # -> ChartTheme
register_theme(theme)            # make a custom theme available by name
```

### ChartTheme

Dataclass of every visual knob. Construct with any subset of fields:

```python
theme = ChartTheme(name='corporate', data_color='#1f77b4', signal_color='#d62728')
register_theme(theme)
result.plot(theme='corporate')
```

Fields and defaults:

| group | fields (default) |
|-------|------------------|
| identity | `name` (`'processbehavior'`) |
| data | `data_color` (`'steelblue'`), `data_marker_size` (5), `data_line_width` (1.0), `data_opacity` (1.0) |
| limits & center | `ucl_color` (`'red'`), `lcl_color` (`'red'`), `center_color` (`'#2E8B57'`), `limit_line_dash` (`'dash'`), `limit_line_width` (1.5), `center_line_width` (1.5) |
| signals | `signal_color` (`'red'`), `signal_marker_size` (5), `signal_marker_symbol` (`'circle'`), `signal_marker_line_width` (0.5), `signal_marker_line_color` (`'darkred'`), `pattern_signal_color` (`'#FF8C00'`) |
| limit summary | `limit_summary_color` (`'#333333'`), `limit_summary_bgcolor` (`'rgba(255,255,255,0.7)'`) |
| zones | `zone_a_color` (`'#FFB3B3'`), `zone_b_color` (`'#FFFFB3'`), `zone_c_color` (`'#B3FFB3'`), `zone_opacity` (0.15) |
| canvas | `plot_bgcolor` (`'white'`), `paper_bgcolor` (`'white'`), `grid_color` (`'#E5E5E5'`), `grid_width` (1.0), `show_grid` (True), `axis_line_color` (`'#999999'`), `axis_line_width` (1.0), `show_axis_line` (True) |
| type | `font_family` (`'Arial, sans-serif'`), `font_size` (12), `font_color` (`'#333333'`), `title_font_size` (16), `title_font_color` (`'#222222'`), `axis_title_font_size` (12), `annotation_font_size` (10) |
| facets | `facet_marker_size` (5), `facet_line_width` (1.5) |
| lanes | `lane_boundary_color` (`'#888888'`), `lane_boundary_dash` (`'dot'`), `lane_boundary_width` (1.0), `lane_boundary_annotation_size` (8) |
| stats box | `stats_box_bgcolor` (`'rgba(255, 255, 255, 0.9)'`), `stats_box_bordercolor` (`'#CCCCCC'`), `stats_box_borderwidth` (1), `stats_box_font_size` (10), `stats_box_font_color` (`'#333333'`) |

## Datasets

```python
from processbehavior import make_design, load_coffee_shop

make_design(state,               # int 1-6 — target design state
            K1=3, K2=2,          # levels of factor 1 / factor 2
            T=8,                 # time points
            seed=None, **kwargs) # -> DataFrame with columns
                                 #    'time', 'factor 1', 'factor 2', 'y'

load_coffee_shop()               # -> ProcessBehavior — the coffee-shop demo data
```

`make_design` is the canonical synthetic generator: it builds data whose
*structure* classifies to the requested design state (per-state `**kwargs` such
as `p_replicated=`, `n_when_replicated=`, `n_min=`, `n_max=` shape the cells).

## Constants (`processbehavior.spc_constants`)

Secondary surface. The control-chart constants are functions of subgroup size:

```python
from processbehavior.spc_constants import c4, b3, b4, VALID_BASE_CHARTS

c4(n)                        # bias-correction factor for s
b3(n, sigma_multiplier=3)    # S-chart lower-limit factor
b4(n, sigma_multiplier=3)    # S-chart upper-limit factor
VALID_BASE_CHARTS            # {'Histogram', 'Xbar', 'S', 'X', 'mR'}
```

## Exceptions

```python
from processbehavior import (
    ProcessBehaviorError, ValidationError, ColumnNotFoundError,
    FactorNotFoundError, ChartNotAvailableError, CalibrationNotSupportedError,
    ProcessBehaviorWarning,
)
```

Hierarchy:

```
Exception
└── ProcessBehaviorError
    └── ValidationError            (also subclasses ValueError)
        ├── ColumnNotFoundError
        ├── FactorNotFoundError
        ├── ChartNotAvailableError
        └── CalibrationNotSupportedError

UserWarning
└── ProcessBehaviorWarning
```

Catching `ProcessBehaviorError` catches everything the library raises
deliberately; `ValidationError` (a `ValueError`) covers every bad-request case,
including unavailable charts and unsupported calibrations.

## Quick reference — the 33 top-level exports

| area | names |
|------|-------|
| Entry & core | `ProcessBehavior`, `formulate`, `ColumnRef`, `Study`, `DesignReport`, `SDSResult`, `AnalysisResult` |
| Analyses | `SpecLimits`, `Calibration`, `CapabilityResult`, `LossResult`, `MaximumInformationResult` |
| Derived variables | `Derivation`, `EvalResult`, `ValidationResult`, `evaluate`, `validate`, `derivations`, `remove_derived`, `replace_derived` |
| Exceptions | `ProcessBehaviorError`, `ValidationError`, `ColumnNotFoundError`, `FactorNotFoundError`, `ChartNotAvailableError`, `CalibrationNotSupportedError`, `ProcessBehaviorWarning` |
| Themes | `ChartTheme`, `get_theme`, `list_themes`, `register_theme` |
| Datasets | `make_design`, `load_coffee_shop` |
