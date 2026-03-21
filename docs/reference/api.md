# API Reference

Complete API documentation for ProcessBehavior.

## Core Classes

### ProcessBehavior

The main entry point for working with process data.

```python
from processbehavior import ProcessBehavior

pb = ProcessBehavior(
    df: pd.DataFrame,
    na_values: list[str] = ['*', '?', 'ND', 'BDL', 'NA', 'N/A', 'n/a',
                            '<LOD', '>LOQ', 'TNTC', 'QNS', '--']
)
```

**Parameters:**
- `df`: pandas DataFrame containing your data
- `na_values`: List of strings to treat as missing values (automatically cleaned)

**Attributes:**
- `.data`: The underlying pandas DataFrame
- `.columns`: Accessor for column auto-completion

**Methods:**

#### formulate()

```python
study = pb.formulate(
    response: str | ColumnRef,              # Response variable (required)
    factors: list[str | ColumnRef] = None,  # Grouping variables (optional)
    time: str | ColumnRef = None,           # Time/sequence variable (optional)
    plan: dict = None,                      # Sampling plan (optional)
    precision: int = 3,                     # Decimal places (optional)
    unit_of_analysis: str = None            # Informational metadata (optional)
) -> Study
```

Creates a Study object with automatic SDS detection.

**Parameters:**

- `response`: The measurement variable to analyze. Supports `pb.cols` auto-completion.
- `factors`: Grouping factors defining rational subgroups. Cannot be used with `plan`.
- `time`: Time/sequence variable for ordering observations.
- `plan`: Sampling plan specifying expected factor structure. Must contain a `'factors'` key mapping column names to expected levels. Optionally include `'T'` (planned time points) and `'N'` (planned cell size). Cannot be used with `factors`.
  ```python
  plan={
      'factors': {'Lane': [1,2,3,4], 'Phase': [1,2,3]},
      'T': 10,
      'N': 2
  }
  ```
- `precision`: Decimal places for calculated statistics.
- `unit_of_analysis`: Informational metadata describing what is being measured (e.g., `'filled cup'`). Does not affect calculations.

---

### Study

Represents a formulated study ready for analysis.

```python
study = pb.formulate(...)
```

**Properties - Formulation:**
- `.response`: Response variable name
- `.factors`: List of factor variable names
- `.time`: Time variable name
- `.precision`: Decimal precision

**Properties - SDS:**
- `.plan_design_state`: `SDSResult | None` -- what was planned (None if no plan)
- `.observed_design_state`: `SDSResult` -- what was observed in raw data
- `.analytical_design_state`: `SDSResult` -- what is analyzable after tidying
- `.sds_reason`: ADS-derived machine token (e.g., `'full_replication'`)
- `.sds_description`: ADS-derived human prose description

**Properties - Charts:**
- `.valid_charts`: List of valid chart types ('Xbar', 'S', 'XmR', 'R')
- `.recommended_chart`: Best chart for this SDS
- `.charts`: Accessor for valid chart types (e.g., `study.charts.Xbar`)
- `.residuals`: Accessor for available residuals (e.g., `study.residuals.R2`)
- `.residual_charts`: List of residual+chart combinations (internal)

**Properties - Data:**
- `.dataset`: Full prepared DataFrame with residuals
- `.support`: DataFrame with chart availability matrix (chart, category, available, recommended, reason, question)

**Methods:**

#### execute()

```python
result = study.execute(
    chart: str = None,           # Chart type: 'Xbar', 'S', 'XmR', 'R'
    by: list[str] = None,        # Grouping/stratification (subset of factors)
    value: str = None,           # What to chart: None (response) or 'R1'-'R5'
    recentered: bool = False,    # Re-center residuals on original scale
    bins: int = None,            # Number of bins for histogram charts
    companion: bool = False      # Return companion charts (Xbar+S or XmR+R)
) -> AnalysisResult
```

**Parameters:**

- `chart`: Base chart type. One of `'Xbar'`, `'S'`, `'XmR'`, `'R'`.
- `by`: Controls grouping/stratification:
  - `None`: Default for chart type (full factors for Xbar/S, required for XmR with factors)
  - `[]`: Collapse all factors
  - `['factor']`: Aggregate/stratify by single factor
  - `['f1', 'f2']`: Aggregate/stratify by multiple factors
- `value`: What to plot:
  - `None`: Chart response variable (default)
  - `'R1'` to `'R5'`: Chart the specified VAS residual
- `recentered`: If True and using residuals, re-center on original scale
- `bins`: Number of bins for histogram charts
- `companion`: If True, returns both charts in a pair (Xbar+S or XmR+R). Either chart in the pair triggers the pair (e.g., `chart='S', companion=True` returns Xbar+S)

**Examples:**

```python
# Xbar chart aggregated by all factors (default)
result = study.execute(chart='Xbar')

# Xbar chart aggregated by single factor
result = study.execute(chart='Xbar', by=['factor 1'])

# XmR chart stratified by factor (one chart per level)
result = study.execute(chart='XmR', by=['lane'])

# Chart R5 residuals on Xbar
result = study.execute(chart='Xbar', value='R5')

# Recentered R4 residuals on stratified XmR
result = study.execute(chart='XmR', by=['lane'], value='R4', recentered=True)
```

#### why_not()

```python
explanation = study.why_not(chart: str) -> str
```

Explains why a chart type is invalid for this SDS.

#### design()

```python
report = study.design() -> DesignReport
```

Returns a `DesignReport` comparing the sampling plan to observed data. Works with or without a sampling plan -- without a plan, it reports observed structure only.

---

### DesignReport

Returned by `study.design()`. Compares sampling plan to observed data structure.

**Properties:**
- `.factors`: DataFrame with columns: factor, planned, observed, missing_levels, extra_levels
- `.K`: Planned number of RSG groups
- `.K_observed`: Observed number of RSG groups
- `.K_missing`: Number of missing RSG groups
- `.T`: Planned time points
- `.T_observed`: Observed time points
- `.T_missing`: Number of missing time points
- `.R`: Planned total cells (K x T)
- `.R_observed`: Observed total cells
- `.R_missing`: Number of missing cells
- `.N`: Planned cell size
- `.N_observed`: Observed cell size as (min, median, max) tuple
- `.missing_levels`: Dict of factor -> list of levels in plan but not observed
- `.extra_levels`: Dict of factor -> list of levels observed but not in plan
- `.missing_combos`: List of RSG groups in plan but not observed
- `.extra_combos`: List of RSG groups observed but not in plan
- `.min_cell_size`: Minimum observations per cell (from SDS detection)
- `.n_empty_cells`: Count of cells with zero observations
- `.coverage`: Ratio of observed to planned cells (0.0-1.0, None without plan)
- `.remediation`: Actionable hint for improving design (or None)
- `.plan_adherence`: Summary of how well data matches plan
- `.structure_summary`: Summary of structure discrepancies
- `.has_plan`: Boolean -- whether a sampling plan was provided
- `.sds_reason`: SDS classification reason from detector
- `.unit_of_analysis`: The fundamental entity being measured

---

### AnalysisResult

Contains all results from an analysis.

```python
result = study.execute()
```

**Properties - Charts:**
- `.all_charts`: List of chart names
- `.charts`: Dict of chart name → (data, statistics)

**Properties - Data:**
- `.dataset`: Full analysis DataFrame
- `.residuals`: DataFrame with R1-R5 (if available)
- `.effects`: Dict with 'k_effects', 't_effects'
- `.interactions`: Dict of interaction terms

**Properties - Flags:**
- `.has_residuals`: Boolean
- `.has_effects`: Boolean
- `.has_interactions`: Boolean

**Properties - SDS:**
- `.analytical_sds`: Analytical Sampling Design State (int)
- `.analytical_sds_info`: Complete analytical SDS details

**Properties - Summary:**
- `.summary`: Dict with all metadata

**Methods - Chart Access:**

```python
# Get chart data as DataFrame
data = result.get_chart(chart_name: str) -> pd.DataFrame

# Get chart statistics
stats = result.get_statistics(chart_name: str) -> dict

# Get summary table
table = result.chart_table(chart_name: str) -> pd.DataFrame

# Get specific residual
r4 = result.get_residual('R4') -> pd.Series

# Get points beyond limits
signals = result.get_signals(chart_name: str) -> pd.DataFrame
```

**Methods - Stratified Analysis:**

```python
# List available strata
strata = result.list_strata() -> list[str]

# Get stratified chart
data = result.get_stratified_chart(stratum: str) -> pd.DataFrame

# Get all stratified charts
charts = result.get_stratified_charts() -> dict[str, pd.DataFrame]

# Iterate through charts
for name, data, stats in result.iter_charts():
    ...
```

**Methods - Focus:**

```python
# Focus on a single stratum from a stratified analysis
focused = result.focus(stratum: str) -> AnalysisResult
```

Returns a new `AnalysisResult` containing only data for the specified stratum. Enables drill-down: `result.focus('Lane_1').plot()`.

**Methods - Visualization:**

```python
fig = result.plot(
    chart: str = None,
    facet: bool = False,
    ncols: int = 2,
    show_limits: bool = True,
    show_zones: bool = False,
    show_signals: bool = False,
    show_rules: bool = False,
    show_stats: bool = False,
    template: str = 'processbehavior',
    width: int = 1000,
    height: int = None,
    title: str = None
) -> ControlChartFigure
```

**Methods - Signal Detection:**

```python
signals = result.detect_signals(
    chart: str = None,       # Specific chart or all
    rules: str|RuleSet = 'standard'  # 'standard', 'extended', or custom
) -> SignalResult
```

**Methods - Export:**

```python
result.to_excel(
    filepath: str,
    include_summary: bool = True,
    include_charts: bool = True,
    include_residuals: bool = True,
    include_effects: bool = True,
    include_interactions: bool = True,
    include_full_dataset: bool = False,
    format_cells: bool = True,
    include_chart_images: bool = True,
    export_html: bool = True
)
```

---

## Signal Detection

### SignalResult

Result from signal detection.

```python
signals = result.detect_signals()
```

**Properties:**
- `.has_signals`: Boolean - any signals detected
- `.count`: Total number of signal violations
- `.violations`: DataFrame with all violations
- `.flagged_observations`: Set of observation indices with signals
- `.summary`: DataFrame with counts by rule
- `.by_rule`: Dict of rule → violations

### RuleSet

Builder for custom rule configurations.

```python
from processbehavior.signals import RuleSet

rules = (
    RuleSet()
    .beyond_limits()           # Rule 1
    .zone_a(consecutive=2)     # Rule 2: 2 of 3 in Zone A
    .zone_b(consecutive=4)     # Rule 3: 4 of 5 in Zone B+
    .run(length=8)             # Rule 4: 8+ same side
    .trend(length=6)           # Rule 5: 6+ trending
    .oscillation(length=14)    # Rule 6: 14+ alternating
    .hugging_center(length=15) # Rule 7: 15+ in Zone C
    .avoiding_center(length=8) # Rule 8: 8+ avoiding Zone C
    .build()
)
```

### SignalConfig

Configuration for signal detection.

```python
from processbehavior.signals import SignalConfig

config = SignalConfig(
    enabled_rules: str|list = 'standard',  # 'standard', 'extended', or list
    min_observations: int = 20
)
```

---

## Plotting

### ControlChartFigure

Wrapper around Plotly figure with convenience methods.

```python
fig = result.plot()
```

**Methods:**
- `.show()`: Display in browser
- `.save_html(path, include_plotlyjs=True)`: Save as HTML
- `.save_image(path)`: Save as PNG/PDF/SVG (requires kaleido)
- `.figure`: Access underlying Plotly Figure

### ChartTheme

Custom theme configuration.

```python
from processbehavior.plotting import ChartTheme, register_theme

theme = ChartTheme(
    name: str,
    data_color: str = 'blue',
    signal_color: str = 'red',
    center_color: str = 'green',
    limit_color: str = 'red',
    data_marker_size: int = 8,
    signal_marker_size: int = 12,
    center_line_width: float = 1.5,
    limit_line_width: float = 1.0,
    limit_line_dash: str = 'dash',
    zone_a_color: str = 'rgba(...)',
    zone_b_color: str = 'rgba(...)',
    zone_c_color: str = 'rgba(...)',
    font_family: str = 'Arial',
    font_size: int = 12,
    title_font_size: int = 16,
    plot_bgcolor: str = 'white',
    paper_bgcolor: str = 'white',
    gridcolor: str = 'lightgray'
)

register_theme(theme)
```

### Theme Functions

```python
from processbehavior.plotting import list_themes, get_theme

# List available themes
themes = list_themes()  # ['processbehavior', 'minimal', 'dark', 'ggplot']

# Get theme object
theme = get_theme('dark')
```

---

## SDS Detection

### SDSRegistry

Automatic SDS detection (used internally).

```python
from processbehavior import SDSRegistry

detector = SDSRegistry()
plan = detector.detect(
    df: pd.DataFrame,
    factors: list[str],
    time: str
) -> SDSAnalysisPlan
```

### SDSAnalysisPlan

Complete specification for an SDS.

**Properties:**
- `.sds`: State number (0-6) (on SDSAnalysisPlan objects)
- `.name`: Human-readable name
- `.description`: Detailed explanation
- `.has_factors`: Boolean
- `.has_time`: Boolean
- `.has_replication`: Boolean
- `.valid_charts`: List of valid chart types
- `.recommended_chart`: Best chart type
- `.invalid_charts`: Charts that won't work
- `.vas_residuals_supported`: Boolean
- `.residuals_available`: List of available residuals
- `.residual_calculation_method`: How R2 is calculated
- `.main_effects_supported`: Boolean
- `.interaction_effects_supported`: Boolean
- `.supports_stratification`: Boolean
- `.typical_use_cases`: List of example scenarios
- `.limitations`: Known limitations
- `.bishop_reference`: Reference to Wheeler/Bishop text

---

## Data Preparation

### DataPreparation

Handles data cleaning and preparation (used internally).

```python
from processbehavior import DataPreparation, DataPrepConfig

config = DataPrepConfig({
    'response_var': 'weight',
    'rsg_vars': ['lane'],
    'time_var': 'batch',
    'round_to': 3
})

prep = DataPreparation(config)
prepared_df = prep.prepare(df)
```

---

## Calculation Classes

### ResidualCalculator

Calculates VAS residuals.

```python
from processbehavior import ResidualCalculator

calc = ResidualCalculator()
df_with_residuals = calc.calculate(df, factors, time, response)
```

### EffectsCalculator

Calculates main effects for factorial designs.

```python
from processbehavior import EffectsCalculator

calc = EffectsCalculator()
effects = calc.calculate(df, factors, time, response)
```

---

## Constants

### SPC Constants

Statistical constants for control charts.

```python
from processbehavior.spc_constants import (
    get_c4,    # c4 constant for n
    get_A3,    # A3 constant for Xbar limits
    get_B3,    # B3 constant for S lower limit
    get_B4,    # B4 constant for S upper limit
    get_d2,    # d2 constant for range
    get_D3,    # D3 constant for R lower limit
    get_D4     # D4 constant for R upper limit
)

c4 = get_c4(5)  # c4 for n=5
```

---

## Type Definitions

### Chart Types

Valid chart type strings:

```python
# Standard charts
'Xbar', 'S', 'XmR', 'R'

# Effects and interaction charts (passed to result.plot())
'Effects'            # All main effects (factor + time) combined
'MainEffects'        # Factor main effects only
'TimeEffects'        # Time main effects only
'TimeInteraction'    # Factor x time interaction
'FactorInteraction'  # Factor x factor interaction (requires 2+ factors)
```

To chart residuals, use the `value` parameter:

```python
# Chart R5 residuals on Xbar
study.execute(chart='Xbar', value='R5')

# Chart R4 residuals on stratified XmR
study.execute(chart='XmR', by=['lane'], value='R4')
```

For stratified charts, use the `by` parameter:

```python
# Stratify by single factor
result = study.execute(chart='XmR', by=['lane'])
# Access strata: result.charts['XmR']['strata']  # ['A', 'B', 'C', 'D']
```

### Rule Types

Valid rule configuration:

```python
# Preset strings
'standard'  # Rules 1-4
'extended'  # Rules 1-8
'all'       # Same as extended

# List of rule names
['rule_1', 'rule_2', 'rule_5']

# RuleSet object
RuleSet().beyond_limits().run().build()
```

---

## Exceptions

ProcessBehavior provides a custom exception hierarchy for better error handling.

### Exception Hierarchy

```
ProcessBehaviorError (base)
├── ValidationError
│   └── ColumnNotFoundError
└── ChartNotAvailableError
```

### ProcessBehaviorError

Base exception for all processbehavior errors. Catch this to handle any library error.

```python
from processbehavior import ProcessBehaviorError

try:
    result = study.execute()
except ProcessBehaviorError as e:
    print(f"Analysis failed: {e}")
```

### ValidationError

Raised when input data, parameters, or configuration is invalid.

```python
from processbehavior import ValidationError

try:
    study = pb.formulate(response='nonexistent')
except ValidationError as e:
    print(f"Check your parameters: {e}")
```

### ColumnNotFoundError

Raised when a required column is missing from the DataFrame. Subclass of `ValidationError`.

```python
from processbehavior import ColumnNotFoundError

try:
    study = pb.formulate(response=pb.cols.missing_column)
except ColumnNotFoundError as e:
    print(f"Column not found: {e}")
    print(f"Available: {pb.data.columns.tolist()}")
```

### ChartNotAvailableError

Raised when a chart type is invalid or unavailable for the current SDS.

```python
from processbehavior import ChartNotAvailableError

try:
    result = study.execute(chart='Xbar')
except ChartNotAvailableError as e:
    print(f"Chart not available: {e}")
    print(f"Valid charts: {study.valid_charts}")
    print(f"Recommended: {study.recommended_chart}")
```

### Catching Errors by Category

```python
from processbehavior import (
    ProcessBehaviorError,
    ValidationError,
    ColumnNotFoundError,
    ChartNotAvailableError
)

try:
    result = study.execute(chart='Xbar')
except ChartNotAvailableError as e:
    # Chart-specific handling
    print(f"Try one of: {study.valid_charts}")
except ColumnNotFoundError as e:
    # Column-specific handling
    print(f"Check column names: {pb.cols}")
except ValidationError as e:
    # General validation errors
    print(f"Invalid input: {e}")
except ProcessBehaviorError as e:
    # Catch-all for any library error
    print(f"Unexpected error: {e}")
```
