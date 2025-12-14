# API Reference

Complete API documentation for ProcessBehavior.

## Core Classes

### ProcessBehavior

The main entry point for working with process data.

```python
from processbehavior import ProcessBehavior

pdf = ProcessBehavior(
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
study = pdf.formulate(
    response: str,              # Response variable (required)
    factors: list[str] = None,  # Grouping variables (optional)
    time: str = None,           # Time/sequence variable (optional)
    precision: int = 3          # Decimal places (optional)
) -> Study
```

Creates a Study object with automatic SDS detection.

---

### Study

Represents a formulated study ready for analysis.

```python
study = pdf.formulate(...)
```

**Properties - Formulation:**
- `.response`: Response variable name
- `.factors`: List of factor variable names
- `.time`: Time variable name
- `.precision`: Decimal precision

**Properties - SDS:**
- `.sds`: Sampling Design State (0-6)
- `.sds_name`: Human-readable SDS name
- `.sds_description`: Detailed SDS explanation

**Properties - Charts:**
- `.valid_charts`: List of valid chart types
- `.recommended_chart`: Best chart for this SDS
- `.residual_charts`: Available VAS residual charts
- `.charts`: Accessor for chart type auto-completion

**Properties - Data:**
- `.dataset`: Full prepared DataFrame with residuals

**Methods:**

#### execute()

```python
result = study.execute(
    chart: str = None,        # Chart type (default: recommended)
    recentered: bool = False  # Re-center residuals on original scale
) -> AnalysisResult
```

#### why_not()

```python
explanation = study.why_not(chart: str) -> str
```

Explains why a chart type is invalid for this SDS.

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
- `.sds`: Sampling Design State
- `.sds_info`: Complete SDS details

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

**Methods - Visualization:**

```python
fig = result.plot(
    chart: str = None,
    facet: bool = False,
    facet_by: str = None,
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
- `.sds`: State number (0-6)
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

Calculates main effects (requires statsmodels).

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
'Xbar', 'S', 'Imr', 'R'

# Residual charts
'R2_S', 'R2_Imr', 'R3_Imr', 'R4_Imr', 'R5_Imr'

# Stratified charts (dynamically named)
'Imr_Lane_A', 'Imr_Lane_B', ...
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

ProcessBehavior raises standard Python exceptions:

- `ValueError`: Invalid parameter values
- `KeyError`: Unknown chart or column names
- `TypeError`: Wrong parameter types

Error messages are descriptive and suggest fixes.
