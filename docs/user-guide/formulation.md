# Study Formulation

The `formulate()` method is the heart of ProcessBehavior. It enables the analyst to specify the key inputs defined during problem forumlation to create a structured study.  The system automatically detects the sampling design and prepares the analysis.

## The Formulation API

```python
from processbehavior import ProcessBehavior

pb = ProcessBehavior(df)

study = pb.formulate(
    response=pb.cols.weight,        # Required: measurement variable
    factors=[pb.cols.lane],         # Optional: grouping variables
    time=pb.cols.batch,             # Optional: time/sequence variable
    precision=3                         # Optional: decimal places
)
```

## IDE Auto-Completion

One of ProcessBehavior's key features is **IDE auto-completion** for column names. This eliminates typos and makes exploration faster.

### Column Auto-Completion

After creating a ProcessBehavior, access columns via the `.cols` accessor:

```python
pb = ProcessBehavior(df)

# Type pb.cols. and your IDE will show all available columns
pb.cols.weight      # Instead of 'weight' string
pb.cols.lane        # Instead of 'lane' string
pb.cols.batch       # Instead of 'batch' string
```

This works in:
- VS Code with Pylance
- PyCharm
- Jupyter notebooks (with Tab completion)
- Any LSP-compatible editor

### Chart Type Auto-Completion

After formulation, the `study.charts` accessor provides auto-completion for valid chart types:

```python
study = pb.formulate(...)

# Type study.charts. and see only valid charts for your SDS
study.charts.Xbar        # Available if SDS supports it
study.charts.XmR         # Always available
study.residuals.R4       # VAS residual (pass to value=)
```

## Parameter Details

### response (required)

The measurement variable to analyze.

```python
# Using auto-completion (recommended)
response=pb.cols.measurement

# Using string (still works)
response='measurement'
```

### factors (optional)

A list of categorical variables that define subgroups. These become the "rational subgroups" in Wheeler's terminology.

```python
# Single factor
factors=[pb.cols.operator]

# Multiple factors (creates combined subgroups)
factors=[pb.cols.machine, pb.cols.shift]
```

When multiple factors are specified, they're combined into a single grouping variable (e.g., `Machine_A_Shift_1`).

### time (optional)

The variable that defines time ordering. This enables time-series analysis and certain signal detection rules.

```python
time=pb.cols.batch
time=pb.cols.timestamp
time=pb.cols.sequence
```

### precision (optional, default=3)

Number of decimal places for calculated statistics.

```python
precision=2  # Round to 2 decimal places
precision=4  # Higher precision for sensitive measurements
```

## What Formulation Does

When you call `formulate()`, ProcessBehavior:

1. **Validates data** - Checks for required columns and data types
2. **Cleans data** - Handles missing values and garbage characters
3. **Detects SDS** - Determines the Sampling Design State (0-6)
4. **Calculates means** - Computes Y̅, Y̅_k, Y̅_t, Y̅_kt
5. **Computes residuals** - Calculates R1-R5
6. **Determines valid charts** - Lists which analyses are appropriate

## The Study Object

Formulation returns a `Study` object with rich information:

```python
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)

# SDS information
print(study.observed_design_state)    # SDSResult(sds=1, reason='full_replication', ...)
print(study.analytical_design_state) # SDSResult (what is analyzable after tidying)
print(study.sds_reason)              # "full_replication" (ADS-derived machine token)
print(study.sds_description)         # Detailed explanation (ADS-derived human prose)

# Chart recommendations
print(study.valid_charts)       # ['Xbar', 'S', 'XmR']
print(study.recommended_chart)  # 'Xbar'
print(study.residuals)          # StudyResidualAccessor(R1, R2, R3, R4, R5)

# Chart auto-completion
result = study.execute(study.charts.Xbar)  # IDE suggests valid charts

# Access the prepared dataset
print(study.dataset.columns.tolist())
# ['lane', 'batch', 'weight', 'Ybar', 'Ybar_k', 'Ybar_t', 'Ybar_kt',
#  'R1', 'R2', 'R3', 'R4', 'R5']
```

## The why_not() Method

If you try to use an invalid chart, `why_not()` explains why:

```python
# If Xbar is not valid for your SDS
study.why_not('Xbar')
# Returns: "Xbar requires subgrouped data (n >= 2 per cell).
#          Your data has n=1 per cell (SDS 2)."
```

## Common Patterns

### Pattern 1: Simple Time Series

No factors, just measurements over time:

```python
study = pb.formulate(
    response=pb.cols.temperature,
    time=pb.cols.day
)
# Results in SDS 4, recommends XmR
```

### Pattern 2: Comparing Groups

Factors but no time dimension:

```python
study = pb.formulate(
    response=pb.cols.yield_pct,
    factors=[pb.cols.machine, pb.cols.operator]
)
# Results in SDS varies, recommends Xbar or XmR
```

### Pattern 3: Groups Over Time

Full analysis with factors and time:

```python
study = pb.formulate(
    response=pb.cols.fillweight,
    factors=[pb.cols.lane],
    time=pb.cols.pull
)
# Results in SDS 1-3, recommends Xbar-S with VAS residuals
```

### Pattern 4: Replicated Design

Multiple observations per factor-time cell:

```python
# 4 lanes x 10 batches x 3 replicates = 120 observations
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)
# Results in SDS 1 (Full Replication) - most powerful design
```

## Data Cleaning

ProcessBehavior automatically cleans common garbage values:

```python
# These are automatically converted to NaN:
default_na_values = [
    '*', '?', 'ND', 'BDL', 'NA', 'N/A', 'n/a',
    '<LOD', '>LOQ', 'TNTC', 'QNS', '--'
]

# Custom NA values
pb = ProcessBehavior(df, na_values=['*', 'missing', '<DL'])
```

## Natural Sorting

Factor levels are automatically sorted naturally:

```python
# Input order: ['Lane_10', 'Lane_1', 'Lane_2']
# Sorted order: ['Lane_1', 'Lane_2', 'Lane_10']
```

This uses `natsort` for intelligent alphanumeric sorting.

## The `plan` Parameter

The `plan` parameter provides an alternative to `factors` for specifying the study structure. Use `plan` when you know the intended design -- especially when factor levels or time points are entirely absent from the data.

`plan` and `factors` are **mutually exclusive** -- use one or the other.

### Format

The plan is a dictionary with a required `'factors'` key and optional `'T'` and `'N'` keys:

```python
study = pb.formulate(
    response=pb.cols.weight,
    time=pb.cols.batch,
    plan={
        'factors': {
            pb.cols.lane: [1, 2, 3, 4],
            pb.cols.shift: ['day', 'night']
        },
        'T': 20,   # Planned number of time points
        'N': 3     # Planned observations per cell
    }
)
```

- **`factors`** (required): Dict mapping column names to lists of expected levels
- **`T`** (optional): Number of planned time points
- **`N`** (optional): Planned observations per cell

### When to Use `plan` vs `factors`

| Scenario | Use |
|----------|-----|
| All factor levels and time points are present in data | Either works |
| Some factor levels are entirely absent from data | `plan` |
| Some time points were entirely skipped | `plan` with `'T'` |
| You want to compare planned vs observed structure | `plan` |

### SDS 4-6 Detection

ProcessBehavior can detect SDS 4-6 **with or without a plan**. SDS detection runs on raw data before NA rows are dropped, so cells where all response values are NA (e.g., from garbage values like `*` or `ND`) are counted as empty cells (N_kt=0). This means:

- **With `factors`**: SDS 4-6 are detected when the data contains factor/time combinations where every response value is NA or garbage. The system sees these as empty cells in the grid.
- **With `plan`**: SDS 4-6 are also detected when factor levels or time points specified in the plan are entirely absent from the data. This catches gaps that `factors` alone cannot see.

A plan is valuable for documenting experimental intent and catching absent factor levels, but it is not required for SDS 4-6 detection.

After formulating with `plan`, use `study.design()` to see planned vs observed structure. See [SDS Detection](sds-detection.md) for details on SDS 4-6.

## The `companion` Parameter

Wheeler recommends reading certain charts as pairs: Xbar with S, and XmR with R (the variation chart first, then the location chart). The `companion` parameter in `execute()` returns both charts together.

```python
# Returns both Xbar and S charts together
result = study.execute(chart='Xbar', companion=True)

# Returns both XmR and R charts together, stratified
result = study.execute(chart='XmR', by=['lane'], companion=True)
```

Key details:

- Either chart in the pair triggers the pair: `chart='S', companion=True` also returns Xbar+S
- Default is `companion=False` (returns a single chart)
- Companion results can be plotted and exported the same way as single-chart results

## Study Inspection Methods

After formulation, the `Study` object provides several methods for inspecting what analyses are available.

### `study.support`

A DataFrame showing all chart types with their availability, recommendations, and the analytical question each answers:

```python
study.support
#   chart  category  available  recommended  reason  question
# 0  Xbar   primary       True         True    None  Are subgroup means stable over time?
# 1     S   primary       True        False    None  Is within-subgroup variation stable?
# ...
```

Filter to available charts:

```python
study.support[study.support['available']]
```

### `study.why_not(chart)`

Explains why a specific chart is unavailable:

```python
study.why_not('R2_XmR')
# "Not available for this SDS"
```

### `study.design()`

Returns a `DesignReport` comparing the sampling plan to observed data:

```python
report = study.design()

# Structure metrics
report.K           # Planned number of RSG groups
report.K_observed  # Observed number of RSG groups
report.T           # Planned time points
report.T_observed  # Observed time points
report.N           # Planned cell size
report.N_observed  # Observed cell size (min, median, max)
report.R           # Planned total cells (K × T)
report.R_observed  # Observed total cells

# Missing/extra structure
report.missing_levels   # Factor levels in plan but not observed
report.extra_levels     # Factor levels observed but not in plan
report.missing_combos   # RSG groups in plan but not observed
report.plan_adherence   # Summary of how well data matches plan
```

Works with or without a sampling plan -- without a plan, it reports observed structure only.

## Best Practices

1. **Use auto-completion** - Prevents typos and speeds up development
2. **Start simple** - Begin with just response, add factors/time as needed
3. **Check the SDS** - Understand your data structure before analyzing
4. **Use why_not()** - Learn why certain charts aren't available
5. **Review the dataset** - Check `study.dataset` to verify residual calculations

## Next Steps

- [Sampling Design States](sds-detection.md) - Understanding Sampling Design States
- [Chart Types](chart-types.md) - Choosing the right chart
- [VAS Residuals](residuals.md) - Working with VAS residuals
