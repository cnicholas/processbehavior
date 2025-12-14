# Study Formulation

The `formulate()` method is the heart of ProcessBehavior. It enables the analyst to specify the key inputs defined during problem forumlation to create a structured study.  The system automatically detects the sampling design and prepares the analysis.

## The Formulation API

```python
from processbehavior import ProcessBehavior

pdf = ProcessBehavior(df)

study = pdf.formulate(
    response=pdf.cols.weight,        # Required: measurement variable
    factors=[pdf.cols.lane],         # Optional: grouping variables
    time=pdf.cols.batch,             # Optional: time/sequence variable
    precision=3                         # Optional: decimal places
)
```

## IDE Auto-Completion

One of ProcessBehavior's key features is **IDE auto-completion** for column names. This eliminates typos and makes exploration faster.

### Column Auto-Completion

After creating a ProcessBehavior, access columns via the `.cols` accessor:

```python
pdf = ProcessBehavior(df)

# Type pdf.cols. and your IDE will show all available columns
pdf.cols.weight      # Instead of 'weight' string
pdf.cols.lane        # Instead of 'lane' string
pdf.cols.batch       # Instead of 'batch' string
```

This works in:
- VS Code with Pylance
- PyCharm
- Jupyter notebooks (with Tab completion)
- Any LSP-compatible editor

### Chart Type Auto-Completion

After formulation, the `study.charts` accessor provides auto-completion for valid chart types:

```python
study = pdf.formulate(...)

# Type study.charts. and see only valid charts for your SDS
study.charts.Xbar      # Available if SDS supports it
study.charts.Imr       # Always available
study.charts.R4_Imr    # VAS residual chart
```

## Parameter Details

### response (required)

The measurement variable to analyze.

```python
# Using auto-completion (recommended)
response=pdf.cols.measurement

# Using string (still works)
response='measurement'
```

### factors (optional)

A list of categorical variables that define subgroups. These become the "rational subgroups" in Wheeler's terminology.

```python
# Single factor
factors=[pdf.cols.operator]

# Multiple factors (creates combined subgroups)
factors=[pdf.cols.machine, pdf.cols.shift]
```

When multiple factors are specified, they're combined into a single grouping variable (e.g., `Machine_A_Shift_1`).

### time (optional)

The variable that defines time ordering. This enables time-series analysis and certain signal detection rules.

```python
time=pdf.cols.batch
time=pdf.cols.timestamp
time=pdf.cols.sequence
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
study = pdf.formulate(
    response=pdf.cols.weight,
    factors=[pdf.cols.lane],
    time=pdf.cols.batch
)

# SDS information
print(study.sds)              # 1, 2, 3, 4, 5, or 6
print(study.sds_name)         # "Full Replication"
print(study.sds_description)  # Detailed explanation

# Chart recommendations
print(study.valid_charts)       # ['Xbar', 'S', 'Imr']
print(study.recommended_chart)  # 'Xbar'
print(study.residual_charts)    # ['R2_S', 'R3_Imr', 'R4_Imr', 'R5_Imr']

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
study = pdf.formulate(
    response=pdf.cols.temperature,
    time=pdf.cols.day
)
# Results in SDS 4, recommends IMR
```

### Pattern 2: Comparing Groups

Factors but no time dimension:

```python
study = pdf.formulate(
    response=pdf.cols.yield_pct,
    factors=[pdf.cols.machine, pdf.cols.operator]
)
# Results in SDS varies, recommends Xbar or IMR
```

### Pattern 3: Groups Over Time

Full analysis with factors and time:

```python
study = pdf.formulate(
    response=pdf.cols.fillweight,
    factors=[pdf.cols.lane],
    time=pdf.cols.pull
)
# Results in SDS 1-3, recommends Xbar-S with VAS residuals
```

### Pattern 4: Replicated Design

Multiple observations per factor-time cell:

```python
# 4 lanes x 10 batches x 3 replicates = 120 observations
study = pdf.formulate(
    response=pdf.cols.weight,
    factors=[pdf.cols.lane],
    time=pdf.cols.batch
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
pdf = ProcessBehavior(df, na_values=['*', 'missing', '<DL'])
```

## Natural Sorting

Factor levels are automatically sorted naturally:

```python
# Input order: ['Lane_10', 'Lane_1', 'Lane_2']
# Sorted order: ['Lane_1', 'Lane_2', 'Lane_10']
```

This uses `natsort` for intelligent alphanumeric sorting.

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
