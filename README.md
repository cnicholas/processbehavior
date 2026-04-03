# processbehavior

A Python library for **Process Behavior Analysis** following the Wheeler/Bishop Variance Analysis System (VAS) methodology.

Unlike traditional SPC packages, processbehavior faithfully implements Wheeler/Bishop equation-by-equation: automatic Sampling Design State (SDS) detection, variance decomposition via R1-R5 residuals, and correct chart selection based on data structure.

## Installation

```bash
pip install processbehavior
```

Plotting (plotly) and Excel export (openpyxl) are included. For static image export:

```bash
pip install processbehavior[images]
```

## Quickstart

```python
import pandas as pd
from processbehavior import ProcessBehavior

# Wrap your DataFrame
pb = ProcessBehavior(df)

# Formulate the study (detect SDS, build analysis dataset)
study = pb.formulate(
    response=pb.cols.measurement,
    time=pb.cols.batch,
    factors=[pb.cols.machine]
)

# See what was detected
print(study)  # Shows SDS, valid charts, design report

# Execute analysis
result = study.execute()

# Access results
print(result.summary)
chart_data = result.get_chart('Xbar')
stats = result.get_statistics('Xbar')

# Plot
result.plot()

# Export to Excel
result.to_excel('analysis.xlsx')
```

## Key Concepts

### Two-Step Workflow

The API mirrors how analysts think:

1. **`formulate()`** - Understand your data structure. Detects the Sampling Design State (SDS 1-6), identifies valid charts, and computes residuals. This is the expensive step.
2. **`execute()`** - Run analysis. Produces charts from the pre-computed data. This is cheap and can be called multiple times for different charts from the same study.

### Sampling Design States (SDS)

processbehavior automatically detects your data's structure:

| SDS | Structure | Example |
|-----|-----------|---------|
| 1 | Factors + Time + Replication | 3 machines x 10 batches x 4 samples each |
| 2 | Factors + Time, no replication | 3 machines x 10 batches, 1 sample each |
| 3 | Factors + Time, partial replication | Mixed sample sizes across cells |
| 4 | Time only (single stream) | 30 sequential measurements |
| 5 | Factors only (no time) | 3 machines, multiple samples, no time order |
| 6 | Individual values only | Flat list of measurements |

The detected SDS determines which charts are valid, how R2 is calculated, and whether variance decomposition is available.

### Residual System (R1-R5)

For factorial designs (SDS 1-3), processbehavior decomposes variation into diagnostic residuals:

- **R1** - Overall deviation from grand mean
- **R2** - Within-cell noise (measurement error, short-term fluctuation)
- **R3** - Interaction between factors and time
- **R4** - Time effects (trends, seasonality, batch effects)
- **R5** - Factor effects (machine-to-machine, operator bias)

```python
# Chart any residual
result = study.execute(chart='XmR', value='R4')  # Time effects on XmR chart
result = study.execute(chart='Xbar', value='R5')  # Factor effects on Xbar chart
```

### Stratified Analysis

For XmR/R charts with grouping factors, processbehavior produces a single combined chart with per-stratum limits:

```python
result = study.execute(chart='XmR', by=['machine'])

# Drill into a specific stratum
for stratum in result.strata:
    focused = result.focus(stratum)
    focused.plot()
```

## Features

- **Auto-detection**: SDS detection on raw data determines valid charts and analysis methods
- **Correct charts**: Xbar-S, XmR (IMR), Range, Histogram with proper limit calculations
- **Variance decomposition**: R1-R5 residuals for factorial designs
- **Effects analysis**: Main effects, time effects, and interaction effects
- **Stratified charts**: Automatic per-stratum limits for grouped individual data
- **Signal detection**: Rule 1 (3-sigma) point classification
- **IDE support**: Column auto-completion via `pb.cols`
- **Self-diagnostic errors**: Helpful messages that say what's available and how to fix it
- **Excel export**: Publication-ready workbooks with charts and statistics
- **Interactive plots**: Plotly-based charts with hover details

## License

Apache 2.0
