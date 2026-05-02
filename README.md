# processbehavior

A Python library for **Process Behavior Analysis** following Thomas A. Bishop's Variance Analysis System (VAS) methodology.

Unlike traditional SPC packages, processbehavior faithfully implements Bishop's VAS equation-by-equation: automatic Design State (DS) detection, variance decomposition via R1-R5 residuals, and correct chart selection based on data structure.

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

# Formulate the study (detect DS, build analysis dataset)
study = pb.formulate(
    response=pb.cols.measurement,
    time=pb.cols.batch,
    factors=[pb.cols.machine]
)

# See what was detected
print(study)  # Shows DS, valid charts, design report

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

1. **`formulate()`** - Understand your data structure. Detects the Design State (DS 1-6), identifies valid charts, and computes residuals. This is the expensive step.
2. **`execute()`** - Run analysis. Produces charts from the pre-computed data. This is cheap and can be called multiple times for different charts from the same study.

### Design States (DS)

processbehavior automatically detects your data's structure:

| DS | Name | Cell Sizes (N_kt) |
|-----|------|--------------------|
| 1 | Full Replication | All N_kt >= 2 |
| 2 | No Replication | All N_kt = 1 |
| 3 | Partial Replication | Mix of N_kt = 1 and N_kt >= 2 |
| 4 | Incomplete, No Singletons | Empty cells + all observed N_kt >= 2 |
| 5 | Incomplete, No Replication | Empty cells + all observed N_kt = 1 |
| 6 | Incomplete, With Singletons | Empty cells + mixed N_kt |

The **Analytical Design State** (ADS) determines which charts are valid, how R2 is calculated, and whether variance decomposition is available. DS 4-6 (Incomplete) collapse to DS 1-3 after data cleansing.

### Residual System (R1-R5)

For factorial designs (DS 1-3), processbehavior decomposes variation into diagnostic residuals:

- **R1** - Overall deviation from grand mean
- **R2** - Within-cell noise (measurement error, short-term fluctuation)
- **R3** - Interaction between factors and time
- **R4** - Time effects (trends, seasonality, batch effects)
- **R5** - Factor effects (machine-to-machine, operator bias)

```python
# Chart any residual
result = study.execute(chart='X', value='R4')  # Time effects on X chart
result = study.execute(chart='Xbar', value='R5')  # Factor effects on Xbar chart
```

### Stratified Analysis

For X/mR charts with grouping factors, processbehavior produces a single combined chart with per-stratum limits:

```python
result = study.execute(chart='X', by=['machine'])

# Drill into a specific stratum
for stratum in result.strata:
    focused = result.focus(stratum)
    focused.plot()
```

## Features

- **Auto-detection**: DS detection on raw data determines valid charts and analysis methods
- **Correct charts**: Xbar-S, X (Individual), mR (Moving Range), Histogram with proper limit calculations
- **Variance decomposition**: R1-R5 residuals for factorial designs
- **Effects analysis**: Main effects, time effects, and interaction effects
- **Stratified charts**: Automatic per-stratum limits for grouped individual data
- **Signal detection**: Rule 1 (3-sigma) point classification
- **IDE support**: Column auto-completion via `pb.cols`
- **Self-diagnostic errors**: Helpful messages that say what's available and how to fix it
- **Excel export**: Publication-ready workbooks with charts and statistics
- **Interactive plots**: Plotly-based charts with hover details

## Scope

processbehavior is the **computational engine** for Bishop's VAS methodology. It handles data ingestion, DS detection, residual computation, chart generation, and export.

For the **curated analyst experience** — guided workflows, interactive dashboards, and collaboration features — see [processbehavior.com](https://processbehavior.com).

For the **methodological foundation** — the theory behind VAS, DS classification, and residual interpretation — see the forthcoming book by Dr. Thomas A. Bishop and Chris Nicholas.

## License

Apache 2.0
