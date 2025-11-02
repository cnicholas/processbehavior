# Quick Start Guide

Get up and running with ProcessBehavior in 5 minutes!

## Installation

```bash
pip install processbehavior
```

## Your First Analysis

### 1. Import and Load Data

```python
import pandas as pd
from processbehavior import ProcessDataFrame

# Load your data (or create sample data)
df = pd.DataFrame({
    'measurement': [23.1, 24.5, 22.8, 25.1, 23.7, 24.2, 23.5, 24.8],
    'operator': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time': [1, 2, 1, 2, 3, 4, 3, 4]
})

# Wrap in ProcessDataFrame
pdf = ProcessDataFrame(df)
```

### 2. Run Basic Analysis

```python
# Simple IMR chart (individuals moving range)
result = pdf.analyze(
    response_var='measurement',
    chart_type='Imr'
).calculate()

# Display the chart
result.plot()
```

That's it! You've created your first control chart.

## Understanding the Result

The `result` object contains everything from your analysis:

```python
# Access charts
print(result.charts.keys())  # Dictionary of available charts

# Get specific chart data
imr_data = result.get_chart('Imr')
print(imr_data.head())

# Get statistics
stats = result.get_statistics('Imr')
print(f"Centerline: {stats['center']}")
print(f"Upper Control Limit: {stats['ucl']}")
print(f"Lower Control Limit: {stats['lcl']}")

# View summary
print(result.summary)
```

## Adding Grouping Variables

For more complex analyses with subgroups:

```python
# Xbar and S charts with subgroups
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['operator'],  # Group by operator
    time_var='time',
    chart_type='Xbar'  # Creates both Xbar and S charts
).calculate()

# Plot Xbar chart
result.plot(chart='Xbar')

# Plot S chart
result.plot(chart='Sbar')
```

## Stratified Analysis

Create separate charts for each group:

```python
# Separate IMR chart for each operator
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['operator'],
    time_var='time',
    chart_type='Imr'  # Automatically stratifies
).calculate()

# Plot all charts together
result.plot(facet=True, ncols=2)

# Or plot just one operator
result.plot(chart='A')  # Just operator A
```

## Detect Signals

Apply Western Electric rules to detect out-of-control patterns:

```python
# Run analysis
result = pdf.analyze(
    response_var='measurement',
    chart_type='Imr'
).calculate()

# Detect signals
signals = result.detect_signals(chart='Imr')

# View violations
print(f"Found {len(signals)} signal violations")
print(signals[['index', 'rule', 'description']])
```

## Export to Excel

Save everything to a multi-sheet Excel workbook:

```python
result.to_excel('analysis_results.xlsx')
```

The Excel file will contain:
- **Summary** tab with analysis metadata
- **Chart** tabs with raw data
- **Residuals** tab (if calculated)
- **Effects** tab (if calculated)
- **Visual_Charts** tab with embedded chart images

## Common Patterns

### Pattern 1: Simple Time Series

Just a single column of measurements over time:

```python
result = pdf.analyze(
    response_var='temperature',
    chart_type='Imr'
).calculate()
```

### Pattern 2: Compare Groups

Compare multiple conditions (no time dimension):

```python
result = pdf.analyze(
    response_var='yield',
    grouping_vars=['machine', 'shift'],
    chart_type='Xbar'
).calculate()
```

### Pattern 3: Track Groups Over Time

Monitor multiple streams over time:

```python
result = pdf.analyze(
    response_var='fillweight',
    grouping_vars=['lane'],
    time_var='batch',
    chart_type='Xbar'
).calculate()
```

### Pattern 4: Stratified Time Series

Separate control charts for each stream:

```python
result = pdf.analyze(
    response_var='fillweight',
    grouping_vars=['lane'],
    time_var='batch',
    chart_type='Imr'  # Creates one chart per lane
).calculate()
```

## Auto-Detection

ProcessBehavior automatically detects your data structure (Sampling Design State):

```python
# The library automatically determines:
# - Whether you have grouping variables
# - Whether you have a time dimension
# - Level of replication (full, partial, none)
# - Which analyses are valid
# - Which chart type to recommend

# You can see what was detected:
print(result.summary['sds'])  # Shows SDS number (0-6)
print(result.summary['sds_name'])  # Shows SDS description
print(result.summary['valid_charts'])  # Shows what you can run
```

## Next Steps

- [Understanding SDS](../tutorials/understanding-sds.md) - Learn about Sampling Design States
- [VAS Residuals](../tutorials/vas-residuals.md) - Variance decomposition
- [Chart Types Guide](../guide/chart-types.md) - When to use each chart type
- [API Reference](../api/process-dataframe.md) - Complete API documentation

## Tips

### IDE Autocomplete

ProcessDataFrame provides autocomplete for columns and chart types:

```python
# Column autocomplete
pdf.cols.measurement  # ← IDE suggests your columns!

# Chart type autocomplete
pdf.chart_types.Imr  # ← IDE suggests valid chart types!
```

### Automatic Data Cleaning

ProcessDataFrame automatically cleans common garbage characters:

```python
# These are automatically removed/converted:
# '*', 'ND', 'BDL', 'NA', '<LOD', etc.

# No manual cleaning needed!
```

### Natural Sorting

Groups are automatically sorted naturally:

```python
# Automatic: ['Group_1', 'Group_2', 'Group_10']
# Not:       ['Group_1', 'Group_10', 'Group_2']
```

### Check What's Possible

Before running analysis, check what's valid for your data:

```python
# This will tell you what you can do
result = pdf.analyze(response_var='y', grouping_vars=['x']).calculate()
print(result.summary['valid_charts'])
print(result.summary['recommended_chart'])
```

## Common Issues

### "Chart type not valid for this SDS"

```python
# If you get this error, check what's valid:
print(result.summary['valid_charts'])

# And use a valid chart type:
result = pdf.analyze(
    response_var='y',
    chart_type='Imr'  # ← Use a valid type
).calculate()
```

### "Insufficient observations"

```python
# Need at least 2 observations per subgroup for Xbar/S
# Use IMR for individual observations:
result = pdf.analyze(
    response_var='y',
    chart_type='Imr'  # ← Works with n=1
).calculate()
```

### "Column not found"

```python
# Check your column names:
print(df.columns.tolist())

# Or use autocomplete:
pdf.cols.<TAB>  # ← Shows available columns
```

## Summary

That's the quick start! You now know how to:

- ✅ Load data and create a ProcessDataFrame
- ✅ Run basic analyses (IMR, Xbar, S, R)
- ✅ Create stratified charts
- ✅ Detect signals
- ✅ Export to Excel
- ✅ Use IDE autocomplete
- ✅ Understand auto-detection

Ready to dive deeper? Check out the [tutorials](../tutorials/index.md)!
