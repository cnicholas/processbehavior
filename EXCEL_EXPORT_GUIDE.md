# Excel Export Guide

## Overview

The `to_excel()` method allows you to export analysis results to professional Excel workbooks with organized multi-sheet layouts. This makes it easy to share results with stakeholders, create reports, and archive analysis data.

## Quick Start

```python
from processbehavior.datasets import make_sds1
import analysis_dataset as ad

# Generate data and run analysis
df = make_sds1(K=3, T=8, n_min=2, n_max=4, seed=42)

spec = {
    'analysis_type': 'Xbar',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y'
}

analysis = ad.Analysis(df, spec)
result = analysis.calculate()

# Export to Excel
result.to_excel('my_analysis.xlsx')
```

That's it! You now have a professionally formatted Excel workbook with all your analysis results.

## What Gets Exported?

The `to_excel()` method creates a multi-sheet workbook containing:

### 1. Summary Tab
- Sampling Design State (SDS) information
- Analysis configuration (type, variables, etc.)
- Data dimensions (observations, charts)
- Capabilities (residuals, effects, interactions)
- Signal counts
- SDS capabilities list

### 2. Chart Tabs
- One tab per chart (e.g., `Chart_Xbar`, `Chart_Sbar`)
- For stratified analyses: `Chart_K1`, `Chart_K2`, etc.
- Includes all chart data: values, centerlines, control limits, signals
- Chart statistics in the data

### 3. Residuals Tab (if available)
- VAS residuals (R1-R5)
- Only included when variance decomposition was calculated
- Typically available for SDS ≥ 1

### 4. Effects Tab (if available)
- Main effects for factors and time
- Only included when effects were calculated
- Format: Effect_Type, Level, Value

### 5. Interactions Tab (if available)
- Interaction terms
- Only included when interaction analysis was performed
- Typically available for higher SDS levels

### 6. Full_Dataset Tab (optional)
- Complete analysis dataset with all calculations
- Includes all intermediate values
- Can be large - disabled by default
- Enable with `include_full_dataset=True`

## Method Signature

```python
def to_excel(
    filepath: str,
    include_summary: bool = True,
    include_charts: bool = True,
    include_residuals: bool = True,
    include_effects: bool = True,
    include_interactions: bool = True,
    include_full_dataset: bool = False,
    format_cells: bool = True
) -> None:
```

### Parameters

- **filepath** (str): Output Excel file path (e.g., `'analysis.xlsx'`)
- **include_summary** (bool): Include summary tab with metadata (default: `True`)
- **include_charts** (bool): Include chart tabs (default: `True`)
- **include_residuals** (bool): Include residuals if available (default: `True`)
- **include_effects** (bool): Include effects if available (default: `True`)
- **include_interactions** (bool): Include interactions if available (default: `True`)
- **include_full_dataset** (bool): Include complete dataset (default: `False`)
- **format_cells** (bool): Apply formatting (default: `True`)

## Formatting Features

When `format_cells=True` (default), the following formatting is applied:

- **Bold headers**: First row of each sheet has bold text
- **Frozen panes**: Top row is frozen for easy scrolling
- **Auto-sized columns**: Columns automatically sized to fit content (max 50 chars)
- **Clean layout**: Professional appearance suitable for reports

## Usage Examples

### Example 1: Default Export (Recommended)

Export everything except the full dataset:

```python
result.to_excel('analysis.xlsx')
```

This creates:
- Summary tab
- All chart tabs
- Residuals (if calculated)
- Effects (if calculated)
- Interactions (if calculated)

### Example 2: Complete Export

Include the full dataset for archival purposes:

```python
result.to_excel('complete_analysis.xlsx', include_full_dataset=True)
```

⚠️ **Warning**: The full dataset can be large for complex analyses.

### Example 3: Minimal Export

Export only summary and charts:

```python
result.to_excel(
    'charts_only.xlsx',
    include_residuals=False,
    include_effects=False,
    include_interactions=False
)
```

Ideal for sharing with stakeholders who only need chart visualizations.

### Example 4: Stratified IMR Charts

The "killer feature" - stratified individuals charts:

```python
df = make_sds1(K=4, T=10, n_min=2, n_max=3, seed=42)

spec = {
    'analysis_type': 'Imr',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y'
}

analysis = ad.Analysis(df, spec)
result = analysis.calculate()

result.to_excel('stratified_imr.xlsx')
```

This creates separate IMR chart tabs for each group (K1, K2, K3, K4), each with group-specific control limits.

### Example 5: Using ProcessDataFrame (Frictionless API)

The easiest way with auto-completion:

```python
from process_dataframe import ProcessDataFrame

pdf = ProcessDataFrame(df)

result = pdf.analyze(
    response_var=pdf.columns.y,
    time_var=pdf.columns.time,
    grouping_vars=[pdf.columns.factor_1]
)

result.to_excel('frictionless.xlsx')
```

### Example 6: Custom Configuration

Fine-tune what gets exported:

```python
result.to_excel(
    'custom_export.xlsx',
    include_summary=True,      # Include summary
    include_charts=True,       # Include charts
    include_residuals=True,    # Include residuals
    include_effects=True,      # Include effects
    include_interactions=False, # Skip interactions
    include_full_dataset=False, # Skip full dataset
    format_cells=True          # Apply formatting
)
```

## Excel Tab Naming

### Standard Charts
- `Summary`: Analysis metadata
- `Chart_Xbar`: Xbar chart data
- `Chart_Sbar`: S chart data
- `Chart_R`: R chart data
- `Residuals`: VAS residuals (R1-R5)
- `Effects`: Main effects
- `Interactions`: Interaction terms
- `Full_Dataset`: Complete analysis data

### Stratified Charts
When using stratified analysis (e.g., IMR with grouping), tabs are named after the groups:
- `Chart_K1`, `Chart_K2`, `Chart_K3`, etc.
- `Chart_GroupA`, `Chart_GroupB`, etc.

⚠️ **Note**: Excel limits tab names to 31 characters. Long names are automatically truncated.

## Requirements

Excel export requires the `openpyxl` package:

```bash
pip install openpyxl
```

If not installed, you'll get a helpful error message with installation instructions.

## Integration with Workflow

### Typical Workflow

```python
# 1. Load data
df = pd.read_csv('production_data.csv')

# 2. Run analysis
spec = {
    'analysis_type': 'Xbar',
    'rsg_vars': ['line'],
    'time_var': 'shift',
    'response_var': 'weight'
}

analysis = ad.Analysis(df, spec)
result = analysis.calculate()

# 3. Inspect results programmatically
print(f"SDS: {result.sds}")
print(f"Signals detected: {result.summary['n_signals_total']}")

# 4. Export to Excel for sharing
result.to_excel('production_report.xlsx')

# 5. Access specific data
xbar_chart = result.get_chart('Xbar')
residuals = result.residuals
```

### Batch Processing

Export multiple analyses:

```python
analyses = [
    ('Line_A', df_a, spec_a),
    ('Line_B', df_b, spec_b),
    ('Line_C', df_c, spec_c)
]

for name, df, spec in analyses:
    analysis = ad.Analysis(df, spec)
    result = analysis.calculate()
    result.to_excel(f'{name}_analysis.xlsx')

    print(f"✓ {name}: {result.summary['n_signals_total']} signals detected")
```

## Tips and Best Practices

### 1. Use Descriptive Filenames

```python
# Good
result.to_excel('line_A_shift_1_2024-01-15.xlsx')

# Less helpful
result.to_excel('output.xlsx')
```

### 2. Include Metadata in Summary

The Summary tab automatically includes:
- SDS information
- Analysis configuration
- Signal counts
- Capabilities

This provides context for anyone opening the file.

### 3. Share Charts Without Residuals

For non-technical stakeholders:

```python
result.to_excel(
    'executive_summary.xlsx',
    include_residuals=False,
    include_effects=False,
    include_interactions=False
)
```

### 4. Archive Complete Analyses

For record-keeping:

```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
result.to_excel(
    f'archive/analysis_{timestamp}.xlsx',
    include_full_dataset=True
)
```

### 5. Verify Exports Programmatically

```python
import os

filepath = 'my_analysis.xlsx'
result.to_excel(filepath)

if os.path.exists(filepath):
    print(f"✓ Export successful: {filepath}")
    print(f"  File size: {os.path.getsize(filepath) / 1024:.1f} KB")
```

## Common Use Cases

### 1. Quality Reports

```python
# Weekly quality report
result.to_excel(
    'quality_report_week_42.xlsx',
    include_full_dataset=False
)
```

### 2. Process Capability Studies

```python
# Capability study with full detail
result.to_excel(
    'capability_study_line_A.xlsx',
    include_full_dataset=True
)
```

### 3. Troubleshooting Sessions

```python
# Include everything for deep dive
result.to_excel(
    'troubleshooting_session.xlsx',
    include_full_dataset=True
)
```

### 4. Stakeholder Presentations

```python
# Clean charts only
result.to_excel(
    'presentation_charts.xlsx',
    include_residuals=False,
    include_effects=False,
    include_interactions=False
)
```

## Troubleshooting

### ImportError: No module named 'openpyxl'

**Solution**: Install openpyxl:
```bash
pip install openpyxl
```

### OSError: [Errno 2] No such file or directory

**Solution**: Ensure the directory exists:
```python
import os
os.makedirs('output', exist_ok=True)
result.to_excel('output/analysis.xlsx')
```

### Tab names too long

**Issue**: Excel limits tab names to 31 characters.

**Solution**: Names are automatically truncated. If you need specific names, rename groups before analysis.

### File already open

**Issue**: Cannot write to Excel file that's already open.

**Solution**: Close the file in Excel, or use a different filename.

## Performance Considerations

- **Small analyses** (< 1000 observations): Export is instant
- **Medium analyses** (1000-10000 observations): < 1 second
- **Large analyses** (> 10000 observations): Consider excluding `include_full_dataset`

The bottleneck is typically writing the full dataset, which is why it's disabled by default.

## Summary

The `to_excel()` method provides:

✅ **User-friendly**: Single method call exports everything
✅ **Customizable**: Control exactly what gets exported
✅ **Professional**: Formatted workbooks suitable for reports
✅ **Comprehensive**: All analysis data in one file
✅ **Pythonic**: Follows pandas conventions (`df.to_excel()`)
✅ **Flexible**: Works with all analysis types and SDS levels

For more examples, see `examples/excel_export_demo.py`.
