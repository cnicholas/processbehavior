# Excel Export

ProcessBehavior can export complete analysis results to Excel workbooks with multiple sheets, formatting, and embedded charts.

## Requirements

Excel export requires the `openpyxl` package:

```bash
pip install processbehavior[excel]
# or
pip install openpyxl
```

## Basic Export

```python
result = study.analyze()

# Export to Excel
result.to_excel('analysis_results.xlsx')
```

## Export Options

```python
result.to_excel(
    'analysis_results.xlsx',
    include_summary=True,        # Metadata and SDS information
    include_charts=True,         # Chart data (one tab per chart)
    include_residuals=True,      # R1-R5 decomposition
    include_effects=True,        # Main effects (factor and time)
    include_interactions=True,   # Interaction terms
    include_full_dataset=False,  # Complete dataset (can be large)
    format_cells=True,           # Apply Excel formatting
    include_chart_images=True,   # Embed chart images
    export_html=True             # Also create HTML files
)
```

## Workbook Structure

A typical export creates these sheets:

### Summary Sheet

Contains metadata about the analysis:
- Sampling Design State (SDS)
- Data dimensions (factors, time points, observations)
- Analysis type and chart used
- Control limit values

### Chart Sheets

One sheet per chart with:
- Subgroup identifier
- Observation count (n)
- Chart value (mean, std dev, etc.)
- Control limits (UCL, LCL)
- Signal flags

### Residuals Sheet

If `include_residuals=True` and residuals are available:
- All observations with R1-R5 values
- Factor and time identifiers
- Calculated means (Y̅, Y̅_k, Y̅_t, Y̅_kt)

### Effects Sheet

If `include_effects=True`:
- Factor effects (Y̅_k - Y̅)
- Time effects (Y̅_t - Y̅)

### Full Dataset Sheet

If `include_full_dataset=True`:
- Complete analysis dataset
- All calculated columns
- Can be large for big datasets

## Example: Complete Export

```python
from processbehavior import ProcessBehavior

# Setup
pdf = ProcessBehavior(df)
study = pdf.formulate(
    response=pdf.cols.weight,
    factors=[pdf.cols.lane],
    time=pdf.cols.batch
)
result = study.analyze()

# Full export with all options
result.to_excel(
    'fillweight_analysis.xlsx',
    include_summary=True,
    include_charts=True,
    include_residuals=True,
    include_effects=True,
    include_chart_images=True
)
```

## Chart Images

When `include_chart_images=True`, ProcessBehavior:

1. Generates PNG images of each chart
2. Embeds them in a "Charts" sheet
3. Sizes them appropriately

Requires kaleido:

```bash
pip install kaleido
```

## HTML Companion Files

When `export_html=True`, creates interactive HTML files alongside the Excel:

```
analysis_results.xlsx      # Main workbook
analysis_results_Xbar.html # Interactive Xbar chart
analysis_results_S.html    # Interactive S chart
```

HTML files retain full Plotly interactivity (zoom, hover, etc.).

## Formatting

When `format_cells=True`:

- Headers are bold with background color
- Numeric columns have appropriate decimal places
- Signal cells are highlighted
- Column widths are auto-adjusted

## Working with the Export

### Reading Back in pandas

```python
import pandas as pd

# Read specific sheet
xbar_data = pd.read_excel('analysis_results.xlsx', sheet_name='Xbar')

# Read all sheets
all_sheets = pd.read_excel('analysis_results.xlsx', sheet_name=None)
for name, df in all_sheets.items():
    print(f"{name}: {len(df)} rows")
```

### Customizing After Export

Open in Excel and:
- Add your own charts
- Apply company formatting
- Add comments or annotations
- Create pivot tables from the data

## Programmatic Access

Before exporting, access results as DataFrames:

```python
# Get chart data
xbar_data = result.get_chart('Xbar')
s_data = result.get_chart('S')

# Get statistics
xbar_stats = result.get_statistics('Xbar')

# Get residuals
residuals = result.residuals

# Get effects
effects = result.effects  # Dict with 'k_effects', 't_effects'
```

This allows custom processing before export.

## Export Best Practices

1. **Start with defaults** - Enable specific options as needed
2. **Check file size** - Large datasets with `include_full_dataset=True` can be big
3. **Use HTML for interactivity** - Excel images are static
4. **Include residuals** - Essential for Wheeler-style analysis
5. **Document the export** - Add notes in Excel after export

## Example: Report Generation

```python
# Analyze
result = study.analyze()

# Detect signals
signals = result.detect_signals()

# Export with relevant options
result.to_excel(
    f'analysis_{datetime.now():%Y%m%d}.xlsx',
    include_summary=True,
    include_charts=True,
    include_residuals=result.has_residuals,
    include_chart_images=True,
    export_html=True
)

print(f"Exported with {signals.count} signals detected")
```

## Troubleshooting

### "No module named 'openpyxl'"

```bash
pip install openpyxl
# or
pip install processbehavior[excel]
```

### Chart Images Not Appearing

```bash
pip install kaleido
```

### Large File Size

- Set `include_full_dataset=False`
- Reduce data before analysis
- Use HTML export for interactive charts instead of images

### Formatting Issues

- Ensure `format_cells=True`
- Check column widths in Excel
- Some very long values may need manual adjustment

## Next Steps

- [Plotting & Themes](plotting.md) - Chart customization before export
- [Xbar-S Analysis](../tutorials/xbar-s-analysis.ipynb) - Complete analysis workflow
- [API Reference](../reference/api.md) - Full to_excel() API
