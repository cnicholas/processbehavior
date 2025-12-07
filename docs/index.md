# ProcessBehavior Documentation

Welcome to the comprehensive documentation for **ProcessBehavior** - a powerful, user-friendly Statistical Process Control (SPC) library for Python implementing the Wheeler/Bishop methodology.

## What is ProcessBehavior?

ProcessBehavior is a production-ready library that brings sophisticated statistical process control to Python with an emphasis on:

- **Intelligent Automation** - Automatic detection of data structure (Sampling Design States)
- **Comprehensive Analysis** - Variance Analysis System (VAS) with R1-R5 residual decomposition
- **Beautiful Visualizations** - Interactive control charts with signal detection
- **Professional Output** - Export to Excel with multiple sheets and embedded charts
- **IDE Auto-completion** - Full support for column names and chart types

## Key Features

### Simple Two-Step API

```python
import pandas as pd
from processbehavior import ProcessDataFrame

# Load your data
df = pd.read_csv('measurements.csv')
pdf = ProcessDataFrame(df)

# Step 1: Formulate the study (auto-detects SDS)
study = pdf.formulate(
    response=pdf.columns.measurement,  # IDE auto-completes column names
    time=pdf.columns.time
)

# Inspect what was detected
print(study.sds_name)        # e.g., "SDS 1: Full Replication"
print(study.valid_charts)    # e.g., ['Xbar', 'S', 'R', 'Imr']
print(study.recommended_chart)  # e.g., 'Xbar'

# Step 2: Run the analysis
result = study.analyze()  # Uses recommended chart
# Or specify: result = study.analyze(chart='Imr')

# View results
result.plot()
result.detect_signals()
result.to_excel('analysis.xlsx')
```

### Multiple Chart Types

- **Xbar Charts** - Compare means across rational subgroups
- **S Charts** - Compare variation across subgroups
- **IMR Charts** - Individual moving range for time series
- **R Charts** - Range charts for subgroup variation
- **Residual Charts** - R2, R3, R4, R5 for variance decomposition

### Automatic SDS Detection

The library automatically detects your data structure (SDS 0-6) and recommends appropriate analysis:

| SDS | Description | Recommended Chart |
|-----|-------------|-------------------|
| **SDS 0** | Simple series (no grouping) | IMR |
| **SDS 1** | Full factorial with replication | Xbar/S |
| **SDS 2** | Full factorial without replication | IMR or Xbar |
| **SDS 3** | Partial replication | Xbar/S |
| **SDS 4** | Time series over single condition | IMR |
| **SDS 5** | Nested design | Xbar/S |
| **SDS 6** | Incomplete/irregular grid | Stratified IMR |

### VAS Framework

Complete variance decomposition with Wheeler/Bishop R1-R5 residuals:

| Residual | Description | Use Case |
|----------|-------------|----------|
| **R1** | Total variation | Overall process behavior |
| **R2** | Within-cell variation | Measurement noise |
| **R3** | Interaction effects | Factor x Time patterns |
| **R4** | Time effects | Temporal drift/cycles |
| **R5** | Factor effects | Between-group differences |

Access residual charts via:
```python
# Available for SDS 1-3
result = study.analyze(chart='R2')  # Within-cell residuals
result = study.analyze(chart='R4')  # Time effects chart
```

### Signal Detection (WECO Rules)

Built-in Western Electric rules for automatic signal detection:

```python
signals = result.detect_signals()
print(signals.has_signals)  # True/False
print(signals.count)        # Number of signals
print(signals.summary())    # Detailed breakdown
```

Rules include:
- **Rule 1**: Point beyond 3 sigma limits
- **Rule 2**: 2 of 3 consecutive points beyond 2 sigma
- **Rule 3**: 4 of 5 consecutive points beyond 1 sigma
- **Rule 4**: 8 consecutive points on one side of center

### Professional Export

Export complete analysis to Excel with:

```python
result.to_excel('analysis.xlsx')
```

Generates:
- Summary tab with analysis metadata
- Chart data tabs with control limits
- Residuals tab (R1-R5, RCR1-RCR5)
- Effects and interactions tabs
- Embedded PNG chart images
- Interactive HTML charts (separate file)

### Stratified Analysis

Automatically create separate control charts for each group:

```python
study = pdf.formulate(
    response=pdf.columns.fill_weight,
    factors=[pdf.columns.lane, pdf.columns.phase],
    time=pdf.columns.pull
)

# IMR creates separate charts per lane/phase combination
result = study.analyze(chart='Imr')

# Access individual charts
for chart_name, chart_data in result.charts.items():
    print(f"{chart_name}: center={chart_data['statistics']['center']}")
```

### Garbage Character Handling

Automatically cleans common data issues:

```python
# These are automatically converted to NA:
# '*', '?', '--', 'ND', 'BDL', 'BQL', '<LOD', etc.

pdf = ProcessDataFrame(df)  # Warns about cleaned values
pdf = ProcessDataFrame(df, na_values=['-999', 'MISSING'])  # Add custom
```

## Quick Links

- [Installation Guide](getting-started/installation.md) - Get started in 5 minutes
- [Quick Start Tutorial](getting-started/quickstart.md) - Your first analysis
- [Understanding SDS](tutorials/understanding-sds.md) - Learn the framework
- [API Reference](api/process-dataframe.md) - Complete API documentation
- [Architecture](architecture.md) - Module and class hierarchy
- [GitHub Repository](https://github.com/cnicholas/processbehavior) - Source code

## Use Cases

ProcessBehavior is ideal for:

- **Manufacturing Quality Control** - Monitor production processes across lines/shifts
- **Laboratory QA** - Track measurement systems and instrument drift
- **Pharmaceutical** - Batch-to-batch variation analysis
- **Healthcare Monitoring** - Patient outcome tracking over time
- **Business Analytics** - KPI monitoring with automatic alerting
- **Research** - Experimental data with factorial designs

## Why ProcessBehavior?

| Feature | ProcessBehavior | Other Libraries |
|---------|----------------|-----------------|
| Automatic SDS detection | Yes | No |
| VAS residuals (R1-R5) | Yes | No |
| Stratified charts | Yes | Limited |
| Signal detection | WECO rules | Basic |
| Excel export | Multi-sheet with charts | CSV only |
| Interactive plots | Plotly | Static |
| IDE auto-completion | Full support | None |
| Residual charts | R2, R3, R4, R5 | No |

## Getting Help

- **Documentation**: You're reading it!
- **GitHub Issues**: [Report bugs or request features](https://github.com/cnicholas/processbehavior/issues)
- **Examples**: Check the `examples/` directory in the repository

## License

ProcessBehavior is released under the MIT License. See [LICENSE](https://github.com/cnicholas/processbehavior/blob/main/LICENSE) for details.

## Credits

Based on the statistical methodology developed by Donald J. Wheeler and Thomas P. Bishop.

---

**Ready to get started?** Head over to the [Quick Start Guide](getting-started/quickstart.md)!
