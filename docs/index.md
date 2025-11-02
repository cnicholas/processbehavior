# ProcessBehavior Documentation

Welcome to the comprehensive documentation for **ProcessBehavior** - a powerful, user-friendly Statistical Process Control (SPC) library for Python implementing the Wheeler/Bishop methodology.

## What is ProcessBehavior?

ProcessBehavior is a production-ready library that brings sophisticated statistical process control to Python with an emphasis on:

- **Intelligent Automation** - Automatic detection of data structure (Sampling Design States)
- **Comprehensive Analysis** - Variance Analysis System (VAS) with R1-R5 residual decomposition
- **Beautiful Visualizations** - Interactive control charts with signal detection
- **Professional Output** - Export to Excel with multiple sheets and embedded charts
- **Maximum Flexibility** - Works with any data structure, from simple series to complex factorial designs

## Key Features

### 🎯 Simple API
```python
import pandas as pd
from processbehavior import ProcessDataFrame

# Load your data
df = pd.read_csv('measurements.csv')
pdf = ProcessDataFrame(df)

# Run analysis
result = pdf.analyze(
    response_var='measurement',
    chart_type='Imr'
).calculate()

# Plot it
result.plot()
```

### 📊 Multiple Chart Types

- **Xbar Charts** - Compare means across rational subgroups
- **S Charts** - Compare variation across subgroups
- **IMR Charts** - Individual moving range for time series
- **R Charts** - Range charts for subgroup variation

### 🔍 Automatic SDS Detection

The library automatically detects your data structure (SDS 0-6) and adapts the analysis:

- **SDS 0**: Simple series (no grouping)
- **SDS 1**: Full factorial with replication
- **SDS 2**: Full factorial without replication
- **SDS 3**: Partial replication
- **SDS 4**: Time series over single condition
- **SDS 5**: Cross-sectional comparison
- **SDS 6**: Incomplete/irregular grid

### 📈 VAS Framework

Complete variance decomposition with Wheeler/Bishop R1-R5 residuals:

- **R1** - Total variation
- **R2** - Within-cell variation
- **R3** - Interaction effects
- **R4** - Time effects
- **R5** - Factor effects

### ⚡ Signal Detection

Built-in Western Electric rules for automatic signal detection:

- Rule 1: Point beyond 3σ
- Rule 2: 2 of 3 consecutive points beyond 2σ
- Rule 3: 4 of 5 consecutive points beyond 1σ
- Rule 4: 8 consecutive points on one side
- Rules 5-8: Additional pattern detection

### 📤 Professional Export

Export complete analysis to Excel with:

- Summary tab with analysis metadata
- Chart tabs with raw data
- Residuals tab with original data context
- Effects and interactions tabs
- Visual charts tab with embedded PNG images
- Interactive HTML charts

## Quick Links

- [Installation Guide](getting-started/installation.md) - Get started in 5 minutes
- [Quick Start Tutorial](getting-started/quickstart.md) - Your first analysis
- [Understanding SDS](tutorials/understanding-sds.md) - Learn the framework
- [API Reference](api/process-dataframe.md) - Complete API documentation
- [GitHub Repository](https://github.com/cnicholas/processbehavior) - Source code

## Use Cases

ProcessBehavior is perfect for:

- **Manufacturing Quality Control** - Monitor production processes
- **Laboratory QA** - Track measurement systems
- **Healthcare Monitoring** - Patient outcome tracking
- **Business Analytics** - KPI monitoring over time
- **Research** - Experimental data analysis

## Why ProcessBehavior?

| Feature | ProcessBehavior | Other Libraries |
|---------|----------------|-----------------|
| Automatic SDS detection | ✅ | ❌ |
| VAS residuals (R1-R5) | ✅ | ❌ |
| Stratified charts | ✅ | Limited |
| Signal detection | ✅ WECO rules | Basic |
| Excel export | ✅ Multi-sheet | CSV only |
| Interactive plots | ✅ Plotly | Static |
| Type safety | ✅ Full hints | Partial |
| Documentation | ✅ Comprehensive | Limited |

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
