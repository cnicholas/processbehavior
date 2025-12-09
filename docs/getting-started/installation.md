# Installation

## Requirements

ProcessBehavior requires Python 3.9 or later.

### Core Dependencies
- **pandas** >= 2.0 - Data manipulation
- **natsort** >= 8.0 - Natural sorting for factor levels

### Optional Dependencies
- **plotly** >= 5.18 - Interactive visualization
- **openpyxl** >= 3.1 - Excel export
- **statsmodels** >= 0.14 - Effects analysis
- **ruptures** >= 1.1 - Changepoint detection

## Installation Methods

### From PyPI (Recommended)

```bash
pip install processbehavior
```

### With Plotting Support

```bash
pip install processbehavior[plotting]
```

### With All Optional Features

```bash
pip install processbehavior[plotting,excel,effects,changepoints]
```

Or install everything:

```bash
pip install processbehavior[all]
```

### From Source (Development)

```bash
git clone https://github.com/cnicholas/processbehavior.git
cd processbehavior
pip install -e ".[dev]"
```

## Verifying Installation

After installation, verify everything works:

```python
import processbehavior as pb

# Check version
print(f"ProcessBehavior version: {pb.__version__}")

# Quick functionality test
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'value': np.random.normal(100, 5, 30),
    'time': range(30)
})

# Run a simple analysis
pdf = pb.ProcessDataFrame(df)
study = pdf.formulate(response=pdf.columns.value, time=pdf.columns.time)
result = study.analyze()

print(f"SDS detected: {study.sds}")
print(f"Charts available: {result.all_charts}")
print("Installation verified!")
```

Expected output:
```
ProcessBehavior version: 0.1.0
SDS detected: 4
Charts available: ['Imr', 'R']
Installation verified!
```

## Optional: Verify Plotting

If you installed with plotting support:

```python
# Test plotting
fig = result.plot()
fig.show()  # Opens interactive chart in browser
```

## Troubleshooting

### ImportError: No module named 'plotly'

You need to install plotting dependencies:
```bash
pip install processbehavior[plotting]
```

### ImportError: No module named 'openpyxl'

For Excel export functionality:
```bash
pip install processbehavior[excel]
```

### Jupyter Notebook Display Issues

For Jupyter notebooks, ensure you have the Plotly extension:
```bash
pip install jupyterlab "ipywidgets>=7.6"
```

## Next Steps

Now that ProcessBehavior is installed, continue to the [Quickstart](quickstart.ipynb) tutorial to create your first process behavior chart.
