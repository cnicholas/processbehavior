# Installation

## Requirements

ProcessBehavior requires Python 3.9 or later.

### Dependencies (installed automatically)
- **numpy** >= 1.23 - Numerical computing
- **pandas** >= 2.0, < 3 - Data manipulation
- **natsort** >= 8.0 - Natural sorting for factor levels
- **plotly** >= 5.18, < 7 - Interactive visualization

### Optional Dependencies
Install via `pip install "processbehavior[<extra>]"`:
- **openpyxl** >= 3.1 — Excel export (`[excel]`)
- **kaleido** — Static image export (`[images]`)

## Installation Methods

### From PyPI (Recommended)

```bash
pip install processbehavior
```

### With Excel Export

```bash
pip install "processbehavior[excel]"
```

### With Static Image Export

```bash
pip install "processbehavior[images]"
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
import processbehavior
print(f"ProcessBehavior version: {processbehavior.__version__}")

# Quick functionality test
import pandas as pd
import numpy as np
from processbehavior import ProcessBehavior

# Create sample data: 3 machines x 10 batches x 1 measurement
np.random.seed(42)
df = pd.DataFrame({
    'value': np.random.normal(100, 5, 30),
    'batch': [f'batch_{i//3 + 1}' for i in range(30)],
    'machine': ['A', 'B', 'C'] * 10
})

# Run a simple analysis
pb = ProcessBehavior(df)
study = pb.formulate(response=pb.cols.value, factors=[pb.cols.machine], time=pb.cols.batch)
result = study.execute()

print(f"Observed:   ODS {study.observed_design_state.sds}")
print(f"Analytical: ADS {study.analytical_design_state.sds}")
print(f"Valid charts: {study.valid_charts}")
print(f"Recommended: {study.recommended_chart}")
print("Installation verified!")
```

Expected output:
```
ProcessBehavior version: 0.1.1
Observed:   ODS 2
Analytical: ADS 2
Valid charts: ['Histogram', 'Xbar', 'S', 'X', 'mR']
Recommended: X
Installation verified!
```

## Verify Plotting

Plotting is included by default:

```python
# Test plotting
fig = result.plot()
fig.show()  # Opens interactive chart in browser
```

## Troubleshooting

### Jupyter Notebook Display Issues

For Jupyter notebooks, ensure you have the Plotly extension:
```bash
pip install jupyterlab "ipywidgets>=7.6"
```

## Next Steps

Now that ProcessBehavior is installed, continue to the [Quickstart](quickstart.ipynb) tutorial to create your first process behavior chart.
