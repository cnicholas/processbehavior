# ProcessBehavior

**Python-native Process Behavior Charts with Bishop's Variance Analysis System**

ProcessBehavior brings Thomas A. Bishop's rigorous Variance Analysis System (VAS) methodology to Python, with a modern API designed for data scientists and quality engineers.

## Analyst Workflow

The ProcessBehavior workflow guides you from raw data to actionable insights:

```{mermaid}
flowchart LR
    subgraph Input
        A[📊 pandas DataFrame]
    end

    subgraph Formulation
        B[ProcessBehavior]
        C[formulate]
        D[Study]
    end

    subgraph Analysis
        E[execute]
        F[AnalysisResult]
    end

    subgraph Output
        G[📈 plot]
        H[📁 to_excel]
        I[🔍 detect_signals]
    end

    A --> B
    B --> C
    C -->|"DS detection"| D
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I

    style A fill:#e1f5fe
    style D fill:#fff3e0
    style F fill:#e8f5e9
    style G fill:#fce4ec
    style H fill:#fce4ec
    style I fill:#fce4ec
```

**Key Steps:**
1. **Load** your process data into a pandas DataFrame
2. **Wrap** with `ProcessBehavior` for IDE auto-completion
3. **Formulate** your study — ProcessBehavior detects the Design State (DS) and determines:
   - Design State (DS 1–6)
   - Valid and recommended charts
   - Available VAS residuals (R1–R5)
   - Main effects analysis
   - Interaction analysis
4. **Analyze** — run calculations and get results with charts, statistics, and residuals
5. **Output** — visualize with interactive plots, export to Excel, or detect Western Electric rule violations

## What Makes ProcessBehavior Different?

Unlike traditional SPC packages that require you to manually select chart types and configure parameters, ProcessBehavior automatically detects your data's **Design State (DS)** and recommends the appropriate analysis approach.

```python
import pandas as pd
from processbehavior import ProcessBehavior

# Load your data
df = pd.read_csv("process_data.csv")

# Formulate your study
pb = ProcessBehavior(df)
study = pb.formulate(
    response=pb.cols.measurement,
    factors=[pb.cols.machine, pb.cols.operator],
    time=pb.cols.timestamp
)

# ProcessBehavior automatically detects DS and recommends charts
print(f"Detected: DS {study.observed_design_state.sds}")
print(f"Recommended: {study.recommended_chart}")

# Analyze and visualize
result = study.execute()
result.plot(show_zones=True, highlight_signals=True)
```

## Key Features

### Automatic Design State Detection
ProcessBehavior identifies six distinct Design States (DS 1-6) and configures the analysis accordingly:

| DS | Name | Cell Sizes (N_kt) |
|-----|------|--------------------|
| 1 | Full Replication | All N_kt >= 2 |
| 2 | No Replication | All N_kt = 1 |
| 3 | Partial Replication | Mix of N_kt = 1 and N_kt >= 2 |
| 4 | Incomplete, No Singletons | Empty cells + all observed N_kt >= 2 |
| 5 | Incomplete, No Replication | Empty cells + all observed N_kt = 1 |
| 6 | Incomplete, With Singletons | Empty cells + mixed N_kt |

### Dr. Thomas A. Bishop's Variance Analysis System (VAS)
For replicated designs, ProcessBehavior computes the complete residual decomposition:
- **R1**: Total deviation (Y - grand mean)
- **R2**: Within-cell variation (unexplained noise)
- **R3**: Interaction residual (factor × time)
- **R4**: Time effect + within-cell variation
- **R5**: Factor effect + within-cell variation
- **R6**: Factor main effect residual (computed on-the-fly per factor)

### Western Electric Rules
Built-in signal detection with configurable rules:
- Rule 1: Point beyond 3σ limits
- Rules 2-8: Pattern detection for runs, trends, and zone violations

### Publication-Quality Visualization
Interactive Plotly charts with multiple themes, zone shading, and professional styling.

## Installation

```bash
pip install processbehavior
```

Plotting (plotly) and Excel export (openpyxl) are included by default. For static image export:
```bash
pip install processbehavior[images]
```

## Quick Links

- [Quickstart](getting-started/quickstart.ipynb) - Get up and running in 5 minutes
- [Basic X Chart](tutorials/basic-imr.ipynb) - Your first control chart
- [Design States](user-guide/sds-detection.md) - Understanding Design States
- [API Reference](reference/api.md) - Complete API reference

## Philosophy

ProcessBehavior follows Wheeler's philosophy that **process behavior charts are not about statistics—they're about understanding variation**. The package is designed to:

1. **Guide, not dictate** - Recommend appropriate analyses while allowing expert override
2. **DataFrame-backed results** - Access chart data, residuals, and effects as standard pandas DataFrames
3. **Separate concerns** - Formulation, analysis, and visualization are distinct steps
4. **Be explicit** - No hidden defaults; all parameters are visible and documented

## License

ProcessBehavior is released under the Apache 2.0 License.

## Citation

If you use ProcessBehavior in your research, please cite:

```bibtex
@software{processbehavior,
  author = {Nicholas, Chris and Bishop, Tom},
  title = {ProcessBehavior: Python Process Behavior Charts},
  year = {2025},
  url = {https://github.com/cnicholas/processbehavior}
}
```
