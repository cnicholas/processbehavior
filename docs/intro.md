# ProcessBehavior

**Python-native Process Behavior Charts with Wheeler's Variance Analysis System**

ProcessBehavior brings Donald Wheeler's rigorous statistical process control methodology to Python, with a modern API designed for data scientists and quality engineers.

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
        E[analyze]
        F[AnalysisResult]
    end

    subgraph Output
        G[📈 plot]
        H[📁 to_excel]
        I[🔍 detect_signals]
    end

    A --> B
    B --> C
    C -->|"SDS detection"| D
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
3. **Formulate** your study — ProcessBehavior detects the Sampling Design State (SDS) and determines:
   - Design state (SDS 1–6)
   - Valid and recommended charts
   - Available VAS residuals (R1–R5)
   - Main effects analysis
   - Interaction analysis
4. **Analyze** — run calculations and get results with charts, statistics, and residuals
5. **Output** — visualize with interactive plots, export to Excel, or detect Western Electric rule violations

## What Makes ProcessBehavior Different?

Unlike traditional SPC packages that require you to manually select chart types and configure parameters, ProcessBehavior automatically detects your data's **Sampling Design State (SDS)** and recommends the appropriate analysis approach.

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

# ProcessBehavior automatically detects SDS and recommends charts
print(f"Detected: SDS {study.observed_design_state.sds}")
print(f"Recommended: {study.recommended_chart}")

# Analyze and visualize
result = study.execute()
result.plot(show_zones=True, show_signals=True)
```

## Key Features

### Automatic Sampling Design Detection
ProcessBehavior identifies six distinct sampling design states (SDS 1-6) and configures the analysis accordingly:

| SDS | Design | Charts |
|-----|--------|--------|
| 1 | Full replication | Xbar-S with VAS residuals |
| 2 | No replication | Xbar-S with moving range |
| 3 | Partial replication | Hybrid approach |
| 4 | Single stream | Stratified XmR |
| 5 | Nested/hierarchical | Multi-level analysis |
| 6 | Unstructured | Special handling |

### Wheeler's Variance Analysis System (VAS)
For replicated designs, ProcessBehavior computes the complete residual decomposition:
- **R1**: Total deviation (Y - grand mean)
- **R2**: Within-cell variation (unexplained noise)
- **R3**: Interaction residual (factor × time)
- **R4**: Time effect + within-cell variation
- **R5**: Factor effect + within-cell variation

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

For plotting support:
```bash
pip install processbehavior[plotting]
```

## Quick Links

- [Quickstart](getting-started/quickstart.ipynb) - Get up and running in 5 minutes
- [Basic XmR Chart](tutorials/basic-imr.ipynb) - Your first control chart
- [Sampling Design States](user-guide/sds-detection.md) - Understanding sampling designs
- [API Reference](reference/api.md) - Complete API reference

## Philosophy

ProcessBehavior follows Wheeler's philosophy that **process behavior charts are not about statistics—they're about understanding variation**. The package is designed to:

1. **Guide, not dictate** - Recommend appropriate analyses while allowing expert override
2. **Return plain DataFrames** - No custom classes; results work with your existing pandas workflow
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
