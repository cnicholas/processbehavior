# processbehavior

[![PyPI version](https://img.shields.io/pypi/v/processbehavior.svg)](https://pypi.org/project/processbehavior/)
[![Python versions](https://img.shields.io/pypi/pyversions/processbehavior.svg)](https://pypi.org/project/processbehavior/)
[![CI](https://github.com/cnicholas/processbehavior/actions/workflows/ci.yml/badge.svg)](https://github.com/cnicholas/processbehavior/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/processbehavior.svg)](https://github.com/cnicholas/processbehavior/blob/main/LICENSE)

A Python library for **Process Behavior Analysis** following Thomas A. Bishop's Variance Analysis System (VAS) methodology.

Unlike traditional SPC packages, processbehavior faithfully implements Bishop's VAS equation-by-equation: automatic detection of the three-state design lineage (PDS / ODS / ADS), variance decomposition via R1-R6 residuals, and chart selection routed by the data your study can actually analyze.

## Installation

```bash
pip install processbehavior
```

Plotting (plotly) is included. For Excel export and static image export, install the corresponding extras:

```bash
pip install "processbehavior[excel]"   # openpyxl, for result.to_excel(...)
pip install "processbehavior[images]"  # kaleido, for static plot images
```

## Quickstart

```python
import processbehavior as pb

# Generate a sample dataset (replace with your own DataFrame)
df = pb.make_design(state=1, seed=42)
# Columns: 'time', 'factor 1', 'factor 2', 'y'

# Formulate — detects PDS / ODS / ADS and builds the analysis dataset
study = pb.formulate(
    df,
    response='y',
    time='time',
    factors=['factor 1', 'factor 2'],
)
print(f"Observed:   ODS {study.observed_design_state.sds}")
print(f"Analytical: ADS {study.analytical_design_state.sds}")
print(f"Recommended chart: {study.recommended_chart}")

# Execute analysis (routes by ADS)
result = study.execute()                       # uses the recommended chart
stats = result.statistics('Xbar')              # always {'N', 'center', 'lpl', 'upl'}
print(f"Center: {stats['center']}")

# When subgroup sizes differ the limits differ per subgroup, so the scalar
# 'lpl'/'upl' are None and stats['limits_vary'] is True. The per-subgroup
# limits live in the chart table:
print(result.get_chart('Xbar')[['center', 'lpl', 'upl']].head())

# Plot (interactive plotly figure)
result.plot()

# Export to Excel (requires the [excel] extra)
result.to_excel('analysis.xlsx')
```

The whole pipeline is one expression when you don't need the intermediate objects:

```python
pb.formulate(df, response='y', factors=['factor 1'], time='time').execute().plot()
```

Use `pb.ProcessBehavior(df)` directly when you want the fluent derived-variable verbs
(`.transform(...)`, `.bin(...)`), which attach before formulating. With that object in hand,
`pbd.cols.<column_name>` provides IDE auto-completion for column references.

## Key Concepts

### Two-Step Workflow

The API mirrors how analysts think:

1. **`formulate()`** - Understand your data structure. Detects PDS / ODS / ADS, identifies valid charts, and computes residuals. This is the expensive step.
2. **`execute()`** - Run analysis. Produces charts from the pre-computed data. This is cheap and can be called multiple times for different charts from the same study.

### Design-State Lineage (PDS / ODS / ADS)

processbehavior reports three design states at three points in the analysis lifecycle:

| State | Computed from | Codomain | Meaning |
|-------|---------------|----------|---------|
| **PDS** — Planned | Your declared `factors` and `time` (or an explicit `plan=`) | {1, 2} | What you intended to collect |
| **ODS** — Observed | Raw data, before NA filtering | {1..6} | What was actually collected (cells with all-NA responses count as "attempted but empty") |
| **ADS** — Analytical | Tidy data, after NA filtering | {0, 1, 2, 3} | What survives tidying; **drives chart selection, residual availability, and variance decomposition** |

The integer codes are Bishop's reference scale ("Bishop Table 1"):

| Code | Cell Sizes (N_kt) |
|------|--------------------|
| 1 | Complete grid, all N_kt >= 2 (full replication) |
| 2 | Complete grid, all N_kt = 1 (no replication) |
| 3 | Complete grid, mix of N_kt = 1 and N_kt >= 2 |
| 4 | Incomplete grid, occupied cells N_kt >= 2 |
| 5 | Incomplete grid, occupied cells N_kt = 1 |
| 6 | Incomplete grid, mixed N_kt |

ODS values in {4, 5, 6} collapse to ADS values in {1, 2, 3} during tidying (empty cells drop and the surviving subset becomes the analytical grid).

Access the lineage on a formulated study:

```python
study.plan_design_state        # PDS (when a plan was supplied)
study.observed_design_state    # ODS — what the raw data showed
study.analytical_design_state  # ADS — what the analysis is fit for
study.design()                 # DesignReport: full lineage in one object
```

### Residual System (R1-R6)

For factorial designs (ADS 1-3), processbehavior decomposes variation into diagnostic residuals:

- **R1** - The response centered at 0 (Bishop §13.1); the building block for the rest
- **R2** - Within-cell noise (measurement error, short-term fluctuation)
- **R3** - Interaction between design conditions and time
- **R4** - Time main effects (trends, seasonality, batch effects)
- **R5** - Design condition main effects, all factors combined
- **R6** - A single design factor's main effect (machine-to-machine, operator bias)

```python
# Chart any residual
result = study.execute(chart='Xbar', value='R4')  # Time main effects
result = study.execute(chart='Xbar', value='R5')  # Design condition main effects
result = study.execute(chart='Xbar', value='R6', by=['machine'])  # One factor's effect
```

Not every chart pairs with every residual — the valid pairs depend on the ADS.
`study.residual_charts` lists them, and `study.why_not('X', value='R4')` explains
any pair it refuses.

### Stratified Analysis

For X/mR charts with grouping factors, processbehavior produces a single combined chart with per-stratum limits:

```python
result = study.execute(chart='X', by=['machine'])

# Drill into a specific stratum
for stratum in result.strata:
    focused = result.focus(stratum)
    focused.plot()
```

## Validation

processbehavior's analytical outputs are continuously verified against Dr. Thomas A. Bishop's Minitab reference results. The validator at `validation/e2e_bishop_report.py` runs **280 numerical assertions** through the full formulate → execute → capability / loss / maximum-information pipeline:

| Analytical Design State | Datasets validated | Assertions | Status |
|---|---|---|---|
| ADS 1 (full replication) | `PM SDS 1` from `PBTESTDATABASE_T100.csv` | 102 | ✅ all pass |
| ADS 2 (no replication) | `PM SDS 2` | 89 | ✅ all pass |
| ADS 3 (partial replication) | `PM SDS 3` | 89 | ✅ all pass |

Reference data is Bishop's published Minitab golden output. The library matches every chart center, control limit, signal classification, capability index, and loss-function value to within the precision Bishop reports.

ODS 4–6 (incomplete-grid scenarios) are detected and routed correctly but their end-to-end Bishop-reference coverage is pending. The synthetic generators in `make_design(state=4|5|6)` produce data with the structural shape Bishop's Table 1 specifies; full numerical validation against Bishop's incomplete-grid Minitab output is on the 0.2.0 roadmap.

The full report renders to `validation/e2e_bishop_report.html`.

## Features

- **Three-state lineage**: PDS / ODS / ADS detected automatically; chart selection routes by ADS
- **Correct charts**: Xbar-S, X (Individual), mR (Moving Range), Histogram with proper limit calculations
- **Variance decomposition**: R1-R6 residuals for factorial designs
- **Effects analysis**: Main effects, time effects, and interaction effects
- **Stratified charts**: Automatic per-stratum limits for grouped individual data
- **Signal detection**: all eight Western Electric rules on X/mR; Rule 1 (3-sigma) on Xbar/S, where the run- and zone-based rules need a time order that subgroup comparisons don't have
- **IDE support**: Column auto-completion via `ProcessBehavior(df).cols`
- **Self-diagnostic errors**: Helpful messages that say what's available and how to fix it
- **Excel export**: Publication-ready workbooks with charts and statistics
- **Interactive plots**: Plotly-based charts with hover details

## Scope

processbehavior is the **computational engine** for Bishop's VAS methodology. It handles data ingestion, design-state lineage detection (PDS / ODS / ADS), residual computation, chart generation, and export.

For the **curated analyst experience** — guided workflows, interactive dashboards, and collaboration features — see [processbehavior.com](https://processbehavior.com).

For the **methodological foundation** — the theory behind VAS, design-state classification, and residual interpretation — see the forthcoming book by Dr. Thomas A. Bishop and Chris Nicholas.

## License

Apache 2.0
