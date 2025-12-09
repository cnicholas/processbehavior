# Chart Types

ProcessBehavior supports multiple chart types, each suited for different data structures and analytical questions. This guide helps you choose the right chart.

## Available Chart Types

### Standard Charts

| Chart | Full Name | Purpose | Requirements |
|-------|-----------|---------|--------------|
| **Xbar** | X-bar Chart | Monitor subgroup means | n >= 2 per subgroup |
| **S** | S Chart | Monitor subgroup variation | n >= 2 per subgroup |
| **Imr** | Individual-Moving Range | Monitor individual values | Any structure |
| **R** | Range Chart | Monitor subgroup ranges | n >= 2 per subgroup |

### VAS Residual Charts

| Chart | Residual | Purpose |
|-------|----------|---------|
| **R2_S** | R2 (within-cell) | Check within-group variation stability |
| **R2_Imr** | R2 (within-cell) | Check within variation (no replication) |
| **R3_Imr** | R3 (interaction) | Detect factor × time interactions |
| **R4_Imr** | R4 (time) | Detect time effects |
| **R5_Imr** | R5 (factor) | Detect factor effects |

## Using Chart Auto-Completion

After formulation, use the `study.charts` accessor for IDE auto-completion:

```python
study = pdf.formulate(
    response=pdf.columns.weight,
    factors=[pdf.columns.lane],
    time=pdf.columns.batch
)

# Type study.charts. and your IDE shows only valid charts
result = study.analyze(study.charts.Xbar)    # Xbar-S analysis
result = study.analyze(study.charts.Imr)     # Stratified IMR
result = study.analyze(study.charts.R4_Imr)  # Time effects chart
```

This prevents errors from using invalid chart types.

## Chart Selection Guide

### Question: "Are my subgroups different from each other?"

**Use: Xbar Chart**

```python
result = study.analyze(study.charts.Xbar)
fig = result.plot(chart='Xbar', show_zones=True, show_signals=True)
```

Points beyond limits indicate subgroups that differ from the overall average.

### Question: "Is within-group variation stable?"

**Use: S Chart**

```python
result = study.analyze(study.charts.Xbar)  # Creates both Xbar and S
fig = result.plot(chart='S', show_zones=True)
```

Points beyond limits indicate subgroups with unusual variation.

### Question: "How is my process behaving over time?"

**Use: IMR Chart**

```python
result = study.analyze(study.charts.Imr)
fig = result.plot(show_zones=True, show_rules=True)
```

For stratified data, this creates one IMR per factor level.

### Question: "Do factor effects change over time?"

**Use: R3_Imr Chart (Interactions)**

```python
result = study.analyze(study.charts.R3_Imr)
fig = result.plot(title='Factor × Time Interactions')
```

Signals indicate that factor effects are not consistent over time.

### Question: "Are there time trends after removing factor effects?"

**Use: R4_Imr Chart**

```python
result = study.analyze(study.charts.R4_Imr)
fig = result.plot(title='Time Effects')
```

Signals indicate process drift or shifts over time.

### Question: "Are there factor differences after removing time effects?"

**Use: R5_Imr Chart**

```python
result = study.analyze(study.charts.R5_Imr)
fig = result.plot(title='Factor Effects')
```

Signals indicate true differences between factor levels.

## Chart Validity by SDS

Not all charts are valid for all Sampling Design States:

| Chart | SDS 1 | SDS 2 | SDS 3 | SDS 4 | SDS 5-6 |
|-------|-------|-------|-------|-------|---------|
| Xbar | ✅ | ✅ | ✅ | ❌ | ❌ |
| S | ✅ | ✅ | ✅ | ❌ | ❌ |
| Imr | ✅ | ✅ | ✅ | ✅ | ✅ |
| R | ✅ | ✅ | ✅ | ✅ | ✅ |
| R2_S | ✅ | ❌ | ⚠️ | ❌ | ❌ |
| R2_Imr | ✅ | ✅ | ✅ | ❌ | ❌ |
| R3_Imr | ✅ | ✅ | ✅ | ❌ | ❌ |
| R4_Imr | ✅ | ✅ | ✅ | ❌ | ❌ |
| R5_Imr | ✅ | ✅ | ✅ | ❌ | ❌ |

Use `study.valid_charts` to see what's available:

```python
print(f"Valid charts: {study.valid_charts}")
print(f"Residual charts: {study.residual_charts}")
```

## Understanding Xbar-S Charts

### The Xbar Chart

Plots the **mean** of each subgroup (factor level at each time point).

- **Centerline**: Grand mean (Y̅)
- **Control Limits**: Based on within-subgroup variation
- **Interpretation**: Points beyond limits indicate subgroups with unusual means

### The S Chart

Plots the **standard deviation** of each subgroup.

- **Centerline**: Pooled within-subgroup standard deviation
- **Control Limits**: Based on chi-square distribution
- **Interpretation**: Points beyond limits indicate subgroups with unusual variation

### Reading Order

1. **First check S chart** - Variation must be stable
2. **Then interpret Xbar** - Only meaningful if S is stable
3. **Investigate signals** - What makes that subgroup different?

## Understanding IMR Charts

### The I (Individual) Chart

Plots each individual observation.

- **Centerline**: Average of all observations (X̅)
- **Control Limits**: X̅ ± 2.66 × R̅ (average moving range)
- **Interpretation**: Points beyond limits indicate special causes

### The R (Moving Range) Chart

Plots the absolute difference between consecutive observations.

- **Centerline**: Average moving range (R̅)
- **UCL**: 3.27 × R̅
- **LCL**: 0 (range cannot be negative)
- **Interpretation**: Large ranges indicate sudden changes

## Stratified IMR Charts

When you have factors and time, IMR creates **one chart per factor level**:

```python
result = study.analyze(study.charts.Imr)

# Creates: Imr_Lane_A, Imr_Lane_B, Imr_Lane_C, Imr_Lane_D
print(result.all_charts)
print(result.list_strata())  # ['Lane_A', 'Lane_B', 'Lane_C', 'Lane_D']
```

Each stratum has its own control limits based on its internal variation:

```python
# View individual lane
fig = result.plot(chart='Imr_Lane_A', show_zones=True)

# View all lanes in faceted plot
fig = result.plot(facet=True, ncols=2, show_zones=True)
```

## Re-centered Residual Charts

By default, residual charts are centered at zero. Use `recentered=True` to show residuals on the original measurement scale:

```python
# Zero-centered (default)
result = study.analyze(study.charts.R4_Imr)

# Re-centered on original scale
result = study.analyze(study.charts.R4_Imr, recentered=True)
```

Re-centering uses:
- R4: RCR4 = R4 + Y̅_t (adds back time mean)
- R5: RCR5 = R5 + Y̅_k (adds back factor mean)

## Decision Tree

```
Do you have factors?
├── No → Do you have time?
│   ├── No → SDS 0: Basic statistics only
│   └── Yes → SDS 4: Use IMR
└── Yes → Do you have time?
    ├── No → Use Xbar to compare factors
    └── Yes → Do you have replication (n>=2 per cell)?
        ├── All cells → SDS 1: Full Xbar-S + VAS residuals
        ├── Some cells → SDS 3: Hybrid Xbar-S + limited VAS
        └── No cells → SDS 2: Xbar-S with MR-based limits
```

## Summary

| Question | Chart | Signal Meaning |
|----------|-------|----------------|
| Are groups different? | Xbar | Group deviates from average |
| Is variation stable? | S | Group has unusual variation |
| Process over time? | Imr | Special cause detected |
| Interactions? | R3_Imr | Factor effect changes over time |
| Time trends? | R4_Imr | Process drift/shift |
| Factor effects? | R5_Imr | True factor differences |

## Next Steps

- [Plotting & Themes](plotting.md) - Visualization options for all charts
- [VAS Residuals](residuals.md) - Deep dive into VAS residuals
- [Stratified Analysis](../tutorials/stratified-analysis.ipynb) - Stratified IMR tutorial
