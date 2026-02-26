# Chart Types

ProcessBehavior supports multiple chart types, each suited for different data structures and analytical questions. This guide helps you choose the right chart.

## Available Chart Types

### Standard Charts

| Chart | Full Name | Purpose | Requirements |
|-------|-----------|---------|--------------|
| **Xbar** | X-bar Chart | Monitor subgroup means | n >= 2 per subgroup |
| **S** | S Chart | Monitor subgroup variation | n >= 2 per subgroup |
| **XmR** | XmR Chart | Monitor individual values | Any structure |
| **R** | Range Chart | Monitor Individual ranges | n >= 2 per subgroup |

### VAS Residual Charts

Use the `value` parameter to chart residuals instead of the response:

| Residual | Chart | Purpose |
|----------|-------|---------|
| **R2** | S or XmR | Check within-group variation stability |
| **R3** | XmR | Detect factor × time interactions |
| **R4** | XmR | Detect time effects |
| **R5** | Xbar or XmR | Detect factor effects |

```python
# Chart R5 residuals on Xbar
result = study.execute(chart='Xbar', value='R5')

# Chart R4 residuals on stratified XmR
result = study.execute(chart='XmR', by=['lane'], value='R4')
```

## The `by` Parameter

The `by` parameter controls grouping and stratification:

```python
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)

# Xbar/S - aggregate by different levels
result = study.execute(chart='Xbar')                  # By all factors (default)
result = study.execute(chart='Xbar', by=['lane'])     # By single factor
result = study.execute(chart='Xbar', by=[])           # Collapse to grand mean

# XmR - stratify by factor(s)
result = study.execute(chart='XmR', by=['lane'])      # One chart per lane
```

Use `study.valid_charts` and `study.available_residuals` to see options.

## Chart Selection Guide

### Question: "Are my subgroups different from each other?"

**Use: Xbar Chart**

```python
result = study.execute(chart='Xbar')
fig = result.plot(chart='Xbar', show_zones=True, show_signals=True)
```

Points beyond limits indicate subgroups that differ from the overall average.

### Question: "Is within-group variation stable?"

**Use: S Chart**

```python
result = study.execute(chart='S')
fig = result.plot(chart='S', show_zones=True)
```

Points beyond limits indicate subgroups with unusual variation.

### Question: "How is my process behaving over time?"

**Use: XmR Chart**

```python
result = study.execute(chart='XmR', by=['lane'])
fig = result.plot(show_zones=True, show_rules=True)
```

For stratified data, this creates one XmR per factor level.

### Question: "Do factor effects change over time?"

**Use: XmR with `value='R3'` (Interactions)**

```python
result = study.execute(chart='XmR', by=['lane'], value='R3')
fig = result.plot(title='Factor × Time Interactions')
```

Signals indicate that factor effects are not consistent over time.

### Question: "Are there time trends after removing factor effects?"

**Use: XmR with `value='R4'`**

```python
result = study.execute(chart='XmR', by=['lane'], value='R4')
fig = result.plot(title='Time Effects')
```

Signals indicate process drift or shifts over time.

### Question: "Are there factor differences after removing time effects?"

**Use: XmR with `value='R5'`**

```python
result = study.execute(chart='XmR', by=['lane'], value='R5')
fig = result.plot(title='Factor Effects')
```

Signals indicate true differences between factor levels.

## Chart Validity by SDS

Not all charts are valid for all Sampling Design States:

| Chart | SDS 1 | SDS 2 | SDS 3 | SDS 4 | SDS 5-6 |
|-------|-------|-------|-------|-------|---------|
| Xbar | ✅ | ✅ | ✅ | ❌ | ❌ |
| S | ✅ | ✅ | ✅ | ❌ | ❌ |
| XmR | ✅ | ✅ | ✅ | ✅ | ✅ |
| R | ✅ | ✅ | ✅ | ✅ | ✅ |

**Residual availability** depends on SDS. Use `study.available_residuals` to check:

```python
print(f"Valid charts: {study.valid_charts}")
print(f"Available residuals: {study.available_residuals}")
```

**Note on R2**: SDS 2 and 6 use the moving average method; SDS 1, 3, 4, 5 use within-cell deviation (R2 = Y - Ȳ_kt). See [VAS Residuals](residuals.md) for details.

## Companion Charts

Wheeler recommends reading Xbar and S charts together (S first, then Xbar), and similarly for XmR and R. The `companion` parameter returns both charts in one result:

```python
# Returns both Xbar and S charts
result = study.execute(chart='Xbar', companion=True)
result.plot(chart='Xbar')  # Xbar chart
result.plot(chart='S')     # S chart

# Returns both XmR and R charts, stratified
result = study.execute(chart='XmR', by=['lane'], companion=True)
```

Either chart in the pair triggers the pair: `chart='S', companion=True` also returns Xbar+S.

## Effects and Interaction Charts

When your study has factors, ProcessBehavior can visualize main effects and interactions. These charts help answer: *Are the factor and time effects practically significant?*

| Chart | What It Shows | Requirements |
|-------|---------------|--------------|
| **Effects** | All main effects (factor + time) combined | Factors and time |
| **MainEffects** | Factor main effects only | Factors |
| **TimeEffects** | Time main effects only | Time |
| **TimeInteraction** | Factor x time interaction | Factors and time |
| **FactorInteraction** | Factor x factor interaction | 2+ factors |

```python
result = study.execute(chart='Xbar')

# All main effects combined
result.plot(chart='Effects')

# Factor effects only
result.plot(chart='MainEffects')

# Time effects only
result.plot(chart='TimeEffects')

# Factor x time interaction
result.plot(chart='TimeInteraction')

# Factor x factor interaction (requires 2+ factors)
result.plot(chart='FactorInteraction')
```

Effects charts require `result.has_effects == True` (i.e., the study must have factors). Interaction charts require the corresponding dimensions (factors + time for TimeInteraction, 2+ factors for FactorInteraction).

You can also access the raw effects data:

```python
result.effects       # Dict with 'k_effects', 't_effects'
result.interactions  # Dict of interaction terms
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

## Understanding XmR Charts

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

## Stratified XmR Charts

When you have factors and time, use the `by` parameter to stratify XmR charts:

```python
# Stratify by lane - creates one chart per lane
result = study.execute(chart='XmR', by=['lane'])

# Check the strata
print(result.charts['XmR']['strata'])  # ['A', 'B', 'C', 'D']
```

Each stratum has its own control limits based on its internal variation:

```python
# View faceted plot with all lanes
fig = result.plot(chart='XmR', show_zones=True)
```

### Lane Boundaries

When you collapse factors (use fewer factors in `by` than exist in the study), **lane boundaries** show where the collapsed factors change:

```python
# Single XmR with lane boundaries
result = study.execute(chart='XmR', by=[])  # Collapse all factors
fig = result.plot(chart='XmR')  # Vertical lines show factor transitions
```

## Re-centered Residual Charts

By default, residual charts are centered at zero. Use `recentered=True` to show residuals on the original measurement scale:

```python
# Zero-centered (default)
result = study.execute(chart='XmR', by=['lane'], value='R4')

# Re-centered on original scale
result = study.execute(chart='XmR', by=['lane'], value='R4', recentered=True)
```

Re-centering uses:
- R4: RCR4 = R4 + Y̅_t (adds back time mean)
- R5: RCR5 = R5 + Y̅_k (adds back factor mean)

## Decision Tree

```
Do you have factors?
├── No → Do you have time?
│   └── Yes → SDS 4: Use XmR
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
| Are groups different? | `chart='Xbar'` | Group deviates from average |
| Is variation stable? | `chart='S'` | Group has unusual variation |
| Process over time? | `chart='XmR', by=[...]` | Special cause detected |
| Interactions? | `chart='XmR', value='R3'` | Factor effect changes over time |
| Time trends? | `chart='XmR', value='R4'` | Process drift/shift |
| Factor effects? | `chart='Xbar', value='R5'` | True factor differences |

## Next Steps

- [Plotting & Themes](plotting.md) - Visualization options for all charts
- [VAS Residuals](residuals.md) - Deep dive into VAS residuals
- [Stratified Analysis](../tutorials/stratified-analysis.ipynb) - Stratified XmR tutorial
