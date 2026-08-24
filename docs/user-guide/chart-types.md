# Chart Types

ProcessBehavior supports multiple chart types, each suited for different data structures and analytical questions. This guide helps you choose the right chart.

## Available Chart Types

### Standard Charts

| Chart | Full Name | Purpose | Requirements |
|-------|-----------|---------|--------------|
| **Xbar** | X-bar Chart | Monitor subgroup means | n >= 2 per subgroup |
| **S** | S Chart | Monitor subgroup variation | n >= 2 per subgroup |
| **X** | X (Individual) Chart | Monitor individual values | Any structure |
| **mR** | mR (Moving Range) Chart | Monitor individual moving ranges | Any structure |

### VAS Residual Charts

Use the `value` parameter to chart residuals instead of the response:

| Residual | Chart | Purpose |
|----------|-------|---------|
| **R2** | S or X | Check within-group variation stability |
| **R3** | Xbar, S, or X | Detect factor × time interactions |
| **R4** | Xbar, S, or X | Detect time effects |
| **R5** | Xbar, S, or X | Detect factor effects |

```python
# Chart R5 residuals on Xbar
result = study.execute(chart='Xbar', value='R5')

# Chart R4 residuals on stratified X
result = study.execute(chart='X', by=['lane'], value='R4')
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

# X - stratify by factor(s)
result = study.execute(chart='X', by=['lane'])      # One chart per lane
```

Use `study.valid_charts` and `study.residuals` to see options.

## Chart Selection Guide

### Question: "Are my subgroups different from each other?"

**Use: Xbar Chart**

```python
result = study.execute(chart='Xbar')
fig = result.plot(chart='Xbar', show_zones=True, highlight_signals=True)
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

**Use: X Chart**

```python
result = study.execute(chart='X', by=['lane'])
fig = result.plot(show_zones=True, show_rules=True)
```

For stratified data, this creates one X chart per factor level.

### Question: "Do factor effects change over time?"

**Use: X with `value='R3'` (Interactions)**

```python
result = study.execute(chart='X', by=['lane'], value='R3')
fig = result.plot(title='Factor × Time Interactions')
```

Signals indicate that factor effects are not consistent over time.

### Question: "Are there time trends after removing factor effects?"

**Use: X with `value='R4'`**

```python
result = study.execute(chart='X', by=['lane'], value='R4')
fig = result.plot(title='Time Effects')
```

Signals indicate process drift or shifts over time.

### Question: "Are there design condition differences after removing time effects?"

**Use: X with `value='R5'`**

```python
result = study.execute(chart='X', by=['lane'], value='R5')
fig = result.plot(title='Design Condition Main Effects')
```

Signals indicate true differences between process design conditions.

## Chart Validity by DS

Not all charts are valid for all Design States:

| Chart | DS 1 | DS 2 | DS 3 | DS 4 | DS 5-6 |
|-------|-------|-------|-------|-------|---------|
| Xbar | ✅ | ✅ | ✅ | ❌ | ❌ |
| S | ✅ | ✅ | ✅ | ❌ | ❌ |
| X | ✅ | ✅ | ✅ | ✅ | ✅ |
| mR | ✅ | ✅ | ✅ | ✅ | ✅ |

**Residual availability** depends on DS. Use `study.residuals` to check:

```python
print(f"Valid charts: {study.valid_charts}")
print(f"Available residuals: {study.residuals}")
```

**Note on R2**: DS 2 and 5 use the moving average method; DS 1, 3, 4, 6 use within-cell deviation (R2 = Y - Ȳ_kt). See [VAS Residuals](residuals.md) for details.

## Companion Charts

Wheeler recommends reading certain charts as pairs: Xbar with S, and X with mR. The `companion` parameter returns both charts in one result:

```python
# Returns both Xbar and S charts
result = study.execute(chart='Xbar', companion=True)
result.plot(chart='Xbar')  # Xbar chart
result.plot(chart='S')     # S chart

# Returns both X and mR charts, stratified
result = study.execute(chart='X', by=['lane'], companion=True)
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
# Keys are named after your own factor columns, plus 'main_effect' and 'time'.
# For factors=['machine', 'shift']:
result.effects['machine']   # per-level main effects  (column: Main_Effect)
result.effects['time']      # time main effects       (column: PT_ME)
result.effects              # see AnalysisResult.effects for the full key list
result.interactions         # Dict of interaction terms
```

## Understanding Xbar-S Charts

### The Xbar Chart

Plots the **mean** of each subgroup (factor level at each time point).

- **Centerline**: Grand mean (Y̅)
- **Control Limits**: Based on within-subgroup variation
- **Interpretation**: Points beyond limits indicate subgroups with unusual means

!!! note "Limits for effect-carrying residuals (R4/R5)"
    When charting R4, R5, or their recentered variants (RCR4, RCR5) on Xbar, limits are based on **R2's within-group standard deviation** (Sbar), not the plotted residual's own standard deviation. This matters when `by` collapses factors — e.g., `by=['factor 1']` in a two-factor study. At collapsed groupings, R5's within-group std would include between-cell variance from the collapsed dimension, inflating limits. Using R2's Sbar isolates within-cell noise as the limit basis. At the full RSG level (all factors in `by`), R5's within-group std equals R2's, so there is no difference. This follows Dr. Tom Bishop's VAS methodology.

### The S Chart

Plots the **standard deviation** of each subgroup.

- **Centerline**: Pooled within-subgroup standard deviation
- **Control Limits**: Based on chi-square distribution
- **Interpretation**: Points beyond limits indicate subgroups with unusual variation

!!! note "S chart with effect residuals (R3/R4/R5)"
    When charting an effect residual on S, the data points show R2's
    within-group standard deviation, not the requested residual's. The S chart always
    measures within-cell noise stability. See [How Effect Residuals Are
    Charted](residuals.md#how-effect-residuals-are-charted).

### Reading Order

1. **First check S chart** - Variation must be stable
2. **Then interpret Xbar** - Only meaningful if S is stable
3. **Investigate signals** - What makes that subgroup different?

## Understanding X and mR Charts

### The X (Individual) Chart

Plots each individual observation.

- **Centerline**: Average of all observations (X̅)
- **Control Limits**: X̅ ± 2.66 × R̅ (average moving range)
- **Interpretation**: Points beyond limits indicate special causes

### The mR (Moving Range) Chart

Plots the absolute difference between consecutive observations.

- **Centerline**: Average moving range (R̅)
- **UCL**: 3.27 × R̅
- **LCL**: 0 (range cannot be negative)
- **Interpretation**: Large ranges indicate sudden changes

## Stratified X Charts

When you have factors and time, use the `by` parameter to stratify X charts:

```python
# Stratify by lane - creates one chart per lane
result = study.execute(chart='X', by=['lane'])

# Check the strata
print(result.charts['X']['strata'])  # ['A', 'B', 'C', 'D']
```

Each stratum has its own control limits based on its internal variation:

```python
# View faceted plot with all lanes
fig = result.plot(chart='X', show_zones=True)
```

### Lane Boundaries

When you collapse factors (use fewer factors in `by` than exist in the study), **lane boundaries** show where the collapsed factors change:

```python
# Single X chart with lane boundaries
result = study.execute(chart='X', by=[])  # Collapse all factors
fig = result.plot(chart='X')  # Vertical lines show factor transitions
```

## Re-centered Residual Charts

By default, residual charts are centered at zero. Use `recentered=True` to show residuals on the original measurement scale:

```python
# Zero-centered (default)
result = study.execute(chart='X', by=['lane'], value='R4')

# Re-centered on original scale
result = study.execute(chart='X', by=['lane'], value='R4', recentered=True)
```

Re-centering uses:
- R4: RCR4 = R4 + Y̅_t (adds back time mean)
- R5: RCR5 = R5 + Y̅_k (adds back factor mean)

## Decision Tree

```
Do you have factors?
├── No → Do you have time?
│   └── Yes → DS 6: Use X
└── Yes → Do you have time?
    ├── No → Use Xbar to compare factors
    └── Yes → Do you have replication (n>=2 per cell)?
        ├── All cells → DS 1: Full Xbar-S + VAS residuals
        ├── Some cells → DS 3: Hybrid Xbar-S + limited VAS
        └── No cells → DS 2: Xbar-S with MR-based limits
```

## Summary

| Question | Chart | Signal Meaning |
|----------|-------|----------------|
| Are groups different? | `chart='Xbar'` | Group deviates from average |
| Is variation stable? | `chart='S'` | Group has unusual variation |
| Process over time? | `chart='X', by=[...]` | Special cause detected |
| Interactions? | `chart='X', value='R3'` | Factor effect changes over time |
| Time trends? | `chart='X', value='R4'` | Process drift/shift |
| Factor effects? | `chart='Xbar', value='R5'` | True factor differences |

## Next Steps

- [Plotting & Themes](plotting.md) - Visualization options for all charts
- [VAS Residuals](residuals.md) - Deep dive into VAS residuals
- [Coffee Shop](../tutorials/coffee-shop.ipynb) - A complete analysis, start to finish
