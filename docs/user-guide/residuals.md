# VAS Residuals

Wheeler's **Variance Analysis System (VAS)** decomposes total variation into meaningful components. ProcessBehavior calculates five residuals (R1-R5) that help answer different analytical questions.

## The Five Residuals

| Residual | Name | Formula | Questions Answered |
|----------|------|---------|-------------------|
| **R1** | Total | Y - Y̅ | How far is each point from the grand mean? |
| **R2** | Within-cell | Y - Y̅<sub>kt</sub> | Is measurement variation stable? |
| **R3** | Interaction | Y - Y̅<sub>k</sub> - Y̅<sub>t</sub> + Y̅ | Do factor effects change over time? |
| **R4** | Time | Y̅<sub>t</sub> - Y̅ + R2 | Are there time trends or shifts? |
| **R5** | Factor | Y̅<sub>k</sub> - Y̅ + R2 | Do factors differ from each other? |

Where:
- Y = individual observation
- Y̅ = grand mean
- Y̅<sub>k</sub> = mean for factor level k
- Y̅<sub>t</sub> = mean at time t
- Y̅<sub>kt</sub> = mean for factor k at time t (cell mean)

## Accessing Residuals

Residuals are calculated during formulation and available after analysis:

```python
study = pdf.formulate(
    response=pdf.columns.weight,
    factors=[pdf.columns.lane],
    time=pdf.columns.batch
)

# Access via study.dataset
print(study.dataset[['R1', 'R2', 'R3', 'R4', 'R5']].head())

# Access via result
result = study.analyze()
print(result.residuals.head())
```

## R2: Within-Cell Residuals

**Purpose**: Assess measurement/within-subgroup variation.

**Formula by SDS**:
- **SDS 1 (Full Replication)**: R2 = Y - Y̅<sub>kt</sub> (exact within-cell deviation)
- **SDS 2 (No Replication)**: R2 = (Y<sub>j</sub> - Y<sub>j-1</sub>) / 2 (backward 2-point moving average)
- **SDS 3 (Partial)**: Hybrid approach

**Chart**: R2_S (for replicated data) or R2_Imr

```python
# Chart the within-cell variation
result = study.analyze(study.charts.R2_S)
fig = result.plot(show_zones=True, title='Within-Cell Variation')
```

**Interpretation**:
- Stable R2 → Consistent measurement process
- Signals in R2 → Special causes within subgroups
- Large R2 variation → Measurement system needs attention

## R3: Interaction Residuals

**Purpose**: Detect factor × time interactions.

**Formula**: R3 = Y - Y̅<sub>k</sub> - Y̅<sub>t</sub> + Y̅

This removes both main effects, leaving only the interaction.

**Chart**: R3_Imr

```python
result = study.analyze(study.charts.R3_Imr)
fig = result.plot(show_zones=True, title='Factor × Time Interactions')
```

**Interpretation**:
- Signals in R3 → Factor behavior changes over time
- Stable R3 → Factor effects are consistent across time periods
- Example: Machine A performs worse only on night shift

## R4: Time Effect Residuals

**Purpose**: Detect time-related patterns (trends, shifts, cycles).

**Formula**: R4 = Y̅<sub>t</sub> - Y̅ + R2

This combines the time effect with within-cell variation.

**Chart**: R4_Imr

```python
result = study.analyze(study.charts.R4_Imr)
fig = result.plot(show_zones=True, show_rules=True, title='Time Effects')
```

**Interpretation**:
- Signals in R4 → Process is changing over time
- Trends → Gradual drift (tool wear, environmental change)
- Shifts → Sudden change (adjustment, material change)
- Cycles → Periodic pattern (daily, weekly)

## R5: Factor Effect Residuals

**Purpose**: Detect true differences between factor levels.

**Formula**: R5 = Y̅<sub>k</sub> - Y̅ + R2

This combines the factor effect with within-cell variation.

**Chart**: R5_Imr

```python
result = study.analyze(study.charts.R5_Imr)
fig = result.plot(show_zones=True, title='Factor Effects')
```

**Interpretation**:
- Signals in R5 → Factors truly differ from each other
- No signals → Factor differences are within normal variation
- Use for equipment comparison, operator comparison, etc.

## Re-centered Residuals

By default, residual charts are centered at zero. Use `recentered=True` to show on the original measurement scale:

```python
# Zero-centered (default)
result = study.analyze(study.charts.R4_Imr)
# Centerline at 0, values show deviation from time mean

# Re-centered on original scale
result = study.analyze(study.charts.R4_Imr, recentered=True)
# Centerline at grand mean, values on original measurement scale
```

Re-centering formulas:
- RCR4 = R4 + Y̅<sub>t</sub>
- RCR5 = R5 + Y̅<sub>k</sub>

## Residual Availability by SDS

| SDS | R1 | R2 | R3 | R4 | R5 |
|-----|----|----|----|----|-----|
| 1 (Full Replication) | ✅ Exact | ✅ Exact | ✅ Full | ✅ Full | ✅ Full |
| 2 (No Replication) | ✅ | ⚠️ MR-based | ⚠️ Approx | ⚠️ Approx | ⚠️ Approx |
| 3 (Partial) | ✅ | ⚠️ Hybrid | ⚠️ Approx | ⚠️ Approx | ⚠️ Approx |
| 4 (Single Stream) | ✅ | ⚠️ Hybrid | ⚠️ Approx | ⚠️ Approx | ⚠️ Approx |
| 5 (Nested) | ✅ | ⚠️ Hybrid | ⚠️ Approx | ⚠️ Approx | ⚠️ Approx |
| 6 (Unstructured) | ✅ | ⚠️ MR-based | ⚠️ Approx | ⚠️ Approx | ⚠️ Approx |

## Analysis Workflow with Residuals

### Step 1: Check R2 (Measurement Stability)

```python
result = study.analyze(study.charts.R2_S)
signals = result.detect_signals(chart='R2_S')

if signals.has_signals:
    print("Within-cell variation is unstable!")
    print("Investigate measurement system before proceeding.")
```

### Step 2: Check R3 (Interactions)

```python
result = study.analyze(study.charts.R3_Imr)
signals = result.detect_signals(chart='R3_Imr')

if signals.has_signals:
    print("Significant factor × time interactions detected.")
    print("Factor effects are not consistent over time.")
```

### Step 3: Check R4 (Time Effects)

```python
result = study.analyze(study.charts.R4_Imr)
fig = result.plot(show_zones=True, show_rules=True)
```

### Step 4: Check R5 (Factor Effects)

```python
result = study.analyze(study.charts.R5_Imr)
fig = result.plot(show_zones=True, show_signals=True)
```

## Interpreting the Complete Picture

| R2 Status | R3 Status | R4 Status | R5 Status | Conclusion |
|-----------|-----------|-----------|-----------|------------|
| Stable | Stable | Stable | Stable | Process in control |
| Unstable | - | - | - | Fix measurement first |
| Stable | Signals | - | - | Investigate interactions |
| Stable | Stable | Signals | - | Time-related changes |
| Stable | Stable | Stable | Signals | True factor differences |

## Example: Complete Residual Analysis

```python
from processbehavior import ProcessDataFrame

pdf = ProcessDataFrame(df)
study = pdf.formulate(
    response=pdf.columns.weight,
    factors=[pdf.columns.lane],
    time=pdf.columns.batch
)

# Check available residual charts
print(f"Residual charts: {study.residual_charts}")

# Analyze each component
for chart in ['R2_S', 'R3_Imr', 'R4_Imr', 'R5_Imr']:
    if chart in study.residual_charts:
        result = study.analyze(chart)
        signals = result.detect_signals()
        print(f"{chart}: {signals.count} signals")
```

## Best Practices

1. **Always check R2 first** - Measurement stability is foundational
2. **Use SDS 1 when possible** - Full replication gives exact residuals
3. **Interpret in sequence** - R2 → R3 → R4 → R5
4. **Consider re-centering** - Easier to explain on original scale
5. **Document findings** - Record which residuals showed signals

## Mathematical Details

### R1: Total Deviation

```
R1 = Y - Ȳ
```

Sum of R1 across all observations = 0

### R2: Within-Cell (SDS 1)

```
R2 = Y - Ȳ_kt
```

Where Y̅<sub>kt</sub> is the mean of observations in cell (k, t).

### R2: Within-Cell (SDS 2, Backward Moving Average)

```
R2_j = (Y_j - Y_{j-1}) / 2
```

For the first observation, R2 = 0 or uses forward difference.

### R3: Interaction

```
R3 = Y - Ȳ_k - Ȳ_t + Ȳ
   = R1 - (Ȳ_k - Ȳ) - (Ȳ_t - Ȳ)
   = R1 - FactorEffect - TimeEffect
```

### R4: Time Effect + Unexplained

```
R4 = Ȳ_t - Ȳ + R2
   = TimeEffect + WithinVariation
```

### R5: Factor Effect + Unexplained

```
R5 = Ȳ_k - Ȳ + R2
   = FactorEffect + WithinVariation
```

## Next Steps

- [Xbar-S Analysis](../tutorials/xbar-s-analysis.ipynb) - Practical VAS analysis
- [Chart Types](chart-types.md) - All residual chart types
- [API Reference](../reference/api.md) - ResidualCalculator API
