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
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)

# Access via study.dataset
print(study.dataset[['R1', 'R2', 'R3', 'R4', 'R5']].head())

# Access via result
result = study.execute()
print(result.residuals.head())
```

## R2: Within-Cell Residuals

**Purpose**: Assess measurement/within-subgroup variation.

**Formula by SDS**:
- **SDS 1 (Full Replication)**: R2 = Y - Y̅<sub>kt</sub> (exact within-cell deviation)
- **SDS 2 (No Replication)**: R2 = (Y<sub>j</sub> - Y<sub>j-1</sub>) / 2 (backward 2-point moving average)
- **SDS 3 (Partial)**: Hybrid approach

**Chart**: S chart with `value='R2'` (for replicated data) or XmR

```python
# Chart the within-cell variation
result = study.execute(chart='S', value='R2')
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

**Chart**: XmR with `value='R3'`

```python
result = study.execute(chart='XmR', by=['lane'], value='R3')
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

**Chart**: XmR with `value='R4'` (stratified by factor), or Xbar with `value='R4'` (aggregated across factors).

```python
# Stratified XmR — one chart per factor level
result = study.execute(chart='XmR', by=['lane'], value='R4')
fig = result.plot(show_zones=True, show_rules=True, title='Time Effects')

# Xbar — subgroup means across factor levels
result = study.execute(chart='Xbar', value='R4')
fig = result.plot(show_zones=True, title='Time Effects (Xbar)')
```

When charting R4 on Xbar, limits use R2's Sbar (within-cell noise), not R4's own within-group std. See [Chart Types: Xbar limits note](chart-types.md#the-xbar-chart) for details.

**Interpretation**:
- Signals in R4 → Process is changing over time
- Trends → Gradual drift (tool wear, environmental change)
- Shifts → Sudden change (adjustment, material change)
- Cycles → Periodic pattern (daily, weekly)

## R5: Factor Effect Residuals

**Purpose**: Detect true differences between factor levels.

**Formula**: R5 = Y̅<sub>k</sub> - Y̅ + R2

This combines the factor effect with within-cell variation.

**Chart**: XmR with `value='R5'` (stratified by factor), or Xbar with `value='R5'` (aggregated by factor).

```python
# Stratified XmR — one chart per factor level
result = study.execute(chart='XmR', by=['lane'], value='R5')
fig = result.plot(show_zones=True, title='Factor Effects')

# Xbar — subgroup means by factor
result = study.execute(chart='Xbar', value='R5')
fig = result.plot(show_zones=True, title='Factor Effects (Xbar)')
```

When charting R5 on Xbar, limits use R2's Sbar (within-cell noise), not R5's own within-group std. This prevents between-cell variance from collapsed dimensions from inflating limits. See [Chart Types: Xbar limits note](chart-types.md#the-xbar-chart) for details.

**Interpretation**:
- Signals in R5 → Factors truly differ from each other
- No signals → Factor differences are within normal variation
- Use for equipment comparison, operator comparison, etc.

## How Effect Residuals Are Charted

When you chart an effect-carrying residual (R3, R4, or R5) on Xbar or S, the system substitutes **R2** as the dispersion basis. Understanding this substitution is key to interpreting these charts correctly.

### Why R2 sets the limits

R3, R4, and R5 contain structural effects by design — that's what makes them useful. But if R4's own standard deviation set the Xbar limits, the time effect would widen them, defeating the purpose of looking for signals *beyond* the expected variation. By substituting R2 (pure within-cell noise), the limits reflect only unexplained variation, making structural effects visible as signals.

### Chart-by-chart behavior

| Chart | What is plotted | What sets the limits |
|-------|----------------|---------------------|
| **Xbar** | Subgroup means of the requested residual | R2's within-group Sbar |
| **S** | R2's within-group std (**not** the requested residual's) | R2's Sbar for CL and limits |
| **XmR** | Individual residual values | Moving range of the residual itself (no R2 substitution) |

### The S chart surprise

This is the most counterintuitive behavior: `execute(chart='S', value='R3')` plots R2's within-group standard deviation, not R3's. The S chart always answers "is within-cell noise stable?" regardless of which residual you request. This is correct — the S chart's job is to verify that the dispersion basis (R2) is stable before you interpret the Xbar chart above it.

### When does this matter?

The R2 substitution only matters when `by` collapses factors. At the full RSG level (all factors in `by`), the residual's within-group standard deviation equals R2's, so there is no visible difference. When you collapse — e.g., `by=['factor1']` in a two-factor study — R2 correctly isolates within-cell noise while the residual's own std would include between-cell variance from the collapsed dimension.

For XmR charts, there is no substitution. The moving range is always computed from the requested residual's own values.

## Re-centered Residuals

By default, residual charts are centered at zero. Use `recentered=True` to show on the original measurement scale:

```python
# Zero-centered (default)
result = study.execute(chart='XmR', by=['lane'], value='R4')
# Centerline at 0, values show deviation from time mean

# Re-centered on original scale
result = study.execute(chart='XmR', by=['lane'], value='R4', recentered=True)
# Centerline at grand mean, values on original measurement scale
```

Re-centering formulas:
- RCR3 = R3 + (Y̅<sub>k</sub> + Y̅<sub>t</sub> - Y̅) — adds back factor and time main effects
- RCR4 = R4 + Y̅<sub>t</sub>
- RCR5 = R5 + Y̅<sub>k</sub>

**Note on recentered moving ranges**: For recentered residuals on XmR, the moving range is computed from the non-recentered version (e.g., RCR3 uses MR from R3). This avoids structural jumps between factor levels inflating the moving ranges.

## Residual Availability by SDS

| SDS | R1 | R2 | R3 | R4 | R5 |
|-----|----|----|----|----|-----|
| 1 (Full Replication) | ✅ | ✅ Within-cell | ✅ | ✅ | ✅ |
| 2 (No Replication) | ✅ | ✅ MR-based | ✅ | ✅ | ✅ |
| 3 (Partial) | ✅ | ✅ Hybrid | ✅ | ✅ | ✅ |
| 4 (Single Stream) | ✅ | ✅ Hybrid | ✅ | ✅ | ✅ |
| 5 (Nested) | ✅ | ✅ Hybrid | ✅ | ✅ | ✅ |
| 6 (Unstructured) | ✅ | ✅ MR-based | ✅ | ✅ | ✅ |

**Note on R2 calculation**: R2 adapts to your sampling structure:
- **SDS 1**: Within-cell deviation (`R2 = Y - Ȳ_kt`)
- **SDS 2, 6**: Moving average method (`R2 = Y - MA2`) for unreplicated/sparse designs
- **SDS 3, 4, 5**: Hybrid approach (within-cell for n>1 cells, zero for n=1 cells)

## Analysis Workflow with Residuals

### Step 1: Check R2 (Measurement Stability)

```python
result = study.execute(chart='S', value='R2')
signals = result.detect_signals(chart='S')

if signals.has_signals:
    print("Within-cell variation is unstable!")
    print("Investigate measurement system before proceeding.")
```

### Step 2: Check R3 (Interactions)

```python
result = study.execute(chart='XmR', by=['lane'], value='R3')
signals = result.detect_signals(chart='XmR')

if signals.has_signals:
    print("Significant factor × time interactions detected.")
    print("Factor effects are not consistent over time.")
```

### Step 3: Check R4 (Time Effects)

```python
result = study.execute(chart='XmR', by=['lane'], value='R4')
fig = result.plot(show_zones=True, show_rules=True)
```

### Step 4: Check R5 (Factor Effects)

```python
result = study.execute(chart='XmR', by=['lane'], value='R5')
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
from processbehavior import ProcessBehavior

pb = ProcessBehavior(df)
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)

# Check available residuals
print(f"Available residuals: {study.residuals}")

# Analyze R2 on S chart
result_r2 = study.execute(chart='S', value='R2')
signals_r2 = result_r2.detect_signals(chart='S')
print(f"R2 on S: {signals_r2.count} signals")

# Analyze R3-R5 on stratified XmR charts
for residual in ['R3', 'R4', 'R5']:
    result = study.execute(chart='XmR', by=['lane'], value=residual)
    signals = result.detect_signals(chart='XmR')
    print(f"{residual} on XmR: {signals.count} signals")
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
