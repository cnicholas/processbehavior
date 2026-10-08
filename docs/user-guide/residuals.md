# VAS Residuals

Dr. Thomas A. Bishop's **Variation Analysis System (VAS)** decomposes total variation into meaningful components. ProcessBehavior calculates six residuals (R1-R6) that help answer different analytical questions. Equation numbers on this page refer to Bishop's VAS documentation manual dated 10-1-2026.

## The Residual Hierarchy

| Residual | Name | Formula | Questions Answered |
|----------|------|---------|-------------------|
| **R1** | Response Centered at 0 | Y - Y̅ | How far is each point from the overall mean? |
| **R2** | Unexplained (noise) | Y - Y̅<sub>kt</sub> (DS 1); scaled difference of the effect-adjusted series (DS 2, 3) — see [R2](#r2-unexplained-residuals) | Is measurement variation stable? |
| **R3** | Interaction | (Y̅<sub>kt</sub> - Y̅<sub>k</sub> - Y̅<sub>t</sub> + Y̅) + R2 | Do factor effects change over time? |
| **R4** | Time Main Effect | Y̅<sub>t</sub> - Y̅ + R2 | Are there time trends or shifts? |
| **R5** | Design Condition Main Effect | Y̅<sub>k</sub> - Y̅ + R2 | Do process design conditions differ from each other? |
| **R6** | Design Factor Main Effect | α<sub>i</sub> + R2 | Does a specific design factor have a significant effect? |

Where:
- Y = individual observation
- Y̅ = grand mean: the mean of the cell means, each cell counted once (10-1 manual Eq 10-10)
- Y̅<sub>k</sub> = mean for factor level k: the mean of that level's cell means
- Y̅<sub>t</sub> = mean at time t: the mean of that period's cell means
- Y̅<sub>kt</sub> = mean for factor k at time t (cell mean, Eq 10-8)

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

## R1: Response Centered at 0

**Purpose**: View the original measurements as ± about zero, so the total range of
variation is read directly rather than against an arbitrary process mean.

**Formula**: R1 = Y - Y̅ (Eq 14-2)

R1 is a pure location shift — it subtracts the overall mean and changes nothing else. It
is also the building block the other residuals are derived from, which makes it the
natural starting point when auditing a decomposition: if R1 doesn't look like your raw
data recentred, nothing downstream will be right either.

**Chart**: Xbar (subgroup means) or X (individual values). There is deliberately no S or
mR chart for R1 — a constant shift leaves dispersion untouched, so those charts would be
numerically identical to the response's.

```python
# Individual values, centred at zero
result = study.execute(chart='X', by=[], value='R1')
fig = result.plot(title='Response Centered at 0')

# Subgroup means, centred at zero
result = study.execute(chart='Xbar', value='R1')
```

**Interpretation**:
- The centre line sits at 0 by construction
- The spread is the total variation in the original data, read as ± about zero
- Limits use the noise floor (R2), as all residual charts do, so points falling
  outside them mark variation the decomposition will attribute to time, condition, or
  interaction

**Availability**: Bishop notes R1 can be calculated for data produced by all six sampling
design states.

(r2-unexplained-residuals)=
## R2: Unexplained Residuals

**Purpose**: Assess the unexplained (noise) variation: what is left once the process mean,
the condition and period effects and their interaction are accounted for (the interaction
exactly in DS 1, approximately in DS 2 and 3, where differencing the adjusted series removes
interaction that changes smoothly over time).

**Formula by DS**:
- **DS 1 (Full Replication)**: R2 = Y - Y̅<sub>kt</sub>, the deviation from the cell mean (Eq 14-3)
- **DS 2 (No Replication)** and **DS 3 (Partial)**: with any one-observation subgroup there is
  no within-cell deviation to use, so R2 is built from the whole series in three steps
  (Eqs 14-4..14-8 for DS 2, 14-9..14-13 for DS 3, 5 and 6):
  1. Remove the process mean and the condition and period effects:
     Z = Y - Y̅<sub>k</sub> - Y̅<sub>t</sub> + Y̅ (Y̅<sub>k</sub>, Y̅<sub>t</sub> and Y̅ are
     unweighted means of cell means).
  2. Difference Z along the condition-then-time sequence (the canonical lane-major order,
     running across condition boundaries with no grouping). Only the first observation has
     no value.
  3. Divide by twice the **R2 scale factor** c(K, M), computed from the layout:
     K is the number of process design conditions present and M the number of observations
     (M = KT in DS 2).

Because Z carries no condition or period effect, the step from one condition's last period
to the next condition's first period no longer carries the level difference between
conditions, and a trend common to all conditions is not read as noise. The scale factor puts
R2 on the noise (sigma) scale, so the standard deviation of R2 estimates sigma directly.
Internally this method is named `ma2`: Bishop's step is a size-2 moving average of Z, and
R2<sub>j</sub> is Z<sub>j</sub> minus that average, divided by c.

**Chart**: S chart with `value='R2'` (for replicated data) or X

```python
# Chart the unexplained (noise) variation
result = study.execute(chart='S', value='R2')
fig = result.plot(show_zones=True, title='Unexplained Variation (R2)')
```

**Interpretation**:
- Stable R2 → Consistent measurement process
- Signals in R2 → Special causes within subgroups
- Large R2 variation → Measurement system needs attention

## R3: Interaction Residuals

**Purpose**: Detect factor × time interactions.

**Formula**: R3 = (Y̅<sub>kt</sub> - Y̅<sub>k</sub> - Y̅<sub>t</sub> + Y̅) + R2 (Eq 14-14)

This removes both main effects, leaving the interaction plus the unexplained noise. In DS 1,
where R2 = Y - Y̅<sub>kt</sub>, it reduces to Y - Y̅<sub>k</sub> - Y̅<sub>t</sub> + Y̅.

**Chart**: X with `value='R3'`

```python
result = study.execute(chart='X', by=['lane'], value='R3')
fig = result.plot(show_zones=True, title='Factor × Time Interactions')
```

**Interpretation**:
- Signals in R3 → Factor behavior changes over time
- Stable R3 → Factor effects are consistent across time periods
- Example: Machine A performs worse only on night shift

## R4: Time Main Effect Residuals

**Purpose**: Detect time-related patterns (trends, shifts, cycles).

**Formula**: R4 = Y̅<sub>t</sub> - Y̅ + R2

This combines the time effect with the unexplained noise (R2).

**Chart**: X with `value='R4'` (stratified by factor), or Xbar with `value='R4'` (aggregated across factors).

```python
# Stratified X — one chart per factor level
result = study.execute(chart='X', by=['lane'], value='R4')
fig = result.plot(show_zones=True, show_rules=True, title='Time Effects')

# Xbar — subgroup means across factor levels
result = study.execute(chart='Xbar', value='R4')
fig = result.plot(show_zones=True, title='Time Effects (Xbar)')
```

When charting R4 on Xbar, limits use R2's Sbar (the noise), not R4's own within-group std. See [Chart Types: Xbar limits note](chart-types.md#the-xbar-chart) for details.

**Interpretation**:
- Signals in R4 → Process is changing over time
- Trends → Gradual drift (tool wear, environmental change)
- Shifts → Sudden change (adjustment, material change)
- Cycles → Periodic pattern (daily, weekly)

## R5: Design Condition Main Effect Residuals

**Purpose**: Detect true differences between process design conditions.

**Formula**: R5 = Y̅<sub>k</sub> - Y̅ + R2

This combines the factor effect with the unexplained noise (R2).

**Chart**: X with `value='R5'` (stratified by factor), or Xbar with `value='R5'` (aggregated by factor).

```python
# Stratified X — one chart per factor level
result = study.execute(chart='X', by=['lane'], value='R5')
fig = result.plot(show_zones=True, title='Design Condition Main Effects')

# Xbar — subgroup means by factor
result = study.execute(chart='Xbar', value='R5')
fig = result.plot(show_zones=True, title='Design Condition Main Effects (Xbar)')
```

When charting R5 on Xbar, limits use R2's Sbar (the noise), not R5's own within-group std. This prevents between-cell variance from collapsed dimensions from inflating limits. See [Chart Types: Xbar limits note](chart-types.md#the-xbar-chart) for details.

**Interpretation**:
- Signals in R5 → Factors truly differ from each other
- No signals → Factor differences are within normal variation
- Use for equipment comparison, operator comparison, etc.

## R6: Design Factor Main Effect Residuals

**Purpose**: Isolate a specific design factor's main effect combined with the unexplained noise (R2).

**Formula**: R6 = α<sub>i</sub> + R2

Where α<sub>i</sub> = mean(R5 | factor level(s)) — the mean of R5 for each level of the specified factor(s).

R6 differs from R5 in that it isolates the effect of *individual factors* when multiple factors exist. R5 contains the combined effect of all factors; R6 lets you examine one factor at a time.

**Chart**: Xbar with `value='R6'` and `by` specifying which factor(s) to examine.

```python
# In a two-factor study, examine FACTOR 1's effect
result = study.execute(chart='Xbar', value='R6', by=['factor 1'])
fig = result.plot(show_zones=True, title='Factor 1 Main Effect')

# Re-centered on original measurement scale
result = study.execute(chart='Xbar', value='R6', by=['factor 1'], recentered=True)
```

**Key details**:
- R6 is computed on-the-fly during `execute()` (not stored in the dataset like R1-R5); it lives on the requesting result — `result.dataset['R6']` / `result.get_residual('R6')`
- The `by` parameter is required and specifies which factor(s) to compute the main effect for
- Available when the study has factors and R5/R2 are present
- Re-centered R6 (RCR6) adds back the grand mean: RCR6 = Ȳ + α<sub>i</sub> + R2

**Interpretation**:
- Signals in R6 → The specified factor has a significant main effect
- Useful for drilling into multi-factor studies: which factor matters most?

## How Effect Residuals Are Charted

When you chart an effect-carrying residual (R3, R4, or R5) on Xbar or S, the system substitutes **R2** as the dispersion basis. Understanding this substitution is key to interpreting these charts correctly.

### Why R2 sets the limits

R3, R4, and R5 contain structural effects by design — that's what makes them useful. But if R4's own standard deviation set the Xbar limits, the time effect would widen them, defeating the purpose of looking for signals *beyond* the expected variation. By substituting R2 (the unexplained noise: within-cell in DS 1, the scaled difference in DS 2 and 3), the limits reflect only unexplained variation, making structural effects visible as signals.

### Chart-by-chart behavior

| Chart | What is plotted | What sets the limits |
|-------|----------------|---------------------|
| **Xbar** | Subgroup means of the requested residual | R2's within-group Sbar |
| **S** | R2's within-group std (**not** the requested residual's) | R2's Sbar for CL and limits |
| **X** | Individual residual values | Moving range of the residual itself (no R2 substitution) |

### The S chart surprise

This is the most counterintuitive behavior: `execute(chart='S', value='R3')` plots R2's within-group standard deviation, not R3's. The S chart always answers "is the noise stable?" regardless of which residual you request. This is correct — the S chart's job is to verify that the dispersion basis (R2) is stable before you interpret the Xbar chart above it.

### When does this matter?

The R2 substitution only matters when `by` collapses factors. At the full RSG level (all factors in `by`), the residual's within-group standard deviation equals R2's, so there is no visible difference. When you collapse — e.g., `by=['factor1']` in a two-factor study — R2 still measures only the noise, while the residual's own std would include between-cell variance from the collapsed dimension.

For X charts, there is no substitution. The moving range is always computed from the requested residual's own values.

## Re-centered Residuals

By default, residual charts are centered at zero. Use `recentered=True` to show on the original measurement scale:

```python
# Zero-centered (default)
result = study.execute(chart='X', by=['lane'], value='R4')
# Centerline at 0, values show the time effect plus R2

# Re-centered on original scale
result = study.execute(chart='X', by=['lane'], value='R4', recentered=True)
# Centerline at grand mean, values on original measurement scale
```

Re-centering formulas:
- RCR = R + Y̅ for R1, R3, R4, R5 and R6 (Eq 14-26) — each adds back the grand mean only, so a
  re-centered chart is the zero-centered chart shifted onto the measurement scale
- RCR2 = Y̅<sub>kt</sub> + R2 — not part of Eq 14-26; in DS 1 this reconstructs Y

So an effect chart of RCR5 by condition, RCR4 by period, or RCR3 by subgroup plots that
effect plus R2 on the measurement scale. Chart period effects from R4: an Xbar of RCR3 by
period is flat, because the interaction sums to zero over conditions. Re-centering shifts the
chart by Y̅; the limit width still comes from R2 (Xbar/S) or the residual's own moving
ranges (X).

**Note on recentered moving ranges**: For recentered residuals on X charts, the moving range is computed from the non-recentered version (e.g., RCR3 uses MR from R3). For RCR1 and RCR3–RCR6 this changes nothing, since re-centering adds a constant; for RCR2 it keeps the steps between cell means out of the moving ranges.

## Residual Availability by DS

| DS | R1 | R2 | R3 | R4 | R5 | R6 |
|-----|----|----|----|----|-----|-----|
| 1 (Full Replication) | ✅ | ✅ Within-cell | ✅ | ✅ | ✅ | ✅ |
| 2 (No Replication) | ✅ | ✅ Scaled difference | ✅ | ✅ | ✅ | ✅ |
| 3 (Partial) | ✅ | ✅ Scaled difference | ✅ | ✅ | ✅ | ✅ |
| 4 (Incomplete, No Singletons) → ADS 1 | ✅ | ✅ Within-cell | ✅ | ✅ | ✅ | ✅ |
| 5 (Incomplete, No Replication) → ADS 2 | ✅ | ✅ Scaled difference | ✅ | ✅ | ✅ | ✅ |
| 6 (Incomplete, With Singletons) → ADS 3 | ✅ | ✅ Scaled difference | ✅ | ✅ | ✅ | ✅ |

R6 requires factors (it is computed from R5 and R2 at `execute()` time).

**Note on R2 calculation**: R2 is the one residual whose formula depends on structure, and the
choice is made on the analytical design state (ADS), after cleaning:
- **ADS 1** (every cell n ≥ 2): within-cell deviation, `R2 = Y - Ȳ_kt` (Eq 14-3)
- **ADS 2 and 3** (any one-observation cell): the condition-and-period-adjusted series Z,
  differenced along the full canonical sequence and divided by twice the R2 scale factor,
  for every observation (Eqs 14-4..14-13); no per-cell mixing. See [R2](#r2-unexplained-residuals).

DS 4, 5 and 6 are sampling states; they collapse to ADS 1, 2 and 3 once empty cells are dropped.

**When R2 is unavailable**: in ADS 2 and 3 the R2 scale factor is undefined when there are
fewer than 2 process design conditions or fewer than K + 2 observations (M < K + 2). R2 then
has no value: `formulate()` warns, R2–R6 (and their re-centered forms) are not offered, and
`execute()` and `why_not()` give the reason. `loss_function()` and `maximum_information()`
raise `ValidationError`, and `capability()` reports potential capability as unavailable with
the same reason. R1 is still available.

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
result = study.execute(chart='X', by=['lane'], value='R3')
signals = result.detect_signals(chart='X')

if signals.has_signals:
    print("Significant factor × time interactions detected.")
    print("Factor effects are not consistent over time.")
```

### Step 3: Check R4 (Time Effects)

```python
result = study.execute(chart='X', by=['lane'], value='R4')
fig = result.plot(show_zones=True, show_rules=True)
```

### Step 4: Check R5 (Design Condition Main Effects)

```python
result = study.execute(chart='X', by=['lane'], value='R5')
fig = result.plot(show_zones=True, highlight_signals=True)
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

# Analyze R3-R5 on stratified X charts
for residual in ['R3', 'R4', 'R5']:
    result = study.execute(chart='X', by=['lane'], value=residual)
    signals = result.detect_signals(chart='X')
    print(f"{residual} on X: {signals.count} signals")
```

## Best Practices

1. **Always check R2 first** - Measurement stability is foundational
2. **Use DS 1 when possible** - Full replication gives exact residuals
3. **Interpret in sequence** - R2 → R3 → R4 → R5
4. **Consider re-centering** - Easier to explain on original scale
5. **Document findings** - Record which residuals showed signals

## Mathematical Details

### R1: Total Deviation

```
R1 = Y - Ȳ
```

R1 averages to zero over the cells: the mean of the cell means of R1 is 0, because Ȳ is the
unweighted mean of the cell means. The sum over all observations is 0 only when every cell
has the same number of observations.

### R2: Within-Cell (DS 1)

```
R2 = Y - Ȳ_kt
```

Where Y̅<sub>kt</sub> is the mean of observations in cell (k, t) (Eq 14-3).

### R2: Scaled Difference (DS 2 and 3, any one-observation cell)

```
Z    = Y - Ȳ_k - Ȳ_t + Ȳ
R2_j = (Z_j - Z_{j-1}) / (2 · c(K, M))      j = 2 … M, condition-then-time order
c(K, M) = sqrt( (K - 1) / (2K) · (1 - K / (M - 1)) )
```

K is the number of process design conditions present and M the number of observations
(M = KT in DS 2); c(K, M) is the R2 scale factor (Eqs 14-4..14-8 for DS 2, 14-9..14-13 for
DS 3, 5 and 6). The first observation in the sequence has no value (NaN). The scale factor
is undefined, and R2 unavailable, when K < 2 or M < K + 2.

### R3: Interaction + Unexplained

```
R3 = (Ȳ_kt - Ȳ_k - Ȳ_t + Ȳ) + R2
   = InteractionEffect + R2
```

(Eq 14-14.) In DS 1 this equals Y - Ȳ_k - Ȳ_t + Ȳ, since Y = Ȳ_kt + R2 there.

### R4: Time Main Effect + Unexplained

```
R4 = Ȳ_t - Ȳ + R2
   = TimeEffect + R2
```

### R5: Design Condition Main Effect + Unexplained

```
R5 = Ȳ_k - Ȳ + R2
   = FactorEffect + R2
```

## Maximum Information Analysis

The **Maximum Information** analysis examines the noise floor of your process by analyzing R2 residuals via an X chart and percentage histogram. It answers: *What variation is inherent to the system, and is it predictable?*

The X chart of R2 has limits mean ± (3/1.128)·mR̄ and σ̂ = mR̄/1.128. The values are whatever R2
is at your design state (the scaled difference in ADS 2 and 3), and `maximum_information()`
raises `ValidationError` when R2 is unavailable.

```python
mi = study.maximum_information()

# Key statistics
print(f"Noise floor sigma: {mi.sigma_hat}")   # mR-bar / 1.128 of R2
print(f"Natural process limits: [{mi.lpl}, {mi.upl}]")
print(f"Signals in noise: {mi.n_signals}")     # Points beyond limits

# Visualize
mi.plot()                        # Combined X + histogram
mi.plot(view='xmr')              # X chart of R2 only
mi.plot(view='histogram', bins=15)  # Percentage histogram only
```

**Interpretation**:
- **Stable R2 on X chart** (no signals) → The noise floor is predictable. Any variation beyond this level is attributable to factors, time, or interactions.
- **Signals in R2** → Special causes exist in the noise itself (*within* subgroups, in DS 1). Investigate the measurement system or the short-term process variation before interpreting R3-R5.
- **σ̂ (sigma_hat)** → The irreducible noise floor. This is the best the process can achieve even if all assignable causes are eliminated.

## Next Steps

- [Coffee Shop](../tutorials/coffee-shop.ipynb) - Practical analysis walkthrough
- [Chart Types](chart-types.md) - All residual chart types
- [API Reference](../reference/api.md) - Complete Study API
