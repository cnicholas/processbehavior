# Understanding Sampling Design States (SDS)

One of ProcessBehavior's most powerful features is **automatic Sampling Design State (SDS) detection**. This guide explains what SDS is, why it matters, and how ProcessBehavior uses it to make your life easier.

## What is SDS?

**Sampling Design State (SDS)** is a classification system developed by Wheeler and Bishop that describes the structure of your data based on three characteristics:

1. **Grouping Factors** - Do you have categorical variables that create subgroups?
2. **Time Dimension** - Do you have measurements over time?
3. **Replication** - Do you have multiple observations per factor×time combination?

Based on these characteristics, ProcessBehavior automatically classifies your data into one of 7 states (SDS 0-6), each with different analysis capabilities.

## Why Does SDS Matter?

Different data structures support different types of analyses:

- **Some data can use Xbar charts**, others cannot
- **Some data supports variance decomposition (VAS)**, others don't
- **Some data allows interaction analysis**, others don't

SDS detection means **you don't have to figure this out yourself**. The library automatically:

- ✅ Determines what analyses are valid for your data
- ✅ Recommends the best chart type
- ✅ Calculates everything your data structure supports
- ✅ Prevents invalid analyses with clear error messages

## The Seven SDS States

### SDS 0: Simple Series

**Data Structure**: Just a single column of measurements with no grouping or time structure.

**Characteristics**:
- No grouping variables
- No time variable
- Individual observations

**Example**:
```python
df = pd.DataFrame({
    'measurement': [23.1, 24.5, 22.8, 25.1, 23.7]
})

# SDS 0 is detected
result = pdf.analyze(response_var='measurement').calculate()
```

**Valid Charts**: IMR, R

**Use Cases**:
- Quick exploratory analysis
- Single vector of data
- Baseline capability studies

**Limitations**:
- Cannot calculate VAS residuals (no factors or time)
- Cannot analyze effects or interactions
- Limited to basic control charts

---

### SDS 1: Full Factorial with Complete Replication

**Data Structure**: Complete factorial design with n ≥ 2 observations in every factor×time cell.

**Characteristics**:
- ✅ Grouping variables present
- ✅ Time variable present
- ✅ Full replication (n ≥ 2 in all cells)

**Example**:
```python
df = pd.DataFrame({
    'measurement': [23.1, 23.5, 24.5, 24.2, 22.8, 23.1, 25.1, 24.9],
    'machine': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
    'time': [1, 1, 1, 1, 2, 2, 2, 2]  # 2 obs per machine×time
})

# SDS 1 is detected
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine'],
    time_var='time'
).calculate()
```

**Valid Charts**: Xbar, S, R, IMR

**Recommended**: Xbar (with companion S chart)

**Capabilities**:
- ✅ VAS residuals (R1-R5) - Full variance decomposition
- ✅ Main effects for all factors
- ✅ Factor × time interactions
- ✅ Exact within-cell variance (R2)

**Use Cases**:
- Designed experiments with replication
- Process capability studies
- Multi-factor ANOVA-style analyses
- Complete factorial designs

**This is the "best case" for analysis** - you can do everything!

---

### SDS 2: Full Factorial with No Replication

**Data Structure**: Complete factorial design with exactly n = 1 observation per factor×time cell.

**Characteristics**:
- ✅ Grouping variables present
- ✅ Time variable present
- ❌ No replication (n = 1 in all cells)

**Example**:
```python
df = pd.DataFrame({
    'measurement': [23.1, 24.5, 22.8, 25.1],
    'machine': ['A', 'B', 'A', 'B'],
    'time': [1, 1, 2, 2]  # Only 1 obs per machine×time
})

# SDS 2 is detected
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine'],
    time_var='time'
).calculate()
```

**Valid Charts**: Xbar, IMR, R

**Invalid**: S (requires n ≥ 2 per subgroup)

**Recommended**: Xbar

**Capabilities**:
- ✅ VAS residuals (R1-R5) - but R2 is approximate
- ✅ Main effects
- ✅ Interactions
- ⚠️ R2 estimated via moving average (not exact)

**Use Cases**:
- Production data (one measurement per condition)
- Unreplicated factorial experiments
- Historical data analysis
- Screening experiments

**Limitations**:
- Cannot use S charts (need n ≥ 2)
- R2 is approximate, not exact
- Interaction confounded with pure error

---

### SDS 3: Partial Replication

**Data Structure**: Mixed replication - some factor×time cells have n ≥ 2, others have n = 1.

**Characteristics**:
- ✅ Grouping variables present
- ✅ Time variable present
- ⚠️ Partial replication (some cells replicated, some not)

**Example**:
```python
df = pd.DataFrame({
    'measurement': [23.1, 23.5, 24.5, 22.8, 25.1],
    'machine': ['A', 'A', 'B', 'A', 'B'],
    'time': [1, 1, 1, 2, 2]  # Machine A/Time 1 has n=2, others n=1
})

# SDS 3 is detected
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine'],
    time_var='time'
).calculate()
```

**Valid Charts**: Xbar, S, R, IMR

**Recommended**: Xbar

**Capabilities**:
- ✅ VAS residuals (R1-R5) - Hybrid calculation
- ✅ Main effects
- ✅ Interactions
- ⚠️ R2 uses hybrid calculation (exact where possible, approximate elsewhere)

**Use Cases**:
- Unbalanced designs
- Real-world data with missing observations
- Opportunistic replication in some cells
- Pilot studies with targeted replication

**Limitations**:
- R2 calculation less precise than SDS 1
- Variance estimates depend on where replication exists
- May have unequal subgroup sizes

---

### SDS 4: Time Series (Single Condition)

**Data Structure**: One condition tracked over time (time series).

**Characteristics**:
- ❌ No grouping variables (or only 1 group)
- ✅ Time variable present
- ⚠️ Varies

**Example**:
```python
df = pd.DataFrame({
    'measurement': [23.1, 24.5, 22.8, 25.1, 23.7, 24.2],
    'time': [1, 2, 3, 4, 5, 6]  # Single process over time
})

# SDS 4 is detected
result = pdf.analyze(
    response_var='measurement',
    time_var='time'
).calculate()
```

**Valid Charts**: Xbar, S, R, IMR

**Recommended**: Xbar (if rational subgroups), IMR (if individuals)

**Use Cases**:
- Single process over time with subgroups
- Repeated measurements at time points
- Time series with natural grouping (e.g., hourly batches)
- Rational subgrouping by time period only

**Limitations**:
- No VAS residuals (requires factors)
- Cannot analyze factor effects
- Cannot detect interactions
- Limited to time-based analysis

---

### SDS 5: Cross-Sectional (No Time)

**Data Structure**: Multiple groups compared without time dimension.

**Characteristics**:
- ✅ Grouping variables present
- ❌ No time variable
- ⚠️ Varies

**Example**:
```python
df = pd.DataFrame({
    'measurement': [23.1, 23.5, 24.5, 24.2, 22.8, 23.1],
    'machine': ['A', 'A', 'B', 'B', 'C', 'C']  # Compare machines
})

# SDS 5 is detected
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine']
).calculate()
```

**Valid Charts**: Xbar, S, R, IMR

**Recommended**: Xbar

**Use Cases**:
- Cross-sectional studies
- Between-group comparisons
- Baseline capability studies
- Multi-stream process monitoring

**Limitations**:
- No VAS residuals (requires time dimension)
- Cannot analyze time trends
- Cannot detect factor × time interactions
- Limited to factor main effects

---

### SDS 6: Incomplete/Irregular Grid

**Data Structure**: Sparse factor × time grid with many missing cells.

**Characteristics**:
- ✅ Grouping variables present
- ✅ Time variable present
- ❌ Incomplete grid (many cells empty)

**Example**:
```python
df = pd.DataFrame({
    'measurement': [23.1, 24.5, 22.8],
    'machine': ['A', 'B', 'C'],
    'time': [1, 5, 10]  # Sparse, irregular time points
})

# SDS 6 is detected
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine'],
    time_var='time'
).calculate()
```

**Valid Charts**: IMR, R

**Invalid**: Xbar, S (require complete grid)

**Recommended**: IMR

**Use Cases**:
- Opportunistic data collection
- Irregular time intervals
- Sparse measurements
- Ad-hoc sampling

**Limitations**:
- Cannot use Xbar/S (need complete grid)
- No VAS residuals
- Cannot analyze effects or interactions
- Limited to tracking trends only

---

## SDS Detection in Action

ProcessBehavior automatically detects SDS when you run `.analyze()`:

```python
# The library examines your data and determines:
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine', 'operator'],
    time_var='batch'
).calculate()

# View what was detected:
print(result.summary['sds'])           # SDS number (0-6)
print(result.summary['sds_name'])      # Human-readable name
print(result.summary['sds_description'])  # What it means

# See what you can do:
print(result.summary['valid_charts'])      # Available chart types
print(result.summary['recommended_chart']) # Best option
print(result.summary['has_residuals'])     # Can calculate VAS?
print(result.summary['has_effects'])       # Can calculate effects?
```

## Validation and Recommendations

### Automatic Validation

If you request an invalid chart type for your SDS, you'll get a clear error:

```python
# Example: SDS 6 (incomplete grid)
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine'],
    time_var='sparse_time',
    chart_type='Xbar'  # ← Not valid for SDS 6
).calculate()

# Error: "Xbar charts require a complete factor×time grid.
#         Your data has SDS 6 (Incomplete Grid).
#         Valid charts: ['Imr', 'R']"
```

### Recommendations

If you don't specify a `chart_type`, the library recommends the best option:

```python
# Auto-recommendation based on SDS
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine'],
    time_var='time'
    # ← No chart_type specified
).calculate()

# Will use recommended chart for detected SDS
print(result.summary['recommended_chart'])  # 'Xbar' for SDS 1
```

## SDS Decision Tree

Use this to understand what SDS your data will be:

```
Do you have grouping variables?
│
├─ NO ──→ Do you have a time variable?
│         │
│         ├─ NO ──→ SDS 0 (Simple Series)
│         │
│         └─ YES ──→ SDS 4 (Time Series)
│
└─ YES ──→ Do you have a time variable?
          │
          ├─ NO ──→ SDS 5 (Cross-Sectional)
          │
          └─ YES ──→ Is the factor×time grid complete?
                     │
                     ├─ NO ──→ SDS 6 (Incomplete Grid)
                     │
                     └─ YES ──→ What's the replication level?
                                │
                                ├─ All cells have n ≥ 2 ──→ SDS 1 (Full Replication)
                                │
                                ├─ All cells have n = 1 ──→ SDS 2 (No Replication)
                                │
                                └─ Mixed (some n≥2, some n=1) ──→ SDS 3 (Partial)
```

## SDS Capabilities Matrix

| SDS | Name | Valid Charts | VAS (R1-R5) | Effects | Interactions |
|-----|------|--------------|-------------|---------|--------------|
| 0 | Simple Series | IMR, R | ❌ | ❌ | ❌ |
| 1 | Full Factorial (Repl.) | Xbar, S, R, IMR | ✅ Exact | ✅ | ✅ |
| 2 | Full Factorial (No Repl.) | Xbar, IMR, R | ⚠️ Approx | ✅ | ✅ |
| 3 | Partial Replication | Xbar, S, R, IMR | ⚠️ Hybrid | ✅ | ✅ |
| 4 | Time Series | Xbar, S, R, IMR | ❌ | ❌ | ❌ |
| 5 | Cross-Sectional | Xbar, S, R, IMR | ❌ | ❌ | ❌ |
| 6 | Incomplete Grid | IMR, R | ❌ | ❌ | ❌ |

## Practical Tips

### Tip 1: Check SDS Before Deep Analysis

```python
# Quick check to see what's possible:
result = pdf.analyze(response_var='y', grouping_vars=['x']).calculate()

if result.summary['sds'] == 1:
    print("Excellent! Full capabilities available.")
    # Run VAS analysis, calculate effects, etc.
elif result.summary['sds'] == 2:
    print("Good. Most features available, but R2 is approximate.")
else:
    print(f"Limited to basic control charts (SDS {result.summary['sds']})")
```

### Tip 2: Understand Your Data Structure

```python
# Let the library tell you about your data:
result = pdf.analyze(response_var='y').calculate()

print(f"Your data is: {result.summary['sds_description']}")
print(f"You can use: {', '.join(result.summary['valid_charts'])}")
print(f"We recommend: {result.summary['recommended_chart']}")
```

### Tip 3: Aim for SDS 1 When Designing Experiments

When designing new data collection:

- **Best**: Aim for SDS 1 (full factorial with replication)
- **Good**: SDS 2 or 3 acceptable for resource constraints
- **Limited**: SDS 0, 4, 5, 6 only when unavoidable

## Summary

Understanding SDS helps you:

- ✅ Know what analyses are possible for your data
- ✅ Understand the limitations of your dataset
- ✅ Design better data collection in the future
- ✅ Interpret results correctly
- ✅ Communicate data structure clearly

**The best part?** ProcessBehavior does all the detection automatically - you just need to understand what it's telling you!

## Next Steps

- [VAS Residuals Tutorial](vas-residuals.md) - Learn about R1-R5 decomposition (SDS 1-3)
- [Chart Types Guide](../guide/chart-types.md) - When to use each chart
- [SDS Reference](../guide/sampling-design-states.md) - Complete SDS documentation
