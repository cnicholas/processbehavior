# Key Concepts

This page introduces the core concepts you need to understand ProcessBehavior effectively.

## The Three-Step Workflow

ProcessBehavior follows a deliberate three-step workflow:

```python
# 1. Wrap your data
pb = ProcessBehavior(df)

# 2. Formulate your study (SDS detection happens here)
study = pb.formulate(response=..., factors=..., time=...)

# 3. Analyze and visualize
result = study.execute()
result.plot()
```

This separation ensures you understand your data structure before analyzing it.

## Sampling Design States (SDS)

The **Sampling Design State** describes the structure of your data. ProcessBehavior automatically detects which of six states applies:

| SDS | Name | Description | Recommended Chart |
|-----|------|-------------|-------------------|
| 1 | Full Replication | All factor-time cells have n >= 2 | Xbar-S |
| 2 | No Replication | All cells have exactly n = 1 | Xbar-S (MR-based) |
| 3 | Partial Replication | Mix of n=1 and n>=2 cells | Xbar-S (hybrid) |
| 4 | Single Stream | One factor level over time | Stratified XmR |
| 5 | Nested Design | Hierarchical factor structure | XmR |
| 6 | Unstructured | Irregular/sporadic collection | XmR |

### Why SDS Matters

The SDS determines:
- Which chart types are valid
- How within-group variance is estimated
- Which VAS residuals can be computed
- What conclusions you can draw

```python
study = pb.formulate(response='weight', factors=['lane'], time='batch')
print(f"SDS: {study.observed_design_state.sds}")   # e.g., 3
print(f"Reason: {study.sds_reason}")             # e.g., "partial_replication"
print(f"Valid: {study.valid_charts}") # e.g., ['Xbar', 'S', 'XmR']
```

## Bishop's Variance Analysis System (VAS)

For replicated designs (SDS 1-3), ProcessBehavior computes five residual decompositions:

| Residual | Formula | Questions Answered |
|----------|---------|-------------------|
| **R1** | Y - Y&#x0304; | Total deviation from grand mean |
| **R2** | Y - Y&#x0304;<sub>kt</sub> | Within-cell variation (unexplained) |
| **R3** | Y - Y&#x0304;<sub>k</sub> - Y&#x0304;<sub>t</sub> + Y&#x0304; | Factor-time interaction |
| **R4** | Y&#x0304;<sub>t</sub> - Y&#x0304; + R2 | Time effects + unexplained |
| **R5** | Y&#x0304;<sub>k</sub> - Y&#x0304; + R2 | Factor effects + unexplained |

### Interpreting VAS Residuals

- **R2 (Within-cell)**: Is measurement variation stable? Are there special causes within subgroups?
- **R3 (Interaction)**: Does the factor effect change over time?
- **R4 (Time)**: Are there trends, shifts, or time-related patterns?
- **R5 (Factor)**: Do factors differ significantly from each other?

```python
# Access residuals after formulation
print(study.dataset[['R1', 'R2', 'R3', 'R4', 'R5']].head())

# Chart residuals using the value parameter
result = study.execute(chart='XmR', by=['lane'], value='R4')
result.plot()
```

## Chart Types

### Standard Charts

| Chart | Use Case | Requirements |
|-------|----------|--------------|
| **Xbar** | Compare subgroup means | n >= 2 per subgroup |
| **S** | Monitor subgroup variation | n >= 2 per subgroup |
| **XmR** | Individual measurements over time | Any structure |
| **R** | Range of subgroups | n >= 2 per subgroup |

### Residual Charts

Use the `value` parameter to chart residuals instead of the response variable:

```python
# Chart R5 residuals (factor effects) on an Xbar chart
result = study.execute(chart='Xbar', value='R5')

# Chart R4 residuals (time effects) on a stratified XmR chart
result = study.execute(chart='XmR', by=['lane'], value='R4')
```

| Residual | Chart Type | Purpose |
|----------|------------|---------|
| **R2** | S or XmR | Within-group variation stability |
| **R3** | XmR | Detect factor-time interactions |
| **R4** | XmR | Detect time effects |
| **R5** | Xbar or XmR | Detect factor effects |

## The `by` Parameter

The `by` parameter controls how data is grouped or stratified:

```python
# Xbar chart aggregated by all factors (default)
result = study.execute(chart='Xbar')

# Xbar chart aggregated by single factor
result = study.execute(chart='Xbar', by=['factor 1'])

# Xbar chart collapsed to grand mean
result = study.execute(chart='Xbar', by=[])

# XmR chart stratified by factor (separate chart per level)
result = study.execute(chart='XmR', by=['lane'])
```

**Key concept**: The `by` parameter creates *views* over the same underlying data. Residuals are computed once during formulation and never change regardless of how you view them.

## Western Electric Rules

Signal detection uses the Western Electric (WECO) rules:

| Rule | Name | Description |
|------|------|-------------|
| 1 | Beyond Limits | Point outside 3-sigma limits |
| 2 | Zone A | 2 of 3 consecutive in Zone A (2-3 sigma) |
| 3 | Zone B | 4 of 5 consecutive in Zone B or beyond |
| 4 | Run | 8+ consecutive same side of centerline |
| 5 | Trend | 6+ consecutive increasing or decreasing |
| 6 | Oscillation | 14+ consecutive alternating up/down |
| 7 | Hugging Center | 15+ consecutive in Zone C (within 1 sigma) |
| 8 | Avoiding Center | 8+ consecutive avoiding Zone C |

### Rule Applicability

- **Xbar/S charts**: Only Rule 1 applies (beyond limits)
- **XmR charts**: All 8 rules apply

```python
# Standard rules (1-4)
signals = result.detect_signals(rules='standard')

# Extended rules (1-8)
signals = result.detect_signals(rules='extended')
```

## Terminology Mapping

ProcessBehavior uses Wheeler's terminology consistently:

| Wheeler Term | Common Term | ProcessBehavior |
|--------------|-------------|-----------------|
| Response Variable | Y, measurement | `response` |
| Rational Subgroup | Factor, group | `factors` |
| Time Sequence | Period, batch | `time` |
| Sampling Design State | - | `observed_design_state` / `analytical_design_state` |
| Process Behavior Chart | Control Chart | Chart |

## Philosophy

ProcessBehavior follows Wheeler's philosophy:

1. **Charts are for understanding, not control** - The goal is insight into variation, not just limit checking.

2. **Let the data speak** - Automatic SDS detection ensures appropriate analysis.

3. **Separate formulation from analysis** - Understanding your data structure comes first.

4. **Plain DataFrames** - Results are standard pandas DataFrames, not custom objects.

## Next Steps

- [Basic XmR Chart](../tutorials/basic-imr.ipynb) - Create your first XmR chart
- [Sampling Design States](../user-guide/sds-detection.md) - Deep dive into SDS
- [VAS Residuals](../user-guide/residuals.md) - Understanding VAS residuals
