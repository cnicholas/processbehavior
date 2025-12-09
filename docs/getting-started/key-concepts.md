# Key Concepts

This page introduces the core concepts you need to understand ProcessBehavior effectively.

## The Three-Step Workflow

ProcessBehavior follows a deliberate three-step workflow:

```python
# 1. Wrap your data
pdf = ProcessDataFrame(df)

# 2. Formulate your study (SDS detection happens here)
study = pdf.formulate(response=..., factors=..., time=...)

# 3. Analyze and visualize
result = study.analyze()
result.plot()
```

This separation ensures you understand your data structure before analyzing it.

## Sampling Design States (SDS)

The **Sampling Design State** describes the structure of your data. ProcessBehavior automatically detects which of six states applies:

| SDS | Name | Description | Recommended Chart |
|-----|------|-------------|-------------------|
| 0 | No Structure | No factors, no time | Basic statistics only |
| 1 | Full Replication | All factor-time cells have n >= 2 | Xbar-S |
| 2 | No Replication | All cells have exactly n = 1 | Xbar-S (MR-based) |
| 3 | Partial Replication | Mix of n=1 and n>=2 cells | Xbar-S (hybrid) |
| 4 | Single Stream | One factor level over time | Stratified IMR |
| 5 | Nested Design | Hierarchical factor structure | IMR |
| 6 | Unstructured | Irregular/sporadic collection | IMR |

### Why SDS Matters

The SDS determines:
- Which chart types are valid
- How within-group variance is estimated
- Which VAS residuals can be computed
- What conclusions you can draw

```python
study = pdf.formulate(response='weight', factors=['lane'], time='batch')
print(f"SDS: {study.sds}")           # e.g., 3
print(f"Name: {study.sds_name}")     # e.g., "Partial Replication"
print(f"Valid: {study.valid_charts}") # e.g., ['Xbar', 'S', 'Imr']
```

## Wheeler's Variance Analysis System (VAS)

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

# Chart residuals
result = study.analyze(chart='R4_Imr')
result.plot()
```

## Chart Types

### Standard Charts

| Chart | Use Case | Requirements |
|-------|----------|--------------|
| **Xbar** | Compare subgroup means | n >= 2 per subgroup |
| **S** | Monitor subgroup variation | n >= 2 per subgroup |
| **Imr** | Individual measurements over time | Any structure |
| **R** | Range of subgroups | n >= 2 per subgroup |

### Residual Charts

| Chart | Based On | Purpose |
|-------|----------|---------|
| **R2_S** | S chart | Within-group variation stability |
| **R2_Imr** | IMR | Within variation (no replication) |
| **R3_Imr** | IMR | Detect factor-time interactions |
| **R4_Imr** | IMR | Detect time effects |
| **R5_Imr** | IMR | Detect factor effects |

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
- **IMR charts**: All 8 rules apply

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
| Sampling Design State | - | `sds` |
| Process Behavior Chart | Control Chart | Chart |

## Philosophy

ProcessBehavior follows Wheeler's philosophy:

1. **Charts are for understanding, not control** - The goal is insight into variation, not just limit checking.

2. **Let the data speak** - Automatic SDS detection ensures appropriate analysis.

3. **Separate formulation from analysis** - Understanding your data structure comes first.

4. **Plain DataFrames** - Results are standard pandas DataFrames, not custom objects.

## Next Steps

- [Basic IMR Chart](../tutorials/basic-imr.ipynb) - Create your first IMR chart
- [Sampling Design States](../user-guide/sds-detection.md) - Deep dive into SDS
- [VAS Residuals](../user-guide/residuals.md) - Understanding VAS residuals
