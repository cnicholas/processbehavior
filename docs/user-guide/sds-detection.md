# Sampling Design States

ProcessBehavior automatically detects your data's **Sampling Design State (SDS)**, which determines the appropriate analysis approach. Understanding SDS helps you interpret results correctly and choose the right charts.

## The Six Sampling Design States

| SDS | Name | Structure | Recommended Chart |
|-----|------|-----------|-------------------|
| 0 | No Structure | No factors, no time | Basic statistics |
| 1 | Full Replication | All cells n >= 2 | Xbar-S |
| 2 | No Replication | All cells n = 1 | Xbar-S (MR-based) |
| 3 | Partial Replication | Mixed n=1 and n>=2 | Xbar-S (hybrid) |
| 4 | Single Stream | One factor, multiple times | Stratified IMR |
| 5 | Nested Design | Hierarchical structure | IMR |
| 6 | Unstructured | Irregular collection | IMR |

## How SDS is Detected

When you call `formulate()`, ProcessBehavior examines:

1. **Presence of factors** - Do you have grouping variables?
2. **Presence of time** - Do you have a time/sequence variable?
3. **Replication** - How many observations per (factor, time) cell?
4. **Structure** - Is the design balanced or irregular?

```python
study = pdf.formulate(
    response=pdf.columns.weight,
    factors=[pdf.columns.lane],
    time=pdf.columns.batch
)

print(f"SDS: {study.sds}")
print(f"Name: {study.sds_name}")
print(f"Description: {study.sds_description}")
```

## SDS 1: Full Replication

**Structure**: Every (factor, time) combination has 2+ observations.

**Example**: 4 lanes × 10 batches × 3 replicates = 120 observations

```python
# Every lane-batch combination has exactly 3 measurements
# Lane A, Batch 1: [99.2, 100.1, 99.8]
# Lane A, Batch 2: [100.3, 99.9, 100.5]
# ...
```

**Capabilities**:
- ✅ Exact within-cell variance estimation
- ✅ All VAS residuals (R1-R5)
- ✅ Full interaction analysis
- ✅ Most powerful statistical tests

**Valid Charts**: Xbar, S, Imr, R2_S, R3_Imr, R4_Imr, R5_Imr

## SDS 2: No Replication

**Structure**: Every (factor, time) combination has exactly 1 observation.

**Example**: 4 lanes × 10 batches × 1 measurement = 40 observations

```python
# Each lane-batch combination has exactly 1 measurement
# Lane A, Batch 1: [99.5]
# Lane A, Batch 2: [100.2]
# ...
```

**Capabilities**:
- ⚠️ Variance estimated via 2-point moving average
- ⚠️ R2 residuals approximate (using backward moving range)
- ✅ Main effects analysis
- ✅ Xbar-S analysis (with MR-based limits)

**Valid Charts**: Xbar, S, Imr

## SDS 3: Partial Replication

**Structure**: Mix of n=1 and n>=2 cells (most common in practice).

**Example**: Some batches have 3 replicates, others have 1

```python
# Lane A, Batch 1: [99.2, 100.1, 99.8]  # 3 reps
# Lane A, Batch 2: [100.3]               # 1 rep
# Lane A, Batch 3: [99.7, 100.0]         # 2 reps
# ...
```

**Capabilities**:
- ⚠️ Hybrid variance estimation (exact for n>1, zero for n=1)
- ⚠️ VAS residuals available but interpretation requires care
- ✅ Xbar-S analysis with hybrid limits

**Valid Charts**: Xbar, S, Imr

## SDS 4: Single Stream Over Time

**Structure**: One factor level (or no factors), multiple time points.

**Example**: Daily temperature readings from one sensor

```python
# Just measurements over time
# Day 1: 72.1
# Day 2: 71.8
# Day 3: 72.5
# ...
```

**Capabilities**:
- ✅ Perfect for time series monitoring
- ✅ All 8 WECO rules applicable
- ❌ No factor comparisons (only one level)

**Valid Charts**: Imr, R

## SDS 5: Nested Design

**Structure**: Hierarchical factor structure with incomplete temporal coverage.

**Example**: Different operators work different shifts on different days

**Capabilities**:
- ⚠️ Limited to IMR analysis
- ⚠️ Stratified analysis recommended

**Valid Charts**: Imr, R

## SDS 6: Unstructured

**Structure**: Irregular or sporadic data collection.

**Example**: Measurements taken whenever convenient, no regular schedule

**Capabilities**:
- ⚠️ Most limited analysis options
- ⚠️ IMR with adaptive limits

**Valid Charts**: Imr, R

## Checking Your SDS

After formulation, inspect the SDS information:

```python
study = pdf.formulate(
    response=pdf.columns.weight,
    factors=[pdf.columns.lane],
    time=pdf.columns.batch
)

# Quick check
print(f"SDS {study.sds}: {study.sds_name}")

# Detailed information
print(study.sds_description)

# What's valid for this SDS?
print(f"Valid charts: {study.valid_charts}")
print(f"Recommended: {study.recommended_chart}")

# VAS residual charts (if available)
print(f"Residual charts: {study.residual_charts}")
```

## Impact on Analysis

The SDS affects three key aspects:

### 1. Variance Estimation

| SDS | Method |
|-----|--------|
| 1 | Within-cell standard deviation (exact) |
| 2 | 2-point backward moving average |
| 3 | Hybrid: exact for n>1, zero for n=1 |
| 4-5 | Hybrid: exact for n>1, zero for n=1 |
| 6 | 2-point backward moving average |

### 2. Available Charts

#### Standard Charts

| SDS | Xbar-S | Stratified IMR |
|-----|--------|----------------|
| 1 | ✅ | ✅ |
| 2 | ✅ (MR-based limits) | ✅ |
| 3 | ✅ (hybrid limits) | ✅ |
| 4 | ❌ | ✅ |
| 5 | ❌ | ✅ |
| 6 | ❌ | ✅ |

#### VAS Residual Charts

The available chart types for each residual depend on the **rational subgrouping structure**:

| Residual | Subgrouping | Xbar/S Available | IMR Available |
|----------|-------------|------------------|---------------|
| **R2** | By cell (k,t) | SDS 1, 3, 4, 5* | All SDS |
| **R3** | By cell (k,t) | SDS 1, 3, 4, 5* | All SDS |
| **R4** | By time (aggregate across factors) | All SDS | All SDS |
| **R5** | By factor (aggregate across time) | All SDS | All SDS |

*When cells have n≥2. SDS 2 and 6 use IMR only.

**Key insight**: R4 and R5 use different rational subgrouping than R2/R3:
- **R4**: Aggregates observations across factor levels for each time point (N_.t = Σ_k N_kt)
- **R5**: Aggregates observations across time for each factor level (N_k. = Σ_t N_kt)

This enables Xbar/S analysis for R4 and R5 even when individual cells have n=1, because the aggregated subgroups have larger sample sizes.

**Note on R2 calculation**: R2 adapts to your sampling structure:
- **SDS 1**: Within-cell deviation (`R2 = Y - Ȳ_kt`)
- **SDS 2, 6**: Moving average method (`R2 = Y - MA2`) for unreplicated/sparse designs
- **SDS 3, 4, 5**: Within-cell deviation (R2=0 for cells with n=1)

### 3. Signal Detection Rules

| SDS | Applicable Rules |
|-----|-----------------|
| 1-3 (Xbar/S) | Rule 1 only |
| 1-6 (IMR) | All 8 rules |

## Improving Your SDS

To get a higher SDS (more analytical power):

### Move from SDS 2 → SDS 1

Add replicates to each cell:
```python
# Instead of 1 measurement per lane-batch
# Take 3 measurements per lane-batch
```

### Move from SDS 3 → SDS 1

Ensure all cells have the same number of replicates:
```python
# Standardize data collection
# Every lane-batch combination gets exactly n measurements
```

### Move from SDS 4 → SDS 1-3

Add factor levels:
```python
# Instead of one sensor
# Monitor multiple sensors simultaneously
```

## Example: Diagnosing SDS

```python
import pandas as pd
from processbehavior import ProcessDataFrame

# Check cell sizes to understand your SDS
study = pdf.formulate(
    response=pdf.columns.weight,
    factors=[pdf.columns.lane],
    time=pdf.columns.batch
)

# Get cell counts
cell_counts = (study.dataset
    .groupby(['lane', 'batch'])
    .size()
    .reset_index(name='n'))

print("Observations per cell:")
print(cell_counts['n'].value_counts())

# If all n >= 2: SDS 1
# If all n == 1: SDS 2
# If mixed: SDS 3
```

## Summary

| If You Have... | SDS | Best Approach |
|----------------|-----|---------------|
| Full replication (n>=2 per cell) | 1 | Xbar-S with full VAS |
| One observation per cell | 2 | Xbar-S with MR limits |
| Mixed replication | 3 | Xbar-S with hybrid limits |
| Single stream over time | 4 | IMR with full WECO rules |
| Nested/hierarchical | 5 | Stratified IMR |
| Irregular collection | 6 | IMR with caution |

## Next Steps

- [Chart Types](chart-types.md) - Choosing the right chart for your SDS
- [VAS Residuals](residuals.md) - VAS residual interpretation by SDS
- [API Reference](../reference/api.md) - Complete SDS detection API
