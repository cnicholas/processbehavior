# Sampling Design States

ProcessBehavior automatically detects your data's **Sampling Design State (SDS)**, which determines the appropriate analysis approach. Understanding SDS helps you interpret results correctly and choose the right charts.

## Expected vs Observed: The Core Model

ProcessBehavior understands every analytic study through the lens of **expected structure** versus **observed structure**:

- **Observed structure** comes from your data—the actual factor levels, time points, and cell sizes present
- **Expected structure** comes from your sampling plan—what the experimental design intended

The system operates effectively with or without a plan:

| Approach | What ProcessBehavior Knows | Capabilities |
|----------|---------------------------|--------------|
| **Without plan** | Observed structure (including all-NA cells) | All SDS detection (1-6), charts, basic design reports |
| **With plan** | Expected + observed structure | All SDS detection (1-6), coverage analysis, rich design reports |

SDS detection runs on raw data before NA rows are dropped, so cells where all response values are NA are counted as empty (N_kt=0). This enables SDS 4-6 detection even without a plan. With a plan, ProcessBehavior can additionally compare what *was* collected against what *should have been* collected—catching entirely absent factor levels and providing detailed reports showing exactly what's missing.

## The Six Sampling Design States

| SDS | Name | Structure | Recommended Chart |
|-----|------|-----------|-------------------|
| 1 | Full Replication | All cells n >= 2 | Xbar-S |
| 2 | No Replication | All cells n = 1 | Xbar-S (MR-based) |
| 3 | Partial Replication | Mixed n=1 and n>=2 | Xbar-S (hybrid) |
| 4 | Single Stream | One factor, multiple times | Stratified XmR |
| 5 | Nested Design | Hierarchical structure | XmR |
| 6 | Unstructured | Irregular collection | XmR |

## How SDS is Detected

When you call `formulate()`, ProcessBehavior examines:

1. **Presence of factors** - Do you have grouping variables?
2. **Presence of time** - Do you have a time/sequence variable?
3. **Replication** - How many observations per (factor, time) cell?
4. **Structure** - Is the design balanced or irregular?
5. **Grid coverage** - What proportion of expected cells are observed?

```python
study = pb.formulate(
    response='weight',
    factors=['lane'],
    time='batch'
)

print(f"SDS: {study.observed_design_state.sds}")
print(f"Reason: {study.sds_reason}")
print(f"Description: {study.sds_description}")
```

## The VAS Problem Formulation

Following Wheeler's Variance Analysis System (VAS) methodology, a rigorous problem formulation includes:

- **Unit of Analysis**: The fundamental entity being measured (e.g., "filled cup", "loan contract")
- **Response**: The measurement variable
- **Factors**: Grouping variables that define rational subgroups
- **Time**: The sequencing variable

```python
from processbehavior import ProcessBehavior

pb = ProcessBehavior(df)

# Basic formulation - factors inferred from observed data
study = pb.formulate(
    response='weight',
    factors=['lane', 'shift'],
    time='batch'
)

print(f"SDS: {study.observed_design_state.sds}")
print(f"Charts: {study.valid_charts}")
```

This works for all SDS types (1-6). SDS detection is determined by properties of the observed data—replication levels, cell sizes, and factor structure—including cells where all response values are NA (counted as empty).

## Extending Formulation with a Sampling Plan

A **sampling plan** extends the formulation by explicitly defining what was *expected* in the experimental design:

```python
plan = {
    'factors': {
        'Machine': ['Machine1', 'Machine2', 'Machine3'],
        'Shift': ['Day', 'Night']
    },
    'T': 80,   # Expected time points
    'N': 2     # Expected observations per cell
}

study = pb.formulate(response='weight', time='time', plan=plan)
```

**The plan provides three key benefits:**

1. **Catches entirely absent factor levels** - Factor levels or time points not present in data at all are only visible with a plan
2. **Design reports** - See exactly what's missing from your data (planned vs observed K, T, N, R)
3. **Documentation** - Captures the intended experimental design for reproducibility

### SDS 4-6 Detection: With and Without a Plan

ProcessBehavior can detect SDS 4-6 **with or without a plan**. SDS detection runs on raw data before NA rows are dropped, so cells where all response values are NA (e.g., from garbage values like `*` or `ND`) are counted as empty cells (N_kt=0).

| Approach | How Empty Cells Are Detected |
|----------|------------------------------|
| **`factors`** | Factor/time combinations where every response value is NA or garbage |
| **`plan`** | Same as above, plus factor levels or time points entirely absent from the data |

A plan adds value by catching gaps that `factors` alone cannot see—for example, if a factor level was planned but never collected at all, no rows exist for that level, so `factors` won't see it. But when the data contains all planned factor levels (even if some cells have all-NA responses), both approaches detect the same SDS.

```python
# With factors: ProcessBehavior sees all-NA cells as empty → SDS 4-6 detected
# With plan: additionally catches factor levels entirely absent from data
```

### Expected vs Observed: How It Works

When you provide a plan, ProcessBehavior compares:

| Metric | Expected (from plan) | Observed (from data) |
|--------|---------------------|---------------------|
| **K** | Product of factor levels | Unique factor combinations |
| **T** | Specified time points | Unique time values |
| **R** | K × T (total cells) | Actual (factor, time) cells |
| **N** | Observations per cell | Cell size distribution |

```python
design = study.design()
print(f"K: planned={design.K}, observed={design.K_observed}, missing={design.K_missing}")
print(f"T: planned={design.T}, observed={design.T_observed}, missing={design.T_missing}")
print(f"R: planned={design.R}, observed={design.R_observed}, missing={design.R_missing}")
```

### Plan Parameter Structure

```python
plan = {
    'factors': {
        'factor_name': [list of expected levels],
        ...
    },
    'T': expected_time_points,      # optional
    'N': expected_obs_per_cell      # optional
}
```

**Notes:**
- Factor levels should include ALL expected levels, even if some are missing from data
- `T` enables detection of missing time points
- `N` documents expected replication (useful for design reports)

### Viewing the Design Report

After formulation, inspect how your data compares to the plan:

```python
design = study.design()
print(design)

# DesignReport(2 factors, with plan)
#   SDS reason: incomplete_grid
#   K: planned=6, observed=6, missing=0
#   T: planned=80, observed=8, missing=72
#   R: planned=480, observed=34, missing=446
#
#   Factors:
#     Machine: planned=['Machine1','Machine2','Machine3'], observed=[...]
#     Shift: planned=['Day','Night'], observed=[...]
#
#   Structure: Incomplete: 72 time points missing
```

The design report shows:
- **K, T, R**: Planned vs observed counts with missing tallies
- **N**: Planned vs observed (min, median, max) cell sizes
- **missing_combos**: Which factor combinations are missing
- **sds_reason**: Why this SDS was detected
- **structure_summary**: Overall assessment

### Summary: When Plans Add Value

| SDS | Plan Required? | Plan Adds |
|-----|---------------|-----------|
| 1 - Full Replication | No | Design comparison, coverage reports |
| 2 - No Replication | No | Design comparison, coverage reports |
| 3 - Partial Replication | No | Design comparison, shows sparse cells |
| 4 - Incomplete with Singletons | No | Catches entirely absent factor levels |
| 5 - Incomplete without Singletons | No | Catches entirely absent factor levels |
| 6 - Incomplete without Replication | No | Catches entirely absent factor levels |

**Best practice:** Always use a plan for rigorous VAS analysis. While not required for SDS detection, plans document your experimental design, catch entirely absent factor levels, and enable rich design reports.

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

**Valid Charts**: Xbar, S, XmR, R2_S, R3_XmR, R4_XmR, R5_XmR

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

**Valid Charts**: Xbar, S, XmR

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

**Valid Charts**: Xbar, S, XmR

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

**Valid Charts**: XmR, R

## SDS 5: Nested Design

**Structure**: Hierarchical factor structure with incomplete temporal coverage.

**Example**: Different operators work different shifts on different days

**Capabilities**:
- ⚠️ Limited to XmR analysis
- ⚠️ Stratified analysis recommended

**Valid Charts**: XmR, R

## SDS 6: Unstructured

**Structure**: Irregular or sporadic data collection.

**Example**: Measurements taken whenever convenient, no regular schedule

**Capabilities**:
- ⚠️ Most limited analysis options
- ⚠️ XmR with adaptive limits

**Valid Charts**: XmR, R

## Checking Your SDS

After formulation, inspect the SDS information:

```python
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)

# Quick check
print(f"SDS {study.observed_design_state.sds}: {study.sds_reason}")

# Detailed information
print(study.sds_description)  # ADS-derived human prose

# What's valid for this SDS?
print(f"Valid charts: {study.valid_charts}")
print(f"Recommended: {study.recommended_chart}")

# VAS residual charts (if available)
print(f"Available residuals: {study.residuals}")
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

| SDS | Xbar-S | Stratified XmR |
|-----|--------|----------------|
| 1 | ✅ | ✅ |
| 2 | ✅ (MR-based limits) | ✅ |
| 3 | ✅ (hybrid limits) | ✅ |
| 4 | ❌ | ✅ |
| 5 | ❌ | ✅ |
| 6 | ❌ | ✅ |

#### VAS Residual Charts

The available chart types for each residual depend on the **rational subgrouping structure**:

| Residual | Subgrouping | Xbar/S Available | XmR Available |
|----------|-------------|------------------|---------------|
| **R2** | By cell (k,t) | SDS 1, 3, 4, 5* | All SDS |
| **R3** | By cell (k,t) | SDS 1, 3, 4, 5* | All SDS |
| **R4** | By time (aggregate across factors) | All SDS | All SDS |
| **R5** | By factor (aggregate across time) | All SDS | All SDS |

*When cells have n≥2. SDS 2 and 6 use XmR only.

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
| 1-6 (XmR) | All 8 rules |

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
from processbehavior import ProcessBehavior

# Check cell sizes to understand your SDS
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
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
| Single stream over time | 4 | XmR with full WECO rules |
| Nested/hierarchical | 5 | Stratified XmR |
| Irregular collection | 6 | XmR with caution |

## Next Steps

- [Chart Types](chart-types.md) - Choosing the right chart for your SDS
- [VAS Residuals](residuals.md) - VAS residual interpretation by SDS
- [API Reference](../reference/api.md) - Complete SDS detection API
