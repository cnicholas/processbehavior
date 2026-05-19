# Design-state lineage (PDS / ODS / ADS)

ProcessBehavior reports the design state at **three points** in your analysis lifecycle, and routes chart selection, residual availability, and variance decomposition by the third one (ADS).

## The three states

| State | Computed from | Codomain | Meaning |
|-------|---------------|----------|---------|
| **PDS** — Planned | Your declared `factors` and `time` (or an explicit `plan=`) | {1, 2} | What you intended to collect |
| **ODS** — Observed | Raw data, before NA-filtering | {1..6} | What was actually collected (cells with all-NA responses count as "attempted but empty") |
| **ADS** — Analytical | Tidy data, after NA-filtering | {0, 1, 2, 3} | What survives tidying; **drives chart selection, residual availability, and variance decomposition** |

The integer 1–6 codes are Bishop's reference scale ("Bishop Table 1"). Each of PDS, ODS, ADS carries a value on that scale via its `.sds` field.

Access the full lineage on a formulated study:

```python
study.plan_design_state        # PDS (or None when no plan was supplied)
study.observed_design_state    # ODS — Bishop's classification of the raw N_kt distribution
study.analytical_design_state  # ADS — what the analysis is fit for
study.design()                 # DesignReport — all three in one report object
```

## Expected vs Observed: the underlying model

The lineage above falls out of ProcessBehavior's expected-vs-observed model:

- **Observed structure** comes from your data — the actual factor levels, time points, and cell sizes present.
- **Expected structure** comes from your sampling plan — what the experimental design intended.

The system operates effectively with or without an explicit `plan=`:

| Approach | What ProcessBehavior Knows | Capabilities |
|----------|---------------------------|--------------|
| **Without plan** | Observed structure (including all-NA cells) | Full ODS / ADS detection, charts, basic design reports |
| **With plan** | Expected + observed structure | Full PDS / ODS / ADS detection, coverage analysis, rich design reports |

ODS detection runs on raw data before NA rows are dropped, so cells where all response values are NA are counted as empty (N_kt=0). This enables ODS 4–6 detection even without a plan. With a plan, ProcessBehavior can additionally compare what *was* collected against what *should have been* collected — catching entirely absent factor levels and providing detailed reports showing exactly what's missing.

## Bishop's six design states (the 1–6 scale)

Each of PDS, ODS, ADS reports a value on this scale.

**Complete/Semi-Complete (no empty cells):**

| Code | Name | Cell Sizes (N_kt) | Recommended Chart |
|-----|------|--------------------|-------------------|
| 1 | Full Replication | All N_kt >= 2 | Xbar |
| 2 | No Replication | All N_kt = 1 | X |
| 3 | Partial Replication | Mix of N_kt = 1 and N_kt >= 2 | X |

**Incomplete (has empty cells — applies to ODS only; collapses during tidying):**

| Code | Name | Cell Sizes (N_kt) | ODS → ADS collapse |
|-----|------|--------------------|--------------------|
| 4 | Incomplete, No Singletons | Empty cells + all observed N_kt >= 2 | ODS 4 → ADS 1 |
| 5 | Incomplete, No Replication | Empty cells + all observed N_kt = 1 | ODS 5 → ADS 2 |
| 6 | Incomplete, With Singletons | Empty cells + mixed N_kt | ODS 6 → ADS 3 |

## How DS is Detected

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

print(f"DS: {study.observed_design_state.sds}")
print(f"Reason: {study.sds_reason}")
print(f"Description: {study.sds_description}")
```

## The VAS Problem Formulation

Following Bishop's Variance Analysis System (VAS) methodology, a rigorous problem formulation includes:

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

print(f"DS: {study.observed_design_state.sds}")
print(f"Charts: {study.valid_charts}")
```

This works for all DS types (1-6). DS detection is determined by properties of the observed data—replication levels, cell sizes, and factor structure—including cells where all response values are NA (counted as empty).

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

### DS 4-6 Detection: With and Without a Plan

ProcessBehavior can detect DS 4-6 **with or without a plan**. DS detection runs on raw data before NA rows are dropped, so cells where all response values are NA (e.g., from garbage values like `*` or `ND`) are counted as empty cells (N_kt=0).

| Approach | How Empty Cells Are Detected |
|----------|------------------------------|
| **`factors`** | Factor/time combinations where every response value is NA or garbage |
| **`plan`** | Same as above, plus factor levels or time points entirely absent from the data |

A plan adds value by catching gaps that `factors` alone cannot see—for example, if a factor level was planned but never collected at all, no rows exist for that level, so `factors` won't see it. But when the data contains all planned factor levels (even if some cells have all-NA responses), both approaches detect the same DS.

```python
# With factors: ProcessBehavior sees all-NA cells as empty → DS 4-6 detected
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
- **sds_reason**: Why this DS was detected
- **structure_summary**: Overall assessment

### Summary: When Plans Add Value

| DS | Plan Required? | Plan Adds |
|-----|---------------|-----------|
| 1 - Full Replication | No | Design comparison, coverage reports |
| 2 - No Replication | No | Design comparison, coverage reports |
| 3 - Partial Replication | No | Design comparison, shows sparse cells |
| 4 - Incomplete without Singletons | No | Catches entirely absent factor levels |
| 5 - Incomplete without Replication | No | Catches entirely absent factor levels |
| 6 - Incomplete with Singletons | No | Catches entirely absent factor levels |

**Best practice:** Always use a plan for rigorous VAS analysis. While not required for DS detection, plans document your experimental design, catch entirely absent factor levels, and enable rich design reports.

## DS 1: Full Replication

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

**Valid Charts**: Xbar, S, X + residual charts (R2 via S, R3-R5 via Xbar/S)

## DS 2: No Replication

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

**Valid Charts**: Xbar, S, X

## DS 3: Partial Replication

**Structure**: Mix of n=1 and n>=2 cells (most common in practice).

**Example**: Some batches have 3 replicates, others have 1

```python
# Lane A, Batch 1: [99.2, 100.1, 99.8]  # 3 reps
# Lane A, Batch 2: [100.3]               # 1 rep
# Lane A, Batch 3: [99.7, 100.0]         # 2 reps
# ...
```

**Capabilities**:
- ⚠️ Hybrid R2 estimation (exact where n >= 2, ma2 where n = 1)
- ⚠️ VAS residuals available but interpretation requires care
- ✅ Xbar-S analysis with hybrid limits

**Valid Charts**: Histogram, Xbar, S, X, mR

!!! note "Why Mixed is treated conservatively"
    For R2 calculation, DS 3 uses the hybrid method: exact within-cell deviation where cells have n >= 2, and the ma2 (moving average) method where cells have n = 1. This conservative approach was validated by Monte Carlo simulation — it produces more reliable variance estimates than attempting to use only the replicated cells. The recommended chart is X (not Xbar) because the mixed replication makes subgroup-mean interpretation less straightforward.

## DS 4: Incomplete, No Singletons

**Structure**: Incomplete grid — some cells have no data, but all present cells have N_kt >= 2.

**Example**: A planned 4-lane x 10-batch study where 3 batches were skipped entirely, but all collected batches have 3+ replicates.

**After cleansing**: Collapses to **ADS 1** (Full Replication) — the empty cells are removed and all remaining cells have full replication.

**Valid Charts**: Histogram, X, mR (plus Xbar, S after collapse to ADS 1)

## DS 5: Incomplete, No Replication

**Structure**: Incomplete grid — some cells have no data, and all present cells have N_kt = 1.

**Example**: A sparse unreplicated factorial where some factor-time combinations were never measured.

**After cleansing**: Collapses to **ADS 2** (No Replication) — the empty cells are removed, leaving an unreplicated structure.

**Valid Charts**: Histogram, X, mR (plus Xbar, S after collapse to ADS 2)

## DS 6: Incomplete, With Singletons

**Structure**: Incomplete grid — some cells have no data, and present cells have a mix of N_kt = 1 and N_kt >= 2.

**Example**: A manufacturing study where some shifts had one measurement, others had three, and some were skipped entirely.

**After cleansing**: Collapses to **ADS 3** (Partial Replication) — the empty cells are removed, leaving a mixed-replication structure.

**Valid Charts**: Histogram, X, mR (plus Xbar, S after collapse to ADS 3)

## From Observation to Analysis: ODS → ADS

ProcessBehavior tracks three Design States as data flows through the system. Understanding this pipeline is key to interpreting your results correctly.

### The Pipeline

1. **Plan Design State (PDS)** — Computed from your sampling plan parameters (K × T × N). Always DS 1 or DS 2. Only available when you provide a `plan` to `formulate()`.

2. **Observed Design State (ODS)** — Detected on **raw data** before any NA filtering. Response rows with garbage values (`*`, `ND`, etc.) are preserved during detection, so cells where all responses are NA count as empty cells (N_kt = 0). This enables detection of DS 4-6 (Incomplete designs).

3. **Analytical Design State (ADS)** — Computed on **tidy data** after data cleansing removes invalid response rows. The ADS reflects the structure that is actually fit for analysis and **drives all analysis decisions**: valid charts, R2 calculation method, residual availability, and interaction analysis.

### Why ODS and ADS Can Differ

After data cleansing, empty cells disappear. DS 4-6 (Incomplete designs) collapse to their Complete/Semi-Complete equivalents:

| ODS | After Cleansing → | ADS | Why |
|-----|-------------------|-----|-----|
| 4 (Incomplete, no singletons) | Empty cells removed | 1 (Full Replication) | All remaining cells have n >= 2 |
| 5 (Incomplete, no replication) | Empty cells removed | 2 (No Replication) | All remaining cells have n = 1 |
| 6 (Incomplete, with singletons) | Empty cells removed | 3 (Partial Replication) | Remaining cells have mixed n |

This separation means the system correctly identifies incomplete data collection (via ODS) while performing the most powerful analysis the clean data supports (via ADS).

### Checking Your Design States

```python
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)

# The ADS drives analysis — use these for chart selection
print(f"ADS: {study.analytical_design_state.sds}")  # e.g., 1
print(f"Reason: {study.ads_reason}")                # e.g., "full_replication"
print(f"Description: {study.ads_description}")       # Human-readable

# The ODS captures the raw data structure (diagnostic)
print(f"ODS: {study.observed_design_state.sds}")    # e.g., 6

# Chart validity is determined by the ADS
print(f"Valid charts: {study.valid_charts}")
print(f"Recommended: {study.recommended_chart}")
print(f"Available residuals: {study.residuals}")

# Full design lineage via study.design()
print(study.design())
```

## Impact on Analysis

The **Analytical Design State** determines three key aspects of the analysis:

### 1. Variance Estimation (R2 Method)

The R2 method is determined by the tidy data structure (ADS), not the raw DS:

| ADS | R2 Method | Description |
|-----|-----------|-------------|
| 1 | exact | Within-cell standard deviation (`R2 = Y - Ȳ_kt`) |
| 2 | ma2 | 2-point moving average for unreplicated designs |
| 3 | hybrid | Exact where n >= 2, ma2 where n = 1 |

### 2. Available Charts

#### Standard Charts

| DS | Xbar-S | Stratified X |
|-----|--------|----------------|
| 1 | ✅ | ✅ |
| 2 | ✅ (MR-based limits) | ✅ |
| 3 | ✅ (hybrid limits) | ✅ |
| 4 | ❌ | ✅ |
| 5 | ❌ | ✅ |
| 6 | ❌ | ✅ |

#### VAS Residual Charts

The available chart types for each residual depend on the **rational subgrouping structure**:

| Residual | Subgrouping | Xbar/S Available | X Available |
|----------|-------------|------------------|---------------|
| **R2** | By cell (k,t) | DS 1, 3, 4, 6* | All DS |
| **R3** | By cell (k,t) | DS 1, 3, 4, 6* | All DS |
| **R4** | By time (aggregate across factors) | All DS | All DS |
| **R5** | By factor (aggregate across time) | All DS | All DS |

*When cells have n≥2. DS 2 and 5 use X only.

**Key insight**: R4 and R5 use different rational subgrouping than R2/R3:
- **R4**: Aggregates observations across factor levels for each time point (N_.t = Σ_k N_kt)
- **R5**: Aggregates observations across time for each factor level (N_k. = Σ_t N_kt)

This enables Xbar/S analysis for R4 and R5 even when individual cells have n=1, because the aggregated subgroups have larger sample sizes.

**Note on R2 calculation**: R2 adapts to your sampling structure:
- **DS 1**: Within-cell deviation (`R2 = Y - Ȳ_kt`)
- **DS 2, 5**: Moving average method (`R2 = Y - MA2`) for unreplicated/sparse designs
- **DS 3, 4, 6**: Within-cell deviation (R2=0 for cells with n=1)

### 3. Signal Detection Rules

| DS | Applicable Rules |
|-----|-----------------|
| 1-3 (Xbar/S) | Rule 1 only |
| 1-6 (X) | All 8 rules |

## Improving Your DS

To get a higher DS (more analytical power):

### Move from DS 2 → DS 1

Add replicates to each cell:
```python
# Instead of 1 measurement per lane-batch
# Take 3 measurements per lane-batch
```

### Move from DS 3 → DS 1

Ensure all cells have the same number of replicates:
```python
# Standardize data collection
# Every lane-batch combination gets exactly n measurements
```

### Move from DS 6 → DS 1-3

Add factor levels:
```python
# Instead of one sensor
# Monitor multiple sensors simultaneously
```

## Example: Diagnosing DS

```python
import pandas as pd
from processbehavior import ProcessBehavior

# Check cell sizes to understand your DS
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

# If all n >= 2: DS 1
# If all n == 1: DS 2
# If mixed: DS 3
```

## Summary

| If You Have... | ODS | ADS (after cleansing) | Best Approach |
|----------------|-----|-----------------------|---------------|
| Full replication (n>=2 per cell) | 1 | 1 | Xbar with full VAS |
| One observation per cell | 2 | 2 | X with MA2-based R2 |
| Mixed replication | 3 | 3 | X with hybrid R2 |
| Incomplete grid, all observed replicated | 4 | 1 | Xbar with full VAS |
| Incomplete grid, all observed n=1 | 5 | 2 | X with MA2-based R2 |
| Incomplete grid, mixed observed | 6 | 3 | X with hybrid R2 |

## Next Steps

- [Chart Types](chart-types.md) - Choosing the right chart for your DS
- [VAS Residuals](residuals.md) - VAS residual interpretation by DS
- [API Reference](../reference/api.md) - Complete DS detection API
