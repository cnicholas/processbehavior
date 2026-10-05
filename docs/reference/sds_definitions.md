# Bishop's design-state reference scale (1–6)

This document provides the formal definitions of the six structural design states defined by Thomas A. Bishop, Ph.D. in the Variance Analysis System (VAS) framework. These are the integer codes carried by each state in the PDS / ODS / ADS lineage via its `.sds` field.

For background on the lineage itself (Planned → Observed → Analytical), see [Design-state detection](../user-guide/sds-detection.md).

## Notation

- **k**: Factor level index (e.g., machine, lane, operator)
- **t**: Time period index (e.g., day, shift, batch)
- **N_kt**: Number of observations in cell (k, t) - the count of measurements for factor level k at time t
- **Rational Subgroup**: A (k, t) cell - observations sharing the same factor level and time period

## DS Classification Table

| Design State | Sample Size N_kt | Design State | Distribution of Sample Sizes Across Rational Subgroups |
|:------------:|------------------|:---------------------:|--------------------------------------------------------|
| **1** | Min N_kt ≥ 2 | Complete | Multiple observations in each of the rational subgroups |
| **2** | Min N_kt = 1 and Max N_kt = 1 | Semi-Complete | A single observation in each of the rational subgroups |
| **3** | Min N_kt = 1 and Max N_kt ≥ 2 | Semi-Complete | Multiple observations in some rational subgroups and a single observation in some rational subgroups |
| **4** | Min N_kt = 0, N_kt ≠ 1 and Max N_kt ≥ 2 | Incomplete | Multiple observations in some rational subgroups and no data in some subgroups |
| **5** | Min N_kt = 0 and Max N_kt = 1 | Incomplete | A single observation in some rational subgroups and no data in some rational subgroups |
| **6** | Min N_kt = 0, N_kt = 1 and Max N_kt ≥ 2 | Incomplete | Multiple observations in some rational subgroups, a single observation in some rational subgroups and no data in some rational subgroups |

## Detailed Definitions

### DS 1: Complete with Full Replication

- **Condition**: Min N_kt ≥ 2
- **Grid Status**: Complete
- **Description**: Every (factor × time) cell has at least 2 observations
- **Implications**:
  - True within-cell variance can be estimated directly
  - Full VAS residual decomposition (R1-R5) supported
  - Xbar-S charts are optimal
  - Interaction effects can be estimated precisely

### DS 2: Semi-Complete with No Replication

- **Condition**: Min N_kt = 1 AND Max N_kt = 1
- **Grid Status**: Semi-Complete
- **Description**: Every (factor × time) cell has exactly 1 observation
- **Implications**:
  - No within-cell variance: R2 is estimated from the condition-and-period-adjusted series, differenced along the condition-then-time sequence and scaled by the R2 scale factor (Eqs 14-4..14-8)
  - Interaction effects confounded with pure error
  - VAS residuals available when the R2 scale factor is defined: at least 2 process design conditions and M ≥ K + 2 observations (see [R2 calculation](#r2-calculation-methods-by-ds))
  - Common in unreplicated factorial designs

### DS 3: Semi-Complete with Partial Replication

- **Condition**: Min N_kt = 1 AND Max N_kt ≥ 2
- **Grid Status**: Semi-Complete
- **Description**: Mix of cells - some have n=1, others have n≥2
- **Implications**:
  - R2 by the same scaled difference as DS 2, over every observation (Eqs 14-9..14-13); any one-observation cell selects `ma2` for every observation
  - Xbar/S is the default chart
  - **Most common in real-world data**
  - Partial interaction effect estimation
  - Requires careful handling of mixed replication

### DS 4: Incomplete with Replication Only

- **Condition**: Min N_kt = 0 AND N_kt ≠ 1 (no cells with exactly 1) AND Max N_kt ≥ 2
- **Grid Status**: Incomplete
- **Description**: Missing cells, but all present cells have n≥2
- **Implications**:
  - Incomplete grid structure
  - Where data exists, full replication is available
  - Can estimate within-cell variance for present cells
  - Common in nested/hierarchical designs with asynchronous coverage

### DS 5: Incomplete with No Replication

- **Condition**: Min N_kt = 0 AND Max N_kt = 1
- **Grid Status**: Incomplete
- **Description**: Missing cells, and all present cells have exactly n=1
- **Implications**:
  - Most limited analytical case
  - Sparse, irregular data structure
  - No within-cell variance estimation possible
  - R2 by the scaled difference (`ma2`) throughout; collapses to ADS 2 after cleaning

### DS 6: Incomplete with Mixed Replication

- **Condition**: Min N_kt = 0 AND some N_kt = 1 AND Max N_kt ≥ 2
- **Grid Status**: Incomplete
- **Description**: Missing cells, plus mix of n=1 and n≥2 in present cells
- **Implications**:
  - Incomplete (factor × time) grid
  - Some cells have no data at all
  - Present cells have mixed replication
  - Complex variance estimation required

## Detection Algorithm

To determine the DS for a dataset:

```
1. Compute N_kt for all (factor, time) combinations
2. Determine:
   - min_n = minimum N_kt among PRESENT cells (excluding missing)
   - max_n = maximum N_kt
   - has_missing = whether any expected (k,t) cells have N_kt = 0
   - has_singles = whether any cells have N_kt = 1
   - has_multiples = whether any cells have N_kt ≥ 2

3. Classification:
   - If NOT has_missing (Complete/Semi-Complete grid):
     - If min_n ≥ 2 → DS 1
     - If min_n = 1 AND max_n = 1 → DS 2
     - If min_n = 1 AND max_n ≥ 2 → DS 3
   - If has_missing (Incomplete grid):
     - If NOT has_singles AND has_multiples → DS 4
     - If has_singles AND NOT has_multiples → DS 5
     - If has_singles AND has_multiples → DS 6
```

(r2-calculation-methods-by-ds)=
## R2 Calculation Methods by DS

Equation numbers refer to Bishop's VAS documentation manual dated 10-1-2026.

| DS | R2 Method | Description |
|-----|-----------|-------------|
| 0 | N/A | No VAS decomposition |
| 1 | Exact (within-cell) | `R2 = Y - Ȳ_kt` (Eq 14-3) |
| 2 | Scaled difference (`ma2`) | Condition-and-period-adjusted series, differenced and divided by 2·c(K, M), M = KT (Eqs 14-4..14-8) |
| 3 | Scaled difference (`ma2`) | Any one-observation cell: the same over every observation (Eqs 14-9..14-13) |
| 4 | Exact (present cells) | Collapses to ADS 1 after cleaning; within-cell for available data |
| 5 | Scaled difference (`ma2`) | Collapses to ADS 2 after cleaning; all present cells are unreplicated (Eqs 14-9..14-13) |
| 6 | Scaled difference (`ma2`) | Collapses to ADS 3 after cleaning (Eqs 14-9..14-13) |

The scaled difference first removes the process mean and the condition and period effects,
Z = Y − Ȳ_k − Ȳ_t + Ȳ (unweighted means of cell means), then differences Z along the
condition-then-time sequence, across condition boundaries with no grouping, so only the
first observation has no value. It divides by twice the **R2 scale factor**, computed from
the layout:

```
c(K, M) = sqrt( (K - 1) / (2K) · (1 - K / (M - 1)) )
```

with K the number of process design conditions present and M the number of observations.
This puts R2 on the noise (sigma) scale. The token `ma2` names Bishop's size-2 moving average
of Z, of which R2 is the scaled deviation. When K < 2 or M < K + 2 the R2 scale factor is
undefined and R2, with everything built on it, is reported unavailable.

## References

- Wheeler, D. J. (1995). *Advanced Topics in Statistical Process Control*. SPC Press, Knoxville, TN.
- Wheeler, D. J. & Chambers, D. S. (1992). *Understanding Statistical Process Control*. SPC Press.
- Bishop, T. A. (2023). Personal communication — Variance Analysis System implementation.
