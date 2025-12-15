# Sampling Design State (SDS) Definitions

This document provides the formal definitions of Sampling Design States (SDS) as defined by Wheeler and Bishop in the Variance Analysis System (VAS) framework.

## Notation

- **k**: Factor level index (e.g., machine, lane, operator)
- **t**: Time period index (e.g., day, shift, batch)
- **N_kt**: Number of observations in cell (k, t) - the count of measurements for factor level k at time t
- **Rational Subgroup**: A (k, t) cell - observations sharing the same factor level and time period

## SDS Classification Table

| Design State | Sample Size N_kt | Sampling Design State | Distribution of Sample Sizes Across Rational Subgroups |
|:------------:|------------------|:---------------------:|--------------------------------------------------------|
| **1** | Min N_kt ≥ 2 | Complete | Multiple observations in each of the rational subgroups |
| **2** | Min N_kt = 1 and Max N_kt = 1 | Semi-Complete | A single observation in each of the rational subgroups |
| **3** | Min N_kt = 1 and Max N_kt ≥ 2 | Semi-Complete | Multiple observations in some rational subgroups and a single observation in some rational subgroups |
| **4** | Min N_kt = 0, N_kt = 1 and Max N_kt ≥ 2 | Incomplete | Multiple observations in some rational subgroups, a single observation in some rational subgroups and no data in some rational subgroups |
| **5** | Min N_kt = 0, N_kt ≠ 1 and Max N_kt ≥ 2 | Incomplete | Multiple observations in some rational subgroups and no data in some subgroups |
| **6** | Min N_kt = 0 and Max N_kt = 1 | Incomplete | A single observation in some rational subgroups and no data in some rational subgroups |

## Detailed Definitions

### SDS 0: No Structure

**Not in original Wheeler/Bishop table** - added for processbehavior library.

- **Condition**: No grouping factors or time variable specified
- **Grid Status**: N/A
- **Description**: Individual measurements with no rational subgrouping structure
- **Analysis**: Limited to IMR (Individual-Moving Range) charts

### SDS 1: Complete with Full Replication

- **Condition**: Min N_kt ≥ 2
- **Grid Status**: Complete
- **Description**: Every (factor × time) cell has at least 2 observations
- **Implications**:
  - True within-cell variance can be estimated directly
  - Full VAS residual decomposition (R1-R5) supported
  - Xbar-S charts are optimal
  - Interaction effects can be estimated precisely

### SDS 2: Semi-Complete with No Replication

- **Condition**: Min N_kt = 1 AND Max N_kt = 1
- **Grid Status**: Semi-Complete
- **Description**: Every (factor × time) cell has exactly 1 observation
- **Implications**:
  - No within-cell variance (must use moving average for R2)
  - Interaction effects confounded with pure error
  - VAS residuals available but R2 is approximate
  - Common in unreplicated factorial designs

### SDS 3: Semi-Complete with Partial Replication

- **Condition**: Min N_kt = 1 AND Max N_kt ≥ 2
- **Grid Status**: Semi-Complete
- **Description**: Mix of cells - some have n=1, others have n≥2
- **Implications**:
  - Hybrid R2 calculation required (exact where n≥2, moving average where n=1)
  - **Most common in real-world data**
  - Partial interaction effect estimation
  - Requires careful handling of mixed replication

### SDS 4: Incomplete with Mixed Replication

- **Condition**: Min N_kt = 0 AND some N_kt = 1 AND Max N_kt ≥ 2
- **Grid Status**: Incomplete
- **Description**: Missing cells, plus mix of n=1 and n≥2 in present cells
- **Implications**:
  - Incomplete (factor × time) grid
  - Some cells have no data at all
  - Present cells have mixed replication
  - Complex variance estimation required

### SDS 5: Incomplete with Replication Only

- **Condition**: Min N_kt = 0 AND N_kt ≠ 1 (no cells with exactly 1) AND Max N_kt ≥ 2
- **Grid Status**: Incomplete
- **Description**: Missing cells, but all present cells have n≥2
- **Implications**:
  - Incomplete grid structure
  - Where data exists, full replication is available
  - Can estimate within-cell variance for present cells
  - Common in nested/hierarchical designs with asynchronous coverage

### SDS 6: Incomplete with No Replication

- **Condition**: Min N_kt = 0 AND Max N_kt = 1
- **Grid Status**: Incomplete
- **Description**: Missing cells, and all present cells have exactly n=1
- **Implications**:
  - Most limited analytical case
  - Sparse, irregular data structure
  - No within-cell variance estimation possible
  - Moving average methods required throughout

## Detection Algorithm

To determine the SDS for a dataset:

```
1. Compute N_kt for all (factor, time) combinations
2. Determine:
   - min_n = minimum N_kt among PRESENT cells (excluding missing)
   - max_n = maximum N_kt
   - has_missing = whether any expected (k,t) cells have N_kt = 0
   - has_singles = whether any cells have N_kt = 1
   - has_multiples = whether any cells have N_kt ≥ 2

3. Classification:
   - If no factors/time defined → SDS 0
   - If NOT has_missing (Complete/Semi-Complete grid):
     - If min_n ≥ 2 → SDS 1
     - If min_n = 1 AND max_n = 1 → SDS 2
     - If min_n = 1 AND max_n ≥ 2 → SDS 3
   - If has_missing (Incomplete grid):
     - If has_singles AND has_multiples → SDS 4
     - If NOT has_singles AND has_multiples → SDS 5
     - If has_singles AND NOT has_multiples → SDS 6
```

## R2 Calculation Methods by SDS

| SDS | R2 Method | Description |
|-----|-----------|-------------|
| 0 | N/A | No VAS decomposition |
| 1 | Exact (within-cell) | Pooled within-cell variance |
| 2 | Moving Average | Approximate via sequential differences |
| 3 | Hybrid | Exact where n≥2, moving average where n=1 |
| 4 | Hybrid | Complex handling of missing + mixed |
| 5 | Exact (present cells) | Within-cell for available data |
| 6 | Moving Average | All present cells are unreplicated |

## References

- Wheeler, D. J. (1995). *Advanced Topics in Statistical Process Control*. SPC Press, Knoxville, TN.
- Wheeler, D. J. & Chambers, D. S. (1992). *Understanding Statistical Process Control*. SPC Press.
- Bishop, D. R. (2023). Personal communication - Variance Analysis System implementation.
