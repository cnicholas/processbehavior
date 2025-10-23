# Sampling Design States (SDS) Reference Guide

## Overview

The Sampling Design State (SDS) classification system describes the structure of statistical process control data and determines what analyses are appropriate. This system is based on the Wheeler/Bishop Variance Analysis System (VAS) methodology.

The SDS determines:
- What type of data structure exists
- Which control chart types are valid
- What residual calculations can be performed
- What variance estimation methods to use
- Whether main effects and interaction analyses are supported

There are **7 Sampling Design States (0-6)**, each with specific capabilities and limitations.

---

## Quick Reference Table

| SDS | Name | Factors | Time | Replication | Valid Charts | VAS Residuals | R2 Method |
|-----|------|---------|------|-------------|--------------|---------------|-----------|
| 0 | Simple Series | No | No | None | Imr | No | N/A |
| 1 | Full Factorial with Complete Replication | Yes | Yes | Full | Xbar, S, Imr | Yes | Exact |
| 2 | Full Factorial with No Replication | Yes | Yes | None | Xbar, Imr | Yes | Moving Average |
| 3 | Partial Replication | Yes | Yes | Partial | Xbar, S, Imr | Yes | Hybrid |
| 4 | Factors Only | Yes | No | Full | Xbar, S, Imr | No | N/A |
| 5 | Time Only | No | Yes | Partial | Xbar, S, Imr | No | N/A |
| 6 | Incomplete/Irregular Grid | Yes | Yes | None | Imr | No | N/A |

---

## SDS 0: Simple Series

### Description
Individual measurements with no rational subgrouping or time structure.

### Data Structure
- **Factors:** No
- **Time:** No
- **Replication:** None

### Valid Chart Types
- **Imr** (Individuals Moving Range)

### Invalid Chart Types
- **Xbar** - Requires rational subgroups
- **S** - Requires rational subgroups
- **R** - Requires rational subgroups

### Recommended Chart
**Imr**

### VAS Residuals
**Not Supported**

No VAS residuals (R1-R5) are calculated because there is no factorial structure to decompose.

### Effects Analysis
- **Main Effects:** Not supported
- **Interaction Effects:** Not supported
- **Stratification:** Not supported

### Typical Use Cases
- Simple process monitoring (temperature, pH, daily output)
- Individual measurements over time
- Quality characteristic tracking with no grouping

### Limitations
- Cannot decompose variance (no factors or time structure)
- Cannot detect interaction effects
- Limited to individuals control chart (IMR)
- No rational subgrouping available

### Reference
Wheeler 'Understanding Variation' Chapter 3: Individuals Charts

---

## SDS 1: Full Factorial with Complete Replication

### Description
All factor × time cells have n ≥ 2 observations. This is the **best case for analysis** with full variance decomposition capabilities.

### Data Structure
- **Factors:** Yes
- **Time:** Yes
- **Replication:** Full (all cells have n ≥ 2)

### Valid Chart Types
- **Xbar** (Mean chart)
- **S** (Standard deviation chart)
- **Imr** (Individuals moving range)

### Invalid Chart Types
None - all chart types are supported.

### Recommended Chart
**Xbar** (with S chart for dispersion)

### VAS Residuals
**Fully Supported** - All residuals (R1-R5) are calculated.

#### Residuals Available
All five VAS residuals are calculated:

**R1: Total Deviation from Grand Mean**
```
R1 = Y - Ȳ
```
- Represents total variation of each observation around the overall average
- Foundation for all other residuals
- Sum of all R1 values = 0

**R2: Within-Cell Variation (Exact Method)**
```
R2 = Y - Ȳ_kt
```
- Deviation from cell mean (unexplained variation)
- **SDS 1 uses exact calculation** because all cells have n ≥ 2
- This is the "pure error" component
- Directly estimated from replicate observations within each cell

**R3: Interaction Effects**
```
R3 = Y - Ȳ_k - Ȳ_t + Ȳ
```
- Captures factor × time interaction
- Variation not explained by main effects alone
- Used to detect non-additive patterns

**R4: Time Effects + Unexplained**
```
R4 = Ȳ_t - Ȳ + R2
```
- Time main effect plus within-cell variation
- Used to assess temporal patterns
- Charted to evaluate time significance

**R5: Factor Effects + Unexplained**
```
R5 = Ȳ_k - Ȳ + R2
```
- Factor main effect plus within-cell variation
- Used to assess factor significance
- Charted to evaluate factor differences

### Effects Analysis
- **Main Effects:** Fully supported
- **Interaction Effects:** Fully supported
- **Stratification:** Supported

### Typical Use Cases
- Designed experiments with replication
- Process capability studies
- Multi-factor ANOVA-style analyses
- Complete factorial designs

### Limitations
None - this is the ideal data structure.

### Reference
Wheeler/Bishop Methodology: Complete Data (SDS 1)

---

## SDS 2: Full Factorial with No Replication

### Description
All factor × time cells have exactly n = 1 observation. Common in production environments where each factor/time combination is measured once.

### Data Structure
- **Factors:** Yes
- **Time:** Yes
- **Replication:** None (all cells have n = 1)

### Valid Chart Types
- **Xbar** (Mean chart)
- **Imr** (Individuals moving range)

### Invalid Chart Types
- **S** - Requires n ≥ 2 per subgroup
- **R** - Requires n ≥ 2 per subgroup

### Recommended Chart
**Xbar**

### VAS Residuals
**Supported** - All residuals (R1-R5) are calculated with R2 approximation.

#### Residuals Available
All five VAS residuals are calculated:

**R1: Total Deviation from Grand Mean**
```
R1 = Y - Ȳ
```
Same as SDS 1.

**R2: Within-Cell Variation (Moving Average Approximation)**
```
Y_ma_j = (Y_j + Y_{j-1}) / 2
R2_j = Y_j - Y_ma_j = (Y_j - Y_{j-1}) / 2 = MR_j / 2
```
- **SDS 2 uses moving average approximation** (Bishop Equations 65-66)
- Since n=1, cannot calculate R2 = Y - Ȳ_kt (would give R2=0)
- Instead uses backward-looking 2-point moving average to approximate local mean
- R2 = half the moving range
- First observation in each group is NaN (no lag available)
- This approximates the unexplained variation

**R3: Interaction Effects**
```
R3 = Y - Ȳ_k - Ȳ_t + Ȳ
```
Same as SDS 1, but interaction is confounded with pure error.

**R4: Time Effects + Unexplained**
```
R4 = Ȳ_t - Ȳ + R2
```
Same as SDS 1.

**R5: Factor Effects + Unexplained**
```
R5 = Ȳ_k - Ȳ + R2
```
Same as SDS 1.

### Effects Analysis
- **Main Effects:** Fully supported
- **Interaction Effects:** Supported (limited - confounded with error)
- **Stratification:** Supported

### Typical Use Cases
- Production data (one measurement per factor/time combination)
- Unreplicated factorial experiments
- Historical data analysis
- Screening experiments

### Limitations
- R2 estimated via moving average (approximate, not exact)
- Cannot use S or R charts (require n≥2)
- Interaction confounded with pure error

### Reference
Wheeler/Bishop Methodology: No Replication (SDS 2), Bishop Equations 64-66

---

## SDS 3: Partial Replication

### Description
Mixed cells: some have n ≥ 2, others have n = 1. This is the **most common** in practice.

### Data Structure
- **Factors:** Yes
- **Time:** Yes
- **Replication:** Partial (mixture of n=1 and n≥2 cells)

### Valid Chart Types
- **Xbar** (Mean chart)
- **S** (Standard deviation chart)
- **Imr** (Individuals moving range)

### Invalid Chart Types
None - all chart types are supported.

### Recommended Chart
**Xbar** (with S chart for dispersion where n≥2)

### VAS Residuals
**Supported** - All residuals (R1-R5) are calculated with hybrid R2.

#### Residuals Available
All five VAS residuals are calculated:

**R1: Total Deviation from Grand Mean**
```
R1 = Y - Ȳ
```
Same as SDS 1.

**R2: Within-Cell Variation (Hybrid Method)**
```
R2 = Y - Ȳ_kt    for cells with n > 1
R2 = 0           for cells with n = 1
```
- **SDS 3 uses hybrid approach**
- For cells with n ≥ 2: use exact calculation (like SDS 1)
- For cells with n = 1: set R2 = 0 (no within-cell variance estimable)
- This preserves exact estimates where possible

**R3: Interaction Effects**
```
R3 = Y - Ȳ_k - Ȳ_t + Ȳ
```
Same as SDS 1.

**R4: Time Effects + Unexplained**
```
R4 = Ȳ_t - Ȳ + R2
```
Same as SDS 1.

**R5: Factor Effects + Unexplained**
```
R5 = Ȳ_k - Ȳ + R2
```
Same as SDS 1.

### Effects Analysis
- **Main Effects:** Fully supported
- **Interaction Effects:** Partially supported
- **Stratification:** Supported

### Typical Use Cases
- Unbalanced designs
- Real-world data with missing observations
- Opportunistic replication in some cells
- Pilot studies with targeted replication

### Limitations
- R2 uses hybrid calculation (exact where possible, approximate elsewhere)
- Variance estimates less precise than SDS 1
- May have unequal subgroup sizes

### Reference
Wheeler/Bishop Methodology: Partial Replication (SDS 3)

---

## SDS 4: Factors Only (No Time)

### Description
Grouping factors present but no time variable.

### Data Structure
- **Factors:** Yes
- **Time:** No
- **Replication:** Full

### Valid Chart Types
- **Xbar** (Mean chart)
- **S** (Standard deviation chart)
- **Imr** (Individuals moving range)

### Invalid Chart Types
None - all chart types are supported.

### Recommended Chart
**Xbar**

### VAS Residuals
**Not Supported**

No VAS residuals because the factorial structure requires both factors AND time dimensions.

### Effects Analysis
- **Main Effects:** Supported (factor effects only)
- **Interaction Effects:** Not supported (requires time dimension)
- **Stratification:** Supported

### Typical Use Cases
- Cross-sectional studies
- Between-group comparisons
- Baseline capability studies
- Multi-stream process monitoring

### Limitations
- No VAS residuals (requires time dimension)
- Cannot analyze time trends
- Cannot detect factor × time interactions
- Limited to factor main effects

### Reference
Wheeler/Bishop Methodology: Factors Only (SDS 4)

---

## SDS 5: Time Only (No Factors)

### Description
Time variable present but no grouping factors.

### Data Structure
- **Factors:** No
- **Time:** Yes
- **Replication:** Partial

### Valid Chart Types
- **Xbar** (Mean chart)
- **S** (Standard deviation chart)
- **Imr** (Individuals moving range)

### Invalid Chart Types
None - all chart types are supported.

### Recommended Chart
**Xbar**

### VAS Residuals
**Not Supported**

No VAS residuals because the factorial structure requires both factors AND time dimensions.

### Effects Analysis
- **Main Effects:** Not supported (no factors)
- **Interaction Effects:** Not supported
- **Stratification:** Not supported

### Typical Use Cases
- Single process over time with subgroups
- Repeated measurements at time points
- Time series with natural grouping (hourly batches)
- Rational subgrouping by time period only

### Limitations
- No VAS residuals (requires factors)
- Cannot analyze factor effects
- Cannot detect interactions
- Limited to time-based grouping

### Reference
Wheeler/Bishop Methodology: Time Only (SDS 5)

---

## SDS 6: Incomplete/Irregular Grid

### Description
Sparse factor × time grid with many missing cells (< 75% coverage).

### Data Structure
- **Factors:** Yes
- **Time:** Yes
- **Replication:** None (irregular)

### Valid Chart Types
- **Imr** (Individuals moving range)

### Invalid Chart Types
- **Xbar** - Requires complete grid
- **S** - Requires complete grid

### Recommended Chart
**Imr**

### VAS Residuals
**Not Supported**

No VAS residuals because the incomplete grid prevents proper factorial decomposition.

### Effects Analysis
- **Main Effects:** Not supported (unreliable due to incomplete data)
- **Interaction Effects:** Not supported
- **Stratification:** Supported

### Typical Use Cases
- Opportunistic data collection
- Real-world incomplete data
- Sparse monitoring programs
- Ad-hoc measurements with irregular sampling

### Limitations
- No VAS residuals (incomplete grid)
- Cannot calculate reliable main effects
- Cannot analyze interactions
- Limited to stratified IMR charts per factor level
- Most limited analytical capabilities

### Reference
Wheeler/Bishop Methodology: Irregular Data (SDS 6)

---

## Residual Calculation Methods Summary

### R1: Total Deviation (All SDS 1-3)
```
R1 = Y - Ȳ
```
- Always calculated the same way
- Foundation residual

### R2: Within-Cell Variation (SDS-Dependent)

**SDS 1 (Full Replication) - Exact Method:**
```
R2 = Y - Ȳ_kt
```
- Direct calculation from cell means
- Most accurate
- All cells have n ≥ 2

**SDS 2 (No Replication) - Moving Average Method:**
```
Y_ma_j = (Y_j + Y_{j-1}) / 2
R2_j = Y_j - Y_ma_j = (Y_j - Y_{j-1}) / 2
```
- Backward-looking 2-point moving average
- R2 = half the moving range (MR/2)
- Approximation of unexplained variation
- All cells have n = 1

**SDS 3 (Partial Replication) - Hybrid Method:**
```
R2 = Y - Ȳ_kt    if n > 1 (exact)
R2 = 0           if n = 1 (no information)
```
- Uses exact where possible
- Sets to zero where no replication
- Mixed cell sizes

### R3: Interaction Effects (All SDS 1-3)
```
R3 = Y - Ȳ_k - Ȳ_t + Ȳ
```
- Always calculated the same way
- Measures non-additive effects

### R4: Time Effects + Unexplained (All SDS 1-3)
```
R4 = Ȳ_t - Ȳ + R2
```
- Uses R2 from appropriate SDS method
- Assesses temporal significance

### R5: Factor Effects + Unexplained (All SDS 1-3)
```
R5 = Ȳ_k - Ȳ + R2
```
- Uses R2 from appropriate SDS method
- Assesses factor significance

---

## Chart Type Compatibility Matrix

### Xbar Chart (Mean Chart)
- **SDS 0:** ✗ (No rational subgroups)
- **SDS 1:** ✓ (Recommended)
- **SDS 2:** ✓ (Recommended)
- **SDS 3:** ✓ (Recommended)
- **SDS 4:** ✓ (Recommended)
- **SDS 5:** ✓ (Recommended)
- **SDS 6:** ✗ (Incomplete grid)

### S Chart (Standard Deviation Chart)
- **SDS 0:** ✗ (No rational subgroups)
- **SDS 1:** ✓
- **SDS 2:** ✗ (Requires n≥2)
- **SDS 3:** ✓ (Where n≥2)
- **SDS 4:** ✓
- **SDS 5:** ✓
- **SDS 6:** ✗ (Incomplete grid)

### Imr Chart (Individuals Moving Range)
- **SDS 0:** ✓ (Recommended)
- **SDS 1:** ✓
- **SDS 2:** ✓
- **SDS 3:** ✓
- **SDS 4:** ✓
- **SDS 5:** ✓
- **SDS 6:** ✓ (Recommended)

### R Chart (Range Chart)
- **SDS 0:** ✗ (No rational subgroups)
- **SDS 1:** ✓
- **SDS 2:** ✗ (Requires n≥2)
- **SDS 3:** ✓ (Where n≥2)
- **SDS 4:** ✓
- **SDS 5:** ✓
- **SDS 6:** ✗

---

## Key Concepts

### What is a "Cell"?
A cell is a unique combination of (factor × time). For example:
- Factor = Lane A, Time = Pull 1 → Cell (A, 1)
- Factor = Lane B, Time = Pull 2 → Cell (B, 2)

The number of observations per cell (n) determines the replication level.

### What is the VAS (Variance Analysis System)?
The VAS is Wheeler/Bishop's framework for decomposing total variation into interpretable components:
- Within-cell variation (R2)
- Factor effects (R5)
- Time effects (R4)
- Interaction effects (R3)
- Total variation (R1)

This allows you to understand WHERE variation is coming from.

### Why Does R2 Calculation Differ by SDS?

**The Challenge:**
R2 represents "unexplained" or "within-cell" variation. To estimate this, we need multiple observations per cell.

**The Solutions:**
- **SDS 1:** We have n≥2 in all cells → Use exact calculation
- **SDS 2:** We have n=1 in all cells → Use moving average to approximate
- **SDS 3:** We have mixed → Use exact where possible, zero elsewhere

### When Are VAS Residuals Calculated?

VAS residuals are calculated when:
1. Analysis type is **Xbar** or **S** (cell-level analysis)
2. AND we have proper factorial structure (SDS 1, 2, or 3)

VAS residuals are NOT calculated when:
- Analysis type is **Imr** or **R** (individual-level analysis uses moving ranges)
- No proper structure (SDS 0, 4, 5, 6)

### Grouping Variable: Two Different Roles

**For Xbar/S Charts:**
Grouping defines CELLS for variance decomposition (VAS residuals)

**For Imr/R Charts:**
Grouping defines STRATA for separate charts (stratification)

---

## Code References

### SDS Detection
`processbehavior/sds_detector.py:247` - `detect_sds()` method

### Residual Calculations
`processbehavior/residual_calculator.py:524` - `calculate_residuals()` method

### Analysis Plans
`processbehavior/sds_detector.py:714` - `get_analysis_plan()` method

### Should Calculate VAS
`processbehavior/sds_detector.py:512` - `should_calculate_vas_residuals()` method

---

## Decision Trees

### "What SDS Do I Have?"

```
Do I have BOTH factors AND time?
├─ NO → Do I have factors?
│  ├─ YES → SDS 4 (Factors Only)
│  └─ NO → Do I have time?
│     ├─ YES → SDS 5 (Time Only)
│     └─ NO → SDS 0 (Simple Series)
│
└─ YES → Continue...
   │
   Count observations per (factor × time) cell
   │
   ├─ Is grid coverage < 75%?
   │  └─ YES → SDS 6 (Incomplete Grid)
   │
   ├─ ALL cells have n ≥ 2?
   │  └─ YES → SDS 1 (Full Replication)
   │
   ├─ ALL cells have n = 1?
   │  └─ YES → SDS 2 (No Replication)
   │
   └─ Mixed (some n=1, some n≥2)?
      └─ YES → SDS 3 (Partial Replication)
```

### "What Chart Should I Use?"

```
What is my SDS?
├─ SDS 0 → Use Imr
├─ SDS 1 → Use Xbar+S (best choice)
├─ SDS 2 → Use Xbar (S not available)
├─ SDS 3 → Use Xbar+S (best choice)
├─ SDS 4 → Use Xbar+S
├─ SDS 5 → Use Xbar+S
└─ SDS 6 → Use Imr (stratified)
```

### "Will I Get VAS Residuals?"

```
Is my analysis_type Xbar or S?
├─ NO (Imr/R) → No VAS residuals
│              → Grouping creates stratified charts
│
└─ YES → Is my SDS 1, 2, or 3?
   ├─ YES → VAS residuals calculated
   │        R2 method depends on SDS:
   │        • SDS 1: Exact
   │        • SDS 2: Moving Average
   │        • SDS 3: Hybrid
   │
   └─ NO → No VAS residuals
```

---

## Validation and Diagnostics

### Get Analysis Plan for Your SDS
```python
from processbehavior.sds_detector import SamplingDesignDetector

# Get plan for SDS 1
plan = SamplingDesignDetector.get_analysis_plan(sds=1)
print(plan)  # Prints detailed capabilities

# Get capability matrix for all SDS
matrix = SamplingDesignDetector.get_capability_matrix()
print(matrix)
```

### Print All Plans
```python
from processbehavior.sds_detector import SamplingDesignDetector

# Print complete reference for all SDS 0-6
SamplingDesignDetector.print_all_analysis_plans()
```

---

## References

1. Wheeler, Donald J. "Understanding Variation: The Key to Managing Chaos"
2. Bishop, Tom. "Understanding Statistical Process Control" (unpublished methodology)
3. Wheeler/Bishop VAS (Variance Analysis System) Framework
4. processbehavior package implementation

---

## Appendix: Mean Definitions

These means are calculated for VAS residuals:

**Ȳ (Ybar)** - Grand Mean
- Average of ALL observations
- Baseline for R1

**Ȳ_k (Ybar_k)** - Factor Mean
- Average for each factor level (averaged across time)
- Used in R3, R5

**Ȳ_t (Ybar_t)** - Time Mean
- Average for each time point (averaged across factors)
- Used in R3, R4

**Ȳ_kt (Ybar_kt)** - Cell Mean
- Average for each (factor × time) cell
- Used in R2 (SDS 1, 3)
