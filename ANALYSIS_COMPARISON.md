# SDS 1 Data Analysis Comparison

## Summary of Three Analysis Approaches

We ran the same SDS 1 dataset (800 observations, 50 time points, 4×2 factorial design) with three different configurations to demonstrate the trade-offs.

---

## 1. COMBINED XBAR ANALYSIS ✅
**File:** `sds1_800_analysis_results.xlsx`

### Configuration:
```python
grouping_vars = ['FACTOR 1', 'FACTOR 2']
stratify = False
chart_type = 'Xbar' (auto-selected for SDS 1)
```

### Result:
- **8 rational subgroups** (4 FACTOR 1 levels × 2 FACTOR 2 levels)
- **ONE set of shared control limits** (LCL=237.396, UCL=238.168)
- Shows differences **BETWEEN** factor combinations
- **5 out of 8 RSG beyond limits** → factor effects present

### Use Case:
✓ Comparing factor combinations against overall process
✓ Identifying which combinations run systematically high/low
✓ Understanding main effects (FACTOR 1, FACTOR 2)
✓ Detecting factor×time interactions

### Key Insight:
This answers: "Which factor settings produce different results?"

---

## 2. STRATIFIED XBAR ANALYSIS ⚠️
**File:** `sds1_800_stratified_results.xlsx`

### Configuration:
```python
grouping_vars = ['FACTOR 1', 'FACTOR 2']
stratify = True
chart_type = 'Xbar' (auto-selected)
```

### Result:
- **8 strata**, 16 chart rows (8 Xbar + 8 Sbar)
- **UNIQUE control limits per stratum**
- Shows summary statistics per factor combination
- **Only 1 data point per stratum** (overall mean across all time)
- **NO time-series within strata**

### Limitation:
⚠️ Warning: "Stratifying Xbar charts is uncommon"
⚠️ Each stratum IS the rational subgroup - no variation to chart
⚠️ Doesn't show time-series behavior within factor combinations

### Why This Happens:
When `grouping_vars = ['FACTOR 1', 'FACTOR 2']`, each rational subgroup IS a factor combination. Stratifying by these same variables creates strata with only one mean value (no time dimension).

---

## 3. STRATIFIED IMR ANALYSIS ✅ BEST FOR DRILL-DOWN
**File:** `sds1_800_stratified_imr_results.xlsx`

### Configuration:
```python
grouping_vars = ['FACTOR 1', 'FACTOR 2']
chart_type = 'Imr' (forced instead of Xbar)
stratify = True
```

### Result:
- **8 separate IMR charts** (one per FACTOR 1 × FACTOR 2 combination)
- **100 observations per stratum** (50 time points × 2 replicates)
- **UNIQUE control limits per stratum**
- Shows **time-series WITHIN each factor combination**
- **62 total signals** across all strata and time

### Stratum Details:
| Stratum          | Observations | Signals |
|------------------|--------------|---------|
| FACTOR 1=1, 2=1  | 100          | 6       |
| FACTOR 1=1, 2=2  | 100          | 3       |
| FACTOR 1=2, 2=1  | 100          | 10      |
| FACTOR 1=2, 2=2  | 100          | 12      |
| FACTOR 1=3, 2=1  | 100          | 6       |
| FACTOR 1=3, 2=2  | 100          | 4       |
| FACTOR 1=4, 2=1  | 100          | 7       |
| FACTOR 1=4, 2=2  | 100          | 14      |

### Use Case:
✓ How does EACH factor combination behave over time?
✓ Which combinations are stable vs drifting?
✓ Drill-down analysis per factor setting
✓ Identifying specific time periods with problems for specific combinations
✓ Comparing stability across factor settings

### Key Insight:
This answers: "Is each factor combination stable over time?"

**Most informative for factorial experiments!**

---

## Recommendations

### Use Combined Xbar When:
- You want to compare factor levels to overall process performance
- Identify which factor combinations differ from the mean
- Understand main effects and interactions
- **Standard analysis for factorial designs**

### Use Stratified IMR When:
- You want to drill down into each factor combination
- Assess stability **within** each combination over time
- Identify which combinations have special causes
- **Best for detailed investigation and troubleshooting**

### Avoid Stratified Xbar When:
- Rational subgroups are already factor combinations
- Results in summary stats only, not time-series

---

## Key Takeaway

**Wheeler's Methodology:**
> "Stratification is most useful with IMR charts for drill-down analysis"

For factorial experiments (SDS 1):
1. **Start with Combined Xbar**: Understand overall factor effects
2. **Follow up with Stratified IMR**: Investigate stability within each factor setting
3. **Use residual analysis**: Understand variance components (VAS decomposition)
