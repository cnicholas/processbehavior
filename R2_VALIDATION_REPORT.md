# R2 Residual Calculation Validation Report

## Comparison Against Tom Bishop's "Understanding Statistical Process Control" (Section 20.2)

### Summary

This report validates the R2 residual calculation implementation against the specifications in Tom Bishop's book for SDS 1, 2, 3, and 5.

---

## Tom's Specification (Section 20.2.1)

### For SDS 1, 3, 4, and 5: Exact Method (Equation 59)

**Formula:** R2 = Y - Ŷ = Y - Ȳ_k

**Description:**
- "The rational subgroup data are simply adjusted by the rational subgroup sample mean"
- R2 captures unexplained variation within rational subgroups
- R2 ≈ λ + η + ε (assignable + common causes within subgroups)
- For each rational subgroup, mean(R2) = 0
- Standard deviation: S_R2 = S (same as original data)

### For SDS 2 and 6: Moving Average Method (Equations 64-66)

**When:** All rational subgroups have N=1 (no replication)

**Why:** When N=1, R2 = Y - Ȳ_k produces R2=0 (provides no information about within-subgroup variation). The moving average method extracts the "unexplained" variation by fitting a smooth curve through the data.

**Procedure:**
1. Sort data by process design conditions (PDC), then by production time (PT) within each PDC
2. Re-index as Y_j where j = 1, 2, ..., J
3. Calculate lagged values: lag(Y_j) = Y_{j-1} for j = 2, 3, ..., J (Equation 64)
4. Calculate moving average: **Y_ma = (Y_j + Y_{j-1}) / 2** (Equation 65)
5. Calculate R2: **R2_j = Y_j - Y_ma** for j = 2, 3, ..., J (Equation 66)

**Mathematical Result:**
- Y_ma = (Y_j + Y_{j-1}) / 2
- R2_j = Y_j - Y_ma = Y_j - (Y_j + Y_{j-1})/2 = **(Y_j - Y_{j-1}) / 2**
- This equals **half of the first difference** (moving range / 2)

**Key Points:**
- This uses a **backward-looking** moving average (current + previous observation)
- The smooth curve removes PDC and PT effects, leaving unexplained variation
- R2 ≈ λ + η + ε (assignable + common causes)
- For j=1 (first observation in each PDC group), R2 is not defined

---

## Our Implementation

### SDS 1: Full Replication ✅ CORRECT

**Method:** `calculate_r2_residual_sds1()`

**Formula:** `R2 = Y - Ȳ_kt`

**Code:**
```python
def calculate_r2_residual_sds1(df, response_var, cell_means):
    return df[response_var] - cell_means
```

**Validation:** ✅ Matches Tom's Equation 59 exactly

---

### SDS 2: No Replication ❌ INCORRECT IMPLEMENTATION

**Method:** `calculate_r2_residual_sds2()` in `residual_calculator.py:278-325`

**Tom's Formula (Equation 65):** `Y_ma = (Y_j + Y_{j-1}) / 2`  (backward MA)

**Our Formula:** `ma2 = (Y_{j-1} + Y_{j+1}) / 2`  (centered MA)

**Our Code:**
```python
def calculate_r2_residual_sds2(df, response_var, rsg_var):
    # ❌ WRONG: Using centered moving average
    ma2 = df.groupby(rsg_var)[response_var].transform(
        lambda s: (s.shift(1) + s.shift(-1)) / 2.0  # ❌ CENTERED
    )

    # For edge points, use forward or backward value only
    fwd = df.groupby(rsg_var)[response_var].shift(-1)
    back = df.groupby(rsg_var)[response_var].shift(1)
    ma2 = ma2.where(ma2.notna(), fwd.where(fwd.notna(), back))

    return df[response_var] - ma2
```

**What Tom's Code Should Be:**
```python
def calculate_r2_residual_sds2(df, response_var, rsg_var):
    # ✅ CORRECT: Backward-looking moving average
    # Y_ma = (Y_j + Y_{j-1}) / 2
    ma2 = df.groupby(rsg_var)[response_var].transform(
        lambda s: (s + s.shift(1)) / 2.0  # ✅ BACKWARD
    )

    # First observation in each group has no R2 (lag doesn't exist)
    # Set to NaN or 0 as appropriate

    return df[response_var] - ma2
```

**Mathematical Comparison:**

| Observation | Tom's R2 (Correct) | Our R2 (Wrong) |
|-------------|-------------------|----------------|
| j=1 (first in group) | NaN (no lag) | Y_1 - Y_2 |
| j=2 | (Y_2 - Y_1)/2 | Y_2 - (Y_1 + Y_3)/2 |
| j=3 | (Y_3 - Y_2)/2 | Y_3 - (Y_2 + Y_4)/2 |
| j=4 | (Y_4 - Y_3)/2 | Y_4 - (Y_3 + Y_5)/2 |

**Why This Matters:**

Tom's method: **R2_j = (Y_j - Y_{j-1}) / 2**
- This is **half of the moving range** (MR/2)
- Aligns with IMR chart methodology where MR = |Y_j - Y_{j-1}|
- Natural connection to process variation estimation
- Purely backward-looking (doesn't use future data)

Our method: **R2_j = Y_j - (Y_{j-1} + Y_{j+1}) / 2**
- This is deviation from **local linear interpolation**
- Uses both past and future data (centered)
- Different statistical properties
- Not what Tom Bishop specified

**Impact:** This is a **methodological error** that affects all R2 residuals for SDS 2 and 6.

---

### SDS 3: Partial Replication ✅ CORRECT (with clarification)

**Method:** `calculate_r2_residual_sds3()`

**Formula:**
```
R2 = Y - Ȳ_kt  when n_kt > 1
R2 = 0         when n_kt = 1
```

**Code:**
```python
def calculate_r2_residual_sds3(df, response_var, cell_means, rsg_var, time_var):
    n_per_cell = df.groupby([rsg_var, time_var])[response_var].transform('count')
    r2_within = df[response_var] - cell_means

    return pd.Series(
        np.where(n_per_cell > 1, r2_within, 0.0),
        index=df.index
    )
```

**Validation:** ✅ Matches Tom's Equation 59
- When n > 1: R2 = Y - Ȳ_kt (exact method)
- When n = 1: Y = Ȳ_kt, so R2 = 0 (our code optimizes this)

**Note:** Tom says to use Equation 59 for SDS 3. For cells with n=1, the equation naturally produces R2=0 because there's only one observation (Y = Ȳ_kt). Our implementation explicitly sets this to 0 for clarity.

---

### SDS 5: Implementation Questions ⚠️

**Current Specification:** `residual_calculation_method='none'`

**Current Implementation:** Uses SDS 3 approach (hybrid)

**Code:**
```python
elif sds == 5:
    logger.warning(
        "SDS 5: Using SDS 3 hybrid approach for R2.\n"
        "Nested designs may require custom variance components."
    )
    return calculate_r2_residual_sds3(df, y, df['Ybar_kt'], ...)
```

**Tom's Book:** Says SDS 4 and 5 should use Equation 59 (exact method)

**Questions:**
1. Does SDS 5 in our numbering match Tom's SDS 5?
2. Our SDS 5 is "Time Only (no factors)" with no VAS support in spec
3. Is SDS 5 intended to have rational subgroups?
4. Should VAS residuals be calculated for SDS 5?

---

### SDS 4: Not Implemented ⚠️

**Current Specification:** `residual_calculation_method='none'`

**Tom's Book:** Says SDS 4 should use Equation 59 (exact method)

**Our SDS 4 Definition:** "Factors Only (no time, no VAS)"

**Question:** Does SDS 4 in our numbering match Tom's SDS 4?

---

### SDS 6: Not Implemented ⚠️

**Current Specification:** `residual_calculation_method='none'`

**Tom's Book:** Says SDS 6 should use moving average method (like SDS 2)

**Our SDS 6 Definition:** "Incomplete/Irregular Grid (limited to stratified IMR)"

**Question:** Should SDS 6 support R2 residuals with moving average method?

---

## Validation Summary

| SDS | Tom's Method | Our Spec | Our Implementation | Status |
|-----|--------------|----------|-------------------|--------|
| 0 | N/A | `none` | Not implemented | ✅ Correct |
| 1 | Exact (Eq. 59: Y - Ȳ_k) | `exact` | Y - Ȳ_kt | ✅ Correct |
| 2 | Backward MA (Eq. 65-66) | `moving_average` | ❌ Centered MA | ❌ **WRONG** |
| 3 | Exact (Eq. 59: Y - Ȳ_k) | `hybrid` | Y - Ȳ_kt when n>1 | ✅ Correct |
| 4 | Exact (Eq. 59: Y - Ȳ_k) | `none` | Not implemented | ❌ Missing |
| 5 | Exact (Eq. 59: Y - Ȳ_k) | `none` | Uses SDS 3 (warns) | ⚠️ Incomplete |
| 6 | Backward MA (Eq. 65-66) | `none` | Not implemented | ❌ Missing |

---

## Key Findings

### ✅ Correct Implementations

1. **SDS 1 (Full Replication):** ✅ Perfect match with Tom's Equation 59
   - R2 = Y - Ȳ_kt
   - Code: `calculate_r2_residual_sds1()`

2. **SDS 3 (Partial Replication):** ✅ Correctly applies Equation 59
   - R2 = Y - Ȳ_kt when n>1, R2 = 0 when n=1
   - Code: `calculate_r2_residual_sds3()`

### ❌ CRITICAL ERROR: SDS 2 (and 6)

**Location:** `processbehavior/residual_calculator.py:278-325`

**The Problem:**
- **Tom's specification (Eq. 65):** Backward MA = (Y_j + Y_{j-1}) / 2
- **Our implementation:** Centered MA = (Y_{j-1} + Y_{j+1}) / 2

**The Fix Required:**
```python
# CHANGE FROM:
ma2 = df.groupby(rsg_var)[response_var].transform(
    lambda s: (s.shift(1) + s.shift(-1)) / 2.0  # ❌ CENTERED
)

# CHANGE TO:
ma2 = df.groupby(rsg_var)[response_var].transform(
    lambda s: (s + s.shift(1)) / 2.0  # ✅ BACKWARD
)
```

**Why This Matters:**
- Tom's method produces R2 = (Y_j - Y_{j-1}) / 2 = MR/2 (half of moving range)
- Aligns with IMR chart methodology
- Our method produces completely different residuals
- This affects all VAS residual analysis for SDS 2 and 6

**Impact:** All R2, R4, and R5 residuals are incorrect for SDS 2 (and 6 if implemented)

### ❌ Missing Implementations

**SDS 4 and 5:**
- Tom's book specifies these should use Equation 59 (exact method)
- Our specs say `residual_calculation_method='none'`
- Need to clarify if these should support VAS residuals

**SDS 6:**
- Tom's book specifies moving average method (like SDS 2)
- Our specs say `residual_calculation_method='none'`
- Currently not implemented

---

## Immediate Actions Required

### 1. **Fix SDS 2 R2 Calculation** (CRITICAL)

**File:** `processbehavior/residual_calculator.py:278-325`

**Change Required:**
```python
def calculate_r2_residual_sds2(df, response_var, rsg_var):
    """
    Calculate R2 for SDS 2: moving average approximation.

    Per Tom Bishop Equation 65: Y_ma = (Y_j + Y_{j-1}) / 2
    Therefore: R2_j = Y_j - Y_ma = (Y_j - Y_{j-1}) / 2 = MR/2
    """
    # Backward-looking moving average: current + previous
    ma2 = df.groupby(rsg_var)[response_var].transform(
        lambda s: (s + s.shift(1)) / 2.0
    )

    # R2 = Y - MA
    r2 = df[response_var] - ma2

    # First observation in each group has no lag (NaN)
    # Keep as NaN or set to 0 depending on downstream handling

    return r2
```

**Tests Affected:** Any tests using SDS 2 with VAS residuals need re-validation

### 2. **Clarify SDS 4, 5, 6 Specifications**

**Questions to resolve:**
1. Should SDS 4 support VAS residuals using Equation 59?
2. Should SDS 5 support VAS residuals using Equation 59?
3. Should SDS 6 support VAS residuals using moving average (like SDS 2)?

**Current Status:**
- SDS 4, 5, 6 specs all say `residual_calculation_method='none'`
- But Tom's book indicates they should support R2 calculations

### 3. **Update Tests**

After fixing SDS 2:
1. Verify all SDS 2 tests still pass with corrected calculation
2. Add tests that validate R2 = MR/2 relationship
3. Add tests comparing to Tom's Figure 30 if data available

### 4. **Update Documentation**

1. Update `SDSAnalysisPlan` docstrings to clarify backward MA for SDS 2
2. Add comments in code explaining connection to MR/2
3. Document the fix in CHANGELOG

---

## References

- Tom Bishop, "Understanding Statistical Process Control," Section 20.2: "VAS Residual Calculations"
- Equation 59: R2 = Y - Ȳ_k (exact method for SDS 1, 3, 4, 5)
- Equations 64-66: Moving average method for SDS 2, 6
