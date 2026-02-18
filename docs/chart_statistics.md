# Chart Statistics Review and Validation

## Summary

| Chart | Center Line | Formula | Status |
|-------|-------------|---------|--------|
| Xbar | Grand mean (Ȳ) | `df[response].mean()` before groupby | CORRECT |
| S | Mean of subgroup std devs (S̄) | `out["s"].mean()` | CORRECT |
| XmR | Mean of individuals (X̄) | `out[response].mean()` | CORRECT |
| R | Mean moving range (m̄R) | `out['mr'].mean()` | CORRECT |

---

## Xbar Chart (analysis.py:614-678)

### Center Line
```python
_Ybar = out[spec.response_var].mean()  # Grand mean BEFORE groupby
_Xbar = _Ybar
```
**Status:** CORRECT - Uses grand mean (weighted by observation count)

### Limits
```python
# Wd = S̄ / c4(n) - within-subgroup standard deviation estimate
Wd = sd / c4(N)

# LPL = X̄ - 3 * Wd / sqrt(n)
# UPL = X̄ + 3 * Wd / sqrt(n)
lpl = mean - (3 * Wd) / sqrt(N)
upl = mean + (3 * Wd) / sqrt(N)
```
**Status:** CORRECT - Standard Shewhart Xbar limits

---

## S Chart (analysis.py:680-721, spc_constants.py:289-299)

### Center Line
```python
_S = out["s"].mean()  # Mean of subgroup standard deviations
```
**Status:** CORRECT - S̄ is the average of subgroup standard deviations

### Limits
```python
# b3(n) = 1 - 3/c4(n) * sqrt(1 - c4(n)^2)
# b4(n) = 1 + 3/c4(n) * sqrt(1 - c4(n)^2)

lpl = sd * b3(N)  # Can be negative for small n
upl = sd * b4(N)
```
**Status:** CORRECT - Uses b3/b4 constants, allows negative LPL per Wheeler

---

## XmR Chart (analysis.py:1142-1183, spc_constants.py:301-311)

### Center Line
```python
mean_ = out[y].mean()  # Mean of all individual values
out['center'] = mean_
```
**Status:** CORRECT - X̄ for individuals chart

### Limits
```python
# E2 = 2.66 (for n=2 moving range)
XMR_LIMIT_MULTIPLIER = 2.66

lpl = mean - (E2 * mR)  # X̄ - 2.66 * m̄R
upl = mean + (E2 * mR)  # X̄ + 2.66 * m̄R
```
**Status:** CORRECT - Standard XmR limits using E2 constant

---

## R Chart (analysis.py:1185-1214, spc_constants.py:313-323)

### Center Line
```python
mR = out['mr'].mean()  # Mean moving range
r_out['center'] = mR
```
**Status:** CORRECT - m̄R is the average moving range

### Limits
```python
# D4 = 3.268 (for n=2 moving range)
R_UPPER_LIMIT_MULTIPLIER = 3.268

lpl = 0                    # Ranges cannot be negative
upl = mR * D4              # m̄R * 3.268
```
**Status:** CORRECT - D4 constant, LPL clamped to 0

---

## Key Constants (spc_constants.py)

| Constant | Value/Formula | Purpose |
|----------|--------------|---------|
| SIGMA_MULTIPLIER | 3 | 3-sigma limits |
| E2 | 2.66 | XmR limit multiplier (d2/d3 for n=2) |
| D4 | 3.268 | R chart upper limit (1 + 3*d3/d2 for n=2) |
| c4(n) | sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2) | Bias correction |
| b3(n) | 1 - 3/c4(n) * sqrt(1 - c4(n)^2) | S chart lower limit |
| b4(n) | 1 + 3/c4(n) * sqrt(1 - c4(n)^2) | S chart upper limit |

---

## Recent Fixes Applied

1. **Xbar center line** (commit 337432e): Changed from mean of subgroup means to grand mean
2. **S chart LPL** (commit 4a1c0c4): Removed clamping to 0, allows negative values per Wheeler
3. **R chart LPL**: Correctly clamped to 0 (ranges cannot be negative)

---

## Validation Summary

All chart statistics are now correctly calculated per Wheeler/Bishop methodology:

- **Xbar**: Grand mean (Ȳ) with limits based on S̄/c4(n)
- **S**: Mean of subgroup std devs (S̄) with b3/b4 limits (can be negative)
- **XmR**: Mean of individuals (X̄) with E2*m̄R limits
- **R**: Mean moving range (m̄R) with D4 upper limit, LPL=0
