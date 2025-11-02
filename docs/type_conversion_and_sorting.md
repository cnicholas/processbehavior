# Type Conversion and Natural Sorting in ProcessBehavior

**New in v0.1.0**: Automatic type conversion and natural sorting for correct chart ordering and mathematical operations.

## Overview

ProcessBehavior now automatically handles common data type issues to ensure:
- **Correct time series ordering**: 1, 2, 3, 10 (not '1', '10', '2', '3')
- **Natural factor level sorting**: Lane 1, Lane 2, Lane 10 (not Lane 1, Lane 10, Lane 2)
- **Accurate moving range calculations**: Uses adjacent observations
- **Valid signal detection**: Checks truly consecutive points

This happens transparently during data preparation - no code changes required!

---

## What Gets Converted?

### Time Variables

| Your Data | ProcessBehavior Converts To | Example |
|-----------|---------------------------|---------|
| String numbers | Numeric | '1', '2', '10' → 1, 2, 10 |
| String dates | datetime | '2024-01-01' → Timestamp('2024-01-01') |
| Numeric | No change | 1, 2, 10 → 1, 2, 10 |
| datetime | No change | Already correct |
| date objects | No change | Already correct |
| Categorical (ordered) | No change | User knows best |
| Period | No change | Already correct |

### Factor Columns (Grouping Variables)

| Your Data | ProcessBehavior Converts To | Example |
|-----------|---------------------------|---------|
| String numbers | Numeric | '1', '10' → 1, 10 |
| Numeric | No change | 1, 10 → 1, 10 |
| Mixed strings | No change | 'A', 'B', '1' stays as-is |
| Categorical | No change | User-specified order respected |

### RSG (Rational Subgroup) Labels

**All RSG labels become categorical with natural sort order:**

```python
# Your factors: lane=[1, 2, 10], phase=[1, 2]
# RSG labels: '1_1', '1_2', '2_1', '2_2', '10_1', '10_2'
#
# Natural sort order (correct):
# → '1_1', '1_2', '2_1', '2_2', '10_1', '10_2'
#
# NOT lexicographic (wrong):
# → '10_1', '10_2', '1_1', '1_2', '2_1', '2_2'
```

Charts will display labels in the natural order automatically!

---

## Examples

### Example 1: String-Numeric Time Variable

**Problem**: Time points stored as strings sort incorrectly

```python
import pandas as pd
from processbehavior import ProcessDataFrame

# Your data has string time values
df = pd.DataFrame({
    'time': ['1', '2', '3', '10', '11', '12'],  # Strings!
    'machine': ['A', 'A', 'A', 'A', 'A', 'A'],
    'measurement': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
})

pdf = ProcessDataFrame(df)
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['machine'],
    time_var='time'
).calculate()

# ProcessBehavior automatically converted 'time' to numeric
# Charts will display in correct order: 1, 2, 3, 10, 11, 12
# Moving range calculated between adjacent points
```

**Before** (lexicographic sort):
```
Time points: 1, 10, 11, 12, 2, 3 ❌
Moving range: |10.1 - 10.4| = 0.3 (WRONG - not adjacent!)
```

**After** (numeric sort):
```
Time points: 1, 2, 3, 10, 11, 12 ✓
Moving range: |10.1 - 10.2| = 0.1 (CORRECT - adjacent)
```

### Example 2: String-Date Time Variable

```python
df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-10', '2024-01-02'],  # String dates
    'product': ['Widget', 'Widget', 'Widget'],
    'defects': [5, 3, 4]
})

pdf = ProcessDataFrame(df)
result = pdf.analyze(
    response_var='defects',
    grouping_vars=['product'],
    time_var='date'
).calculate()

# 'date' automatically converted to datetime
# Charts show chronological order: 2024-01-01, 2024-01-02, 2024-01-10
```

### Example 3: Multi-Factor Natural Sorting

```python
df = pd.DataFrame({
    'lane': [1, 1, 2, 2, 10, 10],  # Numeric
    'head': [1, 10, 1, 10, 1, 10],  # Numeric
    'measurement': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
})

pdf = ProcessDataFrame(df)
result = pdf.analyze(
    response_var='measurement',
    grouping_vars=['lane', 'head'],
    time_var=None
).calculate()

# RSG labels (categorical with natural sort):
# '1_1', '1_10', '2_1', '2_10', '10_1', '10_10'
#
# Chart legend displays in this natural order!
```

**Chart display**:
```
✓ Lane 1, Head 1
✓ Lane 1, Head 10  ← 10 comes after 1 (natural)
✓ Lane 2, Head 1
✓ Lane 2, Head 10
✓ Lane 10, Head 1  ← Lane 10 after Lane 2 (natural)
✓ Lane 10, Head 10
```

---

## How It Works

### 1. Type Detection (Transparent)

During `prepare_dataset()`, ProcessBehavior:

1. Checks time_var and grouping_vars data types
2. Attempts numeric conversion for string columns
3. Attempts datetime conversion if numeric fails
4. Logs INFO messages for any conversions
5. Keeps original types if already correct

```python
# Example log output:
INFO:processbehavior.data_preparation:Converted column 'time' from string to numeric for correct sorting (example: '1' → 1)
INFO:processbehavior.data_preparation:Created categorical column 'rsg' with natural sort order (8 categories)
```

### 2. Natural Sorting (Automatic)

After creating RSG labels, ProcessBehavior:

1. Gets unique RSG values
2. Sorts using `natsort` library (handles '10' > '2')
3. Creates ordered `pd.Categorical` with natural-sorted categories
4. All pandas operations (groupby, sort, plot) respect this order

### 3. Original Columns Preserved

**Important**: Your original columns are preserved!

```python
# After analysis:
result.charts['Xbar']['data'].columns
# ['rsg', 'xbar', 'center', 'lcl', 'ucl', ...]

# Original data still accessible:
result.summary['data_config']
# Shows: lane (numeric), phase (numeric), pull (numeric)
```

---

## User Control

### Pre-Convert for Explicit Control

If you want explicit control over types:

```python
# Convert before passing to ProcessBehavior
df['time'] = pd.to_numeric(df['time'])
df['date'] = pd.to_datetime(df['date'])
df['category'] = pd.Categorical(df['category'], categories=['Low', 'Med', 'High'], ordered=True)

# ProcessBehavior will respect your types
pdf = ProcessDataFrame(df)
```

### Disable Conversion (Not Recommended)

Type conversion is part of `prepare_dataset()` and cannot be disabled, as it's essential for:
- Correct mathematical operations (moving range)
- Valid signal detection (consecutive points)
- Accurate chart display

If you need the old behavior, use ProcessBehavior v0.0.x.

---

## Technical Details

### Dependencies

- **natsort >= 8.0.0**: Natural sorting algorithm
- Handles numbers in strings: 'Item1', 'Item2', 'Item10'
- Supports multiple numeric segments: 'V1.10.2' vs 'V1.2.10'

### Implementation

- **File**: `processbehavior/data_preparation.py`
- **Methods**:
  - `_detect_and_convert_type()`: Type conversion logic
  - `_make_categorical_rsg()`: Natural sorting for RSG labels
- **When**: During `prepare_dataset()` call

### Performance

Minimal overhead:
- Type conversion: O(n) single pass
- Natural sorting: O(k log k) where k = unique RSG values (typically small)
- Categorical operations: Faster than string comparisons

---

## Troubleshooting

### Issue: "My dates are not sorting correctly"

**Cause**: Dates have inconsistent formats

**Solution**:
```python
# Specify format explicitly
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
```

### Issue: "My categories have a specific order"

**Cause**: Natural sort doesn't match your domain logic

**Solution**:
```python
# Use ordered categorical with your preferred order
df['priority'] = pd.Categorical(
    df['priority'],
    categories=['Low', 'Medium', 'High', 'Critical'],
    ordered=True
)
```

### Issue: "I see INFO messages about conversions"

**Not a problem**: These are informational logs showing what was converted

**To suppress**:
```python
import logging
logging.getLogger('processbehavior.data_preparation').setLevel(logging.WARNING)
```

---

## Migration Guide

### From v0.0.x to v0.1.0

**No code changes required!** Type conversion is automatic and backwards compatible.

**What changed**:
- ✓ Charts now display in natural order
- ✓ Moving range calculations use correct adjacent observations
- ✓ Signal detection checks truly consecutive points
- ✓ RSG labels are categorical (still strings, but ordered)

**What stayed the same**:
- API unchanged
- Analysis results structure unchanged
- Chart names unchanged ('Xbar', 'Sbar', 'Imr')

### Verification

To verify natural sorting works:

```python
result = pdf.analyze(...).calculate()
xbar_data = result.charts['Xbar']['data']

# Check RSG is categorical
assert isinstance(xbar_data['rsg'].dtype, pd.CategoricalDtype)

# Check categories are naturally sorted
categories = list(xbar_data['rsg'].cat.categories)
print(f"Categories in order: {categories}")
# Should show: ['1_1', '1_2', '2_1', '2_2', '10_1', '10_2']
```

---

## See Also

- **Fillweight Tutorial**: `examples/tom/fillweight_analysis_tutorial.ipynb`
- **Fillweight Power Demo**: `examples/fillweight_power_demo.ipynb`
- **SDS Detection Demo**: `examples/sds_detection_demo.ipynb`
- **Data Preparation Tests**: `tests/test_data_preparation.py`

---

## Summary

**ProcessBehavior now automatically handles data types to ensure mathematically correct and visually intuitive results.**

✅ String-numeric → Numeric conversion
✅ String-date → Datetime conversion
✅ Natural sorting for display labels
✅ Categorical RSG for correct ordering
✅ Original columns preserved
✅ No code changes required

**Your charts will look right. Your calculations will be correct. Automatically.**
