# SDS Detection: How Sampling Design States Are Classified

This document explains how ProcessBehavior detects and classifies Sampling Design States (SDS). Understanding SDS detection is essential for interpreting what analysis capabilities are available for your data.

## Overview

### What is SDS and Why It Matters

The Sampling Design State (SDS) describes the structure of your data in terms of:
- **Factors (K)**: Categorical variables that define groups (e.g., Lane, Phase, Machine)
- **Time (T)**: The temporal dimension of your process
- **Replication**: How many observations exist per factor-time cell

ProcessBehavior automatically detects your SDS to determine:
- Which chart types are valid
- How to calculate variance (R2 residual)
- What VAS residuals (R1-R5) are available
- Whether interaction analysis is possible

### The Six Sampling Design States

Per Wheeler/Bishop Table 1, there are six distinct sampling design states:

| SDS | Name | N_kt Pattern | Description |
|-----|------|--------------|-------------|
| 1 | Complete | All N_kt ≥ 2 | Full replication in every cell |
| 2 | Semi-Complete | All N_kt = 1 | No replication (all singletons) |
| 3 | Semi-Complete | Mixed (1s and ≥2s) | Partial replication |
| 4 | Incomplete | Has 0s and ≥2s, no 1s | Incomplete without singletons |
| 5 | Incomplete | Has 0s, max = 1 | Incomplete without replication |
| 6 | Incomplete | Has 0s, 1s, and ≥2s | Incomplete with singletons |

**Complete/Semi-Complete** (SDS 1-3): No empty cells (all cells have N_kt ≥ 1)
**Incomplete** (SDS 4-6): Has empty cells (some N_kt = 0)

### When Detection Runs

SDS detection runs **before** `prepare_dataset()`, on the raw data. This timing is critical because:
- Cells with all-NA responses are preserved (they haven't been dropped yet)
- The true intended structure is visible
- Without raw data detection, all-NA cells would vanish and the structure would appear more complete than intended

## Core Concepts

### N_kt: Observations Per Cell

**N_kt** is the count of valid (non-NA) responses for each unique (factor × time) cell.

For example, with factors `Lane` and `Phase`, and time variable `Pull`:

| Lane | Phase | Pull | N_kt |
|------|-------|------|------|
| 1 | 1 | 1 | 3 |
| 1 | 1 | 2 | 2 |
| 1 | 2 | 1 | 1 |
| 2 | 1 | 1 | 0 |

The distribution of N_kt values determines the SDS classification:
- `min(N_kt)` - Are any cells singletons or empty?
- `max(N_kt)` - Is there any replication?
- Presence of zeros - Is the grid incomplete?

### Structure View

SDS detection works on a **structure view** - a minimal projection of the raw data containing only the columns needed for classification:

```python
kt_cols = rsg_vars + [time_var]  # e.g., ['Lane', 'Phase', 'Pull']
structure_view = df[kt_cols + [response_col]]
```

The structure view undergoes:

1. **Canonicalization of kt columns**: Type conversion (e.g., `"1"` → `1`) to ensure consistent grouping
2. **Response normalization**: Missing tokens (`*`, `NA`, `N/A`, `nan`, `null`, `None`, empty string) converted to `pd.NA`
3. **Row filtering**: Rows with NA in any kt column are dropped (can't determine cell membership)
4. **Response NA preservation**: Rows with NA response are **kept** - this reveals cells where data collection was attempted but failed

## Detection WITHOUT a Plan

### When to Use

Detection without a plan is appropriate when:
- The sampling structure is unknown or exploratory
- No formal sampling plan exists
- You want a quick structure check

### How It Works

Without a plan, detection groups raw data by kt_cols and counts valid responses per cell:

```python
nkt_observed = structure_view.groupby(kt_cols)[response_col].apply(
    lambda s: s.notna().sum()
)
```

Cells where all responses are NA will show N_kt = 0 in the observed counts.

### Classification Rules

Without a plan, the same classification logic applies - SDS 1-6 are all possible:

**Complete/Semi-Complete (no empty cells):**

| Condition | SDS | Reason |
|-----------|-----|--------|
| `min(N_kt) ≥ 2` | 1 | `full_replication` |
| `max(N_kt) = 1` | 2 | `no_replication` |
| `min(N_kt) = 1` AND `max(N_kt) ≥ 2` | 3 | `partial_replication` |

**Incomplete (has empty cells with N_kt = 0):**

| Condition | SDS | Reason |
|-----------|-----|--------|
| Has 0s AND no 1s AND has ≥2s | 4 | `incomplete_no_singletons` |
| Has 0s AND `max(N_kt) = 1` | 5 | `incomplete_no_replication` |
| Has 0s AND has 1s AND has ≥2s | 6 | `incomplete_with_singletons` |

Empty cells (N_kt = 0) can appear without a plan when a cell exists in the data but all its responses are NA. For example, if Lane=2, Pull=1 has two rows but both have `Weight = NA`, that cell has N_kt = 0.

### Limitations

Without a plan, detection **cannot**:
- Identify cells that were **never attempted** (no rows exist for that combination)
- Detect incomplete coverage against an intended design
- Know that a 10-cell dataset was supposed to be a 12-cell design

If your experiment intended 12 factor combinations but only 10 have any rows in the data, without a plan this looks like a complete 10-cell design, not an incomplete 12-cell design. The 2 never-attempted cells are invisible.

## Detection WITH a Plan

### When to Use

Provide a sampling plan when:
- You have a designed experiment (DOE) with a known intended structure
- You need to detect cells that were **never attempted** (no rows exist)
- You want to measure deviation from intended structure
- You need to know if observed 10 cells were supposed to be 12 cells

### Plan Format

The plan specifies expected factor levels and optionally time extent:

```python
# Full format with factors, T (time points), and N (observations per cell)
plan = {
    'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]},
    'T': 10,  # Expected number of time points (optional)
    'N': 2   # Expected observations per cell (optional, for diagnostics)
}

# Usage via formulate()
study = pb.formulate(
    response='Weight',
    time='Pull',
    plan={
        'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]},
        'T': 10
    }
)
```

### Plan Canonicalization

Plan values undergo the same type conversion as data columns:
- `'1'` → `1` (if data column is numeric)
- `' A '` → `'A'` (whitespace stripped)

This prevents false "missing cells" from type mismatches between plan and data.

### Time Column Handling

The time column values are determined by priority:

1. **If `T` is specified in plan**: Use `1..T`
2. **If plan factors include the time column**: Use those explicit values
3. **Otherwise**: Use observed time values from data

```python
# T in plan takes priority
plan = {
    'factors': {'Lane': [1, 2]},
    'T': 5
}
# → time values: [1, 2, 3, 4, 5]

# Explicit time in factors
plan = {
    'factors': {'Lane': [1, 2], 'Pull': [1, 2, 3]}
}
# → time values: [1, 2, 3]

# Fallback to observed
plan = {
    'factors': {'Lane': [1, 2]}
}
# → time values: whatever is in the data
```

### Expected Cell Grid

With a plan, the expected cells are the Cartesian product of all factor levels (plus time if specified). Internally:

```python
# After extracting factors from plan['factors'] and time from plan['T']
planned_index = pd.MultiIndex.from_product(
    [canonicalized_plan[c] for c in kt_cols],
    names=kt_cols
)
nkt_counts = nkt_observed.reindex(planned_index, fill_value=0)
```

Cells missing from observed data get N_kt = 0, enabling detection of never-attempted combinations.

### Classification Rules

The same classification rules apply with or without a plan. The plan simply adds more cells to check (never-attempted combinations get N_kt = 0):

| Condition | SDS | Reason |
|-----------|-----|--------|
| Has 0s AND no 1s AND has ≥2s | 4 | `incomplete_no_singletons` |
| Has 0s AND `max(N_kt) = 1` | 5 | `incomplete_no_replication` |
| Has 0s AND has 1s AND has ≥2s | 6 | `incomplete_with_singletons` |

## Classification Decision Tree

```
START with N_kt distribution
│
├─ Has empty cells (N_kt = 0)?
│  │
│  ├─ NO (Complete/Semi-Complete)
│  │  │
│  │  ├─ min(N_kt) ≥ 2?
│  │  │  └─ YES → SDS 1 (full_replication)
│  │  │
│  │  ├─ max(N_kt) = 1?
│  │  │  └─ YES → SDS 2 (no_replication)
│  │  │
│  │  └─ else
│  │     └─ SDS 3 (partial_replication)
│  │
│  └─ YES (Incomplete)
│     │
│     ├─ max(N_kt) = 1?
│     │  └─ YES → SDS 5 (incomplete_no_replication)
│     │
│     ├─ has N_kt = 1 cells?
│     │  └─ YES → SDS 6 (incomplete_with_singletons)
│     │
│     └─ no singletons
│        └─ SDS 4 (incomplete_no_singletons)
```

## Output: SDSResult

Internally, the detection result is stored as an `SDSResult` dataclass:

```python
@dataclass(frozen=True)
class SDSResult:
    sds: int                    # 1-6
    min_cell_size: int          # Minimum N_kt for cells with N_kt > 0
    reason: SDSReasonType       # Why this SDS was classified
    n_empty_cells: int          # Count of cells with N_kt = 0
```

### Fields

- **`sds`**: The detected Sampling Design State (1-6)
- **`min_cell_size`**: The minimum observation count among non-empty cells
- **`reason`**: A string explaining the classification (see Reason Types below)
- **`n_empty_cells`**: How many cells have N_kt = 0 (after plan comparison if provided)

### Accessing Results

The key results are exposed through the `Study` and `DesignReport` APIs:

```python
# Via Study
study.observed_design_state   # SDSResult for observed raw data
study.analytical_design_state # SDSResult for analyzable data after tidying
study.sds_reason              # ADS-derived machine token (e.g., 'full_replication')
study.sds_description         # ADS-derived human prose description

# Via DesignReport
design = study.design()
design.sds_reason      # Why this SDS was classified (e.g., 'full_replication')
```

The `min_cell_size` value is used internally to determine R2 chart selection (S vs XmR).

## min_cell_size Calculation

`min_cell_size` is calculated only from cells with N_kt > 0:

```python
valid_nkt = nkt_counts[nkt_counts > 0]
min_cell_size = int(valid_nkt.min()) if len(valid_nkt) > 0 else 0
```

This value drives R2 chart selection:
- **`min_cell_size ≥ 2`**: Can use S chart (within-cell variance available)
- **`min_cell_size = 1`**: Must use XmR chart (no within-cell variance)

The R2 residual represents within-cell variation. When cells have replication (n ≥ 2), exact variance can be calculated. When cells are singletons (n = 1), variance must be estimated via moving average.

## Why Raw Data Detection Matters

Consider this scenario:

**Raw data:**
| Lane | Pull | Weight |
|------|------|--------|
| 1 | 1 | 10.5 |
| 1 | 1 | 10.3 |
| 2 | 1 | NA |
| 2 | 1 | NA |

**After dropping NA responses:**
| Lane | Pull | Weight |
|------|------|--------|
| 1 | 1 | 10.5 |
| 1 | 1 | 10.3 |

Without raw data detection:
- Only Lane=1 appears → looks like SDS 1 (single factor, full replication)

With raw data detection:
- Lane=2 cell has N_kt = 0 (all-NA) → reveals incomplete structure

This is why SDS detection runs before `prepare_dataset()` drops NA response rows.

## Integration with formulate()

When you call `formulate()`, SDS detection happens automatically:

```python
study = pb.formulate(
    response='Weight',
    time='Pull',
    plan={
        'factors': {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]},
        'T': 10
    }
)

# SDS result is accessible via Study
print(study.observed_design_state.sds)  # 1-6
print(study.sds_reason)                # Machine token (e.g., 'full_replication')
print(study.design().sds_reason)  # Classification reason
```

The sequence is:
1. Raw data passed to detector (NA rows preserved)
2. Plan validated and canonicalized
3. SDS detected from N_kt distribution
4. Result stored in Study object
5. `prepare_dataset()` runs afterward (NA rows dropped for analysis)

## Practical Examples

### Example 1: SDS 1 (Full Replication, No Plan)

```python
import pandas as pd
from processbehavior import ProcessBehavior

df = pd.DataFrame({
    'Lane': [1, 1, 2, 2, 1, 1, 2, 2],
    'Pull': [1, 1, 1, 1, 2, 2, 2, 2],
    'Weight': [10, 11, 9, 10, 12, 11, 10, 9]
})

pb = ProcessBehavior(df)
study = pb.formulate(
    response='Weight',
    time='Pull',
    factors=['Lane']
)

print(f"SDS: {study.observed_design_state.sds}")   # SDS: 1
print(f"Reason: {study.design().sds_reason}")    # Reason: full_replication
```

N_kt distribution: All cells have n = 2 → `min(N_kt) ≥ 2` → SDS 1

### Example 2: SDS 3 (Partial Replication, No Plan)

```python
df = pd.DataFrame({
    'Lane': [1, 1, 2, 1, 1, 2],
    'Pull': [1, 1, 1, 2, 2, 2],
    'Weight': [10, 11, 9, 12, 11, 10]
})

pb = ProcessBehavior(df)
study = pb.formulate(
    response='Weight',
    time='Pull',
    factors=['Lane']
)

print(f"SDS: {study.observed_design_state.sds}")   # SDS: 3
print(f"Reason: {study.design().sds_reason}")    # Reason: partial_replication
```

N_kt distribution: Lane=1 has n=2, Lane=2 has n=1 → mixed → SDS 3

### Example 3: SDS 6 (Incomplete with Singletons, With Plan)

```python
df = pd.DataFrame({
    'Lane': [1, 1, 2],
    'Pull': [1, 1, 1],
    'Weight': [10, 11, 9]
})

pb = ProcessBehavior(df)
study = pb.formulate(
    response='Weight',
    time='Pull',
    plan={'factors': {'Lane': [1, 2, 3]}}  # Lane 3 expected but missing
)

print(f"SDS: {study.observed_design_state.sds}")   # SDS: 6
print(f"Reason: {study.design().sds_reason}")    # Reason: incomplete_with_singletons
```

N_kt distribution: Lane=1 has n=2, Lane=2 has n=1, Lane=3 has n=0 → SDS 6

### Example 4: SDS 4 (Incomplete without Singletons, With Plan)

```python
df = pd.DataFrame({
    'Lane': [1, 1, 2, 2],
    'Pull': [1, 1, 1, 1],
    'Weight': [10, 11, 9, 10]
})

pb = ProcessBehavior(df)
study = pb.formulate(
    response='Weight',
    time='Pull',
    plan={'factors': {'Lane': [1, 2, 3]}}  # Lane 3 expected but missing
)

print(f"SDS: {study.observed_design_state.sds}")   # SDS: 4
print(f"Reason: {study.design().sds_reason}")    # Reason: incomplete_no_singletons
```

N_kt distribution: Lane=1 has n=2, Lane=2 has n=2, Lane=3 has n=0 → SDS 4

## SDS Reason Types

The `reason` field provides a semantic explanation of why a particular SDS was classified:

| Reason | SDS | N_kt Pattern | Description |
|--------|-----|--------------|-------------|
| `full_replication` | 1 | All N_kt ≥ 2 | Every cell has multiple observations |
| `no_replication` | 2 | All N_kt = 1 | Every cell is a singleton |
| `partial_replication` | 3 | Mixed 1s and ≥2s | Some cells replicated, some singleton |
| `incomplete_no_singletons` | 4 | Has 0s and ≥2s, no 1s | Empty cells, all observed cells replicated |
| `incomplete_no_replication` | 5 | Has 0s, max = 1 | Empty cells, no replication |
| `incomplete_with_singletons` | 6 | Has 0s, 1s, and ≥2s | Empty cells with mixed replication |

## Edge Cases

### Single Factor Level (K = 1)

With only one factor level, classification is based purely on replication:
- All n ≥ 2 → SDS 1
- All n = 1 → SDS 2
- Mixed → SDS 3

This is not considered SDS 4 (incomplete) unless a plan specifies additional expected levels.

### Single Time Point (T = 1)

A single time point still classifies normally based on factor replication.

### No Factors, No Time

With neither factors nor time, you have a single cell. Classification:
- Multiple observations → SDS 1 (replicated)
- Single observation → SDS 2 (no replication)

### All Data is NA

If all response values are NA, detection raises a `ValueError`:

```python
ValueError: No valid response values found after filtering
```

### Nested/Hierarchical Designs

Nested designs (where factor B only exists within certain levels of factor A) may classify as SDS 4 or 6 depending on how the plan is specified:
- If the plan expects a full crossing, missing combinations show as empty cells
- If the plan reflects the nested structure, only truly missing combinations appear empty

## Reference

The SDS classification system is based on Wheeler and Bishop's Variance Analysis System (VAS) framework, specifically Table 1 which defines the six sampling design states by their N_kt distribution patterns.

For more information on how SDS affects analysis capabilities, see `SDSRegistry.get_analysis_plan()` which returns the complete specification of valid charts, residuals, and analysis options for each SDS.
