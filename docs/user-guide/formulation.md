# Study Formulation

The `formulate()` method is the heart of ProcessBehavior. It enables the analyst to specify the key inputs defined during problem formulation to create a structured study. The system automatically detects the design state and prepares the analysis.

## The Formulation API

```python
from processbehavior import ProcessBehavior

pb = ProcessBehavior(df)

study = pb.formulate(
    response=pb.cols.weight,        # Required: measurement variable
    factors=[pb.cols.lane],         # Optional: grouping variables
    time=pb.cols.batch,             # Optional: time/sequence variable
    precision=3                         # Optional: decimal places
)
```

## IDE Auto-Completion

One of ProcessBehavior's key features is **IDE auto-completion** for column names. This eliminates typos and makes exploration faster.

### Column Auto-Completion

After creating a ProcessBehavior, access columns via the `.cols` accessor:

```python
pb = ProcessBehavior(df)

# Type pb.cols. and your IDE will show all available columns
pb.cols.weight      # Instead of 'weight' string
pb.cols.lane        # Instead of 'lane' string
pb.cols.batch       # Instead of 'batch' string
```

This works in:
- VS Code with Pylance
- PyCharm
- Jupyter notebooks (with Tab completion)
- Any LSP-compatible editor

### Chart Type Auto-Completion

After formulation, the `study.charts` accessor provides auto-completion for valid chart types:

```python
study = pb.formulate(...)

# Type study.charts. and see only valid charts for your DS
study.charts.Xbar        # Available if DS supports it
study.charts.XmR         # Always available
study.residuals.R4       # VAS residual (pass to value=)
```

## Parameter Details

### response (required)

The measurement variable to analyze.

```python
# Using auto-completion (recommended)
response=pb.cols.measurement

# Using string (still works)
response='measurement'
```

### factors (optional)

A list of categorical variables that define subgroups. These become the "rational subgroups" in Wheeler's terminology.

```python
# Single factor
factors=[pb.cols.operator]

# Multiple factors (creates combined subgroups)
factors=[pb.cols.machine, pb.cols.shift]
```

When multiple factors are specified, they're combined into a single grouping variable (e.g., `Machine_A_Shift_1`).

### time (optional)

The variable that defines time ordering. This enables time-series analysis and certain signal detection rules.

```python
time=pb.cols.batch
time=pb.cols.timestamp
time=pb.cols.sequence
```

### precision (optional, default=3)

Number of decimal places for calculated statistics.

```python
precision=2  # Round to 2 decimal places
precision=4  # Higher precision for sensitive measurements
```

## What Formulation Does

When you call `formulate()`, ProcessBehavior:

1. **Validates data** - Checks for required columns and data types
2. **Cleans data** - Handles missing values and garbage characters
3. **Detects DS** - Determines the Design State (0-6)
4. **Calculates means** - Computes Y̅, Y̅_k, Y̅_t, Y̅_kt
5. **Computes residuals** - Calculates R1-R5
6. **Determines valid charts** - Lists which analyses are appropriate

## The Study Object

Formulation returns a `Study` object with rich information about your data structure and available analyses.

### Design State Traceability

The Study tracks three Design States providing transparent lineage from raw data to analysis. The **Analytical Design State (ADS)** is the authoritative state that drives all analysis decisions. See [Design State Traceability](../getting-started/key-concepts.md#design-state-traceability) for the full explanation.

```python
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)

# Design states
print(study.observed_design_state)    # ODS: what was actually collected (raw data)
print(study.analytical_design_state)  # ADS: what is fit for analysis (tidy data)
print(study.plan_design_state)        # PDS: what was intended (None if no plan)

# ADS-derived properties (these drive chart selection)
print(study.ads_reason)              # e.g., "full_replication"
print(study.ads_description)         # e.g., "Full replication (all cells n>=2)"

# Chart recommendations (determined by ADS)
print(study.valid_charts)       # ['Histogram', 'Xbar', 'S', 'XmR', 'R']
print(study.recommended_chart)  # 'Xbar'
print(study.residuals)          # StudyResidualAccessor(R1, R2, R3, R4, R5)

# Chart auto-completion
result = study.execute(study.charts.Xbar)  # IDE suggests valid charts

# Access the prepared dataset
print(study.dataset.columns.tolist())
# ['lane', 'batch', 'weight', 'Ybar', 'Ybar_k', 'Ybar_t', 'Ybar_kt',
#  'R1', 'R2', 'R3', 'R4', 'R5']
```

## The why_not() Method

If you try to use an invalid chart, `why_not()` explains why:

```python
# If Xbar is not valid for your DS
study.why_not('Xbar')
# Returns: "Xbar requires subgrouped data (n >= 2 per cell).
#          Your data has n=1 per cell (DS 2)."
```

## Common Patterns

### Pattern 1: Simple Time Series

No factors, just measurements over time:

```python
study = pb.formulate(
    response=pb.cols.temperature,
    time=pb.cols.day
)
# Results in DS 4, recommends XmR
```

### Pattern 2: Comparing Groups

Factors but no time dimension:

```python
study = pb.formulate(
    response=pb.cols.yield_pct,
    factors=[pb.cols.machine, pb.cols.operator]
)
# Results in DS varies, recommends Xbar or XmR
```

### Pattern 3: Groups Over Time

Full analysis with factors and time:

```python
study = pb.formulate(
    response=pb.cols.fillweight,
    factors=[pb.cols.lane],
    time=pb.cols.pull
)
# Results in DS 1-3, recommends Xbar-S with VAS residuals
```

### Pattern 4: Replicated Design

Multiple observations per factor-time cell:

```python
# 4 lanes x 10 batches x 3 replicates = 120 observations
study = pb.formulate(
    response=pb.cols.weight,
    factors=[pb.cols.lane],
    time=pb.cols.batch
)
# Results in DS 1 (Full Replication) - most powerful design
```

## Data Cleaning

ProcessBehavior automatically cleans common garbage values:

```python
# These are automatically converted to NaN:
default_na_values = [
    '*', '?', 'ND', 'BDL', 'NA', 'N/A', 'n/a',
    '<LOD', '>LOQ', 'TNTC', 'QNS', '--'
]

# Custom NA values
pb = ProcessBehavior(df, na_values=['*', 'missing', '<DL'])
```

## Natural Sorting

Factor levels are automatically sorted naturally:

```python
# Input order: ['Lane_10', 'Lane_1', 'Lane_2']
# Sorted order: ['Lane_1', 'Lane_2', 'Lane_10']
```

This uses `natsort` for intelligent alphanumeric sorting.

## The `plan` Parameter

The `plan` parameter provides an alternative to `factors` for specifying the study structure. Use `plan` when you know the intended design -- especially when factor levels or time points are entirely absent from the data.

`plan` and `factors` are **mutually exclusive** -- use one or the other.

### Format

The plan is a dictionary with a required `'factors'` key and optional `'T'` and `'N'` keys:

```python
study = pb.formulate(
    response=pb.cols.weight,
    time=pb.cols.batch,
    plan={
        'factors': {
            pb.cols.lane: [1, 2, 3, 4],
            pb.cols.shift: ['day', 'night']
        },
        'T': 20,   # Planned number of time points
        'N': 3     # Planned observations per cell
    }
)
```

- **`factors`** (required): Dict mapping column names to lists of expected levels
- **`T`** (optional): Number of planned time points
- **`N`** (optional): Planned observations per cell

### When to Use `plan` vs `factors`

| Scenario | Use |
|----------|-----|
| All factor levels and time points are present in data | Either works |
| Some factor levels are entirely absent from data | `plan` |
| Some time points were entirely skipped | `plan` with `'T'` |
| You want to compare planned vs observed structure | `plan` |

### DS 4-6 Detection

ProcessBehavior can detect DS 4-6 **with or without a plan**. DS detection runs on raw data before NA rows are dropped, so cells where all response values are NA (e.g., from garbage values like `*` or `ND`) are counted as empty cells (N_kt=0). This means:

- **With `factors`**: DS 4-6 are detected when the data contains factor/time combinations where every response value is NA or garbage. The system sees these as empty cells in the grid.
- **With `plan`**: DS 4-6 are also detected when factor levels or time points specified in the plan are entirely absent from the data. This catches gaps that `factors` alone cannot see.

A plan is valuable for documenting experimental intent and catching absent factor levels, but it is not required for DS 4-6 detection.

After formulating with `plan`, use `study.design()` to see planned vs observed structure. See [Design States](sds-detection.md) for details on DS 4-6.

## The `companion` Parameter

Wheeler recommends reading certain charts as pairs: Xbar with S, and XmR with R (the variation chart first, then the location chart). The `companion` parameter in `execute()` returns both charts together.

```python
# Returns both Xbar and S charts together
result = study.execute(chart='Xbar', companion=True)

# Returns both XmR and R charts together, stratified
result = study.execute(chart='XmR', by=['lane'], companion=True)
```

Key details:

- Either chart in the pair triggers the pair: `chart='S', companion=True` also returns Xbar+S
- Default is `companion=False` (returns a single chart)
- Companion results can be plotted and exported the same way as single-chart results

## Study Inspection Methods

After formulation, the `Study` object provides several methods for inspecting what analyses are available.

### `study.support`

A DataFrame showing all chart types with their availability, recommendations, and the analytical question each answers:

```python
study.support
#   chart  category  available  recommended  reason  question
# 0  Xbar   primary       True         True    None  Are subgroup means stable over time?
# 1     S   primary       True        False    None  Is within-subgroup variation stable?
# ...
```

Filter to available charts:

```python
study.support[study.support['available']]
```

### `study.why_not(chart)`

Explains why a specific chart is unavailable:

```python
study.why_not('XmR', value='R2')
# "'XmR' (R2) unavailable: Not available for this DS"
```

### `study.design()`

Returns a `DesignReport` comparing the sampling plan to observed data:

```python
report = study.design()

# Structure metrics
report.K           # Planned number of RSG groups
report.K_observed  # Observed number of RSG groups
report.T           # Planned time points
report.T_observed  # Observed time points
report.N           # Planned cell size
report.N_observed  # Observed cell size (min, median, max)
report.R           # Planned total cells (K × T)
report.R_observed  # Observed total cells

# Missing/extra structure
report.missing_levels   # Factor levels in plan but not observed
report.extra_levels     # Factor levels observed but not in plan
report.missing_combos   # RSG groups in plan but not observed
report.plan_adherence   # Summary of how well data matches plan
```

Works with or without a sampling plan — without a plan, it reports observed structure only.

### Interpreting the Design Report

The Design Report shows the full **Design State lineage** (PDS → ODS → ADS) and highlights structural discrepancies between your plan and observed data.

**Key metrics to check:**

| Metric | What It Tells You | Action |
|--------|-------------------|--------|
| **K_missing > 0** | Factor combinations in your plan were never observed | Investigate why — were samples lost? Were levels skipped? |
| **T_missing > 0** | Expected time points are absent from data | Check for missing batches or skipped collection periods |
| **N_observed as (min, med, max)** | Cell sizes vary | Large variation may indicate DS 3; consider standardizing collection |
| **coverage < 1.0** | Data doesn't cover the full planned grid | Lower coverage = more incomplete design (DS 4-6) |
| **ODS ≠ ADS** | Raw structure changed after cleansing | Empty cells were removed; check `missing_combos` to understand what was lost |
| **remediation** | Actionable guidance for improving the design | Follow the suggestion to move toward DS 1 |

**Example output:**

```
Design Report (2 factors)
  Unit of analysis: filled container
  Design lineage:
    Planned Design State:    DS 1 (Full Replication)
    Observed Design State:   DS 6 (Incomplete, With Singletons) — 3 empty cells
    Analytical Design State: DS 1 (Full Replication)
  Plan adherence: 3 missing cells out of 480 planned (99.4% coverage)
  K=6, T=80, R=477/480, N=(2, 3, 5)
```

This tells you: the plan called for full replication across 480 cells. Three cells were empty in the raw data (ODS 6), but after cleansing the remaining 477 cells all have n >= 2 (ADS 1), enabling full Xbar-S analysis with exact R2.

### `study.capability()`

Assesses process capability against specification limits (Bishop Ch. 16). Returns both **current capability** (Pp/Ppk, based on overall variation) and **potential capability** (Cp/Cpk, based on R2 within-cell noise only).

```python
from processbehavior import SpecLimits

# Two-sided specifications
cap = study.capability(usl=250.5, lsl=249.5, target=250.0)
print(f"Current:   Pp={cap.pp:.2f}, Ppk={cap.ppk:.2f}")
print(f"Potential: Cp={cap.cp:.2f}, Cpk={cap.cpk:.2f}")
print(f"Outside specs: {cap.pct_outside:.1f}%")

# One-sided (upper limit only)
cap = study.capability(usl=105.0)

# Visualize
cap.plot(values=study.dataset[study.response].dropna().values)
```

| Metric | Based On | Meaning |
|--------|----------|---------|
| **Pp/Ppk** | Overall σ̂ (all variation) | Current capability — process as-is |
| **Cp/Cpk** | R2 σ̂ (within-cell noise only) | Potential capability — achievable if all assignable causes are eliminated |

Potential capability (Cp/Cpk) requires VAS residuals (factors + time). If unavailable, `cap.potential_unavailable_reason` explains why.

### `study.loss_function()`

Decomposes expected loss into five components using the Taguchi Loss Function (Bishop Ch. 15). Identifies the largest sources of variation as a Pareto analysis.

```python
# Use grand mean as target (centering = 0)
loss = study.loss_function()

# Use explicit target
loss = study.loss_function(target=250.0)

# The 5 components (as percentages of total loss)
print(f"Centering:   {loss.pct_centering:.1f}%")
print(f"Unexplained: {loss.pct_unexplained:.1f}%")
print(f"PDC:         {loss.pct_pdc:.1f}%")
print(f"Time:        {loss.pct_time:.1f}%")
print(f"Interaction: {loss.pct_interaction:.1f}%")

# For multi-factor studies, see PDC breakdown by factor
print(loss.pdc_by_factor)  # e.g., {'machine': 12.5, 'operator': 4.3}

# Visualize
loss.plot()                    # 5-bar Pareto
loss.plot(structured=True)     # Expands PDC into per-factor components
```

**The five components:**

| Component | Formula | What It Captures |
|-----------|---------|-----------------|
| **Centering** | (Ȳ - Target)² | Loss from being off-target |
| **Unexplained** | Within-cell variance | Irreducible noise (R2) |
| **PDC** | Between-factor variance | Factor (process design condition) effects |
| **Time** | Between-time variance | Time period effects |
| **Interaction** | Factor × time variance | How factor effects change over time |

These five components always sum to the total expected loss. The largest percentage identifies where to focus improvement efforts.

## Best Practices

1. **Use auto-completion** — Prevents typos and speeds up development
2. **Start simple** — Begin with just response, add factors/time as needed
3. **Check the ADS** — Understand your analytical design state before interpreting
4. **Use why_not()** — Learn why certain charts aren't available
5. **Review the design** — Use `study.design()` to verify structure and lineage

## Next Steps

- [Design States](sds-detection.md) - Understanding Design States
- [Chart Types](chart-types.md) - Choosing the right chart
- [VAS Residuals](residuals.md) - Working with VAS residuals
