# Key Concepts

This page introduces the core concepts you need to understand ProcessBehavior effectively.

## The Three-Step Workflow

ProcessBehavior follows a deliberate three-step workflow:

```python
# 1. Wrap your data
pb = ProcessBehavior(df)

# 2. Formulate your study (DS detection happens here)
study = pb.formulate(response=..., factors=..., time=...)

# 3. Analyze and visualize
result = study.execute()
result.plot()
```

This separation ensures you understand your data structure before analyzing it.

## Design States (DS)

The **Design State** describes the structure of your data. ProcessBehavior automatically detects which of six states applies:

**Complete/Semi-Complete (no empty cells):**

| DS | Name | Cell Sizes | Recommended Chart |
|-----|------|------------|-------------------|
| 1 | Full Replication | All N_kt >= 2 | Xbar |
| 2 | No Replication | All N_kt = 1 | XmR |
| 3 | Partial Replication | Mix of N_kt = 1 and N_kt >= 2 | XmR |

**Incomplete (has empty cells):**

| DS | Name | Cell Sizes | Recommended Chart |
|-----|------|------------|-------------------|
| 4 | Incomplete, No Singletons | Empty cells + all observed N_kt >= 2 | XmR |
| 5 | Incomplete, No Replication | Empty cells + all observed N_kt = 1 | XmR |
| 6 | Incomplete, With Singletons | Empty cells + mixed N_kt | XmR |

See [DS Definitions](../reference/sds_definitions.md) for the formal classification table per Dr. Thomas A. Bishop's VAS methodology.

### Why DS Matters

The DS determines:
- Which chart types are valid
- How within-group variance is estimated (R2 method: exact, ma2, or hybrid)
- Which VAS residuals can be computed
- What conclusions you can draw

```python
study = pb.formulate(response='weight', factors=['lane'], time='batch')
print(f"DS: {study.analytical_design_state.sds}")  # e.g., 1
print(f"Reason: {study.ads_reason}")                # e.g., "full_replication"
print(f"Valid: {study.valid_charts}")                # e.g., ['Histogram', 'Xbar', 'S', 'XmR', 'R']
```

## Design State Traceability

ProcessBehavior tracks **three Design States** as data flows from intent through observation to analysis, providing transparent lineage at every step.

### The Three States

| State | Property | Computed From | Purpose |
|-------|----------|---------------|---------|
| **Plan Design State (PDS)** | `study.plan_design_state` | Sampling plan parameters | What you *intended* to collect |
| **Observed Design State (ODS)** | `study.observed_design_state` | Raw data (before NA filtering) | What was *actually collected* |
| **Analytical Design State (ADS)** | `study.analytical_design_state` | Tidy data (after data cleansing) | What is *fit for analysis* |

**The ADS drives all analysis decisions** — valid charts, residual availability, R2 calculation method, and interaction analysis. The ODS and PDS provide diagnostic lineage so you can trace exactly how your data structure changed through processing.

### Why Three States?

Raw data often contains garbage values, missing cells, and structural irregularities. The ODS captures this reality — including incomplete designs (DS 4-6) where some factor-time cells are entirely empty. After data cleansing removes invalid observations, the empty cells disappear and the remaining structure may be simpler:

- ODS 4 (Incomplete, no singletons) → ADS 1 (Full Replication)
- ODS 5 (Incomplete, no replication) → ADS 2 (No Replication)
- ODS 6 (Incomplete, with singletons) → ADS 3 (Partial Replication)

This separation means the system correctly identifies incomplete data collection (ODS) while still performing the most powerful analysis the clean data supports (ADS).

### Viewing Design State Lineage

Use `study.design()` to see the full lineage:

```python
report = study.design()
print(report)

# Design Report (2 factors)
#   Design lineage:
#     Planned Design State:    DS 1 (Full Replication)
#     Observed Design State:   DS 6 (Incomplete, With Singletons) — 3 empty cells
#     Analytical Design State: DS 1 (Full Replication)
#   ...
```

When ODS and ADS differ, the Study display shows both:

```python
print(study)
# Study(response='weight', factors=[lane], time='pull', ods=6, ads=1)
#   Valid: Histogram, Xbar, S, XmR, R | Recommended: Xbar
```

### Key Properties

```python
# Plan Design State (None if no plan was provided)
study.plan_design_state       # SDSResult or None

# Observed Design State (always available)
study.observed_design_state   # SDSResult — diagnostic/lineage

# Analytical Design State (drives analysis)
study.analytical_design_state # SDSResult — the authoritative state
study.ads_reason              # e.g., "full_replication" (machine-readable)
study.ads_description         # e.g., "Full replication (all cells n>=2)"
```

## Bishop's Variance Analysis System (VAS)

For replicated designs (DS 1-3), ProcessBehavior computes five residual decompositions:

| Residual | Formula | Questions Answered |
|----------|---------|-------------------|
| **R1** | Y - Y&#x0304; | Total deviation from grand mean |
| **R2** | Y - Y&#x0304;<sub>kt</sub> | Within-cell variation (unexplained) |
| **R3** | Y - Y&#x0304;<sub>k</sub> - Y&#x0304;<sub>t</sub> + Y&#x0304; | Factor-time interaction |
| **R4** | Y&#x0304;<sub>t</sub> - Y&#x0304; + R2 | Time effects + unexplained |
| **R5** | Y&#x0304;<sub>k</sub> - Y&#x0304; + R2 | Factor effects + unexplained |

### Interpreting VAS Residuals

- **R2 (Within-cell)**: Is measurement variation stable? Are there special causes within subgroups?
- **R3 (Interaction)**: Does the factor effect change over time?
- **R4 (Time)**: Are there trends, shifts, or time-related patterns?
- **R5 (Factor)**: Do factors differ significantly from each other?

```python
# Access residuals after formulation
print(study.dataset[['R1', 'R2', 'R3', 'R4', 'R5']].head())

# Chart residuals using the value parameter
result = study.execute(chart='XmR', by=['lane'], value='R4')
result.plot()
```

## Chart Types

### Standard Charts

| Chart | Use Case | Requirements |
|-------|----------|--------------|
| **Xbar** | Compare subgroup means | n >= 2 per subgroup |
| **S** | Monitor subgroup variation | n >= 2 per subgroup |
| **XmR** | Individual measurements over time | Any structure |
| **R** | Range of subgroups | n >= 2 per subgroup |

### Residual Charts

Use the `value` parameter to chart residuals instead of the response variable:

```python
# Chart R5 residuals (factor effects) on an Xbar chart
result = study.execute(chart='Xbar', value='R5')

# Chart R4 residuals (time effects) on a stratified XmR chart
result = study.execute(chart='XmR', by=['lane'], value='R4')
```

| Residual | Chart Type | Purpose |
|----------|------------|---------|
| **R2** | S or XmR | Within-group variation stability |
| **R3** | XmR | Detect factor-time interactions |
| **R4** | XmR | Detect time effects |
| **R5** | Xbar or XmR | Detect factor effects |

## The `by` Parameter

The `by` parameter controls how data is grouped or stratified:

```python
# Xbar chart aggregated by all factors (default)
result = study.execute(chart='Xbar')

# Xbar chart aggregated by single factor
result = study.execute(chart='Xbar', by=['factor 1'])

# Xbar chart collapsed to grand mean
result = study.execute(chart='Xbar', by=[])

# XmR chart stratified by factor (separate chart per level)
result = study.execute(chart='XmR', by=['lane'])
```

**Key concept**: The `by` parameter creates *views* over the same underlying data. Residuals are computed once during formulation and never change regardless of how you view them.

## Western Electric Rules

Signal detection uses the Western Electric (WECO) rules:

| Rule | Name | Description |
|------|------|-------------|
| 1 | Beyond Limits | Point outside 3-sigma limits |
| 2 | Zone A | 2 of 3 consecutive in Zone A (2-3 sigma) |
| 3 | Zone B | 4 of 5 consecutive in Zone B or beyond |
| 4 | Run | 8+ consecutive same side of centerline |
| 5 | Trend | 6+ consecutive increasing or decreasing |
| 6 | Oscillation | 14+ consecutive alternating up/down |
| 7 | Hugging Center | 15+ consecutive in Zone C (within 1 sigma) |
| 8 | Avoiding Center | 8+ consecutive avoiding Zone C |

### Rule Applicability

- **Xbar/S charts**: Only Rule 1 applies (beyond limits)
- **XmR charts**: All 8 rules apply

```python
# Standard rules (1-4)
signals = result.detect_signals(rules='standard')

# Extended rules (1-8)
signals = result.detect_signals(rules='extended')
```

## Terminology Mapping

ProcessBehavior uses Wheeler's terminology consistently:

| Wheeler Term | Common Term | ProcessBehavior |
|--------------|-------------|-----------------|
| Response Variable | Y, measurement | `response` |
| Rational Subgroup | Factor, group | `factors` |
| Time Sequence | Period, batch | `time` |
| Design State | - | `observed_design_state` / `analytical_design_state` |
| Process Behavior Chart | Control Chart | Chart |

## Philosophy

ProcessBehavior follows Wheeler's philosophy:

1. **Charts are for understanding, not control** - The goal is insight into variation, not just limit checking.

2. **Let the data speak** - Automatic DS detection ensures appropriate analysis.

3. **Separate formulation from analysis** - Understanding your data structure comes first.

4. **DataFrame-backed results** - Access chart data, residuals, and effects as standard pandas DataFrames.

## Next Steps

- [Basic XmR Chart](../tutorials/basic-imr.ipynb) - Create your first XmR chart
- [Design States](../user-guide/sds-detection.md) - Deep dive into DS
- [VAS Residuals](../user-guide/residuals.md) - Understanding VAS residuals
