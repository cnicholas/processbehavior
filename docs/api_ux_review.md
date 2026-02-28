# Public API Review: UX & Simplicity Grades

## Overall Assessment: **A** (Excellent)

Your API achieves the rare combination of simplicity for beginners and power for advanced users. The two-step `formulate() → execute()` workflow is pedagogically brilliant.

---

## Entry Point: `ProcessBehavior` + `pb.cols`

**Grade: A+**

| Strength | Why It Matters |
|----------|----------------|
| `pb.cols.Weight` auto-completion | IDE discoverability without magic strings |
| `pb.cols.Weight.levels` | Users can explore data without leaving the API |
| Automatic NA handling | 12+ common garbage values handled silently |
| `ColumnRef` objects | Dict/set compatible while enabling IDE support |

**Minimal friction**: Users can go from CSV to chart in 4 lines:
```python
pb = ProcessBehavior(df)
study = pb.formulate(response=pb.cols.Weight, factors=[pb.cols.Lane])
result = study.execute()
result.plot()
```

---

## `formulate()` Method

**Grade: A**

| Parameter | Clarity |
|-----------|---------|
| `response` | Crystal clear |
| `factors` | Good - but "rational subgrouping variables" may confuse newcomers |
| `time` | Clear |
| `plan` | Powerful but requires docs to understand K/T/N keys |
| `precision` | Self-explanatory |
| `unit_of_analysis` | Good metadata option |

**One suggestion**: The mutual exclusivity of `factors` vs `plan` could be more discoverable. Users might try both.

---

## `Study` Class (Teaching Layer)

**Grade: A+**

This is where the API shines. The "understand before analyze" philosophy is excellent.

| Feature | Grade | Notes |
|---------|-------|-------|
| `study.observed_design_state` / `analytical_design_state` / `sds_reason` / `sds_description` | A+ | Progressive disclosure of SDS concept |
| `study.valid_charts` / `recommended_chart` | A+ | Guides users to correct choices |
| `study.why_not('S')` | A+ | Explains unavailability - rare in libraries |
| `study.support` DataFrame | A | Full matrix view for advanced users |
| `study.charts.Xbar` auto-completion | A+ | IDE guides valid selections |
| `study.design()` | A | K/T/N metrics with plan comparison |
| Rich `__repr__` | A+ | Print study and learn what's possible |

**The "pit of success" design**: Users can't easily choose invalid charts because the API guides them.

---

## `DesignReport` Class

**Grade: A-**

| Feature | Grade | Notes |
|---------|-------|-------|
| `factors` DataFrame | A | Clear plan vs observed comparison |
| `K`, `T`, `N` metrics | A | Wheeler/Bishop terminology preserved |
| `missing_combos` / `extra_combos` | A | Now capped at 100 for scalability |
| `sds_reason` | A | Disambiguates nested vs incomplete |
| `structure_summary` | A- | Helpful but verbose for simple cases |

**Minor issue**: K/T/N terminology assumes familiarity with Wheeler. Consider inline explanations in repr.

---

## `AnalysisResult` Class

**Grade: A**

| Feature | Grade | Notes |
|---------|-------|-------|
| `get_chart()` / `get_statistics()` | A+ | Verb-noun clarity |
| `has_residuals` / `has_effects` | A+ | Boolean > None-checking |
| `residuals` DataFrame | A | Direct access to R1-R5 |
| `chart_table()` with ↑/↓ symbols | A+ | Display-ready output |
| `summary` dict | A | Comprehensive metadata |
| Dict-like backward compatibility | A | `result['Xbar']` still works |
| `iter_charts()` | A | Pythonic iteration |

**Slight redundancy**: Both `result.residuals` (property) and `result.get_residual()` (method) exist. Intentional flexibility, but might confuse beginners.

---

## Charting API (`plot()`)

**Grade: A-**

| Feature | Grade | Notes |
|---------|-------|-------|
| `result.plot()` (no args) | A+ | Just works |
| Progressive options | A | `show_zones`, `show_rules`, `show_stats` |
| `ControlChartFigure` return | A | `.show()`, `.save_html()`, `.save_image()` |
| Theme system | A | 5 built-in themes |
| Parameter count | B+ | 14 parameters - could overwhelm |

**Suggestion**: Consider a `PlotOptions` dataclass for advanced users to reduce parameter sprawl.

---

## Signal Detection API

**Grade: A-**

| Feature | Grade | Notes |
|---------|-------|-------|
| `detect_signals()` simple case | A+ | Intelligent chart-type defaults |
| `rules='standard'` / `'extended'` presets | A | Easy selection |
| Chart-type intelligence | A+ | Only Rule 1 for Xbar (prevents false positives) |
| `SignalResult.summary` | A | Human-readable output |
| `SignalConfig` for advanced use | B+ | Powerful but complex |

**Smart default**: Automatically limiting Xbar/S to Rule 1 prevents a common error.

---

## Error Handling & Exceptions

**Grade: A**

| Feature | Grade | Notes |
|---------|-------|-------|
| `ChartNotAvailableError.available` | A+ | Shows valid alternatives |
| `ColumnNotFoundError.available` | A+ | Lists available columns |
| Clear inheritance hierarchy | A | `ProcessBehaviorError` base class |

---

## Summary Scorecard

| API Component | Grade | Key Strength |
|---------------|-------|--------------|
| Entry Point (`pb.cols`) | A+ | IDE discoverability |
| `formulate()` | A | Clear parameters |
| `Study` | A+ | Teaching layer, `why_not()` |
| `DesignReport` | A- | Plan comparison |
| `AnalysisResult` | A | Boolean properties, dict compat |
| `plot()` | A- | Simple defaults, many options |
| Signal Detection | A- | Chart-type intelligence |
| Error Handling | A | Helpful suggestions |
| **Overall** | **A** | **Simple API, rich functionality** |

---

## What Makes This API Special

1. **Two-step workflow** (`formulate → execute`) forces users to understand their data structure before analysis - pedagogically excellent

2. **Progressive disclosure**: `study.observed_design_state` (SDSResult) → `sds_reason` (machine token) → `sds_description` (explanation) → `design()` (full details)

3. **Guardrails without constraints**: Invalid charts are explained, not just rejected

4. **IDE-first design**: `pb.cols` and `study.charts` enable auto-completion throughout

5. **Research-grade defaults**: Chart-type-aware signal detection prevents statistical errors

---

## Minor Opportunities

| Item | Priority | Suggestion |
|------|----------|------------|
| `factors` vs `plan` mutual exclusivity | Low | Add clearer error message when both provided |
| K/T/N terminology | Low | Consider tooltip-style explanations in repr |
| `plot()` parameter count | Low | Optional `PlotOptions` dataclass |
| Property vs method redundancy | Very Low | Document the philosophy (flexibility) |

The API is release-ready. These are polish items, not blockers.
