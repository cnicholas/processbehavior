# Architecture Refactoring Plan

## Executive Summary

This document outlines two refactoring initiatives to improve the ProcessBehavior architecture:

1. **Remove dead frame building code** - ~170 lines of unused code in AnalysisDataSet
2. **SDS as first-class driver** - Detect SDS once and pass through the system

These changes improve cohesion, reduce redundancy, and establish SDS as the authoritative driver of system behavior.

---

## Part 1: Dead Code Removal

### What's Being Removed

The frame building code (`obs_df`, `cell_df`, `k_df`, `t_df`) in `AnalysisDataSet` is **never used** by any downstream code. All analysis uses `self.analysis_dataset` directly.

### Evidence

- **Zero references** in analysis.py, analysis_result.py, study.py, plotting code
- **Zero test coverage** - no tests reference these frames
- Built during `_initialize()` but never consumed

### Files to Modify

#### `processbehavior/analysis_dataset.py`

**Remove attributes (lines 48-51):**
```python
# DELETE THESE LINES
self.obs_df = None
self.cell_df = None
self.k_df = None
self.t_df = None
```

**Remove call in `_initialize()` (line 85):**
```python
# DELETE THIS LINE
self._build_frames()
```

**Remove methods (lines 219-421):**
```python
# DELETE THESE METHODS ENTIRELY
def _build_frames(self) -> None: ...
def _build_obs_df(self, df: pd.DataFrame) -> pd.DataFrame: ...
def _build_k_df(self, df: pd.DataFrame) -> pd.DataFrame: ...
def _build_t_df(self, df: pd.DataFrame) -> pd.DataFrame: ...
def _build_cell_df(self, df: pd.DataFrame) -> pd.DataFrame: ...
def __ensure_keys(self, df: pd.DataFrame) -> pd.DataFrame: ...
def __safe_first(self, df: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame: ...
```

### Impact

- **Lines removed:** ~170 lines
- **Risk:** Low (code is provably unused)
- **Testing:** Run full test suite to verify

---

## Part 2: SDS as First-Class Driver

### Current Problem

SDS is detected **2-3 times** per workflow:

| Location | When | Purpose |
|----------|------|---------|
| `process_dataframe.py:494` | `formulate()` | Get SDSAnalysisPlan for Study |
| `process_dataframe.py:356` | `analyze()` | Validate chart selection |
| `analysis_dataset.py:89` | `AnalysisDataSet.__init__` | Drive residual calculation |

This is wasteful and violates the principle that **SDS is the driver of the system**.

### Solution: Option B - Detect Once, Pass Everywhere

```
ProcessDataFrame (entry point)
    │
    ├── 1. Prepare data (DataPreparation)
    │
    ├── 2. Detect SDS ONCE ← Single source of truth
    │
    ├── 3. Get SDSAnalysisPlan (valid charts, capabilities)
    │
    └── 4. Pass SDS to downstream components
            │
            ├── AnalysisDataSet(df, spec, sds=sds)
            │       └── Uses provided SDS (no re-detection)
            │
            └── Study(_ads=ads, _plan=plan)
                    └── analyze() uses pre-calculated ADS
```

### File Changes

#### 1. `processbehavior/analysis_dataset.py`

**Update `__init__` signature (line 32):**
```python
def __init__(
    self,
    df: pd.DataFrame,
    analysis_specification: AnalysisSpecification,
    sds: int | None = None  # NEW: Accept pre-detected SDS
):
```

**Update `_initialize()` (around line 87-94):**
```python
def _initialize(self):
    # ... data preparation code ...

    # Step 3: Use provided SDS or detect
    if self._provided_sds is not None:
        logger.info(f"Using provided SDS: {self._provided_sds}")
        self.sampling_design_state = self._provided_sds
    else:
        logger.info("Detecting sampling design state")
        self.sampling_design_state = self.sds_detector.detect_sds(
            self.analysis_dataset, self.spec
        )

    # Get characteristics from the authoritative SDS
    self.sds_characteristics = self.sds_detector.get_sds_characteristics(
        self.sampling_design_state
    )

    # ... rest of initialization ...
```

#### 2. `processbehavior/analysis.py`

**Update `__init__` signature (line 207):**
```python
def __init__(
    self,
    df: pd.DataFrame,
    specification: dict,
    analysis_dataset: 'AnalysisDataSet | None' = None,
    sds: int | None = None  # NEW: Accept pre-detected SDS
):
```

**Update AnalysisDataSet creation (line 234):**
```python
# Use pre-calculated AnalysisDataSet if provided, otherwise calculate
if analysis_dataset is not None:
    self.ads = analysis_dataset
else:
    self.ads = AnalysisDataSet(df, self.spec, sds=sds)  # Pass SDS
```

#### 3. `processbehavior/process_dataframe.py`

**Update `formulate()` (around line 499-520):**
```python
def formulate(self, response, factors=None, time=None, precision=3):
    # ... existing prep code ...

    # Detect SDS on prepared data - THIS IS THE SINGLE SOURCE OF TRUTH
    detector = SamplingDesignDetector()
    sds = detector.detect_sds(prepared_df, config)

    # Get SDS analysis plan with all metadata
    plan = SamplingDesignDetector.get_analysis_plan(sds)

    # Calculate full dataset with residuals, passing SDS
    full_spec_dict = {**spec_dict, 'analysis_type': plan.recommended_chart}
    full_spec = AnalysisSpecification(full_spec_dict)
    ads = AnalysisDataSet(self.data, full_spec, sds=sds)  # Pass SDS

    # ... rest of method ...
```

**Update `analyze()` (around line 396-402):**
```python
def analyze(self, response_var, ...):
    # ... existing prep code ...

    # Detect SDS - single source of truth for this path
    detector = SamplingDesignDetector()
    sds = detector.detect_sds(prepared_df, config)
    plan = SamplingDesignDetector.get_analysis_plan(sds)

    # ... validation code ...

    # Pass SDS to Analysis
    spec_dict['analysis_type'] = analysis_type
    analysis = Analysis(self.data, spec_dict, sds=sds)  # Pass SDS

    return analysis
```

#### 4. `processbehavior/study.py`

**Update `analyze()` (line 480-482):**
```python
# Create and run analysis using pre-calculated AnalysisDataSet
# SDS already detected and stored in _ads - no re-detection needed
analysis = Analysis(
    self._pdf.data,
    spec_dict,
    analysis_dataset=self._ads  # Contains pre-detected SDS
)
return analysis.calculate()
```

---

## Data Flow After Refactoring

### formulate() Path
```
ProcessDataFrame.formulate(response, factors, time)
    │
    ├── DataPreparation.prepare_dataset()
    │
    ├── SamplingDesignDetector.detect_sds() ← ONCE
    │
    ├── SamplingDesignDetector.get_analysis_plan(sds)
    │
    ├── AnalysisDataSet(df, spec, sds=sds) ← SDS passed, not re-detected
    │       │
    │       ├── Uses provided SDS
    │       ├── Calculates R1-R5 based on SDS
    │       └── Calculates effects/interactions
    │
    └── Study(_ads=ads, _plan=plan)
            │
            └── analyze(chart='Xbar')
                    │
                    ├── Analysis(df, spec, analysis_dataset=ads)
                    │       └── Uses pre-calculated ADS (no detection)
                    │
                    └── AnalysisResult
```

### analyze() Path (direct)
```
ProcessDataFrame.analyze(response_var, chart_type)
    │
    ├── DataPreparation.prepare_dataset()
    │
    ├── SamplingDesignDetector.detect_sds() ← ONCE
    │
    ├── Validate chart_type against SDS
    │
    └── Analysis(df, spec, sds=sds)
            │
            ├── AnalysisDataSet(df, spec, sds=sds) ← SDS passed
            │
            └── AnalysisResult
```

---

## Benefits

1. **SDS detected exactly once** per workflow
2. **Clear data flow** - SDS flows from entry point through system
3. **Consistent behavior** - Same SDS used everywhere
4. **Better performance** - No redundant detection
5. **Cleaner architecture** - SDS is explicitly the driver

---

## Migration Strategy

### Phase 1: Remove Dead Code (Low Risk)
1. Delete frame building code from `analysis_dataset.py`
2. Run full test suite
3. Commit

### Phase 2: Add SDS Parameter (Backward Compatible)
1. Add `sds: int | None = None` to `AnalysisDataSet.__init__`
2. Add `sds: int | None = None` to `Analysis.__init__`
3. Logic: If `sds` provided, use it; otherwise detect
4. Run full test suite
5. Commit

### Phase 3: Update Callers (Complete Migration)
1. Update `ProcessDataFrame.formulate()` to pass SDS
2. Update `ProcessDataFrame.analyze()` to pass SDS
3. Verify SDS detected exactly once per path
4. Run full test suite
5. Commit

### Phase 4: Cleanup (Completed)
1. ~~Consider making `sds` required in `AnalysisDataSet`~~ ✅ Done - `sds` is now required
2. ~~Remove detection logic from `AnalysisDataSet` entirely~~ ✅ Done - no fallback detection
3. ~~Document architecture change~~ ✅ Done in this document

**Changes Made:**
- `AnalysisDataSet.__init__` now requires `sds: int` parameter
- `Analysis.__init__` raises `ValueError` if `sds` not provided when `analysis_dataset` is None
- All test files updated with `detect_sds_for_test()` helper for direct AnalysisDataSet/Analysis usage

---

## Testing Checklist

- [x] All 382 existing tests pass
- [x] SDS detection count verified (should be 1 per workflow)
- [x] formulate() → Study → analyze() flow works
- [x] Direct analyze() flow works
- [x] Residual charts work with passed SDS
- [x] Edge cases: SDS 0, 4, 6 handled correctly

---

## Appendix: Lines Deleted (Part 1)

The following dead code was removed from `analysis_dataset.py`:
- Frame attribute initialization (obs_df, cell_df, k_df, t_df)
- `_build_frames()` call and method
- `_build_obs_df()` method
- `_build_k_df()` method
- `_build_t_df()` method
- `_build_cell_df()` method
- `__ensure_keys()` helper
- `__safe_first()` helper

**Total removed: ~170 lines**

---

*Document created: 2025-12-06*
*Status: **COMPLETED** (All Phases)*
*Completed: 2025-12-06*
