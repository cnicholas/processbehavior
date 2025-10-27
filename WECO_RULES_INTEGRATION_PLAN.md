# Western Electric Rules Integration Plan

## Executive Summary

**Status**: WECO rules are **fully implemented and tested** but **not fully integrated** into the ProcessBehavior workflow.

**Current State**: Option 3 (Declarative Rule Engine with Fluent API) has been implemented with excellent design quality. The framework exists in `processbehavior/signals/` with:
- ✅ Complete implementation of 8 Western Electric rules
- ✅ Fluent API (RuleSet) and declarative config (SignalConfig)
- ✅ Vectorized detection functions for performance
- ✅ Rich result container (SignalResult) with multiple access patterns
- ✅ Integration method in AnalysisResult (`detect_signals()`)
- ✅ Comprehensive test suite

**Integration Gaps Identified**:
1. **Value column mapping inconsistency** - SignalDetector defaults to `value_col='mean'` but charts use different columns
2. **No automatic signal detection** - Users must manually call `detect_signals()`
3. **Plotting integration incomplete** - Signals not visualized differently from basic `beyond_limits`
4. **Excel export gap** - Signal detection results not included in exports
5. **Limited to Rule 1** - Only `beyond_limits` (Rule 1) is currently applied during chart calculation

---

## Architectural Context

### Current Architecture (from ARCHITECTURE_ASSESSMENT_2025.md)

The ProcessBehavior system follows this flow:

```
ProcessDataFrame
    ↓
SamplingDesignDetector (identifies SDS 0-6)
    ↓
AnalysisSpecification (validates parameters)
    ↓
AnalysisDataSet → Analysis (calculates charts)
    ↓
AnalysisResult (contains charts, residuals, effects)
    ↓
Plotter (visualizes results)
```

### WECO Rules Current Implementation

The signal detection framework is a **separate subsystem** in `processbehavior/signals/`:

```
signals/
├── __init__.py          # Public API exports
├── config.py            # SignalConfig, RuleSet, ZoneDefinition
├── detector.py          # SignalDetector (orchestrator)
├── detectors.py         # Vectorized rule detection functions
└── result.py            # SignalResult (result container)
```

**Integration Point**: `AnalysisResult.detect_signals()` method (lines 1306-1439 in analysis_result.py)

---

## Integration Gaps Analysis

### Gap 1: Value Column Mapping Inconsistency ⚠️ **CRITICAL**

**Problem**: SignalDetector uses `value_col='mean'` as default, but charts use different columns:

| Chart Type | Value Column | Current Default | Status |
|------------|--------------|----------------|---------|
| Xbar | `xbar` | `mean` ❌ | **MISMATCH** |
| Sbar | `s` | `mean` ❌ | **MISMATCH** |
| R | `mr` | `mean` ❌ | **MISMATCH** |
| IMR | `{response_var}` | `mean` ❌ | **MISMATCH** |

**Impact**: `detect_signals()` will fail because the expected column doesn't exist in chart data.

**Evidence**:
```python
# From detector.py:69-75
def detect(
    self,
    data: pd.DataFrame,
    stats: dict,
    config: Optional[SignalConfig] = None,
    value_col: str = 'mean',  # ❌ Wrong default
    chart_name: str = 'Chart'
) -> SignalResult:
```

**Solution**: AnalysisResult must use Plotter's `_get_value_column()` logic to determine correct column.

---

### Gap 2: No Automatic Signal Detection

**Current Behavior**:
- Chart calculation applies **only Rule 1** (beyond_limits) via `_add_beyond_limits_flag()`
- Other 7 rules require manual `detect_signals()` call
- Users must know to call this method

**Implications**:
- Inconsistent UX - some signals detected automatically, others require opt-in
- `beyond_limits` column already in chart data but doesn't use SignalDetector framework
- Duplication: `detect_beyond_limits()` in spc_constants.py vs Rule 1 in signals/

**Design Question**: Should signal detection be automatic or opt-in?

**Recommendation**: Keep opt-in but improve discoverability
- ✅ Professional users want control over which rules to apply
- ✅ Different industries use different rule sets
- ✅ Performance consideration for large datasets
- ✅ Configuration complexity (zones, thresholds)

**BUT**: Improve integration:
1. Add `detect_signals` parameter to `analyze()` method
2. Consolidate `beyond_limits` to use SignalDetector framework
3. Better documentation and examples

---

### Gap 3: Plotting Integration Incomplete

**Current State**:
```python
# Plotter.plot() has highlight_signals parameter
def plot(self, highlight_signals: bool = True, ...):
```

**Problem**: `highlight_signals` only shows `beyond_limits` column (Rule 1), not comprehensive signal detection.

**Evidence from plotter.py:272-287**:
```python
# Highlight signals
if highlight_signals and 'beyond_limits' in data.columns:
    signals = data[data['beyond_limits'] != 0]
    if not signals.empty:
        fig.add_trace(go.Scatter(
            x=signals[x_col],
            y=signals[value_col],
            mode='markers',
            name='Signal',
            marker=dict(size=14, color='red', symbol='x')
        ))
```

**Gap**: No integration with SignalResult object.

**Desired Behavior**:
```python
# User workflow
signals = result.detect_signals(rules='extended')
fig = result.plot(highlight_signals=signals)  # ❌ Not supported
```

**Solution**: Extend `Plotter.plot()` to accept SignalResult and visualize all violations.

---

### Gap 4: Excel Export Missing Signal Information

**Current State**: `AnalysisResult.to_excel()` exports:
- ✅ Summary sheet
- ✅ Chart data sheets
- ✅ Residuals (if available)
- ✅ Effects (if available)
- ❌ Signal detection results (NOT included)

**User Need**: QA reports should include violation tracking.

**Solution**: Add signals sheet if `detect_signals()` was called.

---

### Gap 5: Value Column Determination Logic

**Current Situation**: Two separate implementations of the same logic:

1. **Plotter** (`processbehavior/plotting/plotter.py:417-496`):
   - `_get_value_column()` method determines plotting column
   - Uses chart_name and analysis_type to select column
   - Declarative, explicit, NO FALLBACK

2. **SignalDetector** (`processbehavior/signals/detector.py:69-76`):
   - Accepts `value_col` parameter with default `'mean'`
   - NO logic to determine correct column
   - Caller must provide correct column name

**Problem**: AnalysisResult.detect_signals() doesn't use Plotter's logic.

**Evidence from analysis_result.py:1420-1426**:
```python
chart_info = self.charts[chart]
return detector.detect(
    data=chart_info['data'],
    stats=chart_info['statistics'],
    config=config,
    chart_name=chart
    # ❌ Missing: value_col parameter!
)
```

**Impact**: SignalDetector will use default `'mean'` which doesn't exist in chart data → **FAILURE**.

---

## Root Cause Analysis

### Why Integration is Incomplete

1. **Phase 2 Column Standardization** (completed recently):
   - Charts now use declarative column contract: 'xbar', 's', 'mr', response_var
   - SignalDetector still assumes generic 'mean' column
   - **Timing issue**: WECO implementation predates column standardization

2. **Separate Development**:
   - WECO rules developed as standalone framework
   - Chart calculation uses legacy `detect_beyond_limits()`
   - Integration method added but not tested end-to-end

3. **Missing Bridge Code**:
   - No shared logic between Plotter and SignalDetector for column mapping
   - Each component solves same problem independently

---

## Integration Plan

### Principle: Follow Architectural Patterns

From ARCHITECTURE_ASSESSMENT_2025.md:
- ✅ **Specification-Driven**: Configuration objects define behavior
- ✅ **Declarative**: Explicit over implicit
- ✅ **Composition**: Delegate to focused components
- ✅ **Consistent Naming**: Use Phase 2 column standards

### Phase 1: Fix Critical Value Column Bug 🔴 **HIGH PRIORITY**

**Goal**: Make `detect_signals()` work correctly with current charts.

**Tasks**:

1. **Share column resolution logic**
   ```python
   # Option A: Move _get_value_column to shared utility
   # processbehavior/chart_utils.py (new file)
   def get_chart_value_column(
       chart_name: str,
       data: pd.DataFrame,
       analysis_type: Optional[str] = None,
       response_var: Optional[str] = None
   ) -> str:
       """
       Determine the value column for a chart.

       Uses declarative contract from Phase 2:
       - Xbar charts: 'xbar'
       - S charts: 's'
       - R charts: 'mr'
       - IMR charts: response_var
       """
       # Move Plotter._get_value_column() logic here
   ```

2. **Update AnalysisResult.detect_signals()**
   ```python
   # analysis_result.py:1420-1426
   chart_info = self.charts[chart]
   value_col = get_chart_value_column(
       chart_name=chart,
       data=chart_info['data'],
       analysis_type=self.summary.get('analysis_type'),
       response_var=self.summary.get('response_var')
   )

   return detector.detect(
       data=chart_info['data'],
       stats=chart_info['statistics'],
       config=config,
       value_col=value_col,  # ✅ Correct column
       chart_name=chart
   )
   ```

3. **Update Plotter to use shared logic**
   ```python
   # plotter.py:417
   def _get_value_column(self, data: pd.DataFrame, chart_name: str) -> str:
       """Determine value column using shared logic."""
       return get_chart_value_column(
           chart_name=chart_name,
           data=data,
           analysis_type=self.summary.get('analysis_type'),
           response_var=self.summary.get('response_var')
       )
   ```

**Validation**:
- Run existing tests in `tests/test_signals.py`
- Add integration test: analyze → detect_signals → verify violations

---

### Phase 2: Consolidate Beyond Limits Detection

**Goal**: Remove duplication between `detect_beyond_limits()` and SignalDetector Rule 1.

**Current Duplication**:
1. `spc_constants.py:detect_beyond_limits()` - Used during chart calculation
2. `signals/detectors.py:detect_beyond_limits()` - Rule 1 implementation

**Decision**: Keep both but align behavior.

**Why Keep Both**:
- `spc_constants.py` version is simple, lightweight, used in hot path
- `signals/` version is part of comprehensive framework
- Different use cases: inline flagging vs. comprehensive detection

**Alignment Tasks**:
1. Ensure both return same results for same input
2. Add cross-reference in docstrings
3. Consider: `spc_constants.detect_beyond_limits()` calls `signals.detect_beyond_limits()`?

---

### Phase 3: Enhance Plotting Integration

**Goal**: Enable visualization of comprehensive signal detection results.

**API Design**:
```python
# Current (only shows beyond_limits)
fig = result.plot(highlight_signals=True)

# Enhanced (accepts SignalResult)
signals = result.detect_signals(rules='extended')
fig = result.plot(highlight_signals=signals)

# Backward compatibility
fig = result.plot(highlight_signals=True)  # Still works (uses beyond_limits)
```

**Implementation**:

1. **Update Plotter.plot() signature**:
   ```python
   def plot(
       self,
       chart: Optional[str] = None,
       highlight_signals: bool | SignalResult = True,  # ✅ Accept SignalResult
       ...
   ):
   ```

2. **Add signal highlighting logic**:
   ```python
   # In _plot_single_chart
   if highlight_signals:
       if isinstance(highlight_signals, bool):
           # Legacy behavior: use beyond_limits column
           if 'beyond_limits' in data.columns:
               signals = data[data['beyond_limits'] != 0]
               # ... existing code ...
       else:
           # New behavior: use SignalResult
           signal_result = highlight_signals
           flagged_ids = signal_result.flagged_observations
           signals = data[data.index.isin(flagged_ids)]

           # Color by rule type?
           # Add hover text with rule descriptions?
   ```

3. **Visual enhancements**:
   - Different markers for different rule types?
   - Color coding by severity?
   - Hover text showing which rules violated?
   - Zone visualization (A/B/C boundaries)?

**Open Design Questions**:
1. Should we show zone boundaries when extended rules enabled?
2. Different visual treatment for different rule types?
3. Annotation with rule names?

---

### Phase 4: Excel Export Integration

**Goal**: Include signal detection results in Excel exports.

**Implementation**:

1. **Detect if signals were calculated**:
   ```python
   # AnalysisResult stores last signal detection result
   class AnalysisResult:
       def __init__(self, ...):
           self._signal_results = None

       def detect_signals(self, ...):
           results = detector.detect(...)
           self._signal_results = results  # Cache
           return results
   ```

2. **Add signals sheet in to_excel()**:
   ```python
   def to_excel(self, filepath: str, include_signals: bool = True):
       """Export to Excel."""
       # ... existing code ...

       # Add signals if available
       if include_signals and self._signal_results:
           if isinstance(self._signal_results, dict):
               # Multiple charts
               for chart_name, signal_result in self._signal_results.items():
                   sheet_name = f'Signals_{chart_name}'
                   signal_result.violations.to_excel(
                       writer,
                       sheet_name=sheet_name,
                       index=False
                   )
           else:
               # Single chart
               self._signal_results.violations.to_excel(
                   writer,
                   sheet_name='Signals',
                   index=False
               )
   ```

3. **Add summary statistics**:
   - Total violations by rule
   - Flagged observation count
   - Control status summary

---

### Phase 5: Optional Automatic Detection

**Goal**: Make signal detection easier to discover and use.

**API Enhancement**:
```python
# Add detect_signals parameter to analyze()
pdf = ProcessDataFrame('data.csv')
result = pdf.analyze(
    response_var='measurement',
    factors=['operator'],
    detect_signals='standard'  # ✅ Automatic detection
)

# Signals already detected
print(result.signals.summary)
fig = result.plot(highlight_signals=result.signals)
```

**Implementation**:
```python
# In process_dataframe.py
def analyze(
    self,
    response_var: str,
    factors: Optional[List[str]] = None,
    time: Optional[str] = None,
    detect_signals: Optional[str | List[str] | SignalConfig] = None,  # ✅ New
    ...
) -> AnalysisResult:
    """Analyze process behavior data."""

    # ... existing analysis code ...

    # Automatic signal detection if requested
    if detect_signals is not None:
        result._signal_results = result.detect_signals(rules=detect_signals)

    return result
```

**Benefits**:
- Opt-in: defaults to None (no change in behavior)
- Discoverable: shows up in analyze() signature
- Flexible: accepts all signal configuration formats

---

## Testing Strategy

### Unit Tests (Existing ✅)
- `tests/test_signals.py` - Comprehensive test suite exists
- Tests configuration, detection, results
- All passing

### Integration Tests (NEW - Required)

1. **End-to-End Signal Detection**:
   ```python
   def test_detect_signals_integration():
       """Test complete workflow: analyze → detect_signals → verify."""
       pdf = ProcessDataFrame('test_data.csv')
       result = pdf.analyze(response_var='measurement', factors=['operator'])

       # Detect signals
       signals = result.detect_signals(rules='standard')

       # Verify correct value columns used
       assert signals.count >= 0  # Should not crash

       # Verify violations are traceable
       if signals.has_signals:
           flagged = signals.flagged_observations
           assert len(flagged) > 0
   ```

2. **Value Column Mapping**:
   ```python
   def test_value_column_mapping_xbar():
       """Verify Xbar chart uses 'xbar' column."""
       result = analyze_xbar_data()
       signals = result.detect_signals(chart='Xbar')
       # Should use 'xbar' column, not crash looking for 'mean'

   def test_value_column_mapping_imr():
       """Verify IMR chart uses response_var column."""
       result = analyze_imr_data()
       signals = result.detect_signals(chart='Imr')
       # Should use response variable column
   ```

3. **Plotting Integration**:
   ```python
   def test_plot_with_signal_result():
       """Test plotting with SignalResult highlighting."""
       result = analyze_test_data()
       signals = result.detect_signals(rules='extended')

       fig = result.plot(highlight_signals=signals)

       # Verify figure created without errors
       assert fig is not None
   ```

---

## Open Questions

1. **Zone Visualization**:
   - Should we add zone boundary lines (A/B/C) when extended rules used?
   - Would this clutter the chart or provide value?

2. **Rule-Specific Markers**:
   - Different markers for different rule types?
   - Color coding by rule severity?
   - Or keep simple uniform "signal" marker?

3. **Performance**:
   - Should signal detection be lazy (only when accessed)?
   - Cache signal results per chart?
   - Invalidation strategy if chart data changes?

4. **Backward Compatibility**:
   - `beyond_limits` column still needed for compatibility?
   - How to deprecate gracefully?
   - Timeline for migration?

5. **Documentation**:
   - Where to document WECO rules usage?
   - Tutorial notebook needed?
   - Integration examples in README?

---

## Success Criteria

### Phase 1 Complete When:
- ✅ `result.detect_signals()` works for all chart types (Xbar, Sbar, IMR, R)
- ✅ Correct value columns used automatically
- ✅ Integration tests passing
- ✅ No breaking changes to existing API

### Phase 2 Complete When:
- ✅ `beyond_limits` behavior consistent between implementations
- ✅ Documentation clarifies relationship
- ✅ Unit tests verify alignment

### Phase 3 Complete When:
- ✅ `plot(highlight_signals=SignalResult)` works
- ✅ Backward compatibility maintained
- ✅ Visual distinction between simple and extended signals
- ✅ Examples in documentation

### Phase 4 Complete When:
- ✅ `to_excel()` includes signals sheet
- ✅ Summary statistics included
- ✅ Opt-in/opt-out parameter works

### Phase 5 Complete When:
- ✅ `analyze(detect_signals='standard')` works
- ✅ Discoverability improved
- ✅ Documentation updated
- ✅ Tutorial notebook created

---

## Implementation Priority

### 🔴 **Critical (Do First)**
- **Phase 1**: Fix value column mapping bug
  - Without this, detect_signals() doesn't work
  - Blocks all other integration work

### 🟡 **High Priority**
- **Phase 3**: Plotting integration
  - Provides immediate value
  - Visual feedback for users
  - Completes the analysis → detection → visualization loop

### 🟢 **Medium Priority**
- **Phase 4**: Excel export
  - Nice-to-have for reporting
  - Not blocking core functionality

### 🔵 **Optional Enhancement**
- **Phase 5**: Automatic detection
  - Convenience feature
  - Can be added anytime after Phase 1

### 🟣 **Maintenance**
- **Phase 2**: Consolidate beyond_limits
  - Code quality improvement
  - No user-facing impact

---

## Architectural Consistency Checklist

Ensuring integration follows established patterns:

- ✅ **Specification-Driven**: SignalConfig defines behavior
- ✅ **Declarative**: Explicit column mapping, no inference
- ✅ **Composition**: SignalDetector as separate component
- ✅ **Progressive Disclosure**: Simple → Fluent API → Full Config
- ✅ **Observable Results**: SignalResult provides multiple views
- ✅ **Fail Helpful**: Clear error messages for missing columns
- ✅ **Consistent Naming**: Uses Phase 2 column standards
- ✅ **Type Safety**: Full type hints throughout

---

## Conclusion

The Western Electric Rules implementation is **excellent in design and quality** but requires **integration work** to be fully usable within the ProcessBehavior workflow.

**Key Insight**: This is not a "the rules don't work" problem. This is a "the glue code needs attention" problem.

**Critical Path**: Fix Phase 1 (value column mapping) first. Everything else is enhancement.

**Estimated Effort**:
- Phase 1 (Critical): 4-6 hours
- Phase 2 (Maintenance): 2-3 hours
- Phase 3 (High Priority): 6-8 hours
- Phase 4 (Medium Priority): 3-4 hours
- Phase 5 (Optional): 4-5 hours

**Total**: ~20-26 hours for complete integration

**Recommendation**: Implement Phase 1 immediately to unblock users, then Phase 3 for complete UX. Phases 2, 4, 5 can be done incrementally.
