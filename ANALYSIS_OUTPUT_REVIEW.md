# Analysis Output Structure Review

## Executive Summary

This document analyzes the current output structure of the processbehavior analysis system, with special focus on the **stratified individuals charts feature** - a powerful capability that creates separate IMR charts for each rational subgroup.

## Current Output Structures

### 1. IMR Charts (Simple - No Grouping)

**Structure:**
```python
{
    'all': {
        'data': DataFrame with columns [Time, response_var, mean, lcl, ucl, beyond_limits],
        'statistics': {
            'mean': float,
            'lcl': float,
            'ucl': float,
            'n': int
        }
    }
}
```

**Example:**
```python
result = {
    'all': {
        'data': pd.DataFrame(...),  # 30 rows
        'statistics': {
            'mean': 99.624,
            'lcl': 94.19,
            'ucl': 105.057,
            'n': 30
        }
    }
}
```

### 2. IMR Charts (Stratified - WITH Grouping) ⭐ KILLER FEATURE

**Structure:**
```python
{
    'Group_A': {
        'data': DataFrame with columns [Time, rsg, response_var, mean, lcl, ucl, beyond_limits],
        'statistics': {
            'mean': float,
            'lcl': float,
            'ucl': float,
            'n': int
        }
    },
    'Group_B': {
        'data': DataFrame(...),
        'statistics': {...}
    },
    ...
}
```

**Example:**
```python
result = {
    'A': {
        'data': pd.DataFrame([...]),  # 10 rows for Group A
        'statistics': {
            'mean': 101.259,
            'lcl': 93.145,
            'ucl': 109.373,
            'n': 10
        }
    },
    'B': {
        'data': pd.DataFrame([...]),  # 10 rows for Group B
        'statistics': {
            'mean': 97.597,
            'lcl': 89.232,
            'ucl': 105.962,
            'n': 10
        }
    },
    'C': {
        'data': pd.DataFrame([...]),  # 10 rows for Group C
        'statistics': {...}
    }
}
```

### 3. Xbar/S Charts

**Structure:**
```python
{
    'Xbar': {
        'data': DataFrame with columns [rsg, mean, Xbar, lcl, ucl, beyond_limits],
        'statistics': {
            'Mean': float,
            'N': int or 'Varies',
            'ucl': float or 'Varies',
            'lcl': float or 'Varies'
        }
    },
    'Sbar': {
        'data': DataFrame with columns [rsg, s, S, lcl, ucl, beyond_limits],
        'statistics': {
            'S': float,
            'N': int or 'Varies',
            'ucl': float or 'Varies',
            'lcl': float or 'Varies'
        }
    }
}
```

## The Stratified Individuals Charts Feature 🎯

### What Is It?

When you run an IMR analysis **with grouping variables**, the system creates **separate individuals charts for each unique combination of grouping variables**. This is incredibly powerful for:

1. **Comparing process behavior across conditions** (operators, machines, shifts)
2. **Detecting which groups are in/out of control**
3. **Stratified analysis** - analyzing variation within strata
4. **Multi-stream monitoring** - tracking multiple parallel processes

### How It Works

**Implementation (analysis_dataset.py:270-399):**

```python
def _calculate_imr(self):
    """Calculate IMR charts, stratified by group if grouping vars present."""

    if spec.has_grouping:
        # 1. Calculate moving range per subgroup
        out['mr'] = out.groupby(spec.rsg_var_name, sort=False)[y].diff().abs()

        # 2. Aggregate statistics per group
        grouped = out.groupby(spec.rsg_var_name).agg(
            mean=(y, 'mean'),
            mR=('mr', 'mean')
        )

        # 3. Calculate limits per group
        lims = grouped.apply(lambda row: calculate_limits(
            mean=row['mean'],
            mR=row['mR'],
            limits_type="Imr"
        ))

        # 4. Split into separate charts per group
        split_dict = split_df_by_group(df=out, grouping_var=spec.rsg_var_name)

        # 5. Package with statistics
        out = package_analysis(
            analysis_output=split_dict,
            summary_statistics_output=statistics
        )
```

**Key Helper Functions:**

1. **`split_df_by_group()`** (line 766)
   - Splits DataFrame into dictionary keyed by group
   - Returns: `{'A': df_A, 'B': df_B, ...}`

2. **`gather_analysis_statistics()`** (line 802)
   - Collects summary statistics per group
   - Returns: `{'A': {...}, 'B': {...}, ...}`

3. **`package_analysis()`** (line 854)
   - Combines data + statistics per group
   - Returns: `{'A': {'data': ..., 'statistics': ...}, ...}`

### Example Use Case

**Scenario:** Manufacturing line with 3 operators

```python
spec = {
    'analysis_type': 'Imr',
    'response_var': 'Height',
    'time_var': 'Time',
    'rsg_vars': ['Operator']
}

analysis = Analysis(df, spec)
result = analysis.calculate()

# Get separate chart for each operator
alice_chart = result['Alice']['data']
bob_chart = result['Bob']['data']
charlie_chart = result['Charlie']['data']

# Compare their means
print(f"Alice mean: {result['Alice']['statistics']['mean']}")
print(f"Bob mean: {result['Bob']['statistics']['mean']}")
print(f"Charlie mean: {result['Charlie']['statistics']['mean']}")

# Detect which operators are out of control
for operator, chart_data in result.items():
    signals = chart_data['data']['beyond_limits'].sum()
    if signals > 0:
        print(f"⚠️ {operator} has {signals} points beyond limits!")
```

## Current Data Accessibility

### What's Currently Available

**1. From Analysis object:**
```python
analysis = Analysis(df, spec)
result = analysis.calculate()  # Chart data + statistics

# Access underlying components
analysis.ads.analysis_dataset      # Full prepared dataset
analysis.ads.sampling_design_state # SDS number
analysis.ads.sds_characteristics   # SDS info
```

**2. From AnalysisDataSet (ads):**
```python
ads = analysis.ads

# Core data
ads.analysis_dataset    # DataFrame with all calculations
ads.raw_dataset        # Original input data

# VAS residuals (if calculated)
ads.residuals          # Dict with residual calculations
ads.has_vas_residuals  # Boolean

# Effects and interactions (if calculated)
ads.effects           # Dict: {'k_effects': Series, 't_effects': Series}
ads.interactions      # Dict: {'pdc_by_kt': Series, 'pdc_by_pt': Series, ...}

# Frames for charting
ads.obs_df            # Observation-level frame
ads.cell_df           # Cell-level aggregations
ads.k_df              # Factor-level aggregations
ads.t_df              # Time-level aggregations

# Metadata
ads.sampling_design_state  # SDS number (0-6)
ads.sds_characteristics    # Full SDS info dict
ads.analysis_summary       # Comprehensive summary
```

## Strengths ✅

### 1. Stratified Individuals Charts
**KILLER FEATURE** - Best-in-class capability:
- ✅ Automatic stratification by grouping variables
- ✅ Separate limits per group (critical for accuracy)
- ✅ Clean dictionary structure keyed by group name
- ✅ Statistics packaged with each chart
- ✅ Ready for visualization

### 2. Consistent Nested Structure
```python
{
    'group_or_chart_name': {
        'data': DataFrame,
        'statistics': dict
    }
}
```
- ✅ Same pattern across all chart types
- ✅ Easy to navigate
- ✅ Clear separation of data vs. metadata

### 3. Rich Metadata
- ✅ Summary statistics per chart/group
- ✅ Control limits included in data
- ✅ Signal detection (`beyond_limits`)
- ✅ Subgroup sizes (`n`)

### 4. VAS Residuals Integration
- ✅ R1-R5 residuals in `analysis_dataset`
- ✅ Accessible via `ads.analysis_dataset`
- ✅ Effects and interactions calculated
- ✅ Stored in dedicated dicts

### 5. Multiple Aggregation Levels
- ✅ `obs_df` - observation level
- ✅ `cell_df` - cell level (k × t)
- ✅ `k_df` - factor level
- ✅ `t_df` - time level

## Weaknesses ⚠️

### 1. Inconsistent Return Types

**Problem:**
```python
# IMR with grouping → dict of dicts
result = {'A': {'data': df, 'statistics': {}}, 'B': {...}}

# IMR without grouping → dict of dicts (but only one key 'all')
result = {'all': {'data': df, 'statistics': {}}}

# Xbar → dict of dicts (but different keys)
result = {'Xbar': {'data': df, 'statistics': {}}, 'Sbar': {...}}
```

**Impact:**
- User must know chart type to access data correctly
- Different access patterns for different scenarios
- Hard to write generic code

**Example of confusion:**
```python
result = analysis.calculate()

# Is it stratified IMR?
if 'all' in result:
    chart_data = result['all']['data']
elif 'A' in result:  # How do we know all group names?
    # ???
elif 'Xbar' in result:
    chart_data = result['Xbar']['data']
```

### 2. Residuals Not Easily Accessible

**Problem:**
```python
# Residuals are in the underlying dataset
result = analysis.calculate()
# BUT residuals are NOT in result!

# Must access via ads
residuals_df = analysis.ads.analysis_dataset[['R1', 'R2', 'R3', 'R4', 'R5']]
```

**Impact:**
- Not discoverable from `result` object
- Requires knowledge of internal structure
- Breaks encapsulation

### 3. Effects/Interactions Hidden

**Problem:**
```python
# Effects calculated but not in result
effects = analysis.ads.effects
interactions = analysis.ads.interactions

# What if user just has result?
result = analysis.calculate()
# No way to get effects/interactions from here!
```

**Impact:**
- Two-step access pattern required
- Not part of standard output
- Easy to miss these valuable insights

### 4. No Unified Access Pattern

**Problem:**
```python
# Different patterns for different data
chart_data = result['Xbar']['data']        # For charts
residuals = analysis.ads.analysis_dataset   # For residuals
effects = analysis.ads.effects              # For effects
interactions = analysis.ads.interactions    # For interactions
```

**Impact:**
- Cognitive load on users
- Not discoverable
- Not pythonic

### 5. Limited Summary Information in Result

**Problem:**
```python
result = {'Xbar': {'statistics': {'Mean': 100, ...}}}
# Where's SDS info?
# Where's residual summary?
# Where are effects?

# Must go back to ads
sds = analysis.ads.sampling_design_state
summary = analysis.ads.analysis_summary
```

**Impact:**
- Result object incomplete
- Must maintain reference to analysis object
- Hard to serialize/save results

## Areas for Improvement 🚀

### 1. Unified Result Object

**Goal:** Single, comprehensive result object with everything

**Proposed Structure:**
```python
class AnalysisResult:
    """
    Comprehensive analysis result with all data easily accessible.

    Attributes:
        charts: Dict of chart data (Xbar, Sbar, IMR stratified, etc.)
        dataset: Full analysis dataset with residuals
        residuals: Easy access to R1-R5 (if calculated)
        effects: Main effects (if calculated)
        interactions: Interaction effects (if calculated)
        summary: Metadata (SDS, statistics, etc.)
        sds: Sampling design state info
    """
    def __init__(self, analysis):
        self.charts = ...           # Current result dict
        self.dataset = ...          # ads.analysis_dataset
        self.residuals = ...        # R1-R5 extracted
        self.effects = ...          # ads.effects
        self.interactions = ...     # ads.interactions
        self.summary = ...          # ads.analysis_summary
        self.sds = ...              # SDS info

    def get_chart(self, name):
        """Get specific chart data."""
        return self.charts[name]['data']

    def get_statistics(self, name):
        """Get specific chart statistics."""
        return self.charts[name]['statistics']

    @property
    def all_charts(self):
        """List all available charts."""
        return list(self.charts.keys())
```

**Usage:**
```python
result = analysis.calculate()  # Returns AnalysisResult

# Access charts (backward compatible)
xbar = result.charts['Xbar']['data']

# OR use convenience methods
xbar = result.get_chart('Xbar')
stats = result.get_statistics('Xbar')

# Access residuals directly
residuals = result.residuals  # DataFrame with R1-R5

# Access effects
main_effects = result.effects
interactions = result.interactions

# Get summary
print(result.summary)
# {
#     'sds': 1,
#     'sds_description': 'Full replication',
#     'has_residuals': True,
#     'has_effects': True,
#     'n_observations': 120,
#     'chart_types': ['Xbar', 'Sbar']
# }
```

### 2. Consistent Chart Access Pattern

**Goal:** Same pattern regardless of chart type

**Proposed:**
```python
result = analysis.calculate()

# Unified iteration
for chart_name, chart_info in result.charts.items():
    data = chart_info['data']
    stats = chart_info['statistics']
    print(f"{chart_name}: {len(data)} points, mean={stats.get('mean')}")

# Works for:
# - Xbar → result.charts['Xbar']
# - Stratified IMR → result.charts['Group_A'], result.charts['Group_B']
# - Simple IMR → result.charts['all']
```

### 3. Add Residuals to Result

**Goal:** Make residuals first-class citizens in result

**Proposed:**
```python
result = analysis.calculate()

if result.has_residuals:
    # Access all residuals
    residuals_df = result.residuals  # Columns: [R1, R2, R3, R4, R5]

    # Or specific residual
    r1 = result.get_residual('R1')

    # Or with metadata
    residual_info = result.residuals_summary
    # {
    #     'R1': {'description': '...', 'sd': 2.5, 'n': 120},
    #     'R2': {...},
    #     ...
    # }
```

### 4. Add Effects/Interactions to Result

**Goal:** Include all analytical outputs in one place

**Proposed:**
```python
result = analysis.calculate()

if result.has_effects:
    # Main effects
    k_effects = result.effects['k_effects']  # Series
    t_effects = result.effects['t_effects']  # Series

    # Interactions
    pdc = result.interactions['pdc_by_kt']

    # Summary
    effects_summary = result.effects_summary
    # {
    #     'k_effects': {'min': -2.1, 'max': 3.4, 'range': 5.5},
    #     't_effects': {'min': -1.8, 'max': 2.2, 'range': 4.0}
    # }
```

### 5. Enhanced Summary

**Goal:** Rich metadata in result object

**Proposed:**
```python
result.summary
# {
#     'sds': 1,
#     'sds_description': 'Full replication (all cells n≥2)',
#     'sds_capabilities': ['full_vas', 'all_residuals', 'interactions'],
#     'analysis_type': 'Xbar',
#     'n_observations': 120,
#     'n_cells': 24,
#     'response_var': 'Height',
#     'grouping_vars': ['Operator', 'Machine'],
#     'time_var': 'ProductionTime',
#     'has_residuals': True,
#     'has_effects': True,
#     'has_interactions': True,
#     'chart_types': ['Xbar', 'Sbar'],
#     'n_signals': 3  # Total points beyond limits
# }
```

## Recommendations

### Priority 1: Create AnalysisResult Class ⭐⭐⭐

**Benefits:**
- Single source of truth for all analysis outputs
- Consistent access patterns
- Backward compatible (charts still nested dict)
- Easy to extend
- Serializable

**Implementation:**
1. Create `AnalysisResult` class in new file `analysis_result.py`
2. Modify `Analysis.calculate()` to return `AnalysisResult` instance
3. Keep current dict structure in `result.charts` for compatibility
4. Add convenience properties/methods

### Priority 2: Expose Residuals/Effects in Result ⭐⭐

**Benefits:**
- All data in one place
- Discoverable API
- No need to access `ads` internals

**Implementation:**
1. Add `residuals` property to `AnalysisResult`
2. Add `effects` and `interactions` properties
3. Add `has_residuals`, `has_effects` boolean flags
4. Extract relevant columns from `ads.analysis_dataset`

### Priority 3: Enhanced Documentation ⭐⭐

**Benefits:**
- Users discover stratified charts feature
- Clear examples of accessing all data types
- Best practices for different use cases

**Implementation:**
1. Create examples showing stratified IMR charts
2. Document all data access patterns
3. Show how to combine charts + residuals + effects
4. Add to demo notebook

### Priority 4: Improved Chart Metadata ⭐

**Benefits:**
- Easier to identify chart types
- Better introspection
- Helpful for generic plotting functions

**Implementation:**
```python
result.charts['Xbar'] = {
    'data': DataFrame,
    'statistics': {...},
    'metadata': {
        'chart_type': 'Xbar',
        'chart_description': 'Subgroup Mean Chart',
        'n_points': 24,
        'n_signals': 2,
        'is_stratified': False
    }
}

result.charts['Alice'] = {
    'data': DataFrame,
    'statistics': {...},
    'metadata': {
        'chart_type': 'Imr',
        'chart_description': 'Individual Moving Range',
        'group_name': 'Alice',
        'n_points': 40,
        'n_signals': 0,
        'is_stratified': True
    }
}
```

## Summary

### The Stratified IMR Feature is AWESOME! 🎉

This is a **truly differentiated capability** that makes this library stand out:
- ✅ Automatic stratification
- ✅ Separate limits per group (statistically correct)
- ✅ Clean implementation
- ✅ Ready for visualization

### Main Improvement Needed

**Problem:** Data accessibility is fragmented
- Charts in `result`
- Residuals in `ads.analysis_dataset`
- Effects in `ads.effects`
- Interactions in `ads.interactions`

**Solution:** Unified `AnalysisResult` object
- Everything in one place
- Consistent access patterns
- Backward compatible
- Easy to use, discover, and extend

### Next Steps

1. **Implement `AnalysisResult` class** - Unify all outputs
2. **Update examples** - Show stratified charts feature prominently
3. **Enhance documentation** - Make capabilities discoverable
4. **Add convenience methods** - `get_chart()`, `get_residuals()`, etc.

This will transform the user experience from:
```python
# Current (fragmented)
charts = analysis.calculate()
residuals = analysis.ads.analysis_dataset[['R1', 'R2', 'R3', 'R4', 'R5']]
effects = analysis.ads.effects
```

To:
```python
# Proposed (unified)
result = analysis.calculate()
charts = result.charts
residuals = result.residuals
effects = result.effects
```

Much better! 🚀
