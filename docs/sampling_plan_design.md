# Implementation Plan: Sampling Plan Feature

## Goal
Enable SDS 4-6 detection by allowing users to specify expected factor levels (sampling plan), with full integration into SDS detection and Study reporting.

## Design Decisions (Confirmed)

1. **Separate `plan` parameter** - not overloaded `factors`
2. **`ColumnRef` as dataclass** (NOT str subclass) - avoids pandas/numpy quirks, serialization issues
3. **Plan validation** - warn (not error) if observed levels not in plan, with instructions
4. **Mutual exclusion** - `factors` OR `plan`, not both
5. **Full integration** - SDS detection uses plan, Study exposes design info
6. **Plan as dict** - simple `{col: [levels]}`, refactor to class later if needed
7. **DesignReport object** - `study.design()` returns rich object with nice repr
8. **Accept `str | ColumnRef` everywhere** - users can use plain strings or autocomplete

## API

```python
# Simple case (infer from data) - SDS 1-3
study = pb.formulate(
    response=pb.cols.Weight,
    time=pb.cols.Pull,
    factors=[pb.cols.Lane, pb.cols.Phase]
)

# Explicit plan (specify levels) - enables SDS 4-6
study = pb.formulate(
    response=pb.cols.Weight,
    time=pb.cols.Pull,
    plan={
        pb.cols.Lane: [1, 2, 3, 4, 5, 6],
        pb.cols.Phase: [1, 2, 3]
    }
)

# Discoverability
pb.cols.Lane           # Lane (4): [1, 2, 3, 4]
pb.cols.Lane.levels    # [1, 2, 3, 4]
pb.cols.Lane.count     # 4

# Study design info (NEW)
design = study.design()  # Returns DesignReport object

design                   # Nice repr showing summary
design.factors           # DataFrame: factor, planned, observed, missing_levels, extra_levels

# Factor-level gaps (per column)
design.missing_levels    # {'Phase': [3]} - levels in plan but not in data
design.extra_levels      # {'Phase': []} - levels in data but not in plan

# Future: Combination-level gaps (grid-based, for SDS 4-6)
# design.missing_combos  # [(Lane=2, Phase=3), ...] - expected combos not observed
# design.extra_combos    # [...] - observed combos not in plan
# design.grid            # Full DOE combination grid
# design.coverage        # count_observed / count_planned
```

---

## Files to Modify

### 1. `processbehavior/process_behavior.py`

**Create `ColumnRef` class** (~40 lines) - dataclass, NOT str subclass
```python
@dataclass
class ColumnRef:
    """
    Column reference with level awareness for IDE discoverability.

    NOT a str subclass to avoid pandas/numpy quirks.
    Implements __hash__ and __eq__ for dict key usage.
    Compares equal to strings for flexibility.
    """
    name: str
    _df: pd.DataFrame = field(repr=False)

    @property
    def levels(self) -> list:
        """Sorted unique values from the data."""
        values = self._df[self.name].dropna().unique()
        try:
            return sorted(values.tolist())
        except TypeError:
            return list(values)  # Mixed types, can't sort

    @property
    def count(self) -> int:
        """Number of distinct levels."""
        return len(self.levels)

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, ColumnRef):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def __repr__(self) -> str:
        lvls = self.levels
        if len(lvls) <= 6:
            return f"{self.name} ({len(lvls)}): {lvls}"
        return f"{self.name} ({len(lvls)}): [{lvls[0]}..{lvls[-1]}]"
```

**Update `ColumnAccessor`**
- Change `setattr(self, attr_name, col)` → `setattr(self, attr_name, ColumnRef(col, df))`
- Update `__getitem__` to return `ColumnRef`
- Update `__repr__` to show levels summary for all columns

**Update `formulate()` signature** - accept `str | ColumnRef` everywhere
```python
def formulate(
    self,
    response: str | ColumnRef,
    factors: list[str | ColumnRef] | None = None,
    time: str | ColumnRef | None = None,
    plan: dict[str | ColumnRef, list] | None = None,  # NEW
    precision: int = 3
) -> Study:
```

**Add helper to normalize column names**
```python
def _to_column_name(self, col: str | ColumnRef) -> str:
    """Extract column name from str or ColumnRef."""
    return col.name if isinstance(col, ColumnRef) else col
```

**Add validation in `formulate()`**
- Mutual exclusion check
- Extract factors from plan keys
- Validate plan columns exist
- **Warn (not error)** if observed levels not in plan, with helpful message:
  ```
  Warning: Factor 'Lane' has observed levels not in plan: [5]
  Your plan: [1, 2, 3, 4]
  Observed:  [1, 2, 3, 4, 5]

  To update your plan:
    plan[pb.cols.Lane] = pb.cols.Lane.levels  # Use observed
    # or
    plan[pb.cols.Lane] = [1, 2, 3, 4, 5]      # Add manually
  ```
- Pass plan to SDS detector
- Pass plan to Study for design reporting

**Add `_validate_plan()` method** - returns validated plan dict, logs warnings for extra levels

---

### 2. `processbehavior/study.py`

**Add `_sampling_plan` field to Study dataclass**
```python
@dataclass(frozen=True)
class Study:
    _pdf: ProcessBehavior
    _spec: DataPrepConfig
    _plan: SDSAnalysisPlan
    _ads: AnalysisDataSet
    _sampling_plan: dict[str, list] | None = None  # NEW: The user's plan
```

**Create `DesignReport` class** (~60 lines)
```python
@dataclass
class DesignReport:
    """
    Compares sampling plan to observed data.

    Returned by study.design(). Provides insight into the experimental
    design structure and any mismatches between plan and observation.
    """
    _sampling_plan: dict[str, list] | None
    _observed_levels: dict[str, list]  # Actual levels per factor from data
    _factors: list[str]

    @property
    def factors(self) -> pd.DataFrame:
        """
        Factor-level summary table.

        Returns DataFrame with columns:
        - factor: Factor column name
        - planned: Levels in plan (or observed if no plan)
        - observed: Levels actually in data
        - missing_levels: Levels in plan but not in data
        - extra_levels: Levels in data but not in plan
        """

    @property
    def missing_levels(self) -> dict[str, list]:
        """Levels in plan but not observed, per factor."""

    @property
    def extra_levels(self) -> dict[str, list]:
        """Levels observed but not in plan, per factor."""

    # Future: combination-level properties
    # @property
    # def missing_combos(self) -> list[dict]:
    #     """Combinations in plan grid but not observed."""
    #
    # @property
    # def extra_combos(self) -> list[dict]:
    #     """Combinations observed but not in plan grid."""

    def __repr__(self) -> str:
        """
        Nice summary, e.g.:
        DesignReport(2 factors, 1 missing_levels, 0 extra_levels)
          Lane: planned=[1,2,3,4], observed=[1,2,3,4]
          Phase: planned=[1,2,3], observed=[1,2], missing=[3]
        """
```

**Add `design()` method to Study**
```python
def design(self) -> DesignReport:
    """
    Get design report comparing plan to observed data.

    Returns a DesignReport showing factors, levels, and any mismatches.
    If no plan was provided, shows observed levels only.
    """
```

**Update `__repr__` and `_repr_html_`** - show plan status if provided

---

### 3. `processbehavior/sds_detector.py`

**Update `detect_sds()` signature**
```python
def detect_sds(
    self,
    df: pd.DataFrame,
    spec: DataPrepConfig,
    plan: dict[str, list] | None = None  # NEW
) -> tuple[int, int]:
```

**Two detection modes:**

```python
# Mode 1: Observed structure only (plan=None) → SDS 1-3
# Uses actual data to infer structure
sds, min_n = detector.detect_sds(df, spec)  # Current behavior

# Mode 2: Observed + Plan → enables SDS 4-6
# Uses plan to define INTENDED structure, compares to observed
sds, min_n = detector.detect_sds(df, spec, plan={'Lane': [1,2,3,4], 'Phase': [1,2,3]})
```

**Detection logic:**
```python
def detect_sds(self, df, spec, plan=None):
    if plan is None:
        # Mode 1: Infer from observed (SDS 1-3 only)
        return self._classify_from_observed(df, spec)
    else:
        # Mode 2: Compare observed to plan (enables SDS 4-6)
        return self._classify_with_plan(df, spec, plan)
```

**Add helper method `_classify_with_plan()`**
```python
def _classify_with_plan(
    self,
    df: pd.DataFrame,
    spec: DataPrepConfig,
    plan: dict[str, list]
) -> tuple[int, int]:
    """
    Classify SDS using sampling plan (expected structure).

    Key insight: Plan defines the INTENDED structure.
    - Plan levels not in data = missing (expected but not sampled)
    - Missing levels → incomplete grid → SDS 4, 5, or 6

    Example:
        plan = {'Phase': [1, 2, 3]}
        observed = {'Phase': [1, 2]}  # Phase 3 missing
        → Grid is incomplete → may classify as SDS 4 or 5
    """
```

---

### 4. `processbehavior/exceptions.py`

No new exceptions needed - extra levels trigger warning, not error.
Column-not-found still uses existing `ColumnNotFoundError`.

---

### 5. `processbehavior/__init__.py`

- Export `DesignReport` (for type hints if needed)

---

### 6. Tests: `tests/test_sampling_plan.py` (new file)

**ColumnRef tests:**
- `test_column_ref_levels_returns_sorted_unique`
- `test_column_ref_count_matches_levels`
- `test_column_ref_repr_shows_levels`
- `test_column_ref_works_as_dict_key`
- `test_column_ref_equals_string` (ColumnRef('Lane') == 'Lane')
- `test_column_ref_hash_equals_string_hash`
- `test_column_accessor_returns_column_ref`
- `test_formulate_accepts_plain_strings`

**Plan validation tests:**
- `test_plan_and_factors_mutual_exclusion`
- `test_plan_extracts_factors_from_keys`
- `test_plan_warns_on_extra_observed_levels` (warning, not error)
- `test_plan_column_not_found_raises`

**DesignReport tests:**
- `test_design_report_factors_dataframe`
- `test_design_report_missing_levels_per_factor`
- `test_design_report_extra_levels_per_factor`
- `test_design_report_no_plan_uses_observed`
- `test_design_report_repr`
- `test_design_report_empty_when_plan_matches_observed`

**SDS integration tests:**

*Mode 1: Observed only (no plan) → SDS 1-3:*
- `test_sds_6_no_factors_no_plan`
- `test_sds_1_full_replication_no_plan`
- `test_sds_2_no_replication_no_plan`
- `test_sds_3_partial_replication_no_plan`

*Mode 2: Observed + Plan → enables SDS 4-6:*
- `test_sds_with_plan_matches_observed_same_as_no_plan`
- `test_sds_with_plan_missing_one_factor_level`
  ```python
  # Plan says Phase should be [1, 2, 3]
  # Data only has Phase [1, 2]
  # → Phase 3 is missing → incomplete grid
  plan = {'Lane': [1,2,3,4], 'Phase': [1,2,3]}
  # Data has all Lanes but only Phase 1,2
  assert sds in [4, 5]  # Incomplete structure
  ```
- `test_sds_with_plan_missing_multiple_factor_levels`
- `test_sds_with_plan_missing_all_levels_of_one_factor`
- `test_sds_4_nested_detected_via_plan`
- `test_sds_5_irregular_detected_via_plan`
- `test_sds_6_single_condition_detected_via_plan`

*Backward compatibility:*
- `test_sds_detection_without_plan_unchanged` (existing behavior preserved)

---

## Implementation Order

1. **Phase 1: Core Infrastructure**
   - Create `ColumnRef` class
   - Update `ColumnAccessor` to use `ColumnRef`
   - Write ColumnRef tests

2. **Phase 2: Plan Parameter**
   - Add `plan` parameter to `formulate()`
   - Add `_validate_plan()` method
   - Add mutual exclusion validation
   - Write plan validation tests

3. **Phase 3: Study Integration**
   - Add `_sampling_plan` field to Study
   - Create `DesignReport` class
   - Add `design()` method
   - Update display methods
   - Write Study design tests

4. **Phase 4: SDS Detection Integration**
   - Update `detect_sds()` to accept plan
   - Add `_classify_with_plan()` helper
   - Implement SDS 4-6 detection with plan
   - Write SDS integration tests

5. **Phase 5: Verification**
   - Run full test suite
   - Manual testing with fill weight dataset
   - Verify backward compatibility

---

## Example: Fill Weight Dataset

```python
pb = ProcessBehavior(df)

# Discover levels
pb.cols.Lane           # Lane (4): [1, 2, 3, 4]
pb.cols.Phase          # Phase (2): [1, 2]

# Formulate with plan (expecting Phase 3 that wasn't sampled)
study = pb.formulate(
    response=pb.cols.Weight,
    time=pb.cols.Pull,
    plan={
        pb.cols.Lane: [1, 2, 3, 4],
        pb.cols.Phase: [1, 2, 3]  # Phase 3 expected but missing
    }
)

# Get design report
design = study.design()

design
# DesignReport(2 factors, 1 missing_levels, 0 extra_levels)
#   Lane: planned=[1,2,3,4], observed=[1,2,3,4]
#   Phase: planned=[1,2,3], observed=[1,2], missing=[3]

design.factors
#   factor  planned      observed   missing_levels  extra_levels
# 0   Lane  [1,2,3,4]    [1,2,3,4]              []            []
# 1  Phase  [1,2,3]      [1,2]                 [3]            []

design.missing_levels
# {'Lane': [], 'Phase': [3]}

design.extra_levels
# {'Lane': [], 'Phase': []}

# Future (for SDS 4-6 combo-level analysis):
# design.missing_combos  # [{'Lane': 2, 'Phase': 3}, ...]
# design.extra_combos    # []

# SDS reflects missing data
study.observed_design_state.sds  # Could be SDS 4 or 5 depending on structure
```

---

## Future Work (Phase 2+)

**Combination-level analysis (critical for SDS 4-6):**
- `design.missing_combos` - combinations in plan grid but not observed
- `design.extra_combos` - combinations observed but not in plan grid
- `design.grid` - full DOE combination grid
- `design.coverage` - count_observed / count_planned

**Other:**
- `SamplingPlan` class if dict proves limiting
- Web API serialization helpers

---

## Architecture: Truth vs Label

**Principle: `rsg_key` = truth, `rsg` = label**

| Column | Role | Type | Example | Used For |
|--------|------|------|---------|----------|
| `rsg_key` | **Canonical identity** | Tuple | `(1, "No")` | Comparisons, joins, SDS logic |
| `rsg` | Display label | String | `"1_No"` | Groupby, export, human-readable |
| `cell_key` | Cell identity | Tuple | `(1, "No", 3)` | Factor + time |
| `obs_id` | Row identity | Int | `42` | Unique observation |

**Why tuples for identity:**
- Type-safe: `(1, "No")` != `("1", "No")`
- No delimiter ambiguity: `"1_No"` could be `(1, "No")` or `("1", "No")` or `("1_No",)`
- No escaping rules needed for values containing `_`

---

## Factor Order: Critical for Tuple Consistency

**Rule:** Factor order must be stored and immutable.

```python
# Plan stores factor order (insertion order or explicit)
plan = {
    'Floor': [1, 2, 3],      # First
    'Problem': ['No', 'Yes']  # Second
}
# factor_order = ['Floor', 'Problem']

# All tuples use same order:
rsg_key = (floor_value, problem_value)      # (1, "No")
cell_key = (floor_value, problem_value, t)  # (1, "No", 3)
expected_key = (1, "No")                     # From plan
```

**Store with plan and study:**
- `_factor_order: list[str]` in Study or DesignReport
- Ensures consistent tuple generation

---

## Comparison Flow

```
Plan levels + factor_order
        |
        v
Cartesian product -> Expected rsg_key tuples
        |                           |
        v                           v
    Set diff with           Render to rsg strings
    observed rsg_key              for display
        |
        v
Missing/extra tuples
        |
        v
Multiple output formats
```

---

## DesignReport Combo-Level API (Future)

```python
# User-friendly default (strings)
design.missing_combos      # ["3_Yes", "3_No"]

# Structured access (optional)
design.missing_combo_keys  # [(3, "Yes"), (3, "No")]
design.missing_combo_dicts # [{"Floor": 3, "Problem": "Yes"}, ...]
```

No new `condition_id` needed. Tuple is the truth, string is the label.

---

## Scope of Change (Minimal)

1. **Plan normalization:** Store `factor_order` with plan
2. **DesignReport:** Compare `rsg_key` tuples, render to strings
3. **SDS detector:** Use tuple sets for 4-6 detection
4. **Helper:** `render_rsg_from_tuple(key, factor_order, delim)` in DataPreparation
5. **No changes to:** DataFrame schema, core data pipeline, existing rsg logic
