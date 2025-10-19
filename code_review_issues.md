# Code Review Issues - GitHub Issue Creation

## Issue 1: Remove Duplicate Calculation Functions (Code Duplication)

**Title:** Remove duplicate calculation functions after refactoring

**Body:**
After refactoring to the Strategy Pattern in #9, we have 4 deprecated standalone functions that duplicate logic now in the `Analysis` class:

**Duplicate Functions (lines 751-1073):**
- `calculate_statistics_Imr()` - duplicates `Analysis._calculate_imr()`
- `calculate_statistics_R()` - duplicates `Analysis._calculate_r()`
- `calculate_statistics_S()` - duplicates `Analysis._calculate_s()`
- `calculate_statistics_XbarS()` - duplicates `Analysis._calculate_xbar()`

**Impact:**
- ~322 lines of dead code (18% of file)
- Maintenance burden (bugs could be fixed in one place but not the other)
- Confusion about which functions to use

**Fix:**
Remove lines 751-1073 from `analysis_dataset.py`

**Labels:** refactoring, code-quality, technical-debt
**Priority:** High

---

## Issue 2: Remove Duplicate calculate_limits() Function

**Title:** Remove duplicate calculate_limits() function - defined in two places

**Body:**
The `calculate_limits()` function is defined identically in two files:
- `analysis_dataset.py:1665-1713`
- `objects.py:81-129`

The code uses `obj.calculate_limits()`, so the local version at line 1665 is never used.

**Impact:**
- Maintenance risk (changes in one place won't reflect in the other)
- ~50 lines of duplicate code
- Confusion about which version is canonical

**Fix:**
Remove `calculate_limits()`, `c4()`, `b3()`, `b4()`, and `detect_beyond_limits()` from `analysis_dataset.py` (lines 1665-1734). Keep only versions in `objects.py`.

**Labels:** refactoring, code-quality, duplication
**Priority:** High

---

## Issue 3: Remove Unused AbstractAnalysisSpecification ABC

**Title:** Remove unnecessary AbstractAnalysisSpecification abstract base class

**Body:**
The abstract base class `AbstractAnalysisSpecification` (lines 476-513) has only one concrete implementation and adds no value.

**Issues:**
- Unnecessary complexity with no benefit
- The `@property` methods (lines 478-511) are never actually used polymorphically
- Related: property methods in `AnalysisSpecification` don't return values (see #10)

**Impact:**
- Misleading architecture
- ~38 lines of unnecessary code
- Maintenance burden

**Fix:**
Remove `AbstractAnalysisSpecification` class entirely. Convert `AnalysisSpecification` properties to simple attributes or fix them to return values.

**Labels:** refactoring, code-quality, architecture
**Priority:** Medium

---

## Issue 4: Consolidate Utility Functions from objects.py

**Title:** Inconsistent import usage - utility functions duplicated across modules

**Body:**
The code imports `objects as obj` but also duplicates many functions from that module:

**Duplicated in both `analysis_dataset.py` and `objects.py`:**
- `calculate_limits()`
- `c4()`
- `b3()`
- `b4()`
- `detect_beyond_limits()`

**Impact:**
- Unclear module boundaries
- Risk of divergence between duplicated functions
- Makes refactoring harder

**Fix:**
1. Keep all utility functions in `objects.py` only
2. Remove duplicates from `analysis_dataset.py`
3. Use `obj.function_name()` consistently throughout

**Labels:** refactoring, code-quality, organization
**Priority:** Medium

---

## Issue 5: Replace print() Statements with Logging

**Title:** Replace print() debugging statements with proper logging

**Body:**
There are 20+ `print()` statements scattered throughout production code instead of using proper logging.

**Examples:**
- Line 82: `print(f'\nIn calculate statistics XbarS...')`
- Line 260: `print(f'\nIn calculate statistics IMR...')`
- Line 699: `print(f'\nEntering call to prepare_data...')`
- Line 720: `print(f'\nStarting with {starting_count} groups')`

**Impact:**
- Unprofessional output
- Can't control verbosity
- Hard to debug in production
- Performance impact

**Fix:**
1. Add module-level logger: `logger = logging.getLogger(__name__)`
2. Replace all `print()` with `logger.debug()` or `logger.info()`
3. Remove debugging output from hot paths

**Labels:** code-quality, logging, technical-debt
**Priority:** Medium

---

## Issue 6: Standardize Return Types from perform_analysis()

**Title:** Inconsistent return types from perform_analysis() function

**Body:**
The `perform_analysis()` function returns different data structures depending on analysis type:

- **Xbar:** Returns `dict` with 'Xbar' and 'Sbar' keys, each containing `{'data': df, 'statistics': dict}`
- **Imr/R:** Returns `dict` with RSG keys or 'all', each containing `{'data': df, 'statistics': dict}`
- **S:** Returns a DataFrame directly

**Impact:**
- Users must write conditional logic to handle results
- API is unpredictable and hard to document
- Makes testing harder
- Violates principle of least surprise

**Fix:**
Standardize on a consistent return structure. Options:
1. Always return dict with standardized keys
2. Create a result class (e.g., `AnalysisResult`) with consistent interface
3. Document clearly and add type hints

**Labels:** api-design, breaking-change, enhancement
**Priority:** High

---

## Issue 7: Replace Magic Numbers with Named Constants

**Title:** Replace magic numbers with named constants for statistical multipliers

**Body:**
Statistical constants are hard-coded throughout the code with no explanation:

**Examples:**
- Line 1697: `2.66` (IMR limit multiplier)
- Line 1705: `3.268` (R chart limit multiplier)
- Line 299-300: `beyond_limits` calculation logic
- Various uses of `3` for sigma multipliers

**Impact:**
- Hard to verify correctness
- No documentation of where constants come from
- Hard to modify if different sigma levels needed

**Fix:**
Define named constants at module level with docstrings:
```python
# Statistical constants based on d2 and d3 tables
IMR_LIMIT_MULTIPLIER = 2.66  # E2 constant for n=2 (individuals)
R_LIMIT_MULTIPLIER = 3.268   # D4 constant for n=2 (moving range)
SIGMA_MULTIPLIER = 3         # Standard 3-sigma control limits
```

**Labels:** code-quality, documentation, maintainability
**Priority:** Medium

---

## Issue 8: Add Comprehensive Type Hints

**Title:** Add comprehensive type hints to improve IDE support and static analysis

**Body:**
Many functions lack type hints, and some have incomplete or incorrect hints.

**Examples:**
- `calculate_limits()` returns `pd.Series` but is hinted as returning `dict`
- Many methods in `AnalysisDataSet` lack return type hints
- Specification dictionaries should use `TypedDict`

**Impact:**
- Harder for IDE autocomplete
- No static type checking benefits
- Confusing API for users

**Fix:**
1. Add type hints to all public functions and classes
2. Create `TypedDict` for specification dictionaries
3. Use `from __future__ import annotations` for forward references
4. Run mypy for validation

**Labels:** enhancement, type-safety, developer-experience
**Priority:** Low-Medium

---

## Issue 9: Complete or Document Sampling Design State Logic

**Title:** Sampling Design State detection incomplete (only handles SDS 1-2)

**Body:**
The docstring for `__calculate_sampling_design_state()` (lines 1385-1399) describes 6 sampling design states but the implementation only handles SDS 1 and 2.

**From docstring:**
```python
"""
- SDS1: every (k,t) cell has n>=2
- SDS2: every (k,t) cell has n==1
- SDS3: mixture of replication (some n==1 and some n>=2) and/or missing (k,t) cells
- SDS4: single design condition over time (K==1 with time present)
- SDS5: nested design (>=2 design vars) with asynchronous time coverage
- SDS6: unstructured fallback (no time OR cannot form (k,t) cells)
"""
```

**Actual implementation:**
```python
if all(group_sizes>=2): out = 1
if all(group_sizes==1): out = 2
# Everything else returns 0
```

**Impact:**
- Misleading documentation
- Silent failures for SDS 3-6 cases
- Incomplete feature

**Fix:**
**Option 1:** Implement SDS 3-6 detection
**Option 2:** Update docs to state only SDS 1-2 supported and raise error for other cases

**Labels:** bug, documentation, enhancement
**Priority:** Medium

---

## Issue 10: Fix Property Methods That Don't Return Values

**Title:** Property methods in AnalysisSpecification and AnalysisDataSet don't return values

**Body:**
Multiple property methods reference `self.attribute` but don't return it, so they always return `None`.

**In AnalysisSpecification (lines 603-622):**
```python
def data_prep_output_cols(self) -> list:
    self.data_prep_output_cols  # Missing return!

def analysis_output_cols(self) -> list:
    self.analysis_output_cols  # Missing return!

def has_grouping(self) -> bool:
    self.has_grouping  # Missing return!
```

**In AnalysisDataSet (lines 1240-1256):**
```python
def sampling_design_state(self) -> int:
    self.sampling_design_state  # Missing return!

def raw_dataset(self) -> pd.DataFrame:
    self.raw_dataset  # Missing return!
```

**Impact:**
- **Critical bug:** Properties always return `None`
- Broken API
- Will cause runtime errors if anyone uses these properties

**Fix:**
Add `return` statements to all property methods:
```python
def has_grouping(self) -> bool:
    return self.has_grouping
```

Or remove the property decorators since they're already instance attributes.

**Labels:** bug, critical, high-priority
**Priority:** Critical

---

## Summary

- **Critical bugs:** 1 (Issue #10)
- **High priority:** 3 (Issues #1, #2, #6)
- **Medium priority:** 5 (Issues #3, #4, #5, #7, #9)
- **Low priority:** 1 (Issue #8)
- **Total lines of duplicate/dead code:** ~550 lines (32% of analysis_dataset.py)
