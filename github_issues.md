# GitHub Issues for ProcessBehavior Refactoring

This document contains 25 issues identified for the processbehavior refactoring project.

---

## Issue #1: Duplicate Code Between Classes and Functions
**Labels**: `refactor`, `priority-high`, `technical-debt`
**Milestone**: Phase 1 - Core Architecture

### Description
The calculate_statistics logic is duplicated across multiple locations:
- Xbar class method (lines 123-231)
- IMR class method (lines 269-350)
- R class method (lines 371-449)
- Standalone function calculate_statistics_Imr (lines 760-845)
- Standalone function calculate_statistics_R (lines 848-928)
- Standalone function calculate_statistics_XbarS (lines 972-1081)

This is a massive DRY (Don't Repeat Yourself) violation with nearly identical code blocks repeated 4-6 times.

### Impact
- **Severity**: High
- Makes maintenance difficult - bug fixes must be applied in multiple places
- Increases risk of inconsistencies between implementations
- Code bloat (~1000+ lines of duplicated logic)

### Solution
1. Extract common calculation logic into a base class or shared utility functions
2. Use the Template Method pattern to handle analysis-specific variations
3. Keep only analysis-specific logic in subclasses
4. Consider creating a CalculationStrategy pattern for different analysis types

### Files Affected
- `analysis_dataset.py`

### Acceptance Criteria
- [ ] All duplicate calculation logic consolidated
- [ ] Tests still pass
- [ ] Code coverage maintained or improved
- [ ] Each analysis type has <50 lines of specific implementation code

---

## Issue #2: Inconsistent Type Annotations
**Labels**: `refactor`, `priority-high`, `type-safety`, `technical-debt`
**Milestone**: Phase 1 - Core Architecture

### Description
Classes use a mix of `pd.DataFrame`, `AnalysisDataSet`, `dict`, and `AnalysisSpecification` types inconsistently:
- `Xbar.__init__` expects `AnalysisDataSet` (line 79)
- `Sbar.__init__` expects `pd.DataFrame` (line 236)
- `IMR.__init__` expects `pd.DataFrame` (line 251)
- `R.__init__` expects `pd.DataFrame` (line 355)

Type hints are incomplete throughout the codebase.

### Impact
- **Severity**: High
- No IDE support for type checking
- Runtime errors from type mismatches
- Confusing API for users
- Difficult to refactor safely

### Solution
1. Standardize type signatures across all analysis classes
2. Add complete type hints to all functions and methods
3. Use Protocol or ABC for shared interfaces
4. Add mypy to CI/CD pipeline
5. Consider using `typing.TypedDict` for specification dictionaries

### Files Affected
- `analysis_dataset.py`
- `objects.py`

### Acceptance Criteria
- [ ] All public methods have type hints
- [ ] All analysis classes accept consistent input types
- [ ] mypy passes with strict mode
- [ ] No type: ignore comments without justification

---

## Issue #3: Magic Numbers Throughout Codebase
**Labels**: `refactor`, `priority-high`, `code-quality`
**Milestone**: Phase 1 - Core Architecture

### Description
Hard-coded constants scattered throughout without named constants or explanation:
- `2.66` (IMR limit factor) - lines 313, 791, 806
- `3.268` (R chart limit) - line 1691
- `3` (sigma multiplier) - lines 113, 1699
- Various sqrt calculations without context

### Impact
- **Severity**: High
- Impossible to understand statistical meaning
- Changes require searching entire codebase
- Risk of using wrong constant
- Cannot adjust for different confidence levels

### Solution
1. Create a constants module or section:
```python
# Statistical Constants
CONTROL_LIMIT_SIGMAS = 3  # Standard 3-sigma control limits
IMR_LIMIT_FACTOR = 2.66   # D4 constant for n=2
R_CHART_D4 = 3.268        # Upper control limit constant for moving range
```

2. Document the statistical significance of each constant
3. Add references to statistical literature where applicable

### Files Affected
- `analysis_dataset.py`
- `objects.py`

### Acceptance Criteria
- [ ] All magic numbers extracted to named constants
- [ ] Constants documented with statistical meaning
- [ ] No hardcoded numbers in calculation logic
- [ ] Constants grouped logically (control limits, bias corrections, etc.)

---

## Issue #4: Incomplete Abstract Methods Implementation
**Labels**: `bug`, `priority-high`, `correctness`
**Milestone**: Phase 1 - Core Architecture

### Description
1. `get_dataset()` in `AbstractAnalysis` has no `@abstractmethod` decorator but is implemented in subclasses (lines 73-74)
2. Property methods in `AnalysisSpecification` don't return values (lines 617-636) - they just reference `self` attributes without `return` statements

Example of broken property:
```python
def data_prep_output_cols(self) -> list:
    self.data_prep_output_cols  # Missing return!
```

### Impact
- **Severity**: High
- Properties return `None` instead of expected values
- Abstract interface not enforced
- Subclasses might not implement required methods
- Silent failures

### Solution
1. Add `@abstractmethod` decorator to `get_dataset()`
2. Fix all property methods to return values:
```python
def data_prep_output_cols(self) -> list:
    return self.data_prep_output_cols
```
3. Add tests to verify property values are accessible
4. Consider using `@property` decorator for proper properties

### Files Affected
- `analysis_dataset.py` (lines 73-74, 617-636)

### Acceptance Criteria
- [ ] All abstract methods properly decorated
- [ ] All properties return expected values
- [ ] Tests verify property access
- [ ] pylint/mypy detect missing implementations

---

## Issue #5: Dead/Commented Code Blocks
**Labels**: `technical-debt`, `priority-high`, `cleanup`
**Milestone**: Phase 2 - Code Quality

### Description
Large blocks of commented code in multiple locations:
- `calculate_limits` method (lines 106-117)
- `validate_columns` function (lines 696-698)
- Various TODO sections

This creates confusion about what code is actually being used.

### Impact
- **Severity**: High
- Confuses developers about intended behavior
- Suggests incomplete refactoring
- Makes code harder to read
- Version control already stores old code

### Solution
1. Review each commented block
2. Either:
   - Uncomment and properly implement with tests if needed
   - Delete completely if not needed
3. Use git history for reference to old implementations
4. Document why certain approaches were rejected in commit messages

### Files Affected
- `analysis_dataset.py` (lines 106-117, 696-698)

### Acceptance Criteria
- [ ] No commented code blocks remain
- [ ] All needed functionality is implemented and tested
- [ ] Git history documents why alternatives were rejected

---

## Issue #6: Inconsistent Error Handling
**Labels**: `refactor`, `priority-high`, `error-handling`
**Milestone**: Phase 1 - Core Architecture

### Description
Error messages and validation checks are inconsistent:
- Some use custom messages with context (line 149)
- Some just raise without context (line 733)
- No consistent error hierarchy
- Mix of ValueError, generic exceptions
- Different validation approaches

### Impact
- **Severity**: Medium
- Hard to catch specific errors
- Poor error messages for users
- Difficult to handle errors appropriately
- No error recovery strategies

### Solution
1. Create custom exception hierarchy:
```python
class ProcessBehaviorError(Exception):
    """Base exception for processbehavior package"""
    pass

class AnalysisConfigError(ProcessBehaviorError):
    """Invalid analysis configuration"""
    pass

class DataValidationError(ProcessBehaviorError):
    """Data validation failed"""
    pass

class CalculationError(ProcessBehaviorError):
    """Statistical calculation error"""
    pass
```

2. Standardize error messages: "{what_failed}: {why} (received: {value})"
3. Add context to all exceptions
4. Document all exceptions in docstrings

### Files Affected
- `analysis_dataset.py` (lines 148-149, 732-733, 1356-1357, 998-999)
- `objects.py` (lines 142-143)

### Acceptance Criteria
- [ ] Custom exception classes defined
- [ ] All raises use appropriate exception type
- [ ] Error messages include context and suggestions
- [ ] Documentation lists all possible exceptions

---

## Issue #7: Method Naming Inconsistency and Duplication
**Labels**: `refactor`, `priority-medium`, `code-duplication`
**Milestone**: Phase 2 - Code Quality

### Description
Statistical helper functions defined multiple times in different locations:
- `c4()`: Xbar static method (line 118), standalone (line 1709), objects.py (line 174)
- `b3()`, `b4()`: Similar duplication
- `detect_beyond_limits()`: Duplicated
- `calculate_limits()`: Duplicated

### Impact
- **Severity**: Medium
- Which version gets called is unclear
- Bugs fixed in one place might not be fixed in others
- Import confusion
- Wasted code

### Solution
1. Consolidate all statistical helper functions into `objects.py`
2. Remove duplicates from `analysis_dataset.py`
3. Import consistently: `from objects import c4, b3, b4, calculate_limits`
4. Make functions pure (no side effects)
5. Add comprehensive docstrings

### Files Affected
- `analysis_dataset.py` (lines 118, 174, 1709)
- `objects.py` (line 174)

### Acceptance Criteria
- [ ] Each function exists in only one location
- [ ] All functions in objects.py
- [ ] Consistent imports throughout codebase
- [ ] No duplicate implementations

---

## Issue #8: Missing Docstrings on Key Classes
**Labels**: `documentation`, `priority-medium`
**Milestone**: Phase 2 - Code Quality

### Description
Core analysis classes have no docstrings:
- `Xbar` (line 77)
- `Sbar` (line 234)
- `IMR` (line 248)
- `R` (line 353)

These are the main domain classes but have zero documentation.

### Impact
- **Severity**: Medium
- Users don't know how to use classes
- No parameter documentation
- No examples
- Hard to onboard new developers

### Solution
1. Add comprehensive docstrings following NumPy/Google style:
```python
class Xbar(AbstractAnalysis):
    """
    X-bar and S chart analysis for subgrouped data.

    Calculates control limits for both subgroup means (X-bar) and
    subgroup standard deviations (S) using the unbiased c4 constant.

    Parameters
    ----------
    df : AnalysisDataSet
        Prepared dataset with grouped observations
    analysis_specification : AnalysisSpecification
        Configuration specifying response variable, grouping, etc.

    Attributes
    ----------
    is_N_variable : bool
        Whether subgroup sizes vary
    analysis_result : dict
        Results containing 'Xbar' and 'Sbar' dataframes and statistics

    Examples
    --------
    >>> spec = AnalysisSpecification(...)
    >>> dataset = AnalysisDataSet(df, spec)
    >>> xbar = Xbar(dataset, spec)
    >>> results = xbar.get_dataset()

    References
    ----------
    Wheeler, Donald J. "Understanding Variation" (2000)
    """
```

2. Include parameters, returns, raises, examples, and references
3. Add docstrings to all public methods

### Files Affected
- `analysis_dataset.py` (lines 77, 234, 248, 353)

### Acceptance Criteria
- [ ] All public classes have docstrings
- [ ] Docstrings include parameters, returns, raises
- [ ] At least one example per class
- [ ] Statistical references where applicable
- [ ] Sphinx documentation can be generated

---

## Issue #9: Inconsistent Specification Handling
**Labels**: `refactor`, `priority-high`, `api-design`
**Milestone**: Phase 1 - Core Architecture

### Description
Specification objects handled inconsistently:
- `perform_analysis` creates `AnalysisSpecification` but also calls `prepare_dataset` separately (lines 452-487)
- `Xbar` expects `AnalysisDataSet` but others expect `pd.DataFrame` (line 79 vs 236, 251, 355)
- Spec passed as `dict` in some places, `AnalysisSpecification` object in others
- Dual creation of spec objects (lines 459-460)

### Impact
- **Severity**: High
- Confusing API
- Type safety issues
- Hard to refactor
- Inconsistent behavior

### Solution
1. Standardize on `AnalysisSpecification` object throughout
2. Remove dict-based spec handling
3. Ensure all analysis classes accept same input types
4. Single responsibility: either create spec OR use spec, not both
5. Make `prepare_dataset` a method of `AnalysisSpecification` or `AnalysisDataSet`

Example:
```python
def perform_analysis(df: pd.DataFrame, specification: dict) -> AnalysisResult:
    spec = AnalysisSpecification(specification['analysis_type'], specification)
    dataset = AnalysisDataSet(df, spec)
    factory = AnalysisFactory()
    analyzer = factory.create(spec.analysis_type, dataset, spec)
    return analyzer.get_dataset()
```

### Files Affected
- `analysis_dataset.py` (lines 79-86, 236-239, 452-487)

### Acceptance Criteria
- [ ] Single spec creation point
- [ ] All analysis classes accept same types
- [ ] No dict-based spec passing
- [ ] Clear separation of concerns

---

## Issue #10: Excessive Print Statements for Debugging
**Labels**: `technical-debt`, `priority-medium`, `logging`
**Milestone**: Phase 2 - Code Quality

### Description
Production code contains numerous `print()` statements (20+ instances):
- Lines 92, 131-134, 162, 278-284
- Lines 713, 729-743
- Lines 1343-1367, 1389

These are clearly for debugging but left in production code. Tests use proper logging, but production code doesn't.

### Impact
- **Severity**: Medium
- Pollutes stdout
- Can't control verbosity
- Hard to filter messages
- Not production-ready

### Solution
1. Replace all `print()` with logging:
```python
import logging
logger = logging.getLogger(__name__)

# Instead of: print(f'Starting with {count} groups')
logger.info('Starting with %d groups', count)

# Instead of: print(df.head())
logger.debug('DataFrame preview:\n%s', df.head())
```

2. Use appropriate log levels:
   - DEBUG: Detailed data, variable values
   - INFO: Progress, milestones
   - WARNING: Unusual situations
   - ERROR: Errors that are handled

3. Add logging configuration in main module
4. Allow users to control log level

### Files Affected
- `analysis_dataset.py` (lines 92, 131-134, 162, 278-284, 713, 729-743, 1343-1367, 1389)

### Acceptance Criteria
- [ ] No print() statements in production code
- [ ] All messages use logging module
- [ ] Appropriate log levels used
- [ ] Logger configured at module level
- [ ] Documentation on controlling log output

---

## Issue #11: Duplicate Column Validation Logic
**Labels**: `refactor`, `priority-medium`, `code-duplication`
**Milestone**: Phase 2 - Code Quality

### Description
`validate_columns` exists in two places with nearly identical implementations:
- Standalone function (lines 676-709)
- Private method `__validate_columns` in `AnalysisDataSet` (lines 1267-1292)

~40 lines of duplicated validation logic.

### Impact
- **Severity**: Medium
- Bug fixes needed in two places
- Inconsistencies possible
- Maintenance burden

### Solution
1. Keep single `validate_columns` function in module
2. Have `AnalysisDataSet.__validate_columns` call the shared function
3. Consider making it a static method if needed in class
4. Ensure both code paths have tests

### Files Affected
- `analysis_dataset.py` (lines 676-709, 1267-1292)

### Acceptance Criteria
- [ ] Single source of validation logic
- [ ] All callers use same function
- [ ] Tests cover both usage patterns
- [ ] No duplicate validation code

---

## Issue #12: Unused Function Parameters
**Labels**: `bug`, `priority-low`, `code-quality`
**Milestone**: Phase 3 - Enhancement

### Description
`calculate_limits()` function accepts `round_to` parameter but never uses it (lines 1659-1660).

This suggests either:
- Incomplete implementation
- Parameter should be removed
- Rounding should happen elsewhere

### Impact
- **Severity**: Low
- Misleading API
- Suggests incomplete refactoring
- Unused code

### Solution
1. Check all callers to see if they expect rounding
2. Either:
   - Implement rounding: `return {k: round(v, round_to) for k, v in out.items()}`
   - Remove parameter and document that rounding should happen at caller level
3. Add tests for rounding behavior

### Files Affected
- `analysis_dataset.py` (lines 1659-1660)

### Acceptance Criteria
- [ ] No unused parameters
- [ ] Rounding behavior documented
- [ ] Tests verify rounding works as expected

---

## Issue #13: Inconsistent Return Types
**Labels**: `bug`, `priority-medium`, `api-design`
**Milestone**: Phase 1 - Core Architecture

### Description
`calculate_statistics` methods return different types:
- Some return `dict` with nested structure (line 231)
- Some return processed `dict` via `package_analysis` (line 350)
- Some return `pd.DataFrame`
- No consistent interface

### Impact
- **Severity**: Medium
- Unpredictable API
- Type checking impossible
- Hard to use consistently
- Confusing for users

### Solution
1. Define a standard result class:
```python
@dataclass
class AnalysisResult:
    """Results from statistical analysis."""
    data: pd.DataFrame
    statistics: Dict[str, Union[float, int, str]]
    analysis_type: str
    metadata: Dict[str, Any]
```

2. All analysis methods return `AnalysisResult` or `Dict[str, AnalysisResult]`
3. Standardize on single return pattern
4. Update all callers

### Files Affected
- `analysis_dataset.py` (lines 231, 350, 449, 845, 928, 1081)

### Acceptance Criteria
- [ ] All analysis methods return same type
- [ ] Type hints reflect actual returns
- [ ] Documentation clear on return structure
- [ ] Backward compatibility maintained or documented

---

## Issue #14: Missing Type Safety for Dict Keys
**Labels**: `bug`, `priority-medium`, `type-safety`
**Milestone**: Phase 1 - Core Architecture

### Description
Heavy reliance on `dict.get()` with string keys for spec access:
- `spec.get('rsg_vars')` (line 574)
- `spec.get('time_var')` (line 577)
- `spec.get('round_to', 3)` (line 580)

Typos in keys fail silently, returning None or default values.

### Impact
- **Severity**: Medium
- Typos fail silently
- No IDE autocomplete
- No type checking
- Hard to refactor

### Solution
1. Convert specification dict to dataclass:
```python
@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis."""
    analysis_type: str
    response_var: str
    rsg_vars: Optional[List[str]] = None
    time_var: Optional[str] = None
    rsg_var_name: str = 'rsg'
    round_to: int = 3
    zero_center: bool = False

    def __post_init__(self):
        if self.response_var is None:
            raise ValueError("response_var is required")
```

2. Or use Pydantic for validation:
```python
from pydantic import BaseModel, Field

class AnalysisConfig(BaseModel):
    analysis_type: str
    response_var: str
    rsg_vars: Optional[List[str]] = None
    time_var: Optional[str] = None
    rsg_var_name: str = 'rsg'
    round_to: int = Field(default=3, ge=0, le=10)
    zero_center: bool = False
```

3. Migrate all dict.get() calls to attribute access
4. Add validation at construction time

### Files Affected
- `analysis_dataset.py` (lines 97, 136-139, 574-586, 1197-1199)

### Acceptance Criteria
- [ ] Specification is type-safe class
- [ ] No dict.get() for spec access
- [ ] IDE autocomplete works
- [ ] Validation happens at creation
- [ ] Migration guide for existing code

---

## Issue #15: Duplicate Grouping Logic
**Labels**: `refactor`, `priority-medium`, `code-duplication`
**Milestone**: Phase 2 - Code Quality

### Description
`prepare_dataset` and `__prepare_dataset` contain nearly identical grouping and filtering logic (50+ lines):
- Both add grouping variable column
- Both remove groups with n<=1
- Both perform sorting
- Both select output columns

### Impact
- **Severity**: Medium
- Code duplication
- Maintenance burden
- Risk of divergence
- Wasted code

### Solution
1. Extract common grouping logic:
```python
def _add_grouping_and_filter(df, spec):
    """Add grouping column and remove single-observation groups."""
    # Common logic here
    return filtered_df

def prepare_dataset(df, spec):
    out = df.copy()
    if spec.has_grouping:
        out = _add_grouping_and_filter(out, spec)
    # Rest of logic
    return out
```

2. Both functions call shared implementation
3. Keep only differences in each function
4. Add tests for shared logic

### Files Affected
- `analysis_dataset.py` (lines 721-745, 1346-1369)

### Acceptance Criteria
- [ ] Grouping logic in single location
- [ ] Both functions use shared code
- [ ] Tests cover shared logic
- [ ] No duplicate filtering code

---

## Issue #16: Beyond Limits Calculation Bug
**Labels**: `bug`, `priority-high`, `correctness`
**Milestone**: Phase 1 - Core Architecture

### Description
`beyond_limits` calculation has a logic error in multiple places:

```python
out['beyond_limits'] = np.where(out[spec.response_var] < out['lcl'], -1, 0)
out['beyond_limits'] = np.where(out[spec.response_var] > out['ucl'],  1, 0)
```

Line 2 overwrites line 1's results! Values below LCL get set to -1, then immediately overwritten to 0 (if not above UCL) or 1 (if above UCL). Values below LCL never stay -1.

Found in:
- Lines 317-318
- Lines 811-812
- Lines 896-897

### Impact
- **Severity**: High
- Incorrect violation detection
- Lower control limit violations not reported
- Statistical analysis invalid

### Solution
Fix to use proper chained condition:
```python
out['beyond_limits'] = np.where(
    out[spec.response_var] < out['lcl'],
    -1,
    np.where(out[spec.response_var] > out['ucl'], 1, 0)
)
```

Or use pandas.cut() for more readable code:
```python
out['beyond_limits'] = pd.cut(
    out[spec.response_var],
    bins=[-np.inf, out['lcl'], out['ucl'], np.inf],
    labels=[-1, 0, 1]
)
```

### Files Affected
- `analysis_dataset.py` (lines 317-318, 811-812, 896-897)

### Acceptance Criteria
- [ ] Beyond limits correctly detects both upper and lower violations
- [ ] Tests verify -1, 0, and 1 values
- [ ] All three locations fixed
- [ ] Edge cases tested (exactly at limits)

---

## Issue #17: Hardcoded Column Names
**Labels**: `refactor`, `priority-medium`, `maintainability`
**Milestone**: Phase 2 - Code Quality

### Description
Column names hardcoded throughout:
- 'rsg', 'lcl', 'ucl', 'mean', 'x', 'beyond_limits' repeated everywhere
- Some constants exist in `objects.py` (lines 8-10) but aren't used consistently
- Magic strings scattered through code

### Impact
- **Severity**: Medium
- Typos possible
- Hard to rename columns
- No autocomplete
- Maintenance burden

### Solution
1. Create comprehensive column name constants:
```python
# objects.py or constants.py
class ColumnNames:
    """Standard column names used throughout analysis."""
    RSG = 'rsg'
    TIME = 'time'
    RESPONSE = 'response'
    MEAN = 'mean'
    STD = 's'
    LCL = 'lcl'
    UCL = 'ucl'
    BEYOND_LIMITS = 'beyond_limits'
    MOVING_RANGE = 'mr'
    MEAN_MOVING_RANGE = 'mR'
    N = 'n'
    X = 'x'

    # Ybar variations
    YBAR = 'Ybar'
    YBAR_K = 'Ybar_k'
    YBAR_T = 'Ybar_t'
    YBAR_KT = 'Ybar_kt'

    # Residuals
    R1 = 'R1'
    R2 = 'R2'
    R3 = 'R3'
    R4 = 'R4'
    R5 = 'R5'
```

2. Use constants consistently:
```python
out[ColumnNames.MEAN] = ...
cols_to_keep = [ColumnNames.RSG, ColumnNames.MEAN, ColumnNames.LCL]
```

3. Consider enum for type safety:
```python
from enum import Enum

class Column(str, Enum):
    RSG = 'rsg'
    MEAN = 'mean'
    # ...
```

### Files Affected
- `analysis_dataset.py` (lines 193, 226, 321, 421, 815, 900, 965, 1044, 1076)
- `objects.py` (lines 8-10)

### Acceptance Criteria
- [ ] All column names defined as constants
- [ ] No hardcoded column name strings
- [ ] Constants used throughout codebase
- [ ] Documentation of column meanings

---

## Issue #18: Mixed Calculation Responsibilities
**Labels**: `refactor`, `priority-medium`, `architecture`
**Milestone**: Phase 2 - Code Quality

### Description
`AnalysisDataSet` class has too many responsibilities (25+ private methods):
- Data preparation and validation
- Residual calculations (R1, R2, R3, R4, R5)
- Mean calculations (Ybar, Ybar_k, Ybar_t, Ybar_kt)
- Centered residual calculations (RCR1-RCR5)
- Interaction calculations
- Effect calculations

This violates Single Responsibility Principle.

### Impact
- **Severity**: Medium
- Hard to test individual pieces
- Hard to understand
- Difficult to reuse logic
- Long class (700+ lines)

### Solution
Split into focused classes:

```python
class DataPreparation:
    """Handles data validation and preparation."""
    def validate(self, df, spec) -> pd.DataFrame: ...
    def add_grouping(self, df, spec) -> pd.DataFrame: ...
    def filter_groups(self, df, spec) -> pd.DataFrame: ...

class ResidualCalculator:
    """Calculates residuals R1-R5."""
    def calculate_r1(self, df) -> pd.Series: ...
    def calculate_r2(self, df, sds) -> pd.Series: ...
    # ...

class EffectsCalculator:
    """Calculates main effects and interactions."""
    def calculate_main_effects(self, df, factors) -> Dict: ...
    def calculate_interactions(self, df, factors) -> Dict: ...

class AnalysisDataSet:
    """Coordinates analysis data preparation."""
    def __init__(self, df, spec):
        self.prep = DataPreparation()
        self.residuals = ResidualCalculator()
        self.effects = EffectsCalculator()
        # Compose the pieces
```

### Files Affected
- `analysis_dataset.py` (lines 1400-1657)

### Acceptance Criteria
- [ ] Each class has single responsibility
- [ ] Classes are independently testable
- [ ] Logic can be reused
- [ ] Clear composition/delegation

---

## Issue #19: Incomplete Docstrings with Typos
**Labels**: `documentation`, `priority-low`, `cleanup`
**Milestone**: Phase 3 - Enhancement

### Description
Docstrings contain errors:
- Corrupted text: "Prselves. A family ooducts" (lines 16-22)
- Copy-paste errors from template
- Incomplete documentation (lines 1099-1101)

### Impact
- **Severity**: Low
- Confusing documentation
- Unprofessional
- Hard to understand code

### Solution
1. Review and rewrite all docstrings
2. Remove template/placeholder text
3. Ensure docstrings are:
   - Accurate
   - Complete
   - Proofread
   - Following consistent style (NumPy/Google)

### Files Affected
- `analysis_dataset.py` (lines 16-22, 1099-1101)

### Acceptance Criteria
- [ ] No corrupted text in docstrings
- [ ] All docstrings proofread
- [ ] Consistent style throughout
- [ ] Sphinx can generate docs without warnings

---

## Issue #20: TODOs Indicating Technical Debt
**Labels**: `technical-debt`, `priority-medium`, `tracking`
**Milestone**: Phase 2 - Code Quality

### Description
18 TODO comments throughout the code indicating unfinished work:
- "TODO: update to use analysis spec" (line 125)
- "TODO: Ugh! Refactor this mess..." (line 180)
- "TODO: replace with spec" (line 192)
- "TODO: Centralize constants" (line 678)
- "TODO: Make loop to test all analysis types" (line 21 in tests)

And many more...

### Impact
- **Severity**: Medium
- Technical debt not tracked
- Unclear which TODOs are important
- Lost context over time
- Code feels incomplete

### Solution
1. Review each TODO comment
2. For each TODO, either:
   - **Do it now** if simple
   - **Create GitHub issue** with context and priority
   - **Delete** if no longer relevant
3. Link TODO comments to issues:
```python
# TODO(#42): Refactor to use AnalysisSpecification object
```

4. Set policy: No TODOs without linked issues

### Files Affected
- `analysis_dataset.py` (lines 125, 180, 192, 441, 453-454, 646, 678, 932, 973, 1031, 1043, 1269, 1272-1273, 1387)
- Test files

### Acceptance Criteria
- [ ] All TODOs reviewed
- [ ] TODOs either resolved or linked to issues
- [ ] No orphan TODO comments
- [ ] Policy documented for new TODOs

---

## Issue #21: Inconsistent Method Access Patterns
**Labels**: `refactor`, `priority-low`, `code-quality`
**Milestone**: Phase 3 - Enhancement

### Description
Properties in `AnalysisDataSet` don't use `@property` decorator:
- Methods named like properties (lines 1249-1265)
- Reference `self` attributes without returning them
- Look like getters but don't get

Example:
```python
def sampling_design_state(self) -> int:
    self.sampling_design_state  # No return!
```

### Impact
- **Severity**: Low
- Properties return None
- Confusing API
- Not Pythonic

### Solution
Either:
1. Make them proper properties:
```python
@property
def sampling_design_state(self) -> int:
    return self._sampling_design_state
```

2. Or make them regular methods:
```python
def get_sampling_design_state(self) -> int:
    return self._sampling_design_state
```

3. Choose one pattern and be consistent

### Files Affected
- `analysis_dataset.py` (lines 1249-1265)

### Acceptance Criteria
- [ ] All properties use @property decorator or are renamed
- [ ] All properties return values
- [ ] Consistent naming pattern
- [ ] Tests verify property access

---

## Issue #22: Lambda Functions in Data Processing
**Labels**: `performance`, `priority-low`, `readability`
**Milestone**: Phase 3 - Enhancement

### Description
Complex lambda functions in `apply()` operations:
```python
xbar[['lcl', 'ucl']] = xbar.apply(
    lambda row: obj.calculate_limits(mean=row['Xbar'],
    sd=row['S'], N=row[n_to_use], limits_type='Xbar', round_to=spec.round_to), axis=1
)
```

Found in lines 167-170, 172-175, 790, 872-874, 1017-1020, 1023-1026

### Impact
- **Severity**: Low
- Hard to debug
- Can't test lambda independently
- Can be slower than vectorized operations
- Hard to profile

### Solution
1. Replace with named functions for complex logic:
```python
def calculate_xbar_limits(row, spec, n_col):
    return obj.calculate_limits(
        mean=row['Xbar'],
        sd=row['S'],
        N=row[n_col],
        limits_type='Xbar',
        round_to=spec.round_to
    )

xbar[['lcl', 'ucl']] = xbar.apply(
    calculate_xbar_limits,
    axis=1,
    args=(spec, n_to_use)
)
```

2. Consider vectorized operations where possible:
```python
# Instead of apply with lambda
xbar['lcl'] = xbar['Xbar'] - 3 * xbar['S'] / np.sqrt(xbar['N'])
xbar['ucl'] = xbar['Xbar'] + 3 * xbar['S'] / np.sqrt(xbar['N'])
```

3. Profile to verify performance improvements

### Files Affected
- `analysis_dataset.py` (lines 167-170, 172-175, 790, 872-874, 1017-1020, 1023-1026)

### Acceptance Criteria
- [ ] Complex lambdas replaced with named functions
- [ ] Vectorized operations used where possible
- [ ] Code is more readable
- [ ] Performance benchmarked

---

## Issue #23: Import and Package Structure Issues
**Labels**: `technical-debt`, `priority-medium`, `architecture`
**Milestone**: Phase 2 - Code Quality

### Description
Package structure issues:
- Relative import `import objects as obj` instead of absolute (line 12)
- Code in root directory instead of package
- Test import `import analysis_dataset as ad` without package (line 3 in tests)
- No `__init__.py` for package

### Impact
- **Severity**: Medium
- Hard to install as package
- Import errors in different contexts
- Can't use pip install -e
- No clear package boundary

### Solution
1. Reorganize into proper package structure:
```
processbehavior/
├── setup.py
├── README.md
├── processbehavior/
│   ├── __init__.py
│   ├── analysis.py (was analysis_dataset.py)
│   ├── objects.py
│   ├── constants.py
│   └── exceptions.py
└── tests/
    ├── __init__.py
    ├── test_analysis.py
    └── test_specifications.py
```

2. Use absolute imports:
```python
from processbehavior import objects
from processbehavior.constants import ColumnNames
```

3. Create setup.py for installation:
```python
from setuptools import setup, find_packages

setup(
    name='processbehavior',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scipy'],
    python_requires='>=3.8',
)
```

4. Update all imports

### Files Affected
- `analysis_dataset.py` (line 12)
- `tests/test_analysis_dataset.py` (line 3)
- Project structure

### Acceptance Criteria
- [ ] Proper package structure
- [ ] Absolute imports throughout
- [ ] pip installable
- [ ] Tests work with package imports

---

## Issue #24: Missing Tests for Edge Cases
**Labels**: `testing`, `priority-medium`, `quality`
**Milestone**: Phase 2 - Code Quality

### Description
No tests for important edge cases:
- Empty datasets
- All NaN values
- Single observation (n=1)
- Negative numbers
- Zero values
- Division by zero scenarios
- Invalid specifications
- Type mismatches
- Extremely large/small values

### Impact
- **Severity**: Medium
- Unknown behavior in edge cases
- Crashes in production
- Low confidence in refactoring
- Missing error handling

### Solution
1. Add comprehensive edge case tests:
```python
def test_empty_dataframe(self):
    """Test with empty DataFrame."""
    df = pd.DataFrame()
    spec = {...}
    with self.assertRaises(DataValidationError):
        ad.perform_analysis(df, spec)

def test_all_nan_values(self):
    """Test with all NaN values in response."""
    df = pd.DataFrame({'y': [np.nan, np.nan]})
    # ...

def test_single_observation(self):
    """Test with single observation."""
    df = pd.DataFrame({'y': [1.0]})
    # ...

def test_zero_standard_deviation(self):
    """Test when all values are identical."""
    df = pd.DataFrame({'y': [1.0, 1.0, 1.0]})
    # ...
```

2. Add boundary value tests
3. Add invalid input tests
4. Add type error tests
5. Aim for >90% code coverage

### Files Affected
- `tests/test_analysis_dataset.py`
- `tests/test_analysis_specifications.py`

### Acceptance Criteria
- [ ] Edge case tests added for all major functions
- [ ] Error conditions tested
- [ ] Boundary values tested
- [ ] Code coverage >80%
- [ ] All tests pass

---

## Issue #25: Test Code Should Not Import Logging
**Labels**: `technical-debt`, `priority-low`, `testing`
**Milestone**: Phase 3 - Enhancement

### Description
Test code was recently refactored to use logging instead of print (lines 1-14 in test files). However:
- Tests should be simple and focused on assertions
- Logging in tests creates side effects
- Production code should handle logging
- Tests should verify behavior, not log it

### Impact
- **Severity**: Low
- Tests harder to read
- Unnecessary complexity
- Mixing concerns

### Solution
1. Remove logging imports from test files
2. Keep only assertions in tests
3. Move logging to production code where it belongs
4. If debugging needed, use unittest's built-in verbose mode
5. For data inspection, use unittest's self.subTest or print in failing tests only

Example:
```python
# Instead of:
logger.info("Testing group means")
self.assertEqual(result['mR'], 1)

# Just use:
self.assertEqual(result['mR'], 1)

# For debugging specific failures:
with self.subTest(msg=f"Testing group {key}"):
    self.assertEqual(result[key]['mR'], expected)
```

### Files Affected
- `tests/test_analysis_dataset.py` (lines 1-14, and logger calls throughout)
- `tests/test_analysis_specifications.py` (lines 1-11)

### Acceptance Criteria
- [ ] No logging in test files
- [ ] Tests use only assertions
- [ ] Test output is cleaner
- [ ] subTest used for context where needed

---

## Summary

These 25 issues are prioritized into 3 phases:

**Phase 1 (Critical - 9 issues)**: Core architecture and correctness
- Issues #1, #2, #3, #4, #6, #9, #13, #14, #16

**Phase 2 (Important - 13 issues)**: Code quality and maintainability
- Issues #5, #7, #8, #10, #11, #15, #17, #18, #20, #23, #24

**Phase 3 (Enhancement - 3 issues)**: Polish and optimization
- Issues #12, #19, #21, #22, #25

### Labels to Create:
- `refactor`
- `technical-debt`
- `bug`
- `documentation`
- `testing`
- `performance`
- `priority-high`
- `priority-medium`
- `priority-low`
- `type-safety`
- `code-quality`
- `correctness`
- `cleanup`
- `error-handling`
- `code-duplication`
- `api-design`
- `logging`
- `maintainability`
- `architecture`
- `tracking`
- `readability`
- `quality`

### Milestones to Create:
- `Phase 1 - Core Architecture`
- `Phase 2 - Code Quality`
- `Phase 3 - Enhancement`
