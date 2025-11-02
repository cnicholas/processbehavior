# Package Structure Migration Checklist

## Goal
Move all core modules into `processbehavior/` package directory and update all imports to create a proper installable Python package.

## Estimated Time: 1-2 hours

---

## Phase 1: Preparation (5 min)

### Step 1.1: Create a new branch
```bash
git checkout -b package-structure
```

### Step 1.2: Verify current tests pass
```bash
python -m pytest tests/ -v
# Should show: 235 passed
```

### Step 1.3: Take inventory
Current root-level files to move:
- [ ] analysis_dataset.py
- [ ] analysis_result.py
- [ ] analysis_specification.py
- [ ] analysis.py
- [ ] data_preparation.py
- [ ] effects_calculator.py
- [ ] process_dataframe.py
- [ ] residual_calculator.py
- [ ] sds_detector.py
- [ ] spc_constants.py

---

## Phase 2: Move Files (10 min)

### Step 2.1: Move core modules into processbehavior/
```bash
# Move all core modules
mv analysis_dataset.py processbehavior/
mv analysis_result.py processbehavior/
mv analysis_specification.py processbehavior/
mv analysis.py processbehavior/
mv data_preparation.py processbehavior/
mv effects_calculator.py processbehavior/
mv process_dataframe.py processbehavior/
mv residual_calculator.py processbehavior/
mv sds_detector.py processbehavior/
mv spc_constants.py processbehavior/
```

### Step 2.2: Verify the move
```bash
ls processbehavior/*.py
# Should show all 10 files plus __init__.py
```

**✓ Checkpoint**: Files are now in correct location

---

## Phase 3: Update Package __init__.py (10 min)

### Step 3.1: Update processbehavior/__init__.py

Replace contents with:
```python
"""
ProcessBehavior - Statistical Process Control for Python

A Pythonic library for process behavior analysis following Wheeler/Bishop methodology.
Provides auto-detection of Sampling Design States (SDS) and appropriate control chart
analysis with variance decomposition.

Quick Start
-----------
    from processbehavior import ProcessDataFrame

    # Wrap your DataFrame
    pdf = ProcessDataFrame(df)

    # Auto-detect SDS and run analysis
    result = pdf.analyze(
        response_var=pdf.columns.measurement,
        time_var=pdf.columns.time,
        grouping_vars=[pdf.columns.line]
    )

    # Access results
    print(result.summary)
    xbar_chart = result.get_chart('Xbar')

    # Export to Excel
    result.to_excel('analysis.xlsx')

Main Classes
------------
ProcessDataFrame : Main user-facing API with auto-completion
AnalysisResult : Unified result container
Analysis : Core analysis engine
AnalysisDataSet : Analysis dataset manager
"""

__version__ = "0.3.0"

# Main user-facing API
from processbehavior.process_dataframe import ProcessDataFrame

# Result object
from processbehavior.analysis_result import AnalysisResult

# Core analysis classes
from processbehavior.analysis import Analysis
from processbehavior.analysis_dataset import AnalysisDataSet
from processbehavior.analysis_specification import AnalysisSpecification

# Utility classes (advanced users)
from processbehavior.sds_detector import SamplingDesignDetector
from processbehavior.data_preparation import DataPreparation
from processbehavior.effects_calculator import EffectsCalculator
from processbehavior.residual_calculator import ResidualCalculator

__all__ = [
    # Main API
    'ProcessDataFrame',
    'AnalysisResult',

    # Core classes
    'Analysis',
    'AnalysisDataSet',
    'AnalysisSpecification',

    # Utilities
    'SamplingDesignDetector',
    'DataPreparation',
    'EffectsCalculator',
    'ResidualCalculator',
]
```

**✓ Checkpoint**: Package exports are defined

---

## Phase 4: Update Internal Imports (20 min)

### Step 4.1: Update imports in processbehavior/*.py files

For each module in `processbehavior/`, update imports to use relative imports:

**In processbehavior/analysis_dataset.py:**
```python
# OLD
from analysis_specification import AnalysisSpecification
from data_preparation import DataPreparation
import spc_constants as spc

# NEW
from .analysis_specification import AnalysisSpecification
from .data_preparation import DataPreparation
from . import spc_constants as spc
```

**In processbehavior/analysis_result.py:**
```python
# No changes needed (uses standard library only)
```

**In processbehavior/process_dataframe.py:**
```python
# OLD
from analysis_specification import AnalysisSpecification
from analysis_dataset import Analysis
from sds_detector import SamplingDesignDetector

# NEW
from .analysis_specification import AnalysisSpecification
from .analysis_dataset import Analysis
from .sds_detector import SamplingDesignDetector
```

**In processbehavior/data_preparation.py:**
```python
# Check for any cross-module imports and update
```

**In processbehavior/effects_calculator.py:**
```python
# OLD
import spc_constants as spc

# NEW
from . import spc_constants as spc
```

**In processbehavior/residual_calculator.py:**
```python
# OLD
from effects_calculator import EffectsCalculator
import spc_constants as spc

# NEW
from .effects_calculator import EffectsCalculator
from . import spc_constants as spc
```

**In processbehavior/sds_detector.py:**
```python
# Check for any imports - likely none needed
```

**Strategy**: Search each file for these patterns:
```bash
# Find all absolute imports in processbehavior/*.py
cd processbehavior
grep "^import analysis_" *.py
grep "^from analysis_" *.py
grep "^import data_" *.py
grep "^from data_" *.py
grep "^import effects_" *.py
grep "^import residual_" *.py
grep "^import sds_" *.py
grep "^import spc_" *.py
grep "^import process_" *.py
```

**✓ Checkpoint**: Internal package imports use relative imports

---

## Phase 5: Update Test Imports (30 min)

### Step 5.1: Update all test files

Pattern to change in ALL test files:
```python
# OLD
import analysis_dataset as ad
from analysis_specification import AnalysisSpecification
from sds_detector import SamplingDesignDetector

# NEW
from processbehavior import analysis_dataset as ad
from processbehavior.analysis_specification import AnalysisSpecification
from processbehavior.sds_detector import SamplingDesignDetector
```

Files to update:
- [ ] tests/test_analysis_dataset.py
- [ ] tests/test_analysis_specifications.py
- [ ] tests/test_data_preparation.py
- [ ] tests/test_effects_calculator.py
- [ ] tests/test_excel_export.py
- [ ] tests/test_process_dataframe.py
- [ ] tests/test_residual_calculator.py
- [ ] tests/test_sds_detector.py
- [ ] tests/test_sds.py
- [ ] tests/test_sds1.py
- [ ] tests/test_spc_constants.py

**Quick method**: Use find/replace in your editor
- Find: `import analysis_dataset as ad`
- Replace: `from processbehavior import analysis_dataset as ad`

- Find: `^from analysis_specification import`
- Replace: `from processbehavior.analysis_specification import`

- Find: `^from sds_detector import`
- Replace: `from processbehavior.sds_detector import`

... and so on for all modules

**✓ Checkpoint**: Run tests after each file update
```bash
python -m pytest tests/test_analysis_dataset.py -v
```

---

## Phase 6: Update Example Files (15 min)

### Step 6.1: Update example scripts

Files to update:
- [ ] examples/analysis_result_demo.py
- [ ] examples/demo_new_api.py
- [ ] examples/excel_export_demo.py
- [ ] examples/process_dataframe_demo.ipynb (if needed)

Same pattern:
```python
# OLD
import analysis_dataset as ad
from process_dataframe import ProcessDataFrame

# NEW
from processbehavior import analysis_dataset as ad
from processbehavior import ProcessDataFrame
```

**✓ Checkpoint**: Run each example to verify
```bash
python examples/excel_export_demo.py
```

---

## Phase 7: Update pyproject.toml (5 min)

### Step 7.1: Add openpyxl to optional dependencies

In `pyproject.toml`, update:
```toml
[project.optional-dependencies]
changepoints = ["ruptures>=1.1"]
effects = ["statsmodels>=0.14"]
excel = ["openpyxl>=3.1"]  # ADD THIS LINE
dev = [
  "pytest>=8",
  "pytest-cov>=5",
  "ruff>=0.5",
  "mypy>=1.10",
  "pre-commit>=3.7",
  "ipykernel",
  "openpyxl>=3.1",  # ADD THIS LINE (for testing Excel export)
]
```

### Step 7.2: Update email in authors
```toml
authors = [{name = "cnicholas", email = "chris.nicholas@gmail.com"}]
```

**✓ Checkpoint**: Validate pyproject.toml
```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

---

## Phase 8: Clean Up (5 min)

### Step 8.1: Remove any leftover files at root
```bash
# Check for any .pyc files or __pycache__ at root
rm -rf __pycache__

# Remove any old/temp files
ls *.py  # Should show nothing (all moved to processbehavior/)
```

### Step 8.2: Check for stray imports
```bash
# Search for old-style imports in the codebase
grep -r "^import analysis_dataset" --include="*.py" .
grep -r "^import sds_detector" --include="*.py" .
# Should return nothing (except maybe in this checklist!)
```

**✓ Checkpoint**: Root directory is clean

---

## Phase 9: Testing (20 min)

### Step 9.1: Run full test suite
```bash
python -m pytest tests/ -v
# Should show: 235 passed
```

### Step 9.2: Test package imports
```bash
python -c "from processbehavior import ProcessDataFrame; print('✓ ProcessDataFrame imported')"
python -c "from processbehavior import AnalysisResult; print('✓ AnalysisResult imported')"
python -c "from processbehavior import Analysis; print('✓ Analysis imported')"
python -c "from processbehavior import SamplingDesignDetector; print('✓ SamplingDesignDetector imported')"
```

### Step 9.3: Test a complete workflow
```bash
python -c "
from processbehavior.datasets import make_sds1
from processbehavior import ProcessDataFrame

df = make_sds1(K=2, T=5, n_min=2, n_max=3, seed=42)
pdf = ProcessDataFrame(df)
result = pdf.analyze(
    response_var=pdf.columns.y,
    time_var=pdf.columns.time,
    grouping_vars=[pdf.columns.factor_1]
)
print('✓ Full workflow works!')
print(f'SDS: {result.sds}')
"
```

### Step 9.4: Test Excel export
```bash
python -c "
from processbehavior.datasets import make_sds1
from processbehavior import Analysis

df = make_sds1(K=2, T=5, n_min=2, n_max=3, seed=42)
spec = {
    'analysis_type': 'Xbar',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y'
}
analysis = Analysis(df, spec)
result = analysis.calculate()
result.to_excel('test_output.xlsx')
import os
assert os.path.exists('test_output.xlsx')
os.remove('test_output.xlsx')
print('✓ Excel export works!')
"
```

### Step 9.5: Run examples
```bash
cd examples
python excel_export_demo.py
rm -f example*.xlsx  # Clean up
cd ..
```

**✓ Checkpoint**: All tests pass, all functionality works

---

## Phase 10: Build & Install Test (10 min)

### Step 10.1: Build the package
```bash
pip install build
python -m build
```

Should create:
- `dist/processbehavior-0.3.0.tar.gz`
- `dist/processbehavior-0.3.0-py3-none-any.whl`

### Step 10.2: Test install in a clean environment (optional but recommended)
```bash
# Create a test venv
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install the wheel
pip install dist/processbehavior-0.3.0-py3-none-any.whl[excel]

# Test import
python -c "from processbehavior import ProcessDataFrame; print('✓ Package installed successfully')"

# Deactivate and clean up
deactivate
rm -rf test_env
```

**✓ Checkpoint**: Package builds and installs correctly

---

## Phase 11: Commit (5 min)

### Step 11.1: Review changes
```bash
git status
git diff pyproject.toml
```

### Step 11.2: Commit the restructuring
```bash
git add -A
git commit -m "Restructure project into proper package layout

Move all core modules into processbehavior/ package directory:
- Move 10 core modules from root to processbehavior/
- Update all imports to use package structure
- Update internal imports to use relative imports
- Update test imports to reference processbehavior package
- Update example scripts with correct imports
- Add openpyxl to optional dependencies for Excel export
- Update package __init__.py with proper exports

Package is now properly structured for PyPI distribution.
All 235 tests passing."
```

### Step 11.3: Merge to main (after verification)
```bash
git checkout main
git merge package-structure
git branch -d package-structure
```

**✓ Checkpoint**: Changes committed and merged

---

## Phase 12: Verification Checklist

### Final checks:
- [ ] All 235 tests passing
- [ ] Package builds without errors (`python -m build`)
- [ ] All examples run successfully
- [ ] Package imports work correctly
- [ ] Excel export functionality works
- [ ] No absolute imports remain in package code (only relative imports)
- [ ] pyproject.toml updated with openpyxl dependency
- [ ] README still accurate (check if import examples need updating)

---

## Troubleshooting Guide

### Issue: ImportError: No module named 'analysis_dataset'
**Fix**: Check you updated the import to `from processbehavior import analysis_dataset`

### Issue: ImportError: attempted relative import beyond top-level package
**Fix**: Check you're not using `from ..` when you should use `from .`

### Issue: Circular import detected
**Fix**: Rare, but if occurs:
1. Check which modules import each other
2. Consider moving shared code to a separate module
3. Use TYPE_CHECKING for type hints only imports

### Issue: Tests can't find processbehavior module
**Fix**: Make sure you're running pytest from the project root:
```bash
cd /Users/nicholas/Documents/projects/processbehavior
python -m pytest tests/ -v
```

### Issue: Some tests pass, some fail after import changes
**Fix**: Check each failing test individually - likely missed an import update
```bash
python -m pytest tests/test_failing_test.py -v
```

---

## Quick Reference: Import Patterns

### In package modules (processbehavior/*.py)
```python
# Relative imports within package
from .analysis_specification import AnalysisSpecification
from .data_preparation import DataPreparation
from . import spc_constants as spc
```

### In tests (tests/*.py)
```python
# Absolute imports from package
from processbehavior import analysis_dataset as ad
from processbehavior.analysis_specification import AnalysisSpecification
from processbehavior import ProcessDataFrame
```

### In examples (examples/*.py)
```python
# Absolute imports from package
from processbehavior import ProcessDataFrame, Analysis
from processbehavior.datasets import make_sds1
```

### In user code (after pip install)
```python
# Clean imports for end users
from processbehavior import ProcessDataFrame

# Or more specific
from processbehavior import Analysis, AnalysisResult
from processbehavior.datasets import make_sds1
```

---

## Success Criteria

✅ Package structure matches standard Python package layout
✅ All imports use proper package namespacing
✅ All 235 tests passing
✅ Package builds successfully with `python -m build`
✅ Examples run without errors
✅ Can be installed with `pip install dist/*.whl`
✅ Users can `from processbehavior import ProcessDataFrame`

---

## Next Steps After Completion

Once package structure is complete, you'll be ready for:
1. **PyPI upload**: `twine upload dist/*`
2. **Documentation**: Consider Sphinx docs
3. **CI/CD**: GitHub Actions for automated testing
4. **Version management**: Semantic versioning strategy
5. **CHANGELOG**: Track changes for users

---

## Time Estimates by Phase

- Phase 1: Preparation - 5 min
- Phase 2: Move files - 10 min
- Phase 3: Update __init__ - 10 min
- Phase 4: Internal imports - 20 min
- Phase 5: Test imports - 30 min
- Phase 6: Example imports - 15 min
- Phase 7: pyproject.toml - 5 min
- Phase 8: Clean up - 5 min
- Phase 9: Testing - 20 min
- Phase 10: Build test - 10 min
- Phase 11: Commit - 5 min
- Phase 12: Verification - 5 min

**Total: ~2 hours** (including testing time)

---

Good luck! Take it one phase at a time, and test frequently. The structure is straightforward - it's just moving files and updating imports systematically.
