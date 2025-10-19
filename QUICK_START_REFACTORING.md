# Quick Start: ProcessBehavior Refactoring

## 🚀 Getting Started (First Day)

### 1. Fix the Critical Bug FIRST ⚠️

**Issue #16: Beyond Limits Calculation Bug** (15 minutes)

This bug causes incorrect violation detection. Fix it before anything else:

**File**: `analysis_dataset.py` (lines 317-318, 811-812, 896-897)

**Current (WRONG)**:
```python
out['beyond_limits'] = np.where(out[spec.response_var] < out['lcl'], -1, 0)
out['beyond_limits'] = np.where(out[spec.response_var] > out['ucl'],  1, 0)  # Overwrites!
```

**Fixed**:
```python
out['beyond_limits'] = np.where(
    out[spec.response_var] < out['lcl'],
    -1,
    np.where(out[spec.response_var] > out['ucl'], 1, 0)
)
```

Apply this fix in all 3 locations, commit, and run tests.

### 2. Quick Wins (First Few Hours)

Do these in order for immediate improvements:

```bash
# 1. Remove commented code (30 min)
#    Delete commented blocks in lines 106-117, 696-698

# 2. Fix docstring typos (30 min)
#    Fix corrupted text in lines 16-22, 1099-1101

# 3. Remove unused parameters (15 min)
#    Fix round_to in calculate_limits (line 1660)

# 4. Run tests to ensure nothing broke
python -m unittest discover -s tests -p "test_*.py" -v
```

### 3. Set Up Infrastructure (First Day)

```bash
# Install development tools
pip install mypy pylint black isort

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
black analysis_dataset.py objects.py
isort analysis_dataset.py objects.py
mypy analysis_dataset.py objects.py || true
python -m unittest discover -s tests -p "test_*.py"
EOF
chmod +x .git/hooks/pre-commit

# Initialize type checking
mypy --install-types
```

## 📋 Day-by-Day Plan (First Week)

### Monday: Foundation
- [ ] Fix Issue #16 (critical bug)
- [ ] Complete Quick Wins (#5, #19, #12)
- [ ] Set up type checking infrastructure
- [ ] Review all 25 issues in detail

### Tuesday: Type Safety
- [ ] Issue #3: Extract magic numbers to constants
- [ ] Issue #2: Start adding type hints (Xbar, Sbar classes)
- [ ] Set up mypy in strict mode
- [ ] Run tests after each change

### Wednesday: Error Handling
- [ ] Issue #6: Create custom exception classes
- [ ] Issue #4: Fix abstract methods and properties
- [ ] Update tests to expect new exceptions
- [ ] Document exception hierarchy

### Thursday: Remove Duplication
- [ ] Issue #11: Consolidate column validation
- [ ] Issue #7: Move helper functions to objects.py
- [ ] Remove duplicate imports
- [ ] Clean up objects.py

### Friday: Tests & Documentation
- [ ] Issue #24: Add edge case tests
- [ ] Issue #8: Add docstrings to main classes
- [ ] Review week's progress
- [ ] Commit and push all changes

## 🎯 Top 5 Priority Issues

Focus on these first for maximum impact:

### 1. Issue #16: Beyond Limits Bug 🔴
- **Why**: Active bug in production
- **Effort**: 15 minutes
- **Impact**: Critical correctness fix

### 2. Issue #1: Duplicate Code 🔴
- **Why**: Reduces codebase by 1000+ lines
- **Effort**: 2 days
- **Impact**: Major maintainability improvement

### 3. Issue #3: Magic Numbers 🟡
- **Why**: Makes code understandable
- **Effort**: 1 day
- **Impact**: High clarity improvement

### 4. Issue #2: Type Annotations 🟡
- **Why**: Enables IDE support and catches errors early
- **Effort**: 1 day
- **Impact**: Development experience improvement

### 5. Issue #18: Split Large Class 🟡
- **Why**: Better separation of concerns
- **Effort**: 2 days
- **Impact**: Architecture improvement

## 🛠️ Tools & Commands

### Run Tests
```bash
# All tests
python -m unittest discover -s tests -p "test_*.py" -v

# Specific test
python -m unittest tests.test_analysis_dataset.Test_Dataset_Factory.test_analysis_dataset_sds2 -v

# With coverage
pip install coverage
coverage run -m unittest discover
coverage report
coverage html  # Opens in browser
```

### Type Checking
```bash
# Check types
mypy analysis_dataset.py

# Strict mode
mypy --strict analysis_dataset.py

# Ignore missing imports
mypy --ignore-missing-imports analysis_dataset.py
```

### Code Quality
```bash
# Format code
black analysis_dataset.py objects.py

# Sort imports
isort analysis_dataset.py objects.py

# Lint
pylint analysis_dataset.py

# Find duplicate code
pip install pylint
pylint --disable=all --enable=duplicate-code analysis_dataset.py
```

### Git Workflow
```bash
# Create feature branch
git checkout -b refactor/issue-16-beyond-limits-bug

# Make changes, run tests
python -m unittest discover -s tests -p "test_*.py" -v

# Commit
git add analysis_dataset.py
git commit -m "Fix #16: Correct beyond limits calculation

- Fixed logic to not overwrite lower limit violations
- Applied fix in all 3 locations (lines 317, 811, 896)
- Tests verify both upper and lower violations detected"

# Push
git push origin refactor/issue-16-beyond-limits-bug
```

## 📚 Key Files

| File | Purpose | Lines | Issues |
|------|---------|-------|--------|
| `github_issues.md` | Full issue descriptions | - | All 25 issues |
| `REFACTORING_SUMMARY.md` | Overview and plan | - | Summary |
| `issues.csv` | Import into project tools | - | All issues |
| `create_github_issues.sh` | GitHub issue creation | - | Automation |
| `analysis_dataset.py` | Main code to refactor | 1800 | 23 issues |
| `objects.py` | Utilities and constants | 200 | 5 issues |

## 🎓 Refactoring Principles

### DO:
- ✅ Write tests BEFORE refactoring
- ✅ Make small, incremental changes
- ✅ Run tests after EVERY change
- ✅ Commit frequently with clear messages
- ✅ Extract constants before extracting functions
- ✅ Document why, not what
- ✅ Use type hints everywhere

### DON'T:
- ❌ Refactor without tests
- ❌ Make multiple changes at once
- ❌ Skip running tests "just this once"
- ❌ Leave commented code
- ❌ Use print() for debugging (use logging)
- ❌ Commit broken code
- ❌ Ignore type errors

## 📞 Need Help?

### Questions to Ask:
1. **Does this change break backward compatibility?**
   - If yes, document it and create migration guide

2. **Are tests still passing?**
   - If no, fix immediately before continuing

3. **Is this the minimal change needed?**
   - If no, simplify

4. **Can I test this independently?**
   - If no, extract to smaller function

5. **Would my future self understand this?**
   - If no, add comments/docstrings

### Resources:
- Martin Fowler's Refactoring: https://refactoring.com/
- Python Type Hints: https://mypy.readthedocs.io/
- Clean Code Principles: Robert C. Martin

## 🎉 Success Checklist

After each issue is complete:
- [ ] Tests pass
- [ ] Type checking passes (mypy)
- [ ] Linting passes (pylint)
- [ ] Code formatted (black)
- [ ] Imports sorted (isort)
- [ ] Documentation updated
- [ ] Committed with clear message
- [ ] Issue marked as complete

## 🚦 Red Flags

Stop and reconsider if:
- 🛑 Tests start failing for unclear reasons
- 🛑 Changes are getting too large (>500 lines)
- 🛑 You're not sure why something works
- 🛑 Type errors are accumulating
- 🛑 You're adding technical debt to fix technical debt
- 🛑 You're rushing to meet a deadline

## Next Steps

1. **Today**: Fix Issue #16 (critical bug)
2. **This Week**: Complete Phase 1 foundation
3. **Week 2**: Start Issue #1 (consolidate duplicates)
4. **Week 3-4**: Phase 2 code quality improvements
5. **Week 5-6**: Phase 3 polish and documentation

---

**Remember**: Refactoring is a marathon, not a sprint. Small, tested, incremental changes will get you there safely!

Good luck! 🍀
