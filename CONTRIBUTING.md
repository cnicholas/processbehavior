# Contributing to processbehavior

Thanks for your interest. processbehavior is an implementation of Thomas A.
Bishop's Variance Analysis System (VAS); methodology fidelity is more
important than feature volume. Smaller, well-validated changes are preferred.

## Getting set up

```bash
git clone https://github.com/cnicholas/processbehavior.git
cd processbehavior
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The `[dev]` extra pulls in `[test]`, `[lint]`, `[images]`, and `[excel]`.
If you only need a subset, install them directly:

```bash
pip install -e ".[test]"   # pytest + pyarrow + openpyxl
pip install -e ".[lint]"   # ruff + mypy + pre-commit
pip install -e ".[docs]"   # jupyter-book + ipykernel
```

## Running checks

```bash
pytest tests/                # full suite
pytest tests/ -m "not slow"  # skip the perf benchmarks
ruff check .
mypy processbehavior
```

The full test suite is the gate: if it fails, the change is not ready.

## Validating against Bishop's reference data

The file `validation/TABVASTESTDATABASE.csv` is ground-truth Minitab output
for DS 1–6 from Bishop's reference materials. Numerical changes that touch
residuals (R1–R5), DS detection, limit calculations, or variance
decomposition must keep the relevant Bishop-reference tests green:

```bash
pytest tests/test_bishop_reference.py -v
```

If a Bishop test fails after your change, the change is wrong — not the
reference.

## API contract

The current public surface is the two-step analyst workflow:

```python
import processbehavior as pb

study = pb.ProcessBehavior(df).formulate(response=..., time=..., factors=[...])
result = study.execute()
```

Anything exported from `processbehavior/__init__.py` is part of the public
API. Breaking changes to those symbols require a `## [Unreleased]` entry in
`CHANGELOG.md` flagged as **Breaking**.

## Pull requests

- Branch off `main`, keep PRs focused on one concern
- Add or update tests for behavioral changes
- Update `CHANGELOG.md` under `## [Unreleased]`
- Confirm `pytest tests/`, `ruff check .`, and `mypy processbehavior` pass
- Use the commit-message structure in the PR template

## Reporting bugs

Open an issue at <https://github.com/cnicholas/processbehavior/issues> with:
- a minimal reproducer (DataFrame snippet or `make_sds(...)` call)
- the version of processbehavior, pandas, and Python
- the full traceback or wrong output

For security-sensitive issues, use the private disclosure path in
`SECURITY.md` instead of opening a public issue.

## Code of Conduct

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
