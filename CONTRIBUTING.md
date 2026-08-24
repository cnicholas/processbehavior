# Contributing to processbehavior

Thanks for your interest. processbehavior is an implementation of Thomas A.
Bishop's Variance Analysis System (VAS); methodology fidelity is more
important than feature volume. Smaller, well-validated changes are preferred.

## What this library is, and what it is not

processbehavior is the **computational engine** for VAS. It provides the
building blocks and leaves the analysis to the analyst:

- design-state detection and the PDS / ODS / ADS lineage
- the residual system (R1–R6) and variance decomposition
- chart computation and control limits
- signal detection, capability, loss function, maximum information

What it deliberately does **not** provide is the layer above those: an ordered,
gated analytical workflow — which analyses to run for a given study, in what
sequence, under what conditions, and how to read the result of one before
deciding the next. That sequencing is Dr. Bishop's methodology. It is taught,
and it is the subject of his published work; it is not part of this library and
will not be added to it.

Concretely, pull requests along these lines will be declined regardless of
implementation quality:

- a "run the standard analysis" entry point that executes a fixed sequence of
  analyses and returns a narrative report
- ordering, naming, or gating rules that encode which analysis follows which
- helpers whose purpose is to relieve the caller of deciding what to run next

This is a scope boundary, not a missing feature or an oversight. The library's
job is to make each analysis correct and each result inspectable; deciding
*which* analysis answers the question in front of you is the analyst's job, and
learning to make that decision well is what the methodology teaches. Building
blocks stay here. Judgement stays with the practitioner.

If you are unsure which side of that line a contribution falls on, open an issue
before writing code and we will work it out there.

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

The file `validation/PBTESTDATABASE_T100.csv` is ground-truth Minitab output
for DS 1–6 from Bishop's reference materials. Numerical changes that touch
residuals (R1–R5), DS detection, limit calculations, or variance
decomposition must keep the relevant Bishop-reference tests green:

```bash
pytest tests/test_bishop_reference.py -v
```

There is also a full end-to-end check that compares every chart center, control
limit, capability index and loss component against Bishop's published Minitab
results — 280 assertions across ADS 1–3:

```bash
python validation/e2e_bishop_report.py
```

It runs on every push in CI, writes the summary page at
`docs/reference/validation.md`, and CI fails if that page no longer matches what
the run produces. So a change that moves any validated number cannot land
quietly: it either fails the assertions or visibly rewrites the published
summary.

If a Bishop check fails after your change, the change is wrong — not the
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
- Fill in the PR template (`.github/pull_request_template.md`); it is a
  checklist of the invariants most easily broken by accident

## Reporting bugs

Open an issue at <https://github.com/cnicholas/processbehavior/issues> with:
- a minimal reproducer (DataFrame snippet or `pb.make_design(state=..., seed=...)` call)
- the version of processbehavior, pandas, and Python
- the full traceback or wrong output

For security-sensitive issues, use the private disclosure path in
`SECURITY.md` instead of opening a public issue.

## Releasing

Releases are published to PyPI by the `Publish` workflow on tag push.

1. Confirm CI is green on `main`, including the `audit` and `smoke-test` jobs.
2. Move `## [Unreleased]` content into a new `## [X.Y.Z] - YYYY-MM-DD` section
   in `CHANGELOG.md` and update the compare links at the bottom.
3. Bump the version in `processbehavior/__init__.py` (`__version__`).
   `pyproject.toml` reads it via `[tool.hatch.version]`.
4. Commit with subject `Release vX.Y.Z`, push to `main`.
5. Tag and push:
   ```bash
   git tag -s vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
   The tag push triggers `.github/workflows/publish.yml`, which builds,
   verifies with `twine check`, and uploads to PyPI via OIDC trusted
   publishing — no API tokens.
6. Verify the release on https://pypi.org/project/processbehavior/ and run
   `pip install processbehavior==X.Y.Z` in a fresh venv to confirm the
   README quickstart still runs end-to-end.

The PyPI trusted publisher must be configured **once** at
https://pypi.org/manage/project/processbehavior/settings/publishing/
before the first release. Configure repository
`cnicholas/processbehavior`, workflow `publish.yml`, environment `pypi`.

## Code of Conduct

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
