# Contributing
- Create a virtual env (`python3 -m venv .venv && source .venv/bin/activate`)
- `pip install -e ".[dev]"`
- `pytest -q`, `ruff check .`, `mypy processbehavior`
- Keep public API stable: `analyze(df, AnalysisSpec)` returning {"charts":..., "meta":...}
