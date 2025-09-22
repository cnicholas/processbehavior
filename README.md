# processbehavior

*Python-native Process Behavior & Control Charts.*

**Spec-driven SPC** that returns tidy DataFrames ready for UI/BI:
- Chart sets: **Xbar–S**, **I–MR** (rational subgroups), optional **R**
- Optional residual charts, rule flags, and (later) change-points & effects
- **pandas in, tidy out** — no plotting side-effects

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

```python
import pandas as pd
from processbehavior import AnalysisSpec, analyze, load_demo

df = load_demo()  # tiny synthetic dataset
spec = AnalysisSpec(response_var="y", time_var="t", grouping=["line"])
result = analyze(df, spec)
print(result["charts"].keys(), result["meta"])
```

## Philosophy
- Keep the **spec** simple and declarative
- Return **consistent, tidy frames** (per-row limits for unequal *n*)
- Make rules/residuals/effects **plug-ins** so analysts can toggle depth
