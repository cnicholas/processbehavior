# XmR Chart in Python

An **XmR chart** pairs an individuals chart (X — each measurement plotted in
time order) with a moving-range chart (mR — the gap between consecutive
measurements). It is the right chart whenever you have one measurement per
period: weekly KPIs, daily yields, per-batch readings.

## The 60-second version

```python
import numpy as np
import pandas as pd
from processbehavior import ProcessBehavior

rng = np.random.default_rng(3)
df = pd.DataFrame({
    'week': range(1, 25),
    'on_time_pct': np.clip(rng.normal(92, 2.5, 24), 80, 100).round(1),
})

study = ProcessBehavior(df).formulate(response='on_time_pct', time='week')
result = study.execute(chart='X', companion=True)

print(result.get_statistics('X'))
print(result.get_statistics('mR'))
result.plot()   # interactive plotly figure: X on top, mR below
```

Output:

```
{'N': 1, 'center': 91.883, 'lpl': 83.209, 'upl': 100.557}
{'N': 2, 'center': 3.261, 'lpl': 0.0, 'upl': 10.657}
```

## Reading it

- `center` on the X chart is the process's typical level; `lpl`/`upl` are the
  **natural process limits**, computed from the moving ranges (not from the
  overall standard deviation — that classic shortcut inflates the limits).
- The mR chart watches variation: a spike there with a level shift on X says
  "the level moved once"; a widening mR says the process got noisier.
- `companion=True` is what makes it an X**mR** chart — `execute(chart='X')`
  alone returns just the individuals chart.

## Going further

- Signals beyond eyeballing: `result.detect_signals(chart='X')` applies the
  full set of Western Electric run and zone rules (all eight apply to X and mR
  charts).
- Guided version with a planted shift: [Your First X/mR
  Chart](../tutorials/first-xmr-chart.ipynb).
- Multiple streams (one XmR per machine/lane): [Stratified
  Analysis](../tutorials/stratified-analysis.ipynb).
