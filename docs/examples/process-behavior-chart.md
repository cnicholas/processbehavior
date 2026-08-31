# Process Behavior Chart

**Process behavior chart** is Donald Wheeler's name for what most textbooks
call a control chart — and the rename is the point. "Control" suggests the
chart's job is keeping a process inside specifications; Wheeler's term says
what the chart actually does: it characterizes how the process *behaves*, so
you can tell routine variation from a genuine change. Same math (Shewhart's),
better name. This library takes its own name from Wheeler's usage; the
[terminology appendix](../appendix/wheeler-terminology.md) maps his vocabulary
to the textbook terms.

## The 60-second version

```python
import numpy as np
import pandas as pd
from processbehavior import ProcessBehavior

rng = np.random.default_rng(12)
temps = rng.normal(150.0, 3.0, 30)
temps[20:] -= 8            # the process changed at batch 21

df = pd.DataFrame({'batch': range(1, 31), 'seal_temp': temps.round(1)})

study = ProcessBehavior(df).formulate(response='seal_temp', time='batch')
result = study.execute(chart='X', companion=True)

s = result.get_statistics('X')
print(f"Center line: {s['center']}")
print(f"Natural process limits: ({s['lpl']}, {s['upl']})")
print(f"Signals: {result.detect_signals(chart='X').count}")
result.plot()
```

Output:

```
Center line: 147.38
Natural process limits: (139.07, 155.69)
Signals: 26
```

## Reading it in Wheeler's terms

- The limits are **natural process limits** — the voice of the process, not
  spec limits (the voice of the customer). They say what the process *will*
  do, not what you wish it would.
- The planted change at batch 21 lights the chart up: the run of points above
  the center line before the drop and below it after both violate run rules,
  on top of the points beyond the limits. Routine variation doesn't do that —
  that is the signal/noise distinction the chart exists to make.
- **Predictable** (Wheeler) = "in control" (textbook): only routine variation.
  This process is not predictable — something changed, go find it.

## Going further

- [Your First X/mR Chart](../tutorials/first-xmr-chart.ipynb) — the guided
  version of exactly this example.
- [Wheeler terminology](../appendix/wheeler-terminology.md) — the full
  vocabulary map.
- [Design-State Tour](../tutorials/design-state-tour.ipynb) — what happens
  when your data has structure (factors, replication).
