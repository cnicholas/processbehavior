# Control Chart in Python

"Control chart" covers a family: individuals (X/mR) for one-measurement-per-
period data, Xbar/S for subgrouped data, histograms for distribution checks.
Picking the wrong family is the most common charting error — so
ProcessBehavior looks at your data's structure first and recommends the right
one.

## The 60-second version

```python
from processbehavior import load_coffee_shop

study = load_coffee_shop().formulate(
    response='wait_sec', factors=['daypart'], time='date')

print(f"Recommended chart: {study.recommended_chart}")

result = study.execute(companion=True)
stats = result.get_statistics('Xbar')
print(f"center={stats['center']}, limits=({stats['lpl']}, {stats['upl']})")
print(f"Signals on Xbar: {result.detect_signals(chart='Xbar').count}")
result.plot()
```

Output:

```
Recommended chart: Xbar
center=220.771, limits=(180.351, 261.19)
Signals on Xbar: 36
```

## What just happened

`formulate()` classified the data's structure — repeated measurements per
(daypart × date) cell — and recommended an Xbar chart, whose limits come from
*within-subgroup* variation. The 36 signals are real: the coffee-shop demo
data carries a process-improvement story, and the chart finds it. Every
chart's statistics share the same four-key contract: `{N, center, lpl, upl}`.

If your data had been one measurement per period instead, the recommendation
would have been an X chart — same code, different structure, right limits
either way.

## Going further

- The full story behind this dataset: [Coffee Shop — A Complete
  Story](../tutorials/coffee-shop.ipynb).
- Which chart when, and why: [Chart Types](../user-guide/chart-types.md).
- Subgrouped charts in depth: [Xbar-S
  Analysis](../tutorials/xbar-s-analysis.ipynb).
