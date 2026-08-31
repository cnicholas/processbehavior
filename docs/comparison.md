# Compared to Other Python SPC Tools

Several Python packages draw control charts. This page is an honest map of the
landscape: what to look for, where ProcessBehavior sits, and when a different
tool — or a spreadsheet — is the better choice.

## What to look for in a Python SPC library

- **Limits-math provenance** — where do the formulas come from, and can you
  check the numbers against a published reference?
- **Structure awareness** — does the library ask how your data is organized
  (factors, time, replication), or treat every input as one undifferentiated
  stream?
- **Chart types** — individuals (X/mR), subgrouped (Xbar/S), attribute charts,
  EWMA/CUSUM, multivariate?
- **Signal detection** — beyond-limits only, or the run/zone rules too, with
  chart-appropriate applicability?
- **pandas-native workflow** — DataFrames in, DataFrames out, or its own data
  containers?
- **Plotting** — interactive figures, static images, or numbers-only?
- **Maintenance** — releases, responsiveness, tests.

## Where ProcessBehavior sits

ProcessBehavior implements Thomas A. Bishop's Variance Analysis System (VAS)
on top of Wheeler-style process behavior charts. What makes it different is
the formulate-then-execute split: it classifies your data's *structure* — the
three-state design lineage (PDS / ODS / ADS) — before computing anything, and
only offers the charts that structure supports. Its numbers are pinned to
Bishop's Minitab reference output by [a 280-assertion validation
suite](reference/validation.md) that runs in CI, and signal detection applies
the Western Electric rules with chart-appropriate filtering (run rules on
time-ordered charts only). Everything is pandas-native and plots are
interactive plotly figures.

The trade-off: it is opinionated. If you want a quick c-chart of defect
counts, a general-purpose charting library is less ceremony.

## The alternatives

### pyspc


A general-purpose control-chart library with the broadest chart menu of the
Python options: variables charts (Xbar-R, Xbar-S, X/mR), EWMA, CUSUM,
attribute charts (P, NP, C, U), and multivariate (Hotelling T², MEWMA). Data
can be nested lists, numpy arrays, or DataFrames; rule highlighting is
supported. GPL-3.0 licensed, and maintenance activity has been sparse in
recent years.

**Choose it when** you need attribute or multivariate charts, which
ProcessBehavior does not draw. **Mind** the GPL-3.0 license in proprietary
codebases and the maintenance status.

### statprocon


A deliberately small helper for XmR (process behaviour) charts in the Wheeler
tradition: it computes the chart *data* — limits, center lines, moving ranges
— and stays out of plotting entirely, so it has almost no dependencies. Export
to CSV/Google Sheets or plot the numbers yourself.

**Choose it when** you want XmR limits in a constrained environment (no
plotly/pandas stack) or you plot elsewhere. **Mind** that it is XmR-only: no
subgrouped charts, no structure detection, no rule engine.

### mvSPC


Implements methods from Montgomery's *Statistical Quality Control* (7th ed.),
with a textbook orientation — useful when you want the Montgomery formulation
specifically.

**Choose it when** your organization standardizes on Montgomery's methods.

### Rolling your own with matplotlib/plotly

Always an option for a one-off chart: compute a mean and ±3σ and draw three
lines. The costs arrive later — limits from the *standard deviation of all
data* rather than from within-subgroup or moving-range dispersion (a classic
error that inflates limits), no run rules, and no answer when someone asks
"why these limits?"

## Feature summary


| | ProcessBehavior | pyspc | statprocon |
|---|---|---|---|
| X/mR (individuals) | ✅ | ✅ | ✅ (data only) |
| Xbar/S (subgrouped) | ✅ | ✅ | — |
| Attribute charts (P/NP/C/U) | — | ✅ | — |
| EWMA / CUSUM / multivariate | — | ✅ | — |
| Structure detection (design states) | ✅ | — | — |
| VAS residuals / variance decomposition | ✅ | — | — |
| WECO rules with per-chart applicability | ✅ | partial | — |
| Capability (Cp/Cpk/Pp/Ppk) | ✅ | — | — |
| Taguchi loss decomposition | ✅ | — | — |
| Validated against published reference | ✅ ([in CI](reference/validation.md)) | — | — |
| Plotting | plotly, themeable | yes | none (by design) |
| License | Apache-2.0 | GPL-3.0 | MIT |

## When you don't need ProcessBehavior

- **Attribute data** (defect counts, proportions): pyspc's P/NP/C/U charts.
- **EWMA/CUSUM for small persistent shifts**, or multivariate monitoring:
  pyspc.
- **A single XmR chart with no dependencies**: statprocon, or a spreadsheet —
  Wheeler's own examples are spreadsheet-sized.
- **Teaching from Montgomery**: mvSPC matches the textbook.

If you have measurements arriving over time, possibly structured by factors,
and you care that the limits are defensible — that is the problem
ProcessBehavior is built for. Start with
[Your First X/mR Chart](tutorials/first-xmr-chart.ipynb).
