# Derived variables

A derived variable is a new column computed from an existing one before a study is
formulated: a continuous-to-continuous **transform** (`log`, `sqrt`, `zscore`, …) or a
continuous-to-categorical **bin** (`equal_freq`, `equal_width`, `breaks`, `sd`). Derived
columns can be the response, a factor, or the time variable of the study. The
[derived-variables tutorial](../tutorials/derived-variables.ipynb) walks through the
workflow; this page states the contract.

```python
import processbehavior as pb

pbd = (pb.ProcessBehavior(df)
         .transform("weight", "log")                         # -> weight_log
         .bin("weight", n=4, label="weight_q"))              # -> weight_q, ordered categorical
study = pbd.formulate(response="weight_log", factors=["weight_q"], time="hour")
```

Each verb returns a new `ProcessBehavior`; the original is untouched. The specs are plain
data (`pbd.derivations`, `Derivation.to_dict()` / `from_dict()`), fits are frozen when the
study is formulated (`study.derivations`), and `evaluate(spec, column)` is the single
primitive behind both the fluent verbs and the app's live preview.

## The contract

**`evaluate` and `validate` never raise on routine data.** Whatever the column contains —
missing values, infinities, ties, a constant, a tiny range, a huge scale — the result comes
back as data: `values`, `n_invalid`, `invalid_index`, `fitted`, and a `message`. A seeded
fuzz over every method is part of the test suite to keep it that way. Exceptions are reserved
for two things: a spec that cannot be honoured (raised at construction, see below) and the
`on_invalid='error'` commit at `formulate()`.

**Missing input is not a violation.** A missing value passes through as missing and is never
counted in `n_invalid`.

**A domain violation is any input for which the function has no finite real result.** For
transforms that means: the input is ±inf; the input lies outside the function's domain (log
or log10 of a value ≤ 0, sqrt of a negative, arcsin of a value outside [0, 1], inverse of 0,
a negative base to a fractional power, 0 to a negative power); or the arithmetic overflows
(square of 1e200, inverse of a denormal). All are counted in `n_invalid`, listed in
`invalid_index`, and become missing in the output. A z-score whose sigma is zero or undefined
(a constant or single-value column) makes every present value a violation, rather than a
silently all-missing column.

Two conveniences on the boundary: a value within 1e-9 of a *closed* boundary (sqrt of
−1e-10, arcsin of 1.0000000005) is clamped to the boundary and is not a violation; and
`shift` applies before the function, so `log` with `shift=1` accepts zeros.

**What happens to violations is the analyst's call, at formulation.** `on_invalid='error'`
(the default) makes `formulate()` raise a `ValidationError` that names the derivation, the
count, and the first row labels. `on_invalid='na'` makes the offenders missing and proceeds.

**Bins never drop a finite value.** Every finite, non-missing input lands in a bin, including
values exactly on the minimum or maximum. Infinities cannot be placed in a finite bin: they
leave the fit, become missing, and are counted in `n_invalid` with a message. A bin has no
`on_invalid`; a missing factor value is dropped from the study with a warning at
formulation.

**Fits are reported, not hidden.** `fitted` carries the resolved edges, labels, and bin count
(or mu and sigma for a z-score). When the fit is not what was asked for, `message` says so:
ties that collapse equal-frequency bins, more bins requested than distinct values (fitted one
per distinct value), ordinal labels above five bins (numbered instead), range labels too
close to print distinctly (numbered instead), a column with no spread (no bins), a
non-numeric source (nothing derived).

**Construction rejects what evaluation could not honour**, as a `ValidationError` before any
data is touched:

| Parameter | Must be |
|---|---|
| `n` | a positive integer (numpy integers accepted; booleans and floats rejected) |
| `breaks` | a non-empty, strictly ascending list of finite numbers |
| `shift`, `exponent` | finite numbers |
| `on_invalid` | `'error'` or `'na'` |
| explicit `bin_labels` | non-empty, no nulls, unique |
| output name | not an existing column and not another pending derivation |
| source column | present and numeric (bool counts as numeric; datetime and categorical do not) |

## Reading a preview

```python
from processbehavior import Derivation, evaluate

r = evaluate(Derivation.bin("y", n=4), df["y"])
r.fitted["n_bins"], r.fitted["edges"], r.fitted["labels"]
r.n_invalid, r.message
r.values.value_counts(sort=False)          # bin counts, in bin order
```

Before attaching, `validate(spec, df)` returns the structured issues the attach path would
raise on: column not found, not numeric, output-name collision, breaks out of order, and an
explicit label count that does not match the *fitted* bin count.

## Limits of the current design

- Derivations are evaluated against the original columns. A derivation of a derived column
  is rejected at attach time.
- Box–Cox is not offered.
- Range labels use six significant digits, rising to whatever separates the edges.
