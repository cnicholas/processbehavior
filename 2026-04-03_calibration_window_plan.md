# Calibration Window Feature Plan

Review date: April 3, 2026

Purpose: plan a study-level calibration feature for baseline-window limit estimation. The current gap is that limits are computed from the full analyzable dataset, with no way to select a baseline window for establishing the voice of the process.

## Summary

- Add a study-level calibration concept that lets an analyst define a baseline window on the analytical time axis, then selectively use that window to estimate CL/LPL/UPL while still plotting and signal-scoring the full analyzable series.
- Recommended v1 contract:
  - `study.calibrate(start, end) -> Study`
  - calibration is defined in terms of inclusive `time_var` bounds on the tidy analytical dataset
  - calibration is stored on the returned study but is not automatically applied
  - `study.execute(..., use_calibration: bool = False)` opts into calibrated limits per analysis
  - calibration changes limit estimation only
  - residual values, VAS decomposition, chart values, and the plotted dataset remain global
  - `phased=True` and `use_calibration=True` are mutually exclusive
  - calibration is only valid for analyses with a meaningful time ordering
- This keeps the feature analyst-facing, consistent with the existing study-as-formulation model, and avoids silently changing all downstream analyses once a baseline exists.

## Public API and Behavior

- New study-level interface:
  - add an immutable calibration definition object, `CalibrationSpec`, with:
    - `start`
    - `end`
    - `time_var`
    - `inclusive=True`
  - add `Study.calibrate(start, end) -> Study`
    - validates that the study has a `time_var`
    - validates `start <= end`
    - validates that the filtered analytical dataset contains at least one row
    - returns a new `Study` carrying the calibration spec
  - add `Study.calibration` property
    - returns the saved calibration spec or `None`
  - extend `Study.execute()` with `use_calibration: bool = False`
    - if `False`, behavior is unchanged
    - if `True` and no calibration is stored, raise `ValidationError`
    - if `True`, all chart-limit computation uses the baseline subset only, subject to the chart eligibility rules below

- Where calibration applies:
  - calibration is allowed only when the executed analysis has a real time-sequenced interpretation
  - allow:
    - `XmR` and `R` when the study has time ordering
    - `Xbar` and `S` when the output is time-indexed or full `Kt` indexed
    - default `Xbar`/`S` behavior with grouping plus time qualifies
    - `by=[time_var]` qualifies
    - any execution path whose x-axis removes time does not qualify
  - disallow:
    - `Histogram`
    - `Xbar`/`S` views that aggregate away time, such as `by=['factor']`
    - any study with no `time_var`
  - validation message should explain that calibration is a time-baselining feature and requires a time-ordered chart

- Exact semantics:
  - the baseline subset is filtered from `self._ads.analysis_dataset`, not raw data
  - filtering uses inclusive bounds on `spec.time_var`
  - the full chart dataset remains unchanged after filtering
  - only the statistics used to compute `center`, `lpl`, and `upl` come from the baseline subset
  - signal detection continues across the full plotted dataset using those calibrated limits
  - residual charts use globally computed residual columns, but the residual chart limits come from the calibrated subset of that residual column
  - `capability()`, `loss_function()`, and `maximum_information()` are unchanged in v1

- Recommended toggle model and tradeoffs:
  - recommended design:
    - immutable `Study.calibrate(...) -> Study`
    - explicit `execute(use_calibration=True)`
  - why this is the cleanest shape:
    - preserves the current formulation-object mental model
    - lets the analyst save a baseline definition once and selectively apply it
    - avoids hidden global state on a reused study object
    - avoids forcing calibration into every `execute()` call as duplicated parameters
  - alternatives and why not choose them for v1:
    - mutating `Study`: simpler implementation, but introduces hidden state and makes notebook workflows harder to reason about
    - execute-only parameters: avoids stored state, but makes repeated calibrated analysis noisy and weakens the teaching value of the study object

## Implementation Changes

- Study and request layer:
  - add `CalibrationSpec` alongside `FormulationSpec` and `ChartRequest`
  - add optional calibration field to `Study`
  - add `use_calibration: bool = False` and optional calibration payload to `ChartRequest`
  - add `Study.calibrate(start, end)` validation on analytical time values
  - add execute-time validation:
    - reject `use_calibration=True` with `phased=True`
    - reject `use_calibration=True` for unsupported chart/view shapes
    - reject empty or chart-ineligible baseline subsets with a helpful error

- Analysis layer:
  - introduce a single baseline-resolution helper in `Analysis` that returns:
    - `full_df`: the existing full analyzable dataset
    - `baseline_df`: the time-filtered subset used only for limit estimation
    - baseline metadata:
      - time bounds
      - baseline row count
      - baseline subgroup count if relevant
      - baseline first/last time values
  - refactor each chart family so the current plotting/output path still uses `full_df`, while limit-estimation stats come from `baseline_df`

- XmR / R behavior:
  - compute baseline mean and baseline moving range from the ordered baseline subset
  - merge those fixed calibrated limits onto the full ordered output
  - detect beyond-limits on the full output using calibrated limits

- Xbar / S behavior:
  - derive grouped baseline statistics from the baseline subset only
  - use those baseline stats as the basis for `center`, `lpl`, and `upl`
  - still aggregate and plot the full eligible output data
  - preserve existing `n_sigma` and `n_mode` behavior, but apply them against the baseline-derived inputs

- Important v1 rule:
  - do not recompute residuals, effects, or ADS from the baseline window
  - calibration is a limit-estimation overlay, not a second formulation pass

- Metadata and result exposure:
  - add calibration metadata to each affected chart result, for example:
    - `use_calibration`
    - `calibration_start`
    - `calibration_end`
    - `baseline_points`
    - `baseline_time_var`
    - `baseline_source='analytical_dataset'`
  - this should show up in chart metadata and any summary surface that already reports phased or varying-limit behavior

## Test Plan

- Core contract:
  - `study.calibrate(start, end)` returns a new study and leaves the original unchanged
  - calibrated study exposes `study.calibration`
  - `execute(use_calibration=False)` is identical to current behavior
  - `execute(use_calibration=True)` without saved calibration raises `ValidationError`

- Baseline selection:
  - bounds are inclusive on `time_var`
  - baseline is resolved from tidy analytical data, not raw data
  - selecting a range with no analyzable rows raises a helpful error
  - invalid bound order raises a helpful error

- XmR / R behavior:
  - calibrated `center/lpl/upl` differ from full-data limits when the baseline window differs materially
  - plotted row count remains identical to the uncalibrated chart
  - beyond-limits is recomputed across the full series using calibrated limits
  - baseline with too few ordered points fails with guidance

- Xbar / S behavior:
  - default time-sequenced `Xbar`/`S` can use calibration
  - calibrated limits come from baseline groups only
  - full plotted output still includes all eligible groups outside the baseline window
  - views that collapse time out of the chart reject calibration with a clear message
  - existing `n_sigma` and `n_mode` still work under calibration

- Residual behavior:
  - `execute(chart='XmR', value='R2', use_calibration=True)` succeeds
  - residual values are unchanged relative to uncalibrated execution
  - only `center/lpl/upl/beyond_limits` change

- Interaction with existing features:
  - `use_calibration=True` with `phased=True` raises `ValidationError`
  - companion charts respect calibration consistently
  - stratified time-sequenced charts either both qualify and calibrate correctly, or reject consistently when time is not on the x-axis
  - histogram rejects calibration
  - plotting, stats box, and export paths surface calibration metadata without breaking current output

- Regression guard:
  - all existing uncalibrated tests should remain unchanged
  - add an explicit uncalibrated-equals-prior-behavior regression test for at least one chart in each family

## Assumptions and Defaults

- Calibration is an analyst workflow for establishing the voice of the process from a homogeneous time window.
- Time bounds are the right public selector for v1.
- Calibration applies only to analyses with meaningful time ordering.
- Residual calibration changes limits only, not residual decomposition.
- Calibration and phased limits are mutually exclusive in v1.
- The baseline subset must satisfy the existing chart prerequisites after filtering; otherwise execution fails with guidance rather than falling back.
