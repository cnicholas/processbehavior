# Design State Review

Review date: April 3, 2026.

Scope: review of the new three-state lineage model:
- Plan Design State (PDS): derived from the user-provided plan
- Observed Design State (ODS): derived from raw data before cleansing
- Analytical Design State (ADS): derived from tidy/analyzable data after cleansing

Question addressed: does the new intent come through clearly in the architecture, where are legacy SDS references still driving the codebase, and what actions would tighten the implementation.

Clarification incorporated after review: design state `0` is intended to support analyst-facing single-stream analysis with no explicit factor/time structure, such as a quick individuals-style analysis of a vector pasted from Excel. The review below reflects that intended meaning and the current semantic collision around it.

## Executive Summary

The architecture is directionally correct but only partially coherent.

The strongest part of the design is the actual formulation flow:
- [`ProcessBehavior.formulate()`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/process_behavior.py#L867) computes ODS first and optionally computes PDS.
- [`AnalysisDataSet`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_dataset.py#L100) recomputes structure on tidy data and derives ADS.
- [`Study`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/study.py#L978) exposes the three-state lineage explicitly through `plan_design_state`, `observed_design_state`, and `analytical_design_state`.

The weakest part is the surrounding language model:
- the detector and core types are still SDS-first
- many docs still describe SDS 4/5/6 as scenario archetypes rather than incomplete-grid structural states
- several public APIs still expose generic `sds` aliases that hide the fact that ADS is the analysis-driving state

My bottom line:
- The architectural intent is present in the execution path.
- The conceptual model is not yet clear end to end.
- The largest remaining issue is not control flow, it is vocabulary drift and duplicated semantics.

## What Is Clear Today

The code already contains the correct high-level lineage:

- ODS is detected on raw data before dropping invalid responses in [`process_behavior.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/process_behavior.py#L867).
- PDS is computed from the plan in [`process_behavior.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/process_behavior.py#L879).
- ADS is computed on tidy data in [`analysis_dataset.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_dataset.py#L250).
- Analysis plans are derived from ADS, not ODS, in [`process_behavior.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/process_behavior.py#L893).
- `Study` correctly exposes the three-state view in [`study.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/study.py#L982).
- `DesignReport` already presents a lineage view in [`study.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/study.py#L687).

This is good architecture. The dataflow itself is easy to defend:
- PDS describes intended structure.
- ODS describes collected structure.
- ADS describes analyzable structure.
- ADS is the only state that should drive valid charts and downstream analysis behavior.

## Where The Intent Is Still Blurry

### 1. The core types are still SDS-centric, not design-state-centric

The main detector surface is still built around `SDSRegistry`, `SDSResult`, `detect_sds`, and `get_sds_characteristics` in [`sds_detector.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L323).

This keeps the old mental model alive:
- a single “SDS” appears to be the central truth
- PDS, ODS, and ADS look like wrappers around SDS instead of first-class constructs

The code flow is newer than the type vocabulary.

### 2. ADS collapse from 4/5/6 to 1/2/3 is real, but mostly implicit

Your stated intent is:
- ODS/PDS can be 1-6
- ADS should only be 1-3, with incomplete-grid states collapsing as:
  - 4 -> 1
  - 5 -> 2
  - 6 -> 3

The current system effectively does this, but not in the clearest possible way.

Evidence:
- [`get_analysis_plan()`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L1187) explicitly says only analytical SDS 1-3 are supported.
- The comment at [`sds_detector.py:1317`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L1317) explicitly states `4→1, 5→2, 6→3`.
- ADS is computed from tidy `groupby("cell_key").size()` counts in [`analysis_dataset.py:261-279`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_dataset.py#L261).

The key architectural fact is this:
- once empty cells are dropped during tidying, `tidy_nkt` only contains positive-count cells
- therefore ADS 4/5/6 are impossible by construction

That means the collapse is currently encoded structurally, not semantically.

This works, but it is subtle.

I would describe the current implementation as:
- correct in effect
- under-explained in code
- more implicit than it should be for such a central concept

### 3. The public surface still hides ADS behind generic `sds`

[`AnalysisResult`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_result.py#L163) exposes:
- `observed_sds`
- `analytical_sds`
- plus backward-compatible `sds = analytical_sds` at [`analysis_result.py:169-171`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_result.py#L169)

Its summary also includes:
- `observed_sds`
- `analytical_sds`
- generic `sds`, which is ADS, at [`analysis_result.py:217-223`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_result.py#L217)

This keeps old consumer code working, but it also obscures the new model:
- callers still see “the SDS”
- they do not have to confront whether they are reasoning about ODS or ADS

That is fine as a short-term compatibility bridge, but it weakens the conceptual cleanup.

### 4. Design state `0` has a semantic collision

Your intended meaning for state `0` is analyst-facing and useful:
- no explicit structural design
- treat the input as a single stream
- allow a quick, pragmatic individuals-style analysis

That fits the product well.

The issue is that the current codebase appears to use `0` in two different ways.

What the current implementation says:
- [`analysis_dataset.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_dataset.py#L267) creates `ADS 0` when no valid observations remain after tidying.
- [`study.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/study.py#L1065) makes `valid_charts` empty for state `0`.
- [`study.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/study.py#L1208) and [`study.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/study.py#L1318) explain `0` as “no valid observations after data cleaning”.
- [`sds_detector.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L616) describes `0` as “No grouping or time structure”.

Those are not the same concept.

So even if the system handles single-stream analysis properly in practice, state `0` is still semantically overloaded:
- intended meaning: analyst-facing unstructured single stream
- current guard meaning: empty analytical state after cleaning

This is the most important unresolved state-model collision in the current architecture.

## Concrete Legacy References And Inconsistencies

### High-impact conceptual drift

- [`docs/user-guide/sds-detection.md`](/Users/nicholas/Documents/projects/processbehavior/docs/user-guide/sds-detection.md#L21) still describes:
  - SDS 4 as “Nested Design”
  - SDS 5 as “Unstructured”
  - SDS 6 as “Single Stream”
- [`docs/appendix/wheeler-terminology.md`](/Users/nicholas/Documents/projects/processbehavior/docs/appendix/wheeler-terminology.md#L101) repeats the same older semantics.
- [`processbehavior/datasets/synthetic.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/datasets/synthetic.py#L28) still presents SDS 4/5/6 as scenario families:
  - 4: single condition over time
  - 5: nested design
  - 6: unstructured/regime changes

This is the largest source of confusion in the repo.

The detector now treats 4/5/6 structurally:
- incomplete grid without singletons
- incomplete grid without replication
- incomplete grid with singletons

But parts of the docs and synthetic generator still treat 4/5/6 as domain archetypes.

Those are different concepts.

### Medium-impact naming drift

- The package-level description in [`processbehavior/__init__.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/__init__.py#L4) still describes the system as simple “auto-detection of Sampling Design States (SDS)”.
- [`processbehavior/exceptions.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/exceptions.py#L119) still says a chart is unavailable for “this SDS/data structure”, which is less precise than “the analytical design state”.
- [`docs/reference/api.md`](/Users/nicholas/Documents/projects/processbehavior/docs/reference/api.md#L220) exposes `analytical_sds`, but the broader docs still overwhelmingly use SDS as the dominant term.
- Test names, comments, and fixtures still overwhelmingly encode expectations in raw SDS terms rather than PDS/ODS/ADS lineage terms.
- State `0` is described inconsistently enough that an analyst cannot infer one stable meaning for it from the code and docs alone.

### Concrete correctness issue

The `SDSResult` docstring currently mislabels the mapping of incomplete reason tokens:
- [`sds_detector.py:307-310`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L307) says:
  - `incomplete_with_singletons` -> SDS 4
  - `incomplete_no_singletons` -> SDS 5
  - `incomplete_no_replication` -> SDS 6

That is inconsistent with the actual classifier in [`sds_detector.py:1090-1103`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L1090), which correctly implements:
- SDS 4 -> `incomplete_no_singletons`
- SDS 5 -> `incomplete_no_replication`
- SDS 6 -> `incomplete_with_singletons`

This should be treated as a real bug in internal documentation, not just style drift.

## Does The 4→1, 5→2, 6→3 Intent Come Through Clearly?

Partially.

It comes through clearly in these places:
- [`process_behavior.py:893-897`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/process_behavior.py#L893), where analysis planning is explicitly ADS-driven
- [`sds_detector.py:1191-1194`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L1191), where only analytical SDS 1-3 are allowed for analysis plans
- [`sds_detector.py:1317-1327`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L1317), where the collapse is stated directly
- [`tests/test_sampling_plan.py:873-898`](/Users/nicholas/Documents/projects/processbehavior/tests/test_sampling_plan.py#L873), where the tests already acknowledge that 4/5/6 have characteristics but no analysis plans

It does not come through clearly in these places:
- public narrative docs
- the synthetic generator taxonomy
- backward-compatible `sds` aliases
- any code that still speaks as if 4/5/6 are analytical states with direct chart semantics

So the answer is:
- the intent comes through clearly in the core execution path
- it does not yet come through clearly in the repo’s conceptual surface area

## Recommended Tightening Actions

### 1. Make the three-state lineage a first-class object

Introduce a dedicated lineage object, for example `DesignStateLineage`, containing:
- `plan_design_state`
- `observed_design_state`
- `analytical_design_state`

Use it as the explicit carrier between formulation, analysis dataset construction, results, and reporting.

Why this helps:
- it makes the architecture self-describing
- it removes the appearance that PDS/ODS/ADS are just scattered attributes layered onto an SDS-centric system

### 2. Split structural classification from analytical collapse

Today ADS collapse is mostly implicit.

Add an explicit analytical-state classifier or mapper:
- `classify_observed_state(...) -> SDSResult`
- `classify_plan_state(...) -> SDSResult`
- `classify_analytical_state_from_tidy_counts(...) -> SDSResult`

Or, if you want the collapse to be literal, add a dedicated helper:
- `collapse_incomplete_state_to_analytical_state(...)`

Why this helps:
- it encodes the 4→1, 5→2, 6→3 rule as deliberate domain logic
- it stops relying on the reader to infer that zero cells vanished before ADS classification

### 2a. Resolve state `0` into one explicit concept

You should decide in code, not just in intent, whether state `0` means:
- unstructured single-stream analysis, or
- no analyzable observations after cleaning

Based on your clarification, the cleaner design is:
- `0` = analyst-facing single stream / implicit-order analysis
- a separate sentinel for “no analyzable observations remain”

That sentinel could be:
- `None`
- a dedicated status field
- or a separate non-design-state code

Why this helps:
- it preserves the analyst-facing quick-analysis workflow
- it removes the current overloading where `0` sometimes means “analyze this stream” and sometimes means “nothing can be analyzed”

### 3. Separate “observed/planned characteristics” from “analytical capabilities”

Right now `get_sds_characteristics()` in [`sds_detector.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L584) returns capability-style information for 4/5/6 even though `get_analysis_plan()` only supports ADS 1-3.

That mixes two concerns:
- structural characterization
- analytical capability

Split them conceptually:
- one method for structural/state descriptions across 1-6
- one method for ADS-only analysis capabilities across 1-3

Why this helps:
- ODS/PDS can still talk about incomplete grids
- ADS can remain the sole analysis-driving state
- users stop seeing mixed signals about whether 4/5/6 are “valid analysis states”

### 4. Deprecate generic `.sds` aliases on user-facing results

In [`analysis_result.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/analysis_result.py#L169), `result.sds` is currently an alias for ADS.

Short-term:
- keep it for compatibility
- clearly document it as `analytical_sds`

Medium-term:
- deprecate `result.sds`
- prefer `result.analytical_sds`
- prefer `study.observed_design_state.sds` and `study.analytical_design_state.sds` explicitly

Why this helps:
- it forces call sites to be honest about which state they mean

### 5. Correct the semantic drift in docs and synthetic generators

This is the biggest cleanup priority.

Specifically update:
- [`docs/user-guide/sds-detection.md`](/Users/nicholas/Documents/projects/processbehavior/docs/user-guide/sds-detection.md#L23)
- [`docs/appendix/wheeler-terminology.md`](/Users/nicholas/Documents/projects/processbehavior/docs/appendix/wheeler-terminology.md#L101)
- [`processbehavior/datasets/synthetic.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/datasets/synthetic.py#L28)

Recommended rule:
- SDS 1-6 should always mean structural states defined by `N_kt`
- “nested design”, “single stream”, and “unstructured” should be described as example data scenarios, not as alternate definitions of SDS numbers

### 6. Fix state `0` documentation and guard behavior immediately

At minimum, update descriptions of `0` so they match your actual intent.

That includes:
- [`sds_detector.py`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L616)
- any `Study` or result messaging that currently treats `0` only as “no valid observations after data cleaning”

If the runtime truly handles single-stream analysis properly, the docs and guard paths should say so explicitly.

### 7. Fix the detector documentation bug immediately

Correct the incorrect 4/5/6 reason mapping in [`sds_detector.py:307-310`](/Users/nicholas/Documents/projects/processbehavior/processbehavior/sds_detector.py#L307).

This is small, but it matters because the detector module is the conceptual source of truth.

### 8. Add tests that assert lineage, not just raw SDS values

Most tests still assert a single SDS outcome.

Add lineage-oriented tests that explicitly verify:
- PDS can only be 1 or 2
- ODS can be 1-6
- ADS can only be 0-3 if `0` remains a true design state, otherwise 1-3 plus a separate empty-data sentinel
- ODS 4 collapses to ADS 1
- ODS 5 collapses to ADS 2
- ODS 6 collapses to ADS 3
- analysis plans reject 4/5/6 and only accept ADS 1-3
- state `0` supports analyst-facing single-stream behavior if that remains the intended design
- empty-after-cleaning behavior is tested separately from the single-stream case

This would make your new conceptual model executable and durable.

## Recommended Priority Order

1. Fix the `SDSResult` 4/5/6 docstring mapping bug.
2. Resolve state `0` semantically so it has one meaning in code, docs, and tests.
3. Rewrite docs and synthetic-generator descriptions so SDS 4/5/6 are consistently structural incomplete-grid states.
4. Make the ADS collapse explicit in code, not just implicit in tidy grouping behavior.
5. Separate structural-state descriptions from analytical capability descriptions.
6. Deprecate ambiguous generic `sds` aliases on public results.
7. Add lineage-based tests that encode the new model directly.

## Final Assessment

The architecture is already closer to your intended model than the repo narrative suggests.

The implementation path says:
- detect ODS from raw structure
- compute PDS from plan
- compute ADS from analyzable structure
- drive analysis from ADS only

That is the right design.

What is still holding it back is that the language, docs, fixtures, and compatibility aliases still teach the old “single SDS truth” mental model. If you tighten the vocabulary and make the ADS collapse explicit, the system will read as a coherent design-state architecture instead of a partially migrated SDS architecture.

One additional conclusion after clarification: state `0` itself is valuable, but only if it is allowed to mean exactly one thing. Right now it appears to mean both “single-stream quick analysis” and “nothing analyzable remains after cleaning.” That is the most important unresolved semantic collision in the current model.

## Hardening Checklist

- Promote lineage to the primary public abstraction.
  - Introduce a dedicated lineage object or equivalent internal abstraction for PDS, ODS, and ADS.
  - Make sure formulation, reporting, export, and result summaries all consume the same lineage source.

- Make the ODS to ADS collapse explicit in code.
  - Add a clearly named helper or classifier for analytical-state derivation.
  - Encode the intended mapping explicitly: ODS 4 -> ADS 1, ODS 5 -> ADS 2, ODS 6 -> ADS 3.
  - Stop relying on readers to infer the collapse from dropped zero cells.

- Resolve state `0` into one stable contract.
  - Decide whether `0` is a real analyst-facing design state or a failure/sentinel state.
  - If it is a real design state, remove “no valid observations after cleaning” from its semantic definition.
  - If empty-after-cleaning still needs representation, model it separately.

- Separate structural-state description from analytical capability.
  - Keep 1-6 structural descriptions available for PDS and ODS.
  - Restrict analysis-plan capability definitions to ADS 1-3.
  - Remove mixed messaging where SDS 4-6 appear to be direct analysis states.

- Normalize terminology across the codebase.
  - Replace generic “SDS” wording with `PDS`, `ODS`, or `ADS` where the specific state is known.
  - Reserve plain `SDS` for the Wheeler/Bishop classification system as a whole.
  - Audit error messages, docstrings, comments, and logs for ambiguous state references.

- Fix known detector/source-of-truth inconsistencies.
  - Correct the incorrect 4/5/6 reason-token mapping in `SDSResult` documentation.
  - Review `get_sds_characteristics()` descriptions for consistency with incomplete-grid semantics.

- Remove competing meanings for SDS 4/5/6.
  - Rewrite docs so SDS 4/5/6 are always defined structurally by incomplete `N_kt` grids.
  - Move “nested”, “single stream”, and “unstructured” language into examples or scenario descriptions, not core definitions.
  - Update synthetic data docs and naming so generator scenarios do not redefine SDS semantics.

- Tighten the analyst-facing surface.
  - Ensure `study.design()` foregrounds design lineage clearly.
  - Ensure `Study`, `AnalysisResult`, and exports visibly distinguish observed versus analytical state.
  - Deprecate or clearly label generic `sds` aliases that actually mean ADS.

- Add lineage-specific tests.
  - Add tests that assert PDS can only be 1 or 2.
  - Add tests that assert ODS can be 1-6.
  - Add tests that assert ADS can only be 0-3, or 1-3 plus a separate empty-data sentinel if you split state `0`.
  - Add explicit transition tests for 4 -> 1, 5 -> 2, and 6 -> 3.
  - Add tests that analysis plans reject 4-6 and only accept ADS 1-3.
  - Add tests that distinguish true single-stream state `0` behavior from empty-after-cleaning behavior.

- Add documentation that teaches the concept directly.
  - Add one canonical diagram or table showing `PDS -> ODS -> ADS`.
  - Show one complete example where the states differ and explain why.
  - Explain that lineage, not a single SDS number, is the reason the engine chooses its path.

- Make the feature visible as a differentiator.
  - Surface design-state lineage in examples, docs, and release notes.
  - Teach analysts how to answer: what was planned, what was observed, what survived cleaning, and what analysis path was chosen.
