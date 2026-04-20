# Pre-Release Documentation Audit

**Date:** 2026-04-19
**Scope:** All files in `docs/`, all tutorial notebooks, README, public API docstrings, examples
**Target:** PyPI v0.1.0 release readiness

---

## Executive Summary

The processbehavior documentation is structurally comprehensive: 25 markdown files, 9 Jupyter notebooks, and a well-organized MkDocs site covering getting-started through advanced VAS analysis. The methodology content is largely accurate and the tutorial progression is sound.

However, **the documentation is not ready for PyPI release** due to four categories of risk:

1. **Broken code examples (7 critical issues).** Two plot parameters (`template=` and `show_signals=`) are wrong in every documentation page that shows plotting code. The RuleSet API examples reference a `.build()` method that does not exist. Users who copy-paste from the docs will get `TypeError` on their first attempt to plot or detect signals.

2. **Design State traceability is invisible to users.** The system's three-state model — Plan Design State (PDS), Observed Design State (ODS), and Analytical Design State (ADS) — is a core differentiator providing transparent data lineage from intent through raw observation to analysis-ready structure. The code implementation is excellent (clear docstrings, `DesignReport` lineage display, drift logging). But no user-facing documentation explains that three states exist, what each represents, that ADS drives all analysis decisions, or how ODS 4-6 collapse to ADS 1-3 after data cleansing. Users will encounter `study.observed_design_state` and `study.analytical_design_state` with no context for understanding why they differ or which one matters.

3. **Inconsistent SDS 4-6 naming.** Every documentation page that lists SDS 4, 5, and 6 assigns *different names* to each state. The README says SDS 4 is "Single stream"; the user guide says SDS 4 is "Nested Design." This will confuse users immediately.

4. **Undocumented capabilities.** R6 residuals, Maximum Information analysis, and the phased analysis feature all exist in code but have zero documentation.

**Top four priorities before release:**
1. Fix the two plot parameter names across all docs (~25 occurrences)
2. Fix the RuleSet API examples (~15 occurrences)
3. Document the three Design States and the PDS → ODS → ADS pipeline
4. Standardize SDS 4-6 naming across all pages

---

## Terminology Glossary

| Concept | Canonical Form (in code) | Variants Found in Docs | Proposed Standard | Files Affected |
|---------|--------------------------|----------------------|-------------------|----------------|
| Theme parameter | `theme=` | `template=` | `theme=` | plotting.md, api.md, intro.md |
| Signal highlight param | `highlight_signals=` (default `True`) | `show_signals=` (default `False`) | `highlight_signals=` | plotting.md, api.md, chart-types.md, residuals.md, intro.md |
| RuleSet terminal method | Direct pass or `.to_config()` | `.build()` | Remove `.build()`; show direct pass | weco-rules.md, api.md, key-concepts.md |
| Rule 7 method | `.reduced_variation()` | `.hugging_center()` | `.reduced_variation()` | api.md, weco-rules.md |
| Analysis method | `execute()` | `analyze()` | `execute()` | intro.md flowchart, changelog.md |
| Process behavior chart | "Process Behavior Chart" | "control chart" (key-concepts.md:167, weco-rules.md:1), "behavior chart" | "Process Behavior Chart" per Wheeler | Throughout |
| Sampling Design State | "SDS" or full phrase | "data state", "design state" | "Sampling Design State (SDS)" on first use, "SDS" thereafter | Throughout |
| Plan Design State | PDS (`study.plan_design_state`) | Not named in user docs | "Plan Design State (PDS)" — what user intended | New content needed |
| Observed Design State | ODS (`study.observed_design_state`) | Shown in code examples but never explained as a concept | "Observed Design State (ODS)" — raw data structure | key-concepts.md, sds-detection.md, formulation.md |
| Analytical Design State | ADS (`study.analytical_design_state`) | Not explained in user docs; appears only in code comments and API listings | "Analytical Design State (ADS)" — drives all analysis | New content needed |
| R2 calculation method | `exact` / `ma2` / `hybrid` | "moving average", "backward moving range", "2-point moving average", "backward 2-point moving average" | `exact` / `ma2` / `hybrid` matching code constants, with prose explanation | residuals.md, sds-detection.md, key-concepts.md |
| VAS attribution | Bishop's VAS (CLAUDE.md, README, code) | "Wheeler's VAS" (chart-types.md), "Wheeler's framework" (wheeler-terminology.md:127) | "Bishop's VAS" for the system; "Wheeler's terminology/methodology" for SPC foundations | chart-types.md, wheeler-terminology.md |
| SDS 2/3 recommended chart | `XmR` (sds_detector.py:1282, 1311) | "Xbar-S" (key-concepts.md:29-30), "Xbar-S (MR-based)" (sds-detection.md:25-26) | `XmR` per code | key-concepts.md, sds-detection.md |

---

## Critical Issues

These must be fixed before release. Each causes code examples in the documentation to fail at runtime.

### C-1. `template=` parameter does not exist; actual parameter is `theme=`

**Evidence:** `AnalysisResult.plot()` at `processbehavior/analysis_result.py:1206` defines `theme: str = 'processbehavior'`. No `template` parameter exists anywhere in the plot API.

**Occurrences (10):**

| File | Lines | Code |
|------|-------|------|
| `docs/user-guide/plotting.md` | 31 | `template='processbehavior',  # Theme name` |
| `docs/user-guide/plotting.md` | 110 | `fig = result.plot(template='processbehavior')` |
| `docs/user-guide/plotting.md` | 118 | `fig = result.plot(template='minimal')` |
| `docs/user-guide/plotting.md` | 126 | `fig = result.plot(template='dark')` |
| `docs/user-guide/plotting.md` | 134 | `fig = result.plot(template='ggplot')` |
| `docs/user-guide/plotting.md` | 192 | `fig = result.plot(template='company')` |
| `docs/user-guide/plotting.md` | 324 | `template='processbehavior',` |
| `docs/reference/api.md` | 280 | `template: str = 'processbehavior',` |

**Fix:** Find-and-replace `template=` with `theme=` in all documentation code examples. Update the api.md parameter signature.

---

### C-2. `show_signals=` parameter does not exist; actual parameter is `highlight_signals=`

**Evidence:** `AnalysisResult.plot()` at `processbehavior/analysis_result.py:1200` defines `highlight_signals: bool = True`. No `show_signals` parameter exists. Additionally, the default is `True` (signals are shown by default), not `False` as documented.

**Occurrences (11):**

| File | Lines | Code |
|------|-------|------|
| `docs/user-guide/plotting.md` | 28 | `show_signals=False,      # Highlight out-of-control points` |
| `docs/user-guide/plotting.md` | 45 | `show_signals=True` |
| `docs/user-guide/plotting.md` | 55 | `show_signals=True,` |
| `docs/user-guide/plotting.md` | 83 | `show_signals=True` |
| `docs/user-guide/plotting.md` | 254 | `fig = result.plot(show_signals=True)` |
| `docs/user-guide/plotting.md` | 322 | `show_signals=True,` |
| `docs/reference/api.md` | 277 | `show_signals: bool = False,` |
| `docs/user-guide/chart-types.md` | 65 | `fig = result.plot(chart='Xbar', show_zones=True, show_signals=True)` |
| `docs/user-guide/residuals.md` | 236 | `fig = result.plot(show_zones=True, show_signals=True)` |
| `docs/intro.md` | 88 | `result.plot(show_zones=True, show_signals=True)` |

**Fix:** Replace all `show_signals=` with `highlight_signals=`. Update the api.md parameter signature. Note the default is `True`, so many `highlight_signals=True` calls can simply be removed (they're redundant).

---

### C-3. `RuleSet().build()` method does not exist

**Evidence:** The `RuleSet` class at `processbehavior/signals/config.py:201-451` has `.to_config()` (line 430) and `.get_rules()` (line 441), but no `.build()` method. `RuleSet` objects can be passed directly to `detect_signals(rules=...)` as shown in the `detect_signals` docstring at `analysis_result.py:1055-1061`.

**Occurrences in `docs/reference/weco-rules.md` (12):**

| Lines | Code |
|-------|------|
| 66 | `rules = RuleSet().zone_a(consecutive=2).build()` |
| 85 | `rules = RuleSet().zone_b(consecutive=4).build()` |
| 105 | `rules = RuleSet().run(length=8).build()` |
| 108 | `rules = RuleSet().run(length=7).build()` |
| 128 | `rules = RuleSet().trend(length=6).build()` |
| 131 | `rules = RuleSet().trend(length=5).build()` |
| 151 | `rules = RuleSet().oscillation(length=14).build()` |
| 171 | `rules = RuleSet().hugging_center(length=15).build()` |
| 191 | `rules = RuleSet().avoiding_center(length=8).build()` |
| 231 | `rules = RuleSet().beyond_limits().trend().build()` |
| 234 | `rules = RuleSet().beyond_limits().run(length=7).build()` |
| 243 | `.build()` at end of chain |

**Additional occurrences:**

| File | Lines | Code |
|------|-------|------|
| `docs/reference/api.md` | 350 | `.build()` in RuleSet section |
| `docs/reference/api.md` | 594 | `RuleSet().beyond_limits().run().build()` |

**Fix:** Remove `.build()` from all RuleSet examples. Show RuleSet passed directly:
```python
rules = RuleSet().beyond_limits().zone_a(consecutive=2)
signals = result.detect_signals(rules=rules)
```

---

### C-4. `hugging_center()` method does not exist; actual is `reduced_variation()`

**Evidence:** `processbehavior/signals/config.py:357` defines `def reduced_variation(self, length: int = 15) -> RuleSet`. No `hugging_center` method exists on `RuleSet`.

**Occurrences:**

| File | Lines | Code |
|------|-------|------|
| `docs/reference/weco-rules.md` | 171 | `rules = RuleSet().hugging_center(length=15).build()` |
| `docs/reference/api.md` | 349 | `.hugging_center(length=15)` |

**Note:** The *concept name* "Hugging Center" for Rule 7 is correct (key-concepts.md:142, weco-rules.md:155). Only the *method name* needs fixing.

**Fix:** Replace `hugging_center(` with `reduced_variation(` in code examples.

---

### C-5. `installation.md` expected output is fabricated

**File:** `docs/getting-started/installation.md` lines 69-74

**Documented expected output:**
```
SDS detected: SDSResult(sds=4, reason='single_stream', ...)
Charts available: ['XmR', 'R']
```

**Problems:**
1. `'single_stream'` is not a valid `SDSReasonType`. Valid values from `processbehavior/sds_detector.py:28-37` are: `full_replication`, `no_replication`, `partial_replication`, `incomplete_no_singletons`, `incomplete_no_replication`, `incomplete_with_singletons`.
2. The example data (30 observations with a `time` column, no factors) would produce 30 cells each with N_kt=1 and no empty cells, yielding SDS 2 with reason `no_replication`.
3. SDS 2 valid charts include `['Histogram', 'Xbar', 'S', 'XmR', 'R']`, not just `['XmR', 'R']`.
4. The `result.all_charts` accessor on line 65 returns the charts in the *result* (post-execute), not all valid charts for the study.

**Fix:** Run the actual verification code and paste the real output. Alternatively, simplify to `print(study.observed_design_state)` and `print(study.valid_charts)`.

---

### C-6. `pip install processbehavior[excel]` extra does not exist

**Evidence:** `pyproject.toml` defines only two extras: `images` (kaleido) and `dev` (pytest, etc.). `openpyxl` is a core dependency (line 22), not optional.

**Occurrences:**

| File | Lines | Code |
|------|-------|------|
| `docs/user-guide/excel-export.md` | 10 | `pip install processbehavior[excel]` |
| `docs/user-guide/excel-export.md` | 224 | `pip install processbehavior[excel]` |

**Fix:** Remove these lines or replace with a note that openpyxl is installed automatically with processbehavior.

---

### C-7. Changelog references `analyze()` method; actual is `execute()`

**File:** `docs/appendix/changelog.md` line 44

**Text:** `- \`analyze()\` method for chart calculation`

**Evidence:** `study.py:1653` defines `def execute(...)`. The method was renamed during development. The Migration Notes section (lines 146-153 of the same file) correctly documents the old->new rename, but the 0.1.0 "Added" section still uses the old name.

**Fix:** Change `analyze()` to `execute()`.

---

## Important Issues

These should be fixed before release but are not blocking (users won't get runtime errors, but will be confused).

### I-1. SDS 4/5/6 names are inconsistent across every documentation page

The informal names for SDS 4, 5, and 6 rotate differently across files. No two pages agree:

| SDS | README.md | key-concepts.md | sds-detection.md | wheeler-terminology.md | Code reason type |
|-----|-----------|-----------------|------------------|------------------------|-----------------|
| **4** | Single stream | Time only (single stream) | Nested Design | Nested Design | `incomplete_no_singletons` |
| **5** | Nested/hierarchical | Factors only (no time) | Unstructured | Unstructured | `incomplete_no_replication` |
| **6** | Individual values only | Individual values only | Single Stream Over Time | Single Stream | `incomplete_with_singletons` |

The formal definitions in `docs/reference/sds_definitions.md` (which match the code) use N_kt criteria without informal names. The informal names above appear to describe *practical data structure patterns* (e.g., "single stream" = no factors, time only) rather than the Bishop Table 1 classification, and the mapping between the two is never documented.

**Files:** `README.md:66-73`, `docs/getting-started/key-concepts.md:27-34`, `docs/user-guide/sds-detection.md:22-30`, `docs/appendix/wheeler-terminology.md:100-124`

**Fix:** Adopt one consistent naming scheme across all pages. Recommended: use the `sds_definitions.md` formal descriptions as primary (e.g., "Incomplete, no singletons"), with the code reason types as machine-readable labels. If informal names are desired, pick ONE set and use it everywhere.

---

### I-2. `valid_charts` lists omit `Histogram`

`sds_detector.py` lines 1255, 1281, 1310 all include `'Histogram'` in valid_charts for SDS 1-3. The `execute()` docstring at `study.py:1681` explicitly lists `'Histogram'` as valid. But documentation consistently shows only `'Xbar', 'S', 'XmR', 'R'`.

**Files:**
- `docs/reference/api.md` line 86: `'.valid_charts: List of valid chart types ('Xbar', 'S', 'XmR', 'R')'`
- `docs/getting-started/key-concepts.md` line 48: `# e.g., ['Xbar', 'S', 'XmR']`
- `README.md` line 109: lists four chart types without Histogram

**Fix:** Add `'Histogram'` to all valid_charts references. The canonical order from code is `['Histogram', 'Xbar', 'S', 'XmR', 'R']`.

---

### I-3. `execute()` API docs missing three parameters

**File:** `docs/reference/api.md` lines 101-108

The `execute()` signature in api.md omits three parameters that exist in `study.py:1653-1663`:
- `phased: bool = False` — enables per-phase control limits
- `n_sigma: float = 3.0` — sigma multiplier for limit calculation
- `n_mode: str = "actual"` — subgroup size mode (`'actual'` or `'average'`)

These are documented in the source docstring but absent from the API reference. No user guide mentions phased analysis at all.

**Fix:** Add the three parameters to the api.md `execute()` signature. Consider adding a user guide section on phased analysis.

---

### I-4. "Return plain DataFrames" claim is misleading

**Files:** `docs/intro.md` line 144 ("Return plain DataFrames — No custom classes"), `docs/getting-started/key-concepts.md` line 179 ("Plain DataFrames — Results are standard pandas DataFrames, not custom objects")

**Evidence:** `execute()` returns `AnalysisResult` (a custom class with 30+ properties and methods). `plot()` returns `ControlChartFigure`. `design()` returns `DesignReport`. The *data within* these objects is accessible as DataFrames via accessor methods, but the API surface is custom objects throughout.

**Fix:** Rephrase to something like "DataFrame-backed results — access chart data, residuals, and effects as standard pandas DataFrames through accessor methods."

---

### I-5. VAS attributed to Wheeler in some places; VAS is Bishop's system

VAS (Variance Analysis System) is Thomas A. Bishop's contribution. Wheeler contributed the broader SPC/process behavior chart methodology. Several docs conflate the two:

- `docs/appendix/wheeler-terminology.md` line 127: Section header "Variance Analysis System (VAS)" under Wheeler's terminology, implying it's Wheeler's
- `docs/getting-started/key-concepts.md` line 169-170: "ProcessBehavior follows Wheeler's philosophy" immediately before VAS content
- `docs/intro.md` line 141: "ProcessBehavior follows Wheeler's philosophy" in section that includes VAS

**Fix:** Attribute VAS residuals, SDS classification, and variance decomposition to Bishop. Attribute process behavior chart terminology, 3-sigma philosophy, and rational subgrouping concepts to Wheeler.

---

### I-6. SDS 2 and SDS 3 recommended chart: docs say "Xbar-S" but code says "XmR"

**Files:**
- `docs/getting-started/key-concepts.md` lines 29-30: SDS 2 recommended = "Xbar-S (MR-based)", SDS 3 = "Xbar-S (hybrid)"
- `docs/user-guide/sds-detection.md` lines 25-27: Same

**Evidence:** `sds_detector.py` line 1282 sets `recommended_chart='XmR'` for SDS 2, line 1311 sets `recommended_chart='XmR'` for SDS 3. The CLAUDE.md line 26 confirms: "SDS 2/3 recommended_chart: XmR (not Xbar)".

**Fix:** Change recommended chart for SDS 2/3 to "XmR" in all docs tables.

---

### I-7. intro.md Mermaid flowchart says "analyze" not "execute"

**File:** `docs/intro.md` line 25

The Mermaid flowchart node `E[analyze]` should be `E[execute]` to match the actual API.

---

### I-8. Bishop cited with wrong initials

**File:** `docs/reference/sds_definitions.md` line 131

**Text:** `Bishop, D. R. (2023). Personal communication`

**Evidence:** His name is Dr. Thomas A. Bishop, as correctly used in the same file at line 3, in README.md line 3, and in CLAUDE.md.

**Fix:** Change to `Bishop, T. A. (2023). Personal communication — Variance Analysis System implementation.`

---

### I-9. Typo in formulation.md

**File:** `docs/user-guide/formulation.md` line 3

**Text:** "...during problem forumlation to create..."

**Fix:** Change "forumlation" to "formulation".

---

### I-10. SDS variance estimation table has misleading R2 methods for SDS 4-6

**File:** `docs/user-guide/sds-detection.md` lines 335-342

The table shows R2 methods for SDS 4-6 as if they are directly used for analysis. But SDS 4-6 are Observed Design States (ODS) only; after data tidying they collapse to Analytical Design States 1-3 respectively (4->1, 5->2, 6->3). The analytical R2 method is determined by the collapsed ADS, not the ODS.

The R2 methods shown for SDS 4-6 in this table also contradict the code. For example, the table shows SDS 4 uses "Within-cell standard deviation (exact for present cells)" but after collapsing to ADS 1, R2 uses the standard exact method.

**Fix:** Either add a note that SDS 4-6 collapse to 1-3 for analysis and the R2 method shown is the *post-collapse* method, or restructure the table to show only ADS 1-3.

---

## Nice-to-Have Improvements

### N-1. Tutorials use synthetic data exclusively

All 9 tutorial notebooks generate synthetic data. While this ensures reproducibility, a real-world dataset narrative (e.g., the TABVASTESTDATABASE.csv) would be more compelling for data scientists evaluating the library. The process-capability.ipynb and loss-function.ipynb tutorials already use the validation database, which is a good pattern.

### N-2. Some public method docstrings could be more agent-friendly

For the coming wave of LLM-agent tool use, docstrings should include:
- **Intent** (why you'd call this, not just what it does)
- **Constraints** (what must be true for this to work)
- **Examples** with expected output

Methods that would benefit most: `Study.capability()`, `Study.loss_function()`, `Study.maximum_information()`, `AnalysisResult.focus()`.

### N-3. Error messages for wrong parameter names

When a user passes `template='dark'` to `plot()`, they get a generic `TypeError: unexpected keyword argument`. A custom `__init__` or wrapper could catch common misspellings and suggest the correct parameter name (e.g., "Did you mean `theme=`?").

---

## Documented Gaps

Topics that should be documented but currently aren't.

### G-1. Three Design States: PDS, ODS, ADS (ELEVATED — pre-release priority)

The system implements three Design States that provide a powerful and transparent level of traceability as data is processed. This is a core differentiator of the library. The code implementation is excellent — clear docstrings, `DesignReport` lineage display, drift logging — but user-facing documentation fails to explain the model.

**The three states:**

| State | Property | Computed On | Purpose | Code Location |
|-------|----------|-------------|---------|---------------|
| **Plan Design State (PDS)** | `study.plan_design_state` | Plan parameters (K x T x N) | What the user *intended* to collect | `study.py:1017-1029`, `sds_detector.py:1009-1039` |
| **Observed Design State (ODS)** | `study.observed_design_state` | Raw data (response NAs preserved) | What was *actually collected*, including incomplete cells | `study.py:1031-1043`, `sds_detector.py:394-512` |
| **Analytical Design State (ADS)** | `study.analytical_design_state` | Tidy data (after NA filtering) | What is *fit for analysis* — **drives all decisions** | `study.py:1045-1057`, `analysis_dataset.py:250-290` |

**How ADS drives analysis (verified in code):**
- `valid_charts` — `study.py:1087-1102`: returns charts valid for the ADS
- `recommended_chart` — `study.py:1105-1115`: recommendation based on ADS
- R2 method (exact/ma2/hybrid) — `analysis_dataset.py:127-128`: determined from tidy structure stats
- Interaction method — `analysis_dataset.py:146-148`: uses `self._ads_result.sds`
- Residual availability — filtered against ADS columns at `study.py:1135-1141`

**Design lineage display (already implemented, needs documentation):**
- `DesignReport.__repr__` at `study.py:686-700` shows full three-state lineage:
  ```
  Design lineage:
    Planned Design State:    SDS 1 (Full Replication)
    Observed Design State:   SDS 6 (Incomplete, With Singletons) — 3 empty cells
    Analytical Design State: SDS 3 (Partial Replication)
  ```
- `Study.__repr__` at `study.py:2362-2364` shows drift: `Study(..., ods=6, ads=3)` when they differ
- Design state drift is logged at INFO level: `analysis_dataset.py:284-288`

**Documentation audit — where each state is mentioned:**

| File | PDS | ODS | ADS | Three-state concept explained? |
|------|-----|-----|-----|-------------------------------|
| `docs/reference/api.md` lines 79-83 | Listed | Listed | Listed | No — just one-line API entries |
| `docs/user-guide/formulation.md` lines 126-129 | — | Code example | Code example + "ADS-derived" comments | No |
| `docs/user-guide/sds-detection.md` line 49 | — | Code example | — | No |
| `docs/getting-started/key-concepts.md` line 46, 166 | — | Code example | Mentioned in terminology table | No |
| `docs/intro.md` line 83 | — | Code example | — | No |
| `docs/getting-started/installation.md` line 64 | — | Code example | — | No |
| `docs/appendix/wheeler-terminology.md` line 319 | — | Listed | Listed | No |
| **Any tutorial notebook** | — | — | — | **No** |

**Grade: F** — The three-state model is never presented as a concept. Users see property names but have no context.

**Corrective actions (per file):**

1. **`docs/getting-started/key-concepts.md`** — Add a "Design State Traceability" section explaining PDS → ODS → ADS pipeline. This is where the concept should be introduced for the first time. Include:
   - What each state represents and when it is computed
   - That ADS drives chart selection and all analysis capabilities
   - A simple example showing how ODS can differ from ADS
   - A note that `study.design()` shows the full lineage

2. **`docs/user-guide/sds-detection.md`** — Add a subsection "From Observation to Analysis: ODS → ADS" explaining:
   - ODS is detected on raw data (before NA filtering) to preserve structural information about incomplete designs
   - ADS is computed on tidy data and drives all analysis decisions
   - SDS 4-6 (Incomplete) are ODS-only; after data cleansing they collapse to ADS 1-3
   - The variance estimation table (lines 335-342) should clarify it shows post-collapse (ADS) methods

3. **`docs/reference/api.md`** — Expand the three design state property entries (lines 79-83) with brief explanations, not just one-line labels. Add a note: "The ADS is the authoritative state for analysis. See [Key Concepts: Design State Traceability](../getting-started/key-concepts.md) for details."

4. **`docs/user-guide/formulation.md`** — In the "The Study Object" section (lines 116-143), add prose explaining the three properties and their relationship, not just code examples.

5. **Tutorials** — The `sds-validation.ipynb` tutorial is the natural place to demonstrate design state drift (e.g., show an SDS 6 formulation where ODS=6 but ADS=3, and explain what the `study.design()` lineage reveals).

### G-2. ODS → ADS collapse mechanism

The key insight that SDS 4-6 (Observed Design States detected on raw data) collapse to SDS 1-3 (Analytical Design States after data cleansing) is documented only in code comments (`analysis_dataset.py:112-114`). The collapse mechanism is:
- ODS 4 (Incomplete, no singletons) → ADS 1 (Full Replication) — empty cells removed, all remaining have n >= 2
- ODS 5 (Incomplete, no replication) → ADS 2 (No Replication) — empty cells removed, all remaining have n = 1
- ODS 6 (Incomplete, with singletons) → ADS 3 (Partial Replication) — empty cells removed, mixed n remains

This is not merely academic — it explains why a user's `study.observed_design_state.sds` of 6 still produces valid Xbar/S charts (because ADS is 3, and SDS 3 supports Xbar/S). Without this explanation, users will be confused about the relationship between what they see in the ODS and what charts are available.

**Corrective actions:** Covered by G-1 actions above (key-concepts.md and sds-detection.md).

### G-3. Mixed treated as N_kt = 1 rationale

The decision to treat Mixed (SDS 3) as N_kt = 1 for methodological soundness was validated by Monte Carlo simulation. This is mentioned in CLAUDE.md but not in user documentation. Users encountering SDS 3 would benefit from understanding why the more conservative approach is taken.

### G-4. R6 residual

R6 (main effect residual) is fully implemented in `study.py:1954-1996`, available via `execute(value='R6', by=[...])`, and listed in `SDSAnalysisPlan.residuals_available`. But it is absent from all user-facing documentation: not in `residuals.md`, not in `key-concepts.md`, not in `api.md`, and not in any tutorial.

### G-5. No GDP demo dataset

The user's audit instructions reference a GDP demo dataset with 2008/COVID signals. No such dataset exists in the project. All tutorials use synthetic data or the TABVASTESTDATABASE validation dataset. If a GDP narrative is desired for PyPI launch, it would need to be created.

### G-6. No reference to processbehavior.com or forthcoming book

The user notes that the package is one component of a larger system (hosted application at processbehavior.com, forthcoming book co-authored with Bishop). None of this context appears anywhere in the documentation. The README, which serves as the PyPI storefront, should declare the package's scope and point users to the application and book for the curated experience and methodological text respectively.

### G-7. Maximum Information analysis

`Study.maximum_information()` is implemented (`processbehavior/maximum_information.py`, `study.py:1632-1651`) and returns a `MaximumInformationResult` with XmR chart and histogram. No documentation exists in `docs/` — not in the user guide, not in the API reference, and not in any tutorial.

### G-8. Design Report interpretation guide

`study.design()` returns a rich `DesignReport` object (documented in api.md with all properties), but there is no user guide explaining *how to interpret* the report. What does a coverage ratio of 0.7 mean practically? When should the analyst act on missing_combos? What does plan_adherence tell you?

### G-9. Loss function user guide

The Taguchi loss function is documented only in a tutorial notebook (`docs/tutorials/loss-function.ipynb`). There is no user guide entry parallel to the residuals, chart types, or plotting user guides.

### G-10. Process capability user guide

Process capability analysis is documented only in a tutorial notebook (`docs/tutorials/process-capability.ipynb`). There is no user guide entry for `study.capability()`.

---

## File-by-File Findings

### Root-Level Files

| File | Issues | Summary |
|------|--------|---------|
| `README.md` | I-1, I-2 | SDS 4/5/6 names inconsistent with other docs; valid_charts omits Histogram; no mention of processbehavior.com or book |
| `CHANGELOG.md` | — | Minimal (4 lines), defers to docs/appendix/changelog.md. Fine. |
| `CONTRIBUTING.md` | — | Brief (5 lines). Adequate for launch. |
| `LICENSE` | — | Apache 2.0. Correct. |

### docs/getting-started/

| File | Issues | Summary |
|------|--------|---------|
| `docs/intro.md` | C-2, I-4, I-5, I-7 | `show_signals=` (line 88); "Return plain DataFrames" (line 144); Wheeler/VAS conflation (line 141); flowchart says "analyze" (line 25); Mermaid node I says "detect_signals" which is correct but may confuse since it's a method on AnalysisResult, not a separate step |
| `docs/getting-started/installation.md` | C-5 | Expected output fabricated: `sds=4, reason='single_stream'` invalid (lines 69-74). Also, the variable shadowing (`pb = pb.ProcessBehavior(df)` on line 60 after `import processbehavior as pb` on line 43) would crash. |
| `docs/getting-started/key-concepts.md` | G-1, I-1, I-2, I-4, I-5, I-6 | **No Design State Traceability section** — must add PDS/ODS/ADS explanation; SDS 4/5/6 names don't match any other page (lines 27-34); valid_charts omits Histogram (line 48); "Plain DataFrames" claim (line 179); Wheeler/VAS conflation (line 169); SDS 2/3 recommended chart wrong (lines 29-30); `detect_signals(rules='standard')` pattern at line 151 may not match actual API |

### docs/user-guide/

| File | Issues | Summary |
|------|--------|---------|
| `docs/user-guide/formulation.md` | G-1, I-9 | **Study Object section shows design state properties without explanation** — add prose explaining PDS/ODS/ADS relationship (lines 116-143); typo "forumlation" (line 3) |
| `docs/user-guide/chart-types.md` | C-2, I-5 | `show_signals=True` (line 65); Xbar limits admonition references "Dr. Tom Bishop" (good); R2 note on line 143 has wrong SDS mapping ("SDS 1, 3, 4, 6 use within-cell deviation" — SDS 4 and 6 are ODS, not directly analyzed) |
| `docs/user-guide/sds-detection.md` | G-1, G-2, I-1, I-6, I-10 | **No ODS→ADS section** — must add collapse explanation; SDS 4/5/6 names don't match other pages (lines 22-30); SDS 2/3 recommended chart says Xbar-S (lines 25-27); variance estimation table misleading for SDS 4-6 (lines 335-342) — shows methods as if SDS 4-6 are directly analyzed but they collapse to ADS 1-3; SDS 3 R2 description says "Hybrid: exact for n>1, zero for n=1" — the "zero for n=1" is an implementation detail that may confuse users |
| `docs/user-guide/residuals.md` | C-2 | `show_signals=True` (line 236); R2 SDS 2 formula says "(Y_j - Y_{j-1}) / 2" and labels it "backward 2-point moving average" (line 47) — the code uses "ma2" as the method name; residual availability table (lines 187-194) shows all residuals available for SDS 4-6 which is technically about the ADS after collapsing; detect_signals pattern on lines 207-208 uses `signals.has_signals` and `signals.count` — verify these exist on SignalResult (they do) |
| `docs/user-guide/plotting.md` | C-1, C-2 | 7x `template=` (lines 31, 110, 118, 126, 134, 192, 324); 6x `show_signals=` (lines 28, 45, 55, 83, 254, 322); also missing `highlight_signals`, `show_limit_values`, `aspect_ratio`, `yaxis_padding`, `vertical_spacing` from parameter list |
| `docs/user-guide/excel-export.md` | C-6 | 2x `pip install processbehavior[excel]` (lines 10, 224); openpyxl is a core dependency; the `detect_signals()` call on line 202 may use old return-type assumptions |

### docs/reference/

| File | Issues | Summary |
|------|--------|---------|
| `docs/reference/api.md` | G-1, C-1, C-2, C-3, C-4, I-2, I-3 | **Design state properties are one-line labels with no explanation** (lines 79-83) — expand with brief descriptions and link to key-concepts; `template=` (line 280); `show_signals=` (line 277); `.build()` in RuleSet (line 350, 594); `hugging_center()` (line 349); valid_charts omits Histogram (line 86); execute() missing phased/n_sigma/n_mode (lines 101-108); FactorNotFoundError missing from exception hierarchy diagram (line 608); DataPreparation API section (lines 477-493) shows `DataPrepConfig(dict)` constructor — verify this matches current code |
| `docs/reference/sds_definitions.md` | I-8 | Bishop cited as "D. R." (line 131); otherwise the formal Table 1 definitions are correct and match code exactly; the detection algorithm pseudocode (lines 96-113) accurately reflects `_classify_by_nkt` |
| `docs/reference/weco-rules.md` | C-3, C-4 | 12x `.build()` throughout; 1x `hugging_center()` (line 171); rule descriptions and false alarm rates are accurate; the applicability matrix (lines 253-262) matches code |

### docs/appendix/

| File | Issues | Summary |
|------|--------|---------|
| `docs/appendix/changelog.md` | C-7 | `analyze()` should be `execute()` (line 44); Migration Notes section (lines 146-173) is good but references `SignalDetector` import (line 172) — verify this is exported; version date placeholder "2025-XX-XX" needs updating for release |
| `docs/appendix/wheeler-terminology.md` | I-1, I-5 | SDS 4/5/6 names don't match other pages (lines 100-124); VAS section header at line 127 implies Wheeler owns VAS; otherwise a thorough and valuable reference document |

### docs/tutorials/

| File | Issues | Summary |
|------|--------|---------|
| `docs/tutorials/index.md` | — | Clean; tutorial ordering is logical |
| `docs/tutorials/basic-imr.ipynb` | — | Uses `theme=` correctly; synthetic data; API calls verified correct |
| `docs/tutorials/signal-detection.ipynb` | — | Comprehensive WECO coverage; verify detect_signals API patterns |
| `docs/tutorials/xbar-s-analysis.ipynb` | — | Uses `highlight_signals=True` (correct); good VAS residual walkthrough |
| `docs/tutorials/stratified-analysis.ipynb` | — | Clean |
| `docs/tutorials/process-capability.ipynb` | — | Uses TABVASTESTDATABASE; `SpecLimits` and `capability()` patterns correct |
| `docs/tutorials/sds-validation.ipynb` | — | Uses TABVASTESTDATABASE; exercises all SDS; heavy advanced parameter usage |
| `docs/tutorials/sds1-complete-analysis.ipynb` | — | Deep dive; demonstrates Histogram chart type |
| `docs/tutorials/loss-function.ipynb` | — | Uses TABVASTESTDATABASE; Taguchi loss decomposition |
| `docs/getting-started/quickstart.ipynb` | — | Synthetic data; correct API usage |

**Note:** Tutorial notebooks appear to use the correct API (`theme=`, `highlight_signals=`, no `.build()`). The API errors are confined to the markdown documentation files, not the notebooks.

### Cross-Reference Integrity

Internal links between documentation pages resolve correctly within the MkDocs structure. Relative paths in tutorials (`../tutorials/`, `../user-guide/`) follow the directory hierarchy. No broken cross-references detected.

### Attribution and Credit

- Dr. Thomas A. Bishop is credited correctly in README.md, intro.md, sds_definitions.md (except initials — I-8), and throughout the VAS-specific documentation.
- Donald Wheeler is credited correctly in wheeler-terminology.md and weco-rules.md with appropriate book references.
- The distinction between Bishop's methodological contribution (VAS) and Wheeler's prior art (process behavior charts, SPC philosophy) is generally clear, except where VAS is placed under Wheeler's section header (I-5).
- The intro.md citation bibtex block (lines 157-162) correctly lists both Nicholas and Bishop as authors.

---

## Recommended Sequencing

If prioritizing fixes given limited time, address them in this order:

### Phase 1: Critical fixes (blocks release, ~2 hours)

Order by blast radius (most files affected first):

1. **C-2** `show_signals=` -> `highlight_signals=` — 11 occurrences across 5 files. Also fix default from `False` to `True`.
2. **C-1** `template=` -> `theme=` — 10 occurrences across 2 files.
3. **C-3** Remove `.build()` from RuleSet examples — 14 occurrences across 2 files.
4. **C-4** `hugging_center()` -> `reduced_variation()` — 2 occurrences across 2 files.
5. **C-5** Fix installation.md expected output — 1 file, but requires running the code to get correct output.
6. **C-6** Remove `[excel]` extra references — 1 file, 2 occurrences.
7. **C-7** `analyze()` -> `execute()` in changelog — 1 file.

### Phase 2: Design State Traceability (pre-release, ~3 hours)

The three-state model is a core differentiator. Users must understand PDS → ODS → ADS before they can interpret their results correctly.

1. **G-1** Add "Design State Traceability" section to `docs/getting-started/key-concepts.md` — introduce PDS, ODS, ADS as a pipeline with a concrete example.
2. **G-1 + G-2** Add "From Observation to Analysis: ODS → ADS" subsection to `docs/user-guide/sds-detection.md` — explain the collapse mechanism (ODS 4→ADS 1, 5→2, 6→3) and why ODS runs on raw data.
3. **G-1** Expand design state property entries in `docs/reference/api.md` (lines 79-83) — from one-line labels to brief explanations with cross-reference.
4. **G-1** Add prose to `docs/user-guide/formulation.md` Study Object section (lines 116-143) explaining what the three properties represent.
5. **I-10** Fix the variance estimation table in `docs/user-guide/sds-detection.md` (lines 335-342) to clarify it shows post-collapse (ADS) methods, not ODS methods.

### Phase 3: Important fixes (should ship with, ~3 hours)

1. **I-1** Standardize SDS 4/5/6 names — 5+ files. This is the most impactful consistency fix. Decide on canonical names first. The code's formal reason types (`incomplete_no_singletons`, `incomplete_no_replication`, `incomplete_with_singletons`) and `sds_definitions.md` Table 1 are the source of truth.
2. **I-6 + I-10** Fix recommended charts and variance table for SDS 2/3/4-6 — 2 files.
3. **I-2** Add Histogram to valid_charts lists — 3 files.
4. **I-3** Add phased/n_sigma/n_mode to api.md execute() — 1 file.
5. **I-5** Fix VAS attribution — 2+ files.
6. **I-7** Fix flowchart "analyze" -> "execute" — 1 file.
7. **I-4** Rephrase "plain DataFrames" claim — 2 files.
8. **I-8** Fix Bishop initials ("Dr. Thomas A. Bishop") — 1 file.
9. **I-9** Fix typo — 1 file.

### Phase 4: Scope and positioning (pre-release, ~1 hour)

1. **G-6** Add processbehavior.com and book references to README — establishes the package's place in the larger system.
2. Update README to meet PyPI storefront criteria (one-sentence description, simplest useful example within first screen, scope declaration).

### Phase 5: Documentation gaps (post-release OK, track as issues)

1. **G-4** Document R6 residual
2. **G-7** Document Maximum Information analysis
3. **G-8** Design Report interpretation guide
4. **G-9** Loss function user guide
5. **G-10** Process capability user guide
6. **G-3** Mixed-as-N=1 rationale
7. **G-5** GDP demo dataset (if desired)

### Phase 5: Polish (backlog)

1. **N-1** Real-world dataset narrative
2. **N-2** Agent-friendly docstrings
3. **N-3** Helpful error messages for wrong parameter names
