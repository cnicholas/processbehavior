# Wheeler and Bishop References Inventory

Review date: April 3, 2026

Purpose: locate Wheeler and Bishop references across the repository, with emphasis on Bishop/Wheeler methodology references that may need review as the design-state model matures. Tom Bishop is treated here as the methodology expert for ProcessBehavior.

## Notes

- This inventory is based on case-insensitive search hits for `wheeler`, `bishop`, `tom bishop`, `donald wheeler`, `wheeler/bishop`, and related variants.
- I grouped the results by production code, docs, tests, validation assets, fixtures, and generated artifacts.
- Fixture JSON files contain many repeated `prepared_by: "Bishop & Associates"` entries. I kept the exact locations because you asked for file and line inventory.
- This is an index of references, not a judgment about whether each reference should remain.

## Production Code

- `processbehavior/__init__.py`
  - line 4: package docstring says the library follows Wheeler/Bishop methodology
- `processbehavior/analysis.py`
  - line 399: correct dispersion basis per Wheeler/Bishop
  - line 631: XmR rationale tied to Wheeler ordering assumptions
  - line 1361: Xbar and S calculation described as Wheeler methodology
  - line 2333: XmR and R calculation described as Wheeler methodology
- `processbehavior/analysis_dataset.py`
  - line 19: orchestrates SPC analysis using Wheeler/Bishop methodology
- `processbehavior/capability.py`
  - line 2: module docstring references Wheeler/Bishop Chapter 16
  - line 22: bibliography-style Wheeler/Bishop Chapter 16 reference
  - line 97: capability result described as Wheeler/Bishop Chapter 16
  - line 349: implementation note cites Wheeler/Bishop Chapter 16
- `processbehavior/datasets/synthetic.py`
  - line 5: references Dr. Donald Wheeler
  - line 6: references Dr. Davis Bishop
  - line 39: Wheeler citation
  - line 42: Bishop citation
- `processbehavior/loss_function.py`
  - line 2: module docstring references Wheeler/Bishop Chapter 15
  - line 12: bibliography-style Wheeler/Bishop Chapter 15 reference
  - line 37: result described as Wheeler/Bishop Chapter 15
  - line 325: Eq 15.18 attributed to Wheeler/Bishop
  - line 411: assessment described as Wheeler/Bishop Chapter 15
- `processbehavior/maximum_information.py`
  - line 2: module docstring references Wheeler/Bishop
  - line 12: Wheeler/Bishop citation
- `processbehavior/process_behavior.py`
  - line 786: Wheeler XmR assumption about temporal ordering
  - line 799: Wheeler SDS classification assumption about factor x time structure
  - line 826: error/help text references Wheeler SDS 1-6
  - line 868: Tom Bishop Minitab approach note
- `processbehavior/residual_calculator.py`
  - line 4: module docstring says Wheeler/Bishop VAS residuals
  - line 213: Equation 56 from Wheeler
  - line 257: Equation 66 from Wheeler
  - line 298: Equation 72 from Wheeler
  - line 328: Equation 75 from Wheeler
  - line 373: Wheeler equations 13.7-13.9
  - line 402: Bishop leaves first MA2 residual blank
  - line 492: Wheeler/Bishop VAS weighting note
  - line 519: Wheeler/Bishop note for MA2 states 2-3
- `processbehavior/sds_detector.py`
  - line 5: module docstring references Wheeler and Bishop VAS
  - line 27: SDS vocabulary comment says per Wheeler/Bishop Table 1
  - line 87: validation against Bishop methodology
  - line 125: `bishop_reference` field
  - line 126: `bishop_reference` docstring
  - line 167: `bishop_reference` attribute
  - line 207: sections 20.6.1-4 per Wheeler/Bishop
  - line 282: formatted reason includes Bishop reference
  - line 297: 1-6 per Wheeler/Bishop Table 1
  - line 326: classifier docstring says implements Wheeler/Bishop Table 1
  - line 334: SDS classification system per Wheeler/Bishop Table 1
  - line 390: Tom Bishop Minitab approach note
  - line 540: classification logic per Wheeler/Bishop Table 1
  - line 705: deterministic rule based on Wheeler/Bishop methodology
  - line 1032: direct SDS classification per Wheeler/Bishop Table 1
  - line 1256: Bishop methodology reference string for SDS 1
  - line 1285: Bishop methodology reference string for SDS 2
  - line 1315: Bishop methodology reference string for SDS 3
  - line 1342: validating implementation against Bishop methodology
- `processbehavior/spc_constants.py`
  - line 11: Wheeler citation
  - line 80: Wheeler citation for chapter 3
  - line 129: Wheeler citation for chapter 3
  - line 177: Wheeler citation for chapter 3
  - line 266: Wheeler citation for chapters 3-5
  - line 373: Wheeler and Chambers citation
- `processbehavior/study.py`
  - line 1075: Wheeler and Bishop methodology note
  - line 1559: maximum information analysis of R2 residuals attributed to Wheeler/Bishop
  - line 1635: Wheeler/Bishop VAS section 15.5 reference

## Documentation

- `README.md`
  - line 3: package described as following Wheeler/Bishop VAS methodology
  - line 5: claims equation-by-equation Wheeler/Bishop implementation
- `docs/intro.md`
  - line 3: tagline references Wheeler's VAS
  - line 5: intro says package brings Donald Wheeler methodology to Python
  - line 105: section header for Wheeler's VAS
  - line 141: Wheeler philosophy reference
  - line 158: citation block includes Tom Bishop
- `docs/getting-started/key-concepts.md`
  - line 51: section header for Wheeler's VAS
  - line 159: says ProcessBehavior uses Wheeler terminology consistently
  - line 161: Wheeler terminology mapping table
  - line 171: Wheeler philosophy reference
- `docs/reference/api.md`
  - line 470: `.bishop_reference` documented as a Wheeler/Bishop text reference
- `docs/reference/sds_definitions.md`
  - line 3: SDS defined as Wheeler and Bishop VAS framework
  - line 129: Wheeler citation
  - line 130: Wheeler and Chambers citation
  - line 131: Bishop personal communication citation
- `docs/reference/weco-rules.md`
  - line 328: Wheeler contributions section
  - line 346: Wheeler citation
  - line 347: Wheeler and Chambers citation
- `docs/sds-detection.md`
  - line 22: six SDS states per Wheeler/Bishop Table 1
  - line 505: SDS classification system tied to Wheeler and Bishop VAS framework
- `docs/user-guide/sds-detection.md`
  - line 56: formulation guidance says following Wheeler's VAS methodology
- `docs/user-guide/chart-types.md`
  - line 147: Wheeler chart-pair reading recommendation
  - line 212: Dr. Tom Bishop methodology note for R2/R5 limit basis
- `docs/user-guide/residuals.md`
  - line 3: Wheeler's VAS residual decomposition overview
- `docs/user-guide/excel-export.md`
  - line 192: Wheeler-style analysis note
- `docs/user-guide/formulation.md`
  - line 72: rational subgroups in Wheeler's terminology
  - line 286: Wheeler chart-pair recommendation
- `docs/chart_statistics.md`
  - line 53: allows negative LPL per Wheeler
  - line 115: negative S-chart LPL per Wheeler
  - line 122: says calculations are correct per Wheeler/Bishop methodology
- `docs/api_ux_review.md`
  - line 74: Wheeler/Bishop terminology preserved
  - line 79: K/T/N terminology assumes familiarity with Wheeler
- `docs/appendix/wheeler-terminology.md`
  - line 1: document title
  - line 3: glossary says ProcessBehavior follows Donald Wheeler terminology and methodology
  - line 9: Wheeler term for process behavior chart
  - line 15: Wheeler insight quote/explanation
  - line 21: Wheeler term for common cause variation
  - line 27: Wheeler insight
  - line 33: Wheeler term for special cause variation
  - line 39: Wheeler insight
  - line 45: Wheeler term for rational subgroup
  - line 57: Wheeler term for natural process limits
  - line 63: Wheeler insight
  - line 69: Wheeler identifies six sampling design states
  - line 129: Wheeler framework for decomposing variation
  - line 145: Wheeler insight
  - line 155: Wheeler insight
  - line 165: Wheeler insight
  - line 175: Wheeler insight
  - line 183: Wheeler term for X chart / Individual chart
  - line 193: Wheeler term for mR chart
  - line 203: Wheeler term for XmR chart
  - line 215: Wheeler term for Average chart / Xbar chart
  - line 227: Wheeler term for s chart
  - line 239: standard SPC constants from Shewhart's work in Wheeler framing
  - line 255: Wheeler principle about 3-sigma limits
  - line 263: Wheeler statement about chart power
  - line 271: Wheeler principle on natural limits vs specification limits
  - line 279: Wheeler principle on understanding variation before improvement
  - line 285: section header for Wheeler's books
  - line 310: Wheeler term to API mapping table
  - line 326: Wheeler and Chambers citation
  - line 327: Wheeler citation
  - line 328: Wheeler citation
- `docs/myst.yml`
  - line 5: site description references Wheeler's VAS
  - line 10: metadata keyword `Wheeler`
  - line 15: metadata author entry for Tom Bishop
  - line 64: nav entry for `appendix/wheeler-terminology.md`
  - line 65: nav title `Wheeler Terminology`
- `mkdocs.yml`
  - line 87: nav entry for `appendix/wheeler-terminology.md`

## Tests

- `tests/test_bishop_reference.py`
  - line 2: validates residuals against Tom Bishop Minitab reference output
  - line 45: test name references Bishop reference
  - line 84: assertion message references Bishop reference
- `tests/test_by_parameter.py`
  - line 463: Wheeler/Bishop rationale for R2 basis
- `tests/test_capability.py`
  - line 2: tests for Wheeler/Bishop Chapter 16
- `tests/test_edge_cases.py`
  - line 316: Wheeler/Bishop note on SDS being based on N_kt cells
  - line 320: assertion comment references Wheeler/Bishop SDS 2 rule
- `tests/test_loss_function.py`
  - line 2: tests for Wheeler/Bishop Chapter 15
- `tests/test_residual_calculator.py`
  - line 9: Tom Bishop validation data
  - line 162: MA2 behavior per Tom Bishop
  - line 174: Bishop leaves first point blank
  - line 187: Wheeler equations 13.7-13.9
  - line 366: Bishop leaves first observation blank
  - line 516: Bishop leaves j=1 MA2 blank
  - line 548: Tom Bishop validation test label
  - line 553: Tom Bishop formula note
  - line 585: Wheeler equations 13.7-13.9
  - line 616: test name references Tom Bishop example
  - line 618: Figure 30 example attributed to Tom Bishop
  - line 763: Wheeler/Bishop VAS weighting note
- `tests/test_sampling_plan.py`
  - line 1952: Bishop validation CSV note
- `tests/test_sds.py`
  - line 89: test docstring references Wheeler/Bishop SDS 2 semantics
  - line 91: methodology note for N_kt definition
  - line 111: assertion message references Wheeler/Bishop rule
- `tests/test_sds_detector.py`
  - line 183: docstring says SDS detection uses N_kt per Wheeler/Bishop
  - line 194: Wheeler/Bishop SDS 2 rule
  - line 450: docstring says SDS detection uses N_kt per Wheeler/Bishop
  - line 468: Wheeler/Bishop SDS 2 rule
  - line 624: Wheeler/Bishop section 20.6.1 reference
  - line 707: Wheeler/Bishop sections 20.6.3 and 20.6.4 reference
- `tests/test_spc_constants.py`
  - line 50: Wheeler Table A.1 known values
  - line 106: Wheeler Table A.1 known values
  - line 140: Wheeler Table A.1 known values

## Validation and Research Scripts

- `validation/README.txt`
  - line 97: SDS detection logic labeled Wheeler/Bishop methodology
- `validation/sds_raw_vs_tidy.py`
  - line 4: two-state problem described as Wheeler/Bishop issue
- `validation/e2e_bishop_report.py`
  - line 2: title references Tom Bishop VAS analyses
  - line 4: compares library output against Tom Bishop analyses
  - line 8: usage example uses file name
  - line 21: fixture path points at `bishop_analyses`
  - line 22: output HTML path uses `e2e_bishop_report.html`
  - line 348: HTML title references Bishop
  - line 375: HTML heading references Bishop
  - line 376: text compares output against Tom Bishop VAS Minitab analyses
- `simulation/mixed_subgroup_estimation.py`
  - line 12: range-based method attributed to Wheeler
  - line 23: Wheeler and Chambers citation
  - line 25: Wheeler citation
  - line 26: Bishop and Wheeler citation
  - line 65: Wheeler table citation
  - line 165: Wheeler range-based estimator description
- `create_sds_charts_doc.py`
  - line 157: column header `Wheeler Eq`
  - line 220: column header `Wheeler Eq`
- `visualize_design_states.py`
  - line 288: chart subtitle references Wheeler/Bishop Table 1 classification
- `pythonic_hadley_persona.md`
  - line 254: references Wheeler and Bishop methodology for detecting states

## Fixtures

- `tests/fixtures/bishop_analyses/vassds1analysis.json`
  - repeated `prepared_by: "Bishop & Associates"` entries at lines:
    - 23, 46, 69, 92, 115, 138, 161, 184, 207, 230, 253, 276, 299, 322, 345, 368, 391, 414, 437, 460, 483, 506, 529, 552, 575, 598, 621, 644, 667, 690, 713, 736, 759, 782
- `tests/fixtures/bishop_analyses/vassds2analysis.json`
  - repeated `prepared_by: "Bishop & Associates"` entries at lines:
    - 23, 46, 69, 92, 115, 138, 161, 184, 207, 230, 253, 276, 299, 322, 345, 368, 391, 414, 437, 460, 483, 506, 529, 552, 575, 598, 621, 644, 667, 690, 713, 736, 759, 782
- `tests/fixtures/bishop_analyses/vassds3analysis.json`
  - repeated `prepared_by: "Bishop & Associates"` entries at lines:
    - 23, 46, 69, 92, 115, 138, 161, 184, 207, 230, 253, 276, 299, 322, 345, 368, 391, 414, 437, 460, 483, 506, 529, 552, 575, 598, 621, 644, 667, 690, 713, 736, 759, 782

## Generated or Derived Artifacts

- `validation/e2e_bishop_report.html`
  - line 5: HTML title references Bishop
  - line 32: report heading references Bishop
  - line 33: report text compares output against Tom Bishop VAS Minitab analyses
- `sds_design_states.html`
  - line 3886: embedded generated Plotly HTML includes text `Wheeler/Bishop Table 1 classification`

## Repo-Meta and Prior Review Files

- `CLAUDE.md`
  - line 20: says library faithfully implements Wheeler/Bishop VAS
  - line 29: methodology fidelity statement says library implements Wheeler/Bishop equation-by-equation
  - line 31: says reference data should be validated against Bishop Minitab output
  - line 54: references Bishop validation dataset
  - line 58: section header `Wheeler/Bishop Terminology`
- `COMPREHENSIVE_PRE_RELEASE_EVALUATION.md`
  - line 19: Tom Bishop alignment score
  - line 32: says variance decomposition follows Bishop methodology
  - line 60: section on alignment with Tom Bishop methodology
  - line 64: says Tom Bishop philosophy emphasizes specific points
  - line 969: Tom Bishop methodology score summary
- `2026-04-03_design_state_review.md`
  - line 142: references `docs/appendix/wheeler-terminology.md`
  - line 291: references `docs/appendix/wheeler-terminology.md`
  - line 380: says plain `SDS` should be reserved for the Wheeler/Bishop classification system
- `2026-04-04_test_coverage_feedback.md`
  - line 79: references `tests/test_bishop_reference.py`

## Short Read on What Stands Out

- The strongest concentration of Bishop/Wheeler references is in `processbehavior/sds_detector.py`, `processbehavior/residual_calculator.py`, `processbehavior/process_behavior.py`, and the docs around SDS and residuals.
- Bishop is used in three different ways across the repo:
  - as a methodology authority
  - as a validation-data source
  - as embedded metadata in fixture files (`Bishop & Associates`)
- Wheeler terminology is deeply embedded in both the public docs and internal reasoning comments, especially around SDS, residual equations, XmR assumptions, and SPC constants.
- If you later want to distinguish legacy SDS vocabulary from the newer PDS/ODS/ADS model, `sds_detector.py`, `docs/reference/sds_definitions.md`, `docs/sds-detection.md`, and `docs/appendix/wheeler-terminology.md` are the highest-leverage places to start.
