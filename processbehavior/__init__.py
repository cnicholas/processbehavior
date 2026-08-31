"""
ProcessBehavior - Statistical Process Control for Python

A Pythonic library for process behavior analysis following Bishop's
Variance Analysis System (VAS). The library models the data lifecycle
through three design states and routes analysis by what your data
actually supports:

- **Planned Design State (PDS)** — what you intended to collect
- **Observed Design State (ODS)** — what was actually collected (raw data)
- **Analytical Design State (ADS)** — what survives tidying and drives
  chart selection, residual availability, and variance decomposition

Quick Start
-----------
    import processbehavior as pb

    # Generate sample data (or supply your own DataFrame)
    df = pb.make_design(state=1, seed=42)

    # Formulate the study (detects PDS / ODS / ADS, builds analysis dataset)
    study = pb.formulate(df, response='y', time='time', factors=['factor 1', 'factor 2'])
    print(f"Observed:   ODS {study.observed_design_state.sds}")
    print(f"Analytical: ADS {study.analytical_design_state.sds}")
    print(f"Recommended chart: {study.recommended_chart}")

    # Execute analysis (routes by ADS)
    result = study.execute()
    stats = result.get_statistics('Xbar')
    print(f"Center: {stats['center']}, UPL: {stats['upl']}")

    # Export to Excel (requires the [excel] extra)
    result.to_excel('analysis.xlsx')

Main Classes
------------
ProcessBehavior : Main user-facing API with auto-completion
Study           : Formulated study carrying PDS / ODS / ADS lineage
DesignReport    : Plan-vs-observed-vs-analytical lineage report
AnalysisResult  : Unified result container

A Note on Terminology
---------------------
The integer codes 1-6 are Bishop's reference scale ("Bishop Table 1").
Each design state - PDS, ODS, or ADS - reports a value on that scale via
its ``.sds`` field. Internally some modules still use the legacy "SDS"
prefix (e.g. ``sds_detector.py``); the model itself is three-state.
"""

__version__ = '0.2.0'

# Result object
from processbehavior.analysis_result import AnalysisResult
from processbehavior.calibration import Calibration
from processbehavior.capability import CapabilityResult, SpecLimits
from processbehavior.datasets.loaders import load_coffee_shop
from processbehavior.datasets.synthetic import make_design

# Derived variables (transforms + binning)
from processbehavior.derivations import (
    Derivation,
    EvalResult,
    ValidationResult,
    derivations,
    evaluate,
    remove_derived,
    replace_derived,
    validate,
)

# Exceptions
from processbehavior.exceptions import (
    CalibrationNotSupportedError,
    ChartNotAvailableError,
    ColumnNotFoundError,
    FactorNotFoundError,
    ProcessBehaviorError,
    ProcessBehaviorWarning,
    ValidationError,
)
from processbehavior.loss_function import LossResult
from processbehavior.maximum_information import MaximumInformationResult

# Plotting/theming
from processbehavior.plotting import ChartTheme, get_theme, list_themes, register_theme
from processbehavior.process_behavior import ColumnRef, ProcessBehavior, formulate

# Design-state lineage type (re-exported for type-hint use)
from processbehavior.sds_detector import SDSResult

# Study class (formulation layer)
from processbehavior.study import DesignReport, Study

__all__ = [
    # Main API
    'ProcessBehavior',
    'formulate',
    'ColumnRef',
    'Study',
    'DesignReport',
    'SDSResult',
    'AnalysisResult',
    'SpecLimits',
    'Calibration',
    'CapabilityResult',
    'LossResult',
    'MaximumInformationResult',
    # Derived variables
    'Derivation',
    'EvalResult',
    'ValidationResult',
    'evaluate',
    'validate',
    'derivations',
    'remove_derived',
    'replace_derived',
    # Exceptions
    'ProcessBehaviorError',
    'ValidationError',
    'ColumnNotFoundError',
    'FactorNotFoundError',
    'ChartNotAvailableError',
    'CalibrationNotSupportedError',
    'ProcessBehaviorWarning',
    # Plotting/theming
    'ChartTheme',
    'get_theme',
    'list_themes',
    'register_theme',
    # Datasets
    'make_design',
    'load_coffee_shop',
]
