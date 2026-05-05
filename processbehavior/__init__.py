"""
ProcessBehavior - Statistical Process Control for Python

A Pythonic library for process behavior analysis following Bishop's VAS
methodology. Provides auto-detection of Design States (DS) and appropriate
control chart analysis with variance decomposition.

Quick Start
-----------
    import processbehavior as pb

    # Generate sample data (replace with your own DataFrame)
    df = pb.make_sds(sds=1, seed=42)

    # Formulate the study (detect DS, build analysis dataset)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1', 'factor 2'],
    )
    print(study)

    # Execute analysis
    result = study.execute()
    stats = result.get_statistics('Xbar')
    print(f"Center: {stats['center']}, UPL: {stats['upl']}")

    # Export to Excel (requires the [excel] extra)
    result.to_excel('analysis.xlsx')

Main Classes
------------
ProcessBehavior : Main user-facing API with auto-completion
AnalysisResult : Unified result container
"""

__version__ = "0.1.0"

# Result object
from processbehavior.analysis_result import AnalysisResult
from processbehavior.capability import CapabilityResult, SpecLimits
from processbehavior.datasets.synthetic import make_sds

# Exceptions
from processbehavior.exceptions import (
    ChartNotAvailableError,
    ColumnNotFoundError,
    FactorNotFoundError,
    ProcessBehaviorError,
    ValidationError,
)
from processbehavior.loss_function import LossResult
from processbehavior.maximum_information import MaximumInformationResult

# Plotting/theming
from processbehavior.plotting import ChartTheme, get_theme, list_themes, register_theme
from processbehavior.process_behavior import ColumnRef, ProcessBehavior

# Study class (formulation layer)
from processbehavior.study import DesignReport, Study

__all__ = [
    # Main API
    'ProcessBehavior',
    'ColumnRef',
    'Study',
    'DesignReport',
    'AnalysisResult',
    'SpecLimits',
    'CapabilityResult',
    'LossResult',
    'MaximumInformationResult',

    # Exceptions
    'ProcessBehaviorError',
    'ValidationError',
    'ColumnNotFoundError',
    'FactorNotFoundError',
    'ChartNotAvailableError',

    # Plotting/theming
    'ChartTheme',
    'get_theme',
    'list_themes',
    'register_theme',

    # Datasets
    'make_sds',
]
