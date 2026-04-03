"""
ProcessBehavior - Statistical Process Control for Python

A Pythonic library for process behavior analysis following Wheeler/Bishop methodology.
Provides auto-detection of Sampling Design States (SDS) and appropriate control chart
analysis with variance decomposition.

Quick Start
-----------
    from processbehavior import ProcessBehavior

    # Wrap your DataFrame
    pb = ProcessBehavior(df)

    # Formulate the study (detect SDS, get recommendations)
    study = pb.formulate(
        response=pb.cols.measurement,
        time=pb.cols.time,
        factors=[pb.cols.line]
    )

    # Run analysis
    result = study.execute()

    # Access results
    print(result.summary)
    xbar_chart = result.get_chart('Xbar')

    # Export to Excel
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
]
