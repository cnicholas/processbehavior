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
    result = study.analyze()

    # Access results
    print(result.summary)
    xbar_chart = result.get_chart('Xbar')

    # Export to Excel
    result.to_excel('analysis.xlsx')

Main Classes
------------
ProcessBehavior : Main user-facing API with auto-completion
AnalysisResult : Unified result container
Analysis : Core analysis engine
AnalysisDataSet : Analysis dataset manager
"""

__version__ = "0.1.0"

# Main user-facing API
# Core analysis classes
from processbehavior.analysis import Analysis
from processbehavior.analysis_dataset import AnalysisDataSet

# Result object
from processbehavior.analysis_result import AnalysisResult
from processbehavior.analysis_specification import AnalysisSpecification
from processbehavior.data_preparation import DataPreparation
from processbehavior.effects_calculator import EffectsCalculator

# Plotting/theming
from processbehavior.plotting import ChartTheme, get_theme, list_themes, register_theme
from processbehavior.process_behavior import ProcessBehavior
from processbehavior.residual_calculator import ResidualCalculator

# Utility classes (advanced users)
from processbehavior.sds_detector import SamplingDesignDetector, SDSAnalysisPlan

# Study class (formulation layer)
from processbehavior.study import Study

__all__ = [
    # Main API
    'ProcessBehavior',
    'Study',
    'AnalysisResult',

    # Core classes
    'Analysis',
    'AnalysisDataSet',
    'AnalysisSpecification',

    # Utilities
    'SamplingDesignDetector',
    'SDSAnalysisPlan',
    'DataPreparation',
    'EffectsCalculator',
    'ResidualCalculator',

    # Plotting/theming
    'ChartTheme',
    'get_theme',
    'list_themes',
    'register_theme',
]
