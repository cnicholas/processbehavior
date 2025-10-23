"""
ProcessBehavior - Statistical Process Control for Python

A Pythonic library for process behavior analysis following Wheeler/Bishop methodology.
Provides auto-detection of Sampling Design States (SDS) and appropriate control chart
analysis with variance decomposition.

Quick Start
-----------
    from processbehavior import ProcessDataFrame

    # Wrap your DataFrame
    pdf = ProcessDataFrame(df)

    # Auto-detect SDS and run analysis
    result = pdf.analyze(
        response_var=pdf.columns.measurement,
        time_var=pdf.columns.time,
        grouping_vars=[pdf.columns.line]
    )

    # Access results
    print(result.summary)
    xbar_chart = result.get_chart('Xbar')

    # Export to Excel
    result.to_excel('analysis.xlsx')

Main Classes
------------
ProcessDataFrame : Main user-facing API with auto-completion
AnalysisResult : Unified result container
Analysis : Core analysis engine
AnalysisDataSet : Analysis dataset manager
"""

__version__ = "0.3.0"

# Main user-facing API
# Core analysis classes
from processbehavior.analysis_dataset import Analysis, AnalysisDataSet

# Result object
from processbehavior.analysis_result import AnalysisResult
from processbehavior.analysis_specification import AnalysisSpecification
from processbehavior.data_preparation import DataPreparation
from processbehavior.effects_calculator import EffectsCalculator
from processbehavior.process_dataframe import ProcessDataFrame
from processbehavior.residual_calculator import ResidualCalculator

# Utility classes (advanced users)
from processbehavior.sds_detector import SamplingDesignDetector, SDSAnalysisPlan

__all__ = [
    # Main API
    'ProcessDataFrame',
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
]
