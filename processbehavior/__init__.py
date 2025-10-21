"""
ProcessBehavior - Statistical Process Control for Python

A Pythonic library for process behavior analysis following Wheeler/Bishop methodology.

Quick Start:
    from processbehavior import ProcessDataFrame

    # Wrap your data
    data = ProcessDataFrame(df)

    # Auto-detect SDS and run appropriate analysis
    analysis = data.analyze(
        response_var=data.columns.Measurement,
        time_var=data.columns.Time
    )

    # Get results
    result = analysis.calculate()
"""

__version__ = "0.3.0"

# Export main user-facing classes
from process_dataframe import ProcessDataFrame

__all__ = ['ProcessDataFrame']
