"""
Custom exceptions for processbehavior.

This module provides a lightweight exception hierarchy for better error handling UX.
Users can catch errors by category and get self-diagnostic error messages.

Exception Hierarchy
-------------------
ProcessBehaviorError (base)
├── ValidationError - Invalid input data, parameters, or configuration
│   └── ColumnNotFoundError - Required column missing from DataFrame
└── ChartNotAvailableError - Chart type invalid or unavailable for this SDS

Examples
--------
Catch by category:

>>> try:
...     result = study.execute(chart='Xbar')
... except ChartNotAvailableError as e:
...     print(f"Try one of: {study.valid_charts}")
... except ValidationError as e:
...     print(f"Check your data: {e}")

Catch all library errors:

>>> try:
...     result = study.execute()
... except ProcessBehaviorError as e:
...     print(f"Analysis failed: {e}")
"""


class ProcessBehaviorError(Exception):
    """
    Base exception for all processbehavior errors.

    Catch this to handle any error from the library in a single except block.
    """

    pass


class ValidationError(ProcessBehaviorError):
    """
    Invalid input data, parameters, or configuration.

    Raised when user input fails validation before analysis runs.
    Check error message for specific issue and suggested fix.

    Common causes:
    - Invalid column names in formulate()
    - Incompatible parameter combinations
    - Data structure doesn't match requirements
    """

    pass


class ColumnNotFoundError(ValidationError):
    """
    Required column missing from DataFrame.

    Raised when a specified column (response, factor, time) doesn't exist
    in the input DataFrame.

    Suggested actions:
    - Check `pb.cols` for available columns with auto-completion
    - Verify column names match exactly (case-sensitive)
    - Check `df.columns` to see all available columns
    """

    pass


class ChartNotAvailableError(ProcessBehaviorError):
    """
    Chart type invalid or unavailable for this SDS/data structure.

    Raised when requesting a chart that cannot be produced given the
    current data structure and Sampling Design State (SDS).

    Suggested actions:
    - Check `study.valid_charts` for available chart types
    - Use `study.why_not(chart_type)` for explanation
    - Use `study.support` DataFrame for full availability matrix
    """

    pass
