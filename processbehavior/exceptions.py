"""
Custom exceptions for processbehavior.

This module provides a lightweight exception hierarchy for better error handling UX.
Users can catch errors by category and get self-diagnostic error messages.

Exception Hierarchy
-------------------
ProcessBehaviorError (base)
├── ValidationError - Invalid input data, parameters, or configuration
│   ├── ColumnNotFoundError - Required column missing from DataFrame
│   └── FactorNotFoundError - Invalid factor/variable name specified
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

from __future__ import annotations


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

    Attributes
    ----------
    column : str | None
        The column name that was not found
    available : list[str] | None
        List of available column names

    Suggested actions:
    - Check `pb.cols` for available columns with auto-completion
    - Verify column names match exactly (case-sensitive)
    - Check `df.columns` to see all available columns
    """

    def __init__(self, message: str, column: str | None = None, available: list[str] | None = None):
        super().__init__(message)
        self.column = column
        self.available = available


class FactorNotFoundError(ValidationError):
    """
    Invalid factor/variable name specified.

    Raised when a factor or variable name doesn't match any known factors
    in the study. Common causes include typos and case mismatches.

    Attributes
    ----------
    factor : str | None
        The factor name that was not found
    suggestion : str | None
        A suggested correction (e.g., case-corrected name)
    available : list[str] | None
        List of valid factor/variable names

    Suggested actions:
    - Check spelling and case (names are case-sensitive)
    - Use `study.factors` to see available factor names
    """

    def __init__(
        self, message: str, factor: str | None = None, suggestion: str | None = None, available: list[str] | None = None
    ):
        super().__init__(message)
        self.factor = factor
        self.suggestion = suggestion
        self.available = available


class ChartNotAvailableError(ProcessBehaviorError):
    """
    Chart type invalid or unavailable for this SDS/data structure.

    Raised when requesting a chart that cannot be produced given the
    current data structure and Sampling Design State (SDS).

    Attributes
    ----------
    chart : str | None
        The chart type that was requested
    available : list[str] | None
        List of available chart types

    Suggested actions:
    - Check `study.valid_charts` for available chart types
    - Use `study.why_not(chart_type)` for explanation
    - Use `study.support` DataFrame for full availability matrix
    """

    def __init__(self, message: str, chart: str | None = None, available: list[str] | None = None):
        super().__init__(message)
        self.chart = chart
        self.available = available
