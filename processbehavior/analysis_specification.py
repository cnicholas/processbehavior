"""
AnalysisSpecification - Configuration and validation for process behavior analyses.

This module provides two classes:

1. DataPrepConfig: Base configuration for data preparation (no analysis_type needed)
2. AnalysisSpecification: Extended configuration with analysis_type validation

The separation allows data preparation and SDS detection to work without knowing
the analysis type, which is determined after SDS detection.

Usage:
    # For data preparation only (no analysis_type):
    config = DataPrepConfig({
        'response_var': 'Height',
        'time_var': 'Time',
        'rsg_vars': ['Operator', 'Machine'],
        'round_to': 3
    })

    # For full analysis (with analysis_type):
    spec = AnalysisSpecification(
        analysis_type='Xbar',
        analysis_specification={
            'response_var': 'Height',
            'time_var': 'Time',
            'rsg_vars': ['Operator', 'Machine'],
            'round_to': 3
        }
    )
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class DataPrepConfig:
    """
    Base configuration for data preparation (no analysis_type required).

    This class handles structural configuration needed for data preparation
    and SDS detection. It has no knowledge of analysis_type, making it suitable
    for use before the analysis type is determined.

    Parameters
    ----------
    specification : dict
        Dictionary containing data preparation parameters:

        - **response_var** (str, required): Response variable being analyzed.
        - **rsg_vars** (list, optional): Rational subgrouping variables.
        - **time_var** (str, optional): Time dimension or ordering variable.
        - **rsg_var_name** (str, optional): Label for rational subgroup column.
          Defaults to 'rsg'.
        - **rsg_var_delim** (str, optional): Delimiter for multi-variable grouping.
          Default is '_'.
        - **round_to** (int, optional): Decimal places for rounding. Defaults to 3.
        - **zero-center** (bool, optional): Whether to center data at zero.
          Defaults to False.

    Attributes
    ----------
    rsg_vars : list or None
        Rational subgrouping variables
    rsg_var_name : str
        Name for rational subgroup column
    rsg_var_delim : str
        Delimiter for multi-variable groups
    time_var : str or None
        Time/sequence variable
    response_var : str
        Response variable name
    round_to : int
        Decimal places for rounding
    zero_center : bool
        Whether to center data at zero
    has_grouping : bool
        True if rational subgrouping variables are defined
    has_time : bool
        True if time variable is defined
    requires_sort : bool
        True if data needs sorting

    Raises
    ------
    ValueError
        If response_var is not provided
        If zero-center is not boolean

    Examples
    --------
    >>> config = DataPrepConfig({
    ...     'response_var': 'Height',
    ...     'rsg_vars': ['Operator'],
    ...     'time_var': 'Time'
    ... })
    >>> config.has_grouping
    True
    >>> config.response_var
    'Height'
    """

    VALID_TIME_UNITS = ['Year', 'Quarter', 'Month', 'Week']

    def __init__(self, specification: dict):
        """
        Initialize data preparation configuration.

        Parameters
        ----------
        specification : dict
            Configuration parameters (no analysis_type required)
        """
        # Store raw specification
        self.spec = specification

        # Extract parameters
        self.rsg_vars = self.spec.get('rsg_vars')
        self.rsg_var_name = self.spec.get('rsg_var_name', 'rsg')
        self.rsg_var_delim = self.spec.get('rsg_var_delim', '_')
        self.time_var = self.spec.get('time_var')
        self.response_var = self.spec.get('response_var')
        self.round_to = self.spec.get('round_to', 3)
        # Support both 'zero_center' (preferred) and 'zero-center' (legacy) for migration
        self.zero_center = self.spec.get('zero_center', self.spec.get('zero-center', False))

        # Initialize lists
        self.data_prep_output_cols = []
        self.sort_cols = []
        self.time_grouping_units = []
        self.time_grouping_cols = {}
        self.grouping_cols = []

        # Validate response variable (always required)
        if self.response_var is None:
            raise ValueError('A response variable is required!')

        # Validate zero_center parameter
        if self.zero_center not in [True, False]:
            raise ValueError('Supplied value for zero_center needs to be True or False')

        # Set derived properties
        self.has_time = self.time_var is not None
        self.grouping_cols = self.rsg_var_name if self.has_grouping else []

        # Determine if sorting is required
        self.requires_sort = bool(self.has_grouping and self.has_time or self.has_time)

        # Build column specifications
        self._build_data_prep_cols()
        self._build_sort_cols()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def has_grouping(self) -> bool:
        """Return True if rational subgrouping variables are defined."""
        return self.rsg_vars is not None

    # =========================================================================
    # Column Building Methods
    # =========================================================================

    def _build_sort_cols(self):
        """Build list of columns to sort by."""
        # TODO: Need to factor in time unit
        # Both grouping var and time need to be provided to enable sorting
        if self.has_grouping and self.has_time:
            self.sort_cols = [self.rsg_var_name, self.time_var]
        elif self.has_time:
            self.sort_cols = [self.time_var]

    def _build_data_prep_cols(self) -> list:
        """
        Build list of column names to keep at end of data preparation step.

        Includes time units for aggregation if specified.

        Returns
        -------
        list
            Column names to retain
        """
        self.data_prep_output_cols.insert(0, self.response_var)

        # Add time unit cols
        for col in self.time_grouping_cols:
            self.data_prep_output_cols.append(self.time_grouping_cols[col])

        if self.has_grouping:
            self.data_prep_output_cols.insert(0, 'n')
            self.data_prep_output_cols.insert(0, self.rsg_var_name)
            self.data_prep_output_cols.extend(self.rsg_vars)

        if self.has_time:
            # Avoid duplicating time_var if it's already in rsg_vars
            if self.time_var not in self.data_prep_output_cols:
                self.data_prep_output_cols.insert(0, self.time_var)


class AnalysisSpecification(DataPrepConfig):
    """
    Extended configuration with analysis_type validation (inherits from DataPrepConfig).

    This class extends DataPrepConfig with analysis_type-specific validation
    and output column configuration.

    Parameters
    ----------
    specification : dict
        Dictionary containing analysis parameters, including:

        - **analysis_type** (str, required): Type of analysis ('Xbar', 'S', 'Imr', 'R')
        - **response_var** (str, required): Response variable
        - **rsg_vars** (list, optional): Rational subgrouping variables
        - **time_var** (str, optional): Time/sequence variable
        - **zero_center** (bool, optional): Center data at zero (default: False)
        - **round_to** (int, optional): Decimal places (default: 3)

    Attributes
    ----------
    analysis_type : str
        Type of analysis being performed
    analysis_output_cols : list
        Columns to include in analysis output

    Inherits all attributes from DataPrepConfig:
        rsg_vars, rsg_var_name, rsg_var_delim, time_var, response_var,
        round_to, zero_center, has_grouping, has_time, requires_sort

    Raises
    ------
    ValueError
        If analysis_type is not provided or not supported
        If Xbar/S analysis is requested without grouping variables

    Examples
    --------
    >>> spec = AnalysisSpecification({
    ...     'analysis_type': 'Xbar',
    ...     'response_var': 'Height',
    ...     'rsg_vars': ['Operator', 'Machine'],
    ...     'time_var': 'Time'
    ... })
    >>> spec.has_grouping
    True
    >>> spec.analysis_type
    'Xbar'
    """

    def __init__(self, specification: dict):
        """
        Initialize and validate analysis specification.

        Parameters
        ----------
        specification : dict
            Configuration parameters including 'analysis_type'
        """
        SUPPORTED_ANALYSIS_TYPES = ['Xbar', 'S', 'Imr', 'R']
        GROUPED_ANALYSES = ['Xbar', 'S']

        # Extract and validate analysis_type from specification dict
        self.analysis_type = specification.get('analysis_type')

        if self.analysis_type is None:
            raise ValueError(
                "specification must include 'analysis_type'. "
                f"Valid types: {SUPPORTED_ANALYSIS_TYPES}"
            )

        if self.analysis_type not in SUPPORTED_ANALYSIS_TYPES:
            raise ValueError(
                f'Analysis type: {self.analysis_type} is not supported, '
                f'specify one of: {SUPPORTED_ANALYSIS_TYPES}!'
            )

        # Initialize base class (DataPrepConfig)
        super().__init__(specification)

        # Validate grouped analyses have grouping variables
        if self.analysis_type in GROUPED_ANALYSES and self.rsg_vars is None:
            raise ValueError(
                f'A grouping variable is required to produce a {self.analysis_type} analysis!'
            )

        # Extract residual chart parameters (for VAS residual charting)
        # These are optional - only used when charting residuals
        self.residual = specification.get('residual')  # e.g., 'R2', 'R3', 'R4', 'R5'
        self.residual_chart_type = specification.get('residual_chart_type')  # e.g., 'S', 'Imr'
        self.recentered = specification.get('recentered', False)  # Use RCR columns

        # Initialize analysis output columns
        self.analysis_output_cols = [self.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']
        self._build_output_cols()

    # =========================================================================
    # Column Building Methods (analysis_type-specific)
    # =========================================================================

    def _build_output_cols(self):
        """Build list of output columns based on configuration."""
        # Address Grouping var in output
        if self.has_grouping:
            self.analysis_output_cols.insert(0, self.rsg_var_name)

        # Address time var in output
        if self.has_time:
            self.analysis_output_cols.insert(0, self.time_var)
        else:
            self.analysis_output_cols.insert(0, "x")
