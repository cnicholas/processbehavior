"""
AnalysisSpecification - Configuration and validation for process behavior analyses.

This module provides the AnalysisSpecification class which:
- Validates analysis parameters
- Stores configuration for chart calculations
- Determines data preparation requirements
- Ensures analysis type compatibility with provided data structure

Usage:
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


class AnalysisSpecification:
    """
    Configuration and validation for statistical process control analyses.

    This class is responsible for validating and processing analysis specifications.
    It determines how an analysis is executed and what results are returned.

    Parameters
    ----------
    analysis_type : str
        Type of analysis: 'Xbar', 'S', 'Imr', or 'R'
    analysis_specification : dict
        Dictionary containing analysis parameters:

        - **rsg_vars** (list, optional): Rational subgrouping variables. If provided,
          the input dataset will be grouped by these columns. A concatenated column
          will be created with name specified by rsg_var_name.

        - **time_var** (str, optional): Time dimension or ordering variable. If provided,
          data will be sorted by rsg and time or time only.

        - **response_var** (str, required): Response variable being analyzed.

        - **rsg_var_name** (str, optional): Label for rational subgroup column.
          Defaults to 'rsg'.

        - **rsg_var_delim** (str, optional): Delimiter for multi-variable grouping.
          Default is '_'. For example, col_a and col_b will create 'col_a_col_b'.

        - **time_unit** (str, optional): Time aggregation unit (Year, Quarter, Month, Week).
          Not currently implemented.

        - **round_to** (int, optional): Decimal places for rounding. Defaults to 3.

        - **zero-center** (bool, optional): Whether to center data at zero. Defaults to False.

    Attributes
    ----------
    analysis_type : str
        Type of analysis being performed
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
        If analysis_type is not supported
        If required parameters are missing
        If Xbar/S analysis is requested without grouping variables

    Examples
    --------
    >>> spec = AnalysisSpecification(
    ...     analysis_type='Xbar',
    ...     analysis_specification={
    ...         'rsg_vars': ['Operator', 'Machine'],
    ...         'time_var': 'Time',
    ...         'response_var': 'Height'
    ...     }
    ... )
    >>> spec.has_grouping
    True
    >>> spec.response_var
    'Height'
    """

    VALID_TIME_UNITS = ['Year', 'Quarter', 'Month', 'Week']

    def __init__(self, analysis_type: str, analysis_specification: dict):
        """
        Initialize and validate analysis specification.

        Parameters
        ----------
        analysis_type : str
            Type of analysis: 'Xbar', 'S', 'Imr', or 'R'
        analysis_specification : dict
            Configuration parameters
        """
        SUPPORTED_ANALYSIS_TYPES = ['Xbar', 'S', 'Imr', 'R']
        GROUPED_ANALYSES = ['Xbar', 'S']

        # Store raw specification
        self.analysis_type = analysis_type
        self.spec = analysis_specification

        # Extract parameters
        self.rsg_vars = self.spec.get('rsg_vars')
        self.rsg_var_name = self.spec.get('rsg_var_name', 'rsg')
        self.rsg_var_delim = self.spec.get('rsg_var_delim', '_')
        self.time_var = self.spec.get('time_var')
        self.response_var = self.spec.get('response_var')

        self.round_to = self.spec.get('round_to', 3)  # default round to 3 if none
        self.data_prep_output_cols = []
        self.sort_cols = []
        self.time_grouping_units = []
        self.time_grouping_cols = {}
        self.grouping_cols = []
        self.zero_center = self.spec.get('zero-center', False)

        # Validate analysis type
        if self.analysis_type not in SUPPORTED_ANALYSIS_TYPES:
            raise ValueError(
                f'Analysis type: {self.analysis_type} is not supported, '
                f'specify one of: {SUPPORTED_ANALYSIS_TYPES}!'
            )

        # Validate grouped analyses have grouping variables
        if self.analysis_type in GROUPED_ANALYSES and self.rsg_vars is None:
            raise ValueError(
                f'A grouping variable is required to produce a {self.analysis_type} analysis!'
            )

        # Validate response variable provided
        if self.response_var is None:
            raise ValueError(
                f'A response variable is required to produce a {self.analysis_type} analysis!'
            )

        # Validate zero_center parameter
        if self.zero_center not in [True, False]:
            raise ValueError('Supplied value for zero-center needs to be True or False')

        # Set derived properties
        self.has_time = True if self.time_var is not None else False
        self.grouping_cols = self.rsg_var_name if self.has_grouping else []

        # Determine if sorting is required
        self.requires_sort = True if (self.has_grouping and self.has_time) or self.has_time else False

        # Build column specifications
        self._build_data_prep_cols()

        # Initialize output cols
        self.analysis_output_cols = [self.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']

        self._build_sort_cols()
        self._build_output_cols()

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
        self.data_prep_output_cols.insert(0, self.response_var)  # this is the default value

        # Add time unit cols
        for col in self.time_grouping_cols:
            self.data_prep_output_cols.append(self.time_grouping_cols[col])

        if self.has_grouping:
            self.data_prep_output_cols.insert(0, 'n')
            self.data_prep_output_cols.insert(0, self.rsg_var_name)
            self.data_prep_output_cols.extend(self.rsg_vars)

        if self.has_time:
            self.data_prep_output_cols.insert(0, self.time_var)

    # =========================================================================
    # Legacy Methods (for backward compatibility)
    # =========================================================================

    def data_prep_output_cols(self) -> list:
        """Get data preparation output columns."""
        return self.data_prep_output_cols

    def analysis_output_cols(self) -> list:
        """Get analysis output columns."""
        return self.analysis_output_cols

    def sort_cols(self) -> list:
        """Get sort columns."""
        return self.sort_cols

    def grouping_cols(self) -> list:
        """Get grouping columns."""
        return self.grouping_cols

    def has_time(self) -> bool:
        """Check if time variable is defined."""
        return self._has_time

    def requires_sort(self) -> bool:
        """Check if sorting is required."""
        return self.requires_sort
