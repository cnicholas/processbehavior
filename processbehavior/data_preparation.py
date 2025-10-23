"""
Data validation and preparation for process behavior analysis.

This module handles the initial stages of analysis:
- Column validation (types, presence)
- Grouping variable creation
- Data filtering and sorting
- Key generation for stable merges

Follows the Pythonic Hadley philosophy:
- Pure functions where possible
- Fail fast with helpful errors
- Explicit over implicit
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)


class DataPreparation:
    """
    Validates and prepares raw data for statistical process control analysis.

    This class handles the transformation from raw user data to a clean,
    validated dataset ready for analysis. It:

    1. Validates column existence and types
    2. Creates composite grouping variables (e.g., "lane_head")
    3. Filters out invalid subgroups (n ≤ 1)
    4. Sorts data appropriately
    5. Generates stable keys for reproducible analysis

    The class is designed to be reusable across different analysis types
    and testable in isolation from the heavier analysis logic.

    Examples
    --------
    Basic usage with grouping:

    >>> from analysis_dataset import AnalysisSpecification
    >>> spec = AnalysisSpecification('Xbar', {
    ...     'rsg_vars': ['lane', 'head'],
    ...     'time_var': 'pull',
    ...     'response_var': 'weight'
    ... })
    >>> prep = DataPreparation()
    >>> clean_df = prep.prepare_dataset(raw_df, spec)

    The prepared dataset will have:
    - A 'rsg' column: "lane_head"
    - A 'n' column: observations per group
    - Sorted by ['rsg', 'pull']
    - Only groups with n > 1
    """

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        spec: AnalysisSpecification
    ) -> pd.DataFrame:
        """
        Prepare raw data for analysis.

        Takes user's raw DataFrame and specification, returns a clean dataset
        ready for statistical analysis. This is a pure transformation - the
        input DataFrame is never modified.

        Parameters
        ----------
        df : DataFrame
            Raw input data from user
        spec : AnalysisSpecification
            Configuration specifying analysis parameters

        Returns
        -------
        DataFrame
            Validated, filtered, sorted dataset with:
            - All required columns present and validated
            - Composite grouping column if needed
            - Invalid groups removed (n ≤ 1)
            - Sorted appropriately
            - Only requested output columns
            - No missing values

        Raises
        ------
        ValueError
            If required columns are missing, wrong type, or all groups invalid

        Examples
        --------
        >>> spec = AnalysisSpecification('Xbar', {
        ...     'rsg_vars': ['lane'],
        ...     'time_var': 'pull',
        ...     'response_var': 'weight'
        ... })
        >>> df = pd.DataFrame({
        ...     'lane': ['A', 'A', 'B'],
        ...     'pull': [1, 2, 1],
        ...     'weight': [10.1, 10.3, 9.9]
        ... })
        >>> prep = DataPreparation()
        >>> result = prep.prepare_dataset(df, spec)
        >>> 'rsg' in result.columns
        True
        >>> result['rsg'].tolist()
        ['A', 'A', 'B']
        """
        logger.debug('Entering prepare_dataset')
        out = df.copy()

        # Validate columns early (fail fast!)
        self.validate_columns(out, spec)

        # Add composite grouping variable if needed
        if spec.has_grouping:
            out = self._add_grouping_column(out, spec)
            out = self._filter_small_groups(out, spec)

        # Sort if required
        if spec.requires_sort:
            out = out.sort_values(spec.sort_cols, kind='stable')

        # Keep only requested columns
        out = out[spec.data_prep_output_cols]

        # Drop any rows with missing data
        out = out.dropna()

        return out

    def validate_columns(
        self,
        df: pd.DataFrame,
        spec: AnalysisSpecification
    ) -> None:
        """
        Validate that required columns exist and have correct types.

        Checks:
        - All grouping variables exist
        - Time variable exists (if specified)
        - Response variable exists and is numeric

        This function follows "Fail Fast, Fail Helpful" - it raises
        immediately with a message that explains WHAT is wrong, WHY
        it matters, and HOW to fix it.

        Parameters
        ----------
        df : DataFrame
            Input data to validate
        spec : AnalysisSpecification
            Specification with column requirements

        Raises
        ------
        ValueError
            If required columns missing or wrong type, with helpful message
            suggesting fixes

        Examples
        --------
        >>> spec = AnalysisSpecification('Xbar', {
        ...     'rsg_vars': ['lane'],
        ...     'response_var': 'weight'
        ... })
        >>> df = pd.DataFrame({'weight': [10.1, 10.2]})
        >>> prep = DataPreparation()
        >>> prep.validate_columns(df, spec)  # Raises: lane not found
        Traceback (most recent call last):
            ...
        ValueError: One or more grouping variables not found...
        """
        df_cols = df.columns.tolist()

        # Validate grouping variables
        if spec.has_grouping:
            missing = set(spec.rsg_vars) - set(df_cols)
            if missing:
                raise ValueError(
                    f"One or more grouping variables not found in dataset.\n"
                    f"Missing: {sorted(missing)}\n"
                    f"Required: {spec.rsg_vars}\n"
                    f"Available columns: {df_cols}\n"
                    f"Fix: Check spelling or provide correct column names"
                )

        # Validate time variable
        if spec.has_time and spec.time_var not in df_cols:
            raise ValueError(
                f"Time variable '{spec.time_var}' not found in dataset.\n"
                f"Available columns: {df_cols}\n"
                f"Fix: Check spelling or specify correct time column"
            )

        # Validate response variable
        if spec.response_var not in df_cols:
            raise ValueError(
                f"Response variable '{spec.response_var}' not found in dataset.\n"
                f"Available columns: {df_cols}\n"
                f"Fix: Check spelling or specify correct measurement column"
            )

        # Validate response variable is numeric
        if not is_numeric_dtype(df[spec.response_var]):
            raise ValueError(
                f"Response variable '{spec.response_var}' must be numeric.\n"
                f"Current type: {df[spec.response_var].dtype}\n"
                f"Fix: Convert to numeric or choose a different column"
            )

        logger.info('Column validation passed')

    def build_keys(
        self,
        df: pd.DataFrame,
        spec: AnalysisSpecification
    ) -> pd.DataFrame:
        """
        Add stable key columns for reproducible analysis.

        Creates three types of keys:
        - obs_id: Unique ID per observation (for stable sorting)
        - rsg_key: Tuple key for factor combinations
        - cell_key: Tuple key for (factor × time) cells

        These tuple keys are essential for mathematical operations
        (avoiding string comparison issues) while the composite string
        column (e.g., 'rsg') is kept for user-facing charts.

        Parameters
        ----------
        df : DataFrame
            Input data (must already have grouping/time columns)
        spec : AnalysisSpecification
            Specification with grouping and time variables

        Returns
        -------
        DataFrame
            Input data with added key columns

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'lane': ['A', 'B'],
        ...     'pull': [1, 1],
        ...     'weight': [10.1, 9.9]
        ... })
        >>> spec = AnalysisSpecification('Xbar', {
        ...     'rsg_vars': ['lane'],
        ...     'time_var': 'pull',
        ...     'response_var': 'weight'
        ... })
        >>> prep = DataPreparation()
        >>> result = prep.build_keys(df, spec)
        >>> 'obs_id' in result.columns
        True
        >>> 'rsg_key' in result.columns
        True
        """
        out = df.copy()

        # Stable row ID for reproducible merges
        out['obs_id'] = np.arange(len(out), dtype=np.int64)

        k_vars = spec.rsg_vars or []
        t = spec.time_var

        # RSG key: tuple of factor values
        if k_vars:
            out['rsg_key'] = list(map(tuple, out[k_vars].astype(object).values))
        else:
            out['rsg_key'] = [()] * len(out)

        # Cell key: tuple of (factor + time) values
        if k_vars and t:
            out['cell_key'] = list(map(tuple, out[k_vars + [t]].astype(object).values))
        elif t:
            out['cell_key'] = out[t].astype(object).apply(lambda x: (x,))
        else:
            out['cell_key'] = out['rsg_key']

        return out

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _add_grouping_column(
        self,
        df: pd.DataFrame,
        spec: AnalysisSpecification
    ) -> pd.DataFrame:
        """
        Add composite grouping column (e.g., 'lane_head').

        Creates a user-friendly string column combining multiple factors.
        For single factors, just copies the column with a standard name.
        """
        out = df.copy()

        # Drop 'n' column if it exists (will be recalculated)
        if 'n' in out.columns:
            logger.debug('Dropping existing "n" column (will be recalculated)')
            out = out.drop(columns=['n'])

        # Create composite column
        if len(spec.rsg_vars) > 1:
            # Multiple factors: combine with delimiter
            out = self._add_composite_column(
                df=out,
                cols_to_combine=spec.rsg_vars,
                col_name=spec.rsg_var_name,
                col_delim=spec.rsg_var_delim
            )
        else:
            # Single factor: copy with standard name
            out = self._add_column(
                df=out,
                new_col_name=spec.rsg_var_name,
                existing_column=spec.rsg_vars[0]
            )

        return out

    def _filter_small_groups(
        self,
        df: pd.DataFrame,
        spec: AnalysisSpecification
    ) -> pd.DataFrame:
        """
        Remove groups with n ≤ 1 (can't calculate variance).

        For grouped analyses (Xbar, S), we need at least 2 observations
        per group to estimate within-group variance.
        """
        # Count observations per group
        grouped = df.groupby(spec.rsg_var_name).size()
        starting_count = grouped.count()
        logger.debug('Starting with %s groups', starting_count)

        # Keep only groups with n > 1
        grouped = grouped[grouped > 1]

        if grouped.shape[0] == 0:
            raise ValueError(
                "All subgroups have 1 or fewer observations!\n"
                f"Analysis type '{spec.analysis_type}' requires multiple "
                f"observations per group.\n"
                f"Fix: Add more data or use 'Imr' analysis for individual values"
            )

        grouped = grouped.reset_index()
        grouped = grouped.rename(columns={0: 'n'})

        ending_count = grouped.shape[0]
        logger.debug('Groups remaining: %s', ending_count)
        logger.debug('Removed %s group(s)', starting_count - ending_count)

        # Merge to filter
        out = pd.merge(df, grouped, how='inner', on=spec.rsg_var_name)

        return out

    def _add_column(
        self,
        df: pd.DataFrame,
        new_col_name: str,
        existing_column: str
    ) -> pd.DataFrame:
        """Copy an existing column with a new name."""
        if existing_column not in df.columns:
            raise ValueError(
                f"Cannot copy column '{existing_column}' - not in dataset.\n"
                f"Available: {df.columns.tolist()}"
            )

        out = df.copy()
        out[new_col_name] = df[existing_column]
        return out

    def _add_composite_column(
        self,
        df: pd.DataFrame,
        cols_to_combine: list[str],
        col_name: str,
        col_delim: str = '_'
    ) -> pd.DataFrame:
        """
        Create composite column by combining multiple columns.

        Example: ['lane', 'head'] → 'lane_head' with values like 'A_1', 'B_2'
        """
        missing = set(cols_to_combine) - set(df.columns)
        if missing:
            raise ValueError(
                f"Cannot create composite column - some columns missing.\n"
                f"Missing: {sorted(missing)}\n"
                f"Available: {df.columns.tolist()}"
            )

        if len(cols_to_combine) == 1:
            # Single column - just copy
            return self._add_column(df, col_name, cols_to_combine[0])

        # Multiple columns - combine with delimiter
        out = df.copy()
        len_delim = len(col_delim)

        # Build concatenated string
        combined = (df[cols_to_combine].astype(str) + col_delim).cumsum(1).iloc[:, -1].values

        # Remove trailing delimiter
        combined = [x[:-len_delim] for x in combined]

        out[col_name] = combined

        return out
