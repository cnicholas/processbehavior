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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from natsort import natsorted
from pandas.api.types import is_numeric_dtype

if TYPE_CHECKING:
    from .analysis_specification import DataPrepConfig

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
        spec: DataPrepConfig
    ) -> pd.DataFrame:
        """
        Prepare raw data for analysis with automatic type conversion.

        Takes user's raw DataFrame and specification, returns a clean dataset
        ready for statistical analysis. This is a pure transformation - the
        input DataFrame is never modified.

        **Type Conversion for Correct Sorting:**
        - String-numeric columns ('1', '2', '10') → numeric (1, 2, 10)
        - String-date columns ('2024-01-01') → datetime
        - Numeric, date, datetime, categorical, Period types → unchanged
        - RSG column → categorical with natural sort order

        This ensures correct sorting for:
        - Time series: 1, 2, 3, 10 (not '1', '10', '2', '3')
        - Factor levels: Lane 1, Lane 2, Lane 10 (not Lane 1, Lane 10, Lane 2)
        - Moving range calculations (adjacent observations)
        - Signal detection rules (consecutive points)

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
            - Type conversion applied for correct sorting
            - Composite grouping column if needed (categorical with natural sort)
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

        # Convert types for correct sorting (time_var and factor columns)
        # This handles string-numeric ('1', '10') and string-dates
        if spec.has_time:
            out[spec.time_var], msg = self._detect_and_convert_type(
                out[spec.time_var],
                spec.time_var
            )

        if spec.has_grouping:
            for col in spec.rsg_vars:
                out[col], msg = self._detect_and_convert_type(out[col], col)

        # Add composite grouping variable if needed
        if spec.has_grouping:
            out = self._add_grouping_column(out, spec)
            out = self._filter_small_groups(out, spec)

            # Make RSG categorical with natural sort order
            # This ensures 'Lane_1', 'Lane_2', 'Lane_10' (not 'Lane_1', 'Lane_10', 'Lane_2')
            out[spec.rsg_var_name] = self._make_categorical_rsg(
                out[spec.rsg_var_name],
                spec.rsg_var_name
            )

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
        spec: DataPrepConfig
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
        spec: DataPrepConfig
    ) -> pd.DataFrame:
        """
        Add stable key columns for reproducible analysis.

        Creates three types of keys:
        - obs_id: Unique ID per observation (for stable sorting)
        - rsg_key: Tuple key for factor combinations
        - cell_key: Tuple key for (factor × time) cells

        Dual-column strategy:
        - **rsg_key (tuple)**: Available for internal operations (fast lookups, hierarchical ops)
          Preserves factor types after type conversion: (1, 1), (1, 2), (1, 10)
        - **rsg (string)**: Used for display (chart labels, user output)
          Created separately in _add_grouping_column() as categorical with natural sort

        Note: As of the type conversion implementation, tuple keys are created but
        not used for primary sorting. Type conversion + categorical RSG handles
        correct ordering. Tuple keys remain available for future enhancements
        (fast lookups, hierarchical operations, etc.).

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

        Note: This creates the 'rsg' STRING column for user-facing display
        (chart labels, summaries). The 'rsg_key' TUPLE column (created in
        build_keys()) is used for internal computation to ensure correct
        numeric sorting.
        """
        out = df.copy()

        # Drop 'n' column if it exists (will be recalculated)
        if 'n' in out.columns:
            logger.debug('Dropping existing "n" column (will be recalculated)')
            out = out.drop(columns=['n'])

        # Create composite column (always as string for display)
        if len(spec.rsg_vars) > 1:
            # Multiple factors: combine with delimiter
            out = self._add_composite_column(
                df=out,
                cols_to_combine=spec.rsg_vars,
                col_name=spec.rsg_var_name,
                col_delim=spec.rsg_var_delim
            )
        else:
            # Single factor: copy with standard name, convert to string
            out = self._add_column(
                df=out,
                new_col_name=spec.rsg_var_name,
                existing_column=spec.rsg_vars[0]
            )
            # Ensure RSG is string (even if source column is numeric)
            out[spec.rsg_var_name] = out[spec.rsg_var_name].astype(str)

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
        grouped = df.groupby(spec.rsg_var_name, observed=True).size()
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

    def _detect_and_convert_type(
        self,
        series: pd.Series,
        col_name: str
    ) -> tuple[pd.Series, str | None]:
        """
        Detect and convert string columns to appropriate types.

        Attempts conversion in this order:
        1. Try pd.to_numeric() for numeric strings ('1', '2', '10' → 1, 2, 10)
        2. Try pd.to_datetime() for date strings ('2024-01-01' → datetime)
        3. Keep original if both fail

        This ensures correct sorting:
        - Numeric: 1, 2, 10 (not '1', '10', '2')
        - Date/datetime: chronological order
        - Categorical/string: natural sort will be applied

        Parameters
        ----------
        series : pd.Series
            Column to potentially convert
        col_name : str
            Name of column (for logging)

        Returns
        -------
        tuple[pd.Series, str | None]
            - Converted series (or original if no conversion)
            - Conversion message (None if no conversion)

        Examples
        --------
        >>> prep = DataPreparation()
        >>> s = pd.Series(['1', '2', '10'])
        >>> converted, msg = prep._detect_and_convert_type(s, 'time')
        >>> converted.dtype
        dtype('int64')
        >>> '1' in msg  # Should mention conversion
        True
        """
        # Skip if already numeric or datetime
        if pd.api.types.is_numeric_dtype(series):
            return series, None
        if pd.api.types.is_datetime64_any_dtype(series):
            return series, None
        if isinstance(series.dtype, pd.PeriodDtype):
            return series, None

        # Skip if contains Python date/datetime objects (already proper type)
        if len(series) > 0 and isinstance(series.iloc[0], (type(None), type(pd.NaT))):
            # Handle NaT/None
            first_valid_idx = series.first_valid_index()
            if first_valid_idx is not None:
                first_val = series.loc[first_valid_idx]
            else:
                return series, None
        else:
            first_val = series.iloc[0] if len(series) > 0 else None

        if first_val is not None:
            import datetime
            if isinstance(first_val, (datetime.date, datetime.datetime)):
                return series, None

        # Skip if already categorical (assume user set ordering intentionally)
        if isinstance(series.dtype, pd.CategoricalDtype):
            return series, None

        # Try numeric conversion
        try:
            numeric_vals = pd.to_numeric(series, errors='coerce')
            # Only convert if ALL values succeeded (no NaNs introduced)
            if (not numeric_vals.isna().any() or series.isna().any()) and numeric_vals.notna().any():
                msg = (
                    f"Converted column '{col_name}' from string to numeric "
                    f"for correct sorting (example: '{series.iloc[0]}' → {numeric_vals.iloc[0]})"
                )
                logger.info(msg)
                return numeric_vals, msg
        except (ValueError, TypeError):
            pass

        # Try datetime conversion
        try:
            datetime_vals = pd.to_datetime(series, errors='coerce')
            # Only convert if most values succeeded
            success_rate = datetime_vals.notna().sum() / len(series)
            if success_rate > 0.5:  # At least 50% converted
                msg = (
                    f"Converted column '{col_name}' from string to datetime "
                    f"for correct chronological sorting"
                )
                logger.info(msg)
                return datetime_vals, msg
        except (ValueError, TypeError):
            pass

        # No conversion possible/needed
        return series, None

    def _make_categorical_rsg(
        self,
        series: pd.Series,
        col_name: str
    ) -> pd.Series:
        """
        Convert RSG column to categorical with natural sort order.

        Uses natsort to handle numeric parts in strings correctly:
        - Natural sort: 'Lane_1', 'Lane_2', 'Lane_10'
        - Lexicographic: 'Lane_1', 'Lane_10', 'Lane_2' (wrong!)

        This ensures groupby, sort_values, and plotting operations
        all respect the correct numeric ordering within strings.

        Parameters
        ----------
        series : pd.Series
            RSG column (string values like 'Lane_1_Head_10')
        col_name : str
            Name of column (for logging)

        Returns
        -------
        pd.Series
            Categorical series with natural-sorted categories

        Examples
        --------
        >>> prep = DataPreparation()
        >>> s = pd.Series(['Lane_1_Head_10', 'Lane_1_Head_2', 'Lane_10_Head_1'])
        >>> cat = prep._make_categorical_rsg(s, 'rsg')
        >>> cat.dtype.ordered
        True
        >>> list(cat.cat.categories)
        ['Lane_1_Head_2', 'Lane_1_Head_10', 'Lane_10_Head_1']
        """
        # Get unique values and sort naturally
        unique_vals = series.unique()
        sorted_categories = natsorted(unique_vals)

        # Create ordered categorical
        categorical = pd.Categorical(
            series,
            categories=sorted_categories,
            ordered=True
        )

        logger.info(
            f"Created categorical column '{col_name}' with natural sort order "
            f"({len(sorted_categories)} categories)"
        )

        return pd.Series(categorical, index=series.index)
