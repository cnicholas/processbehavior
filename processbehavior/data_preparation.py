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

from .exceptions import ColumnNotFoundError, FactorNotFoundError, ValidationError

if TYPE_CHECKING:
    from .formulation_spec import FormulationSpec

logger = logging.getLogger(__name__)


def natural_sort_key(s: str) -> list:
    """
    Key function for natural sorting of strings with embedded numbers.

    This ensures '1_10' comes after '1_2', not before (lexicographic would put 10 before 2).

    Parameters
    ----------
    s : str
        String to generate sort key for

    Returns
    -------
    list
        Sort key that handles embedded numbers correctly

    Examples
    --------
    >>> sorted(['1_2', '1_10', '1_1'], key=natural_sort_key)
    ['1_1', '1_2', '1_10']
    """
    import re

    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def encode_rsg(factor_values, delimiter: str = '_') -> str:
    """
    Encode factor values into RSG (Rational Subgroup) string.

    This is the single source of truth for RSG encoding.
    Used by both:
    - DataPreparation._add_composite_column() for observed data
    - Plan expansion for expected data (coverage calculation, missing_combos)
    - analysis_result.py for stratum comparisons
    - plotter.py for chart IDs

    Handles both tuple/list (multi-factor) and scalar (single-factor) strata.

    Note: RSG identity assumes canonical factor ordering defined upstream.
    Factor order in encode_rsg() calls must match order in RSG column creation.
    The `by` parameter only affects layout/presentation, not stratum identity.

    Parameters
    ----------
    factor_values : tuple, list, or scalar
        Values for each factor, in factor order.
        Example: (1, 'A'), [2, 'B'], or just 'A' for single factor
    delimiter : str
        Delimiter between values (default: '_')

    Returns
    -------
    str
        Encoded RSG string (e.g., "1_A", "2_B", "A")

    Examples
    --------
    >>> encode_rsg((1, 'A'))
    '1_A'
    >>> encode_rsg([2, 'B'], delimiter='-')
    '2-B'
    >>> encode_rsg((42,))
    '42'
    >>> encode_rsg('Machine_1')  # Scalar - returned as-is (not iterated!)
    'Machine_1'
    >>> encode_rsg(42)  # Numeric scalar
    '42'
    """
    if isinstance(factor_values, (tuple, list)):
        return delimiter.join(str(v) for v in factor_values)
    return str(factor_values)


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

    >>> from processbehavior.formulation_spec import FormulationSpec
    >>> spec = FormulationSpec(
    ...     response_var='weight',
    ...     rsg_vars=('lane', 'head'),
    ...     time_var='pull'
    ... )
    >>> prep = DataPreparation()
    >>> clean_df = prep.prepare_dataset(raw_df, spec)

    The prepared dataset will have:
    - A 'rsg' column: "lane_head"
    - A 'n' column: observations per group
    - Sorted by ['rsg', 'pull']
    - Only groups with n > 1
    """

    def _get_sort_cols(self, spec: FormulationSpec) -> list[str]:
        """Sort order for prepared data."""
        if spec.has_grouping and spec.has_time:
            return [spec.rsg_var_name, spec.time_var]
        elif spec.has_time:
            return [spec.time_var]
        return []

    def _get_output_cols(self, spec: FormulationSpec) -> list[str]:
        """Columns to keep after data preparation."""
        cols = [spec.response_var]
        if spec.has_grouping:
            cols = [spec.rsg_var_name, 'n'] + spec.rsg_vars_list + cols
        if spec.has_time and spec.time_var not in cols:
            cols.insert(0, spec.time_var)
        return cols

    def prepare_dataset(self, df: pd.DataFrame, spec: FormulationSpec) -> pd.DataFrame:
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
        spec : FormulationSpec
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
        >>> spec = FormulationSpec(
        ...     response_var='weight',
        ...     rsg_vars=('lane',),
        ...     time_var='pull'
        ... )
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
            out[spec.time_var], msg = self._detect_and_convert_type(out[spec.time_var], spec.time_var)

        if spec.has_grouping:
            for col in spec.rsg_vars:
                out[col], msg = self._detect_and_convert_type(out[col], col)

        # Add composite grouping variable if needed
        if spec.has_grouping:
            out = self._add_grouping_column(out, spec)

            # Drop rows with missing values in analysis-critical columns BEFORE
            # calculating n. This ensures n reflects actual usable observations,
            # not the raw count which may include rows that will be dropped later.
            drop_cols = [spec.response_var, spec.rsg_var_name]
            if spec.has_time:
                drop_cols.append(spec.time_var)
            out = out.dropna(subset=drop_cols)

            # FormulationSpec is chart-agnostic — filtering for Xbar/S (n≥2)
            # happens at analysis time, not during data preparation.
            # Just add group sizes without filtering.
            out = self._add_group_sizes(out, spec)

            # Make RSG categorical with natural sort order
            # This ensures 'Lane_1', 'Lane_2', 'Lane_10' (not 'Lane_1', 'Lane_10', 'Lane_2')
            out[spec.rsg_var_name] = self._make_categorical_rsg(out[spec.rsg_var_name], spec.rsg_var_name)

        # Sort if required
        sort_cols = self._get_sort_cols(spec)
        if sort_cols:
            out = out.sort_values(sort_cols, kind='stable')

        # Keep only requested columns
        output_cols = self._get_output_cols(spec)
        out = out[output_cols]

        # Final safety net: drop any remaining rows with missing data
        # (Primary dropna for grouped data happens earlier, before n calculation)
        out = out.dropna()

        return out

    def validate_columns(self, df: pd.DataFrame, spec: FormulationSpec) -> None:
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
        spec : FormulationSpec
            Specification with column requirements

        Raises
        ------
        ValueError
            If required columns missing or wrong type, with helpful message
            suggesting fixes

        Examples
        --------
        >>> spec = FormulationSpec(
        ...     response_var='weight',
        ...     rsg_vars=('lane',)
        ... )
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
                raise FactorNotFoundError(
                    f'One or more grouping variables not found in dataset.\n'
                    f'Missing: {sorted(missing)}\n'
                    f'Required: {spec.rsg_vars}\n'
                    f'Available columns: {df_cols}\n'
                    f'Fix: Check spelling or provide correct column names',
                    factor=str(sorted(missing)),
                    available=df_cols,
                )

        # Validate time variable
        if spec.has_time and spec.time_var not in df_cols:
            raise ColumnNotFoundError(
                f"Time variable '{spec.time_var}' not found in dataset.\n"
                f'Available columns: {df_cols}\n'
                f'Fix: Check spelling or specify correct time column',
                column=spec.time_var,
                available=df_cols,
            )

        # Validate response variable
        if spec.response_var not in df_cols:
            raise ColumnNotFoundError(
                f"Response variable '{spec.response_var}' not found in dataset.\n"
                f'Available columns: {df_cols}\n'
                f'Fix: Check spelling or specify correct measurement column',
                column=spec.response_var,
                available=df_cols,
            )

        # Validate response variable is numeric
        if not is_numeric_dtype(df[spec.response_var]):
            raise ValidationError(
                f"Response variable '{spec.response_var}' must be numeric.\n"
                f'Current type: {df[spec.response_var].dtype}\n'
                f'Fix: Convert to numeric or choose a different column'
            )

        logger.debug('Column validation passed')

    def build_keys(self, df: pd.DataFrame, spec: FormulationSpec) -> pd.DataFrame:
        """
        Add stable key columns for reproducible analysis.

        Creates four types of keys:
        - obs_id: Row identity (assigned BEFORE canonical sort, for merges/joins)
        - rsg_key: Tuple key for factor combinations
        - cell_key: Tuple key for (factor × time) cells
        - sort_key: Canonical ordering (assigned AFTER canonical sort)

        Canonical sort order: (cell_key ascending, obs_id ascending as tie-breaker)

        Dual-column strategy:
        - **rsg_key (tuple)**: Available for internal operations (fast lookups, hierarchical ops)
          Preserves factor types after type conversion: (1, 1), (1, 2), (1, 10)
        - **rsg (string)**: Used for display (chart labels, user output)
          Created separately in _add_grouping_column() as categorical with natural sort

        Note: sort_key is study-instance specific and must never be persisted/compared
        across different Studies unless paired with formulation context.

        Parameters
        ----------
        df : DataFrame
            Input data (must already have grouping/time columns)
        spec : FormulationSpec
            Specification with grouping and time variables

        Returns
        -------
        DataFrame
            Input data with added key columns, sorted by canonical order

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'lane': ['A', 'B'],
        ...     'pull': [1, 1],
        ...     'weight': [10.1, 9.9]
        ... })
        >>> spec = FormulationSpec(
        ...     response_var='weight',
        ...     rsg_vars=('lane',),
        ...     time_var='pull'
        ... )
        >>> prep = DataPreparation()
        >>> result = prep.build_keys(df, spec)
        >>> 'obs_id' in result.columns
        True
        >>> 'sort_key' in result.columns
        True
        """
        out = df.copy()

        # 1. Assign obs_id FIRST (row identity, before any sorting)
        # This captures the row order as it enters build_keys() (post-cleaning/filtering)
        out['obs_id'] = np.arange(len(out), dtype=np.int64)

        k_vars = spec.rsg_vars_list
        t = spec.time_var

        # 2. Build tuple keys
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

        # 3. Sort by canonical order: (cell_key, obs_id)
        out = out.sort_values(['cell_key', 'obs_id'], kind='stable')

        # 4. Assign sort_key AFTER sorting (canonical ordering 0..N-1)
        out['sort_key'] = np.arange(len(out), dtype=np.int64)

        return out

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _add_grouping_column(self, df: pd.DataFrame, spec: FormulationSpec) -> pd.DataFrame:
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
                df=out, cols_to_combine=spec.rsg_vars_list, col_name=spec.rsg_var_name, col_delim=spec.rsg_var_delim
            )
        else:
            # Single factor: copy with standard name, convert to string
            out = self._add_column(df=out, new_col_name=spec.rsg_var_name, existing_column=spec.rsg_vars[0])
            # Ensure RSG is string (even if source column is numeric)
            out[spec.rsg_var_name] = out[spec.rsg_var_name].astype(str)

        return out

    def _add_group_sizes(self, df: pd.DataFrame, spec: FormulationSpec) -> pd.DataFrame:
        """
        Add 'n' column with kt cell sizes without filtering.

        Groups by kt (factor × time) columns to get true subgroup sizes.
        Used for IMR/R analyses and SDS detection where n=1 per cell is valid.
        """
        # Build kt grouping columns (factor + time if both present)
        kt_cols = [spec.rsg_var_name, spec.time_var] if spec.has_time else [spec.rsg_var_name]

        grouped = df.groupby(kt_cols, observed=True).size()
        grouped = grouped.reset_index(name='n')

        out = pd.merge(df, grouped, how='left', on=kt_cols)
        return out

    def _add_column(self, df: pd.DataFrame, new_col_name: str, existing_column: str) -> pd.DataFrame:
        """Copy an existing column with a new name."""
        if existing_column not in df.columns:
            raise ValueError(
                f"Cannot copy column '{existing_column}' - not in dataset.\nAvailable: {df.columns.tolist()}"
            )

        out = df.copy()
        out[new_col_name] = df[existing_column]
        return out

    def _add_composite_column(
        self, df: pd.DataFrame, cols_to_combine: list[str], col_name: str, col_delim: str = '_'
    ) -> pd.DataFrame:
        """
        Create composite column by combining multiple columns.

        Uses encode_rsg() for consistent encoding with plan expansion.

        Example: ['lane', 'head'] → 'lane_head' with values like 'A_1', 'B_2'
        """
        missing = set(cols_to_combine) - set(df.columns)
        if missing:
            raise ValueError(
                f'Cannot create composite column - some columns missing.\n'
                f'Missing: {sorted(missing)}\n'
                f'Available: {df.columns.tolist()}'
            )

        # Validate no missing values in source columns before encoding
        # str(None) → "None" and str(np.nan) → "nan" which won't trigger isna()
        # on the output, so we must check inputs
        if df[cols_to_combine].isna().any().any():
            missing_counts = df[cols_to_combine].isna().sum()
            raise ValueError(
                f"Cannot build RSG '{col_name}': missing values in factor columns. "
                f'Missing counts: {missing_counts.to_dict()}'
            )

        if len(cols_to_combine) == 1:
            # Single column - just copy
            return self._add_column(df, col_name, cols_to_combine[0])

        # Multiple columns - combine with delimiter using shared encode_rsg()
        out = df.copy()
        out[col_name] = df[cols_to_combine].apply(lambda row: encode_rsg(tuple(row), delimiter=col_delim), axis=1)

        return out

    def _detect_and_convert_type(self, series: pd.Series, col_name: str) -> tuple[pd.Series, str | None]:
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
            datetime_vals = pd.to_datetime(series, errors='coerce', format='mixed')
            # Only convert if most values succeeded
            success_rate = datetime_vals.notna().sum() / len(series)
            if success_rate > 0.5:  # At least 50% converted
                msg = f"Converted column '{col_name}' from string to datetime for correct chronological sorting"
                logger.info(msg)
                return datetime_vals, msg
        except (ValueError, TypeError):
            pass

        # No conversion possible/needed
        return series, None

    def _make_categorical_rsg(self, series: pd.Series, col_name: str) -> pd.Series:
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
        categorical = pd.Categorical(series, categories=sorted_categories, ordered=True)

        logger.info(
            f"Created categorical column '{col_name}' with natural sort order ({len(sorted_categories)} categories)"
        )

        return pd.Series(categorical, index=series.index)
