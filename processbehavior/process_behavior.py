"""
ProcessBehavior - Main entry point for process behavior analysis.

This module provides a user-friendly interface with IDE auto-completion for column names
and automatic chart selection driven by the analytical design state (ADS).

Usage:
    from processbehavior import ProcessBehavior

    pb = ProcessBehavior(df)

    # Step 1: formulate() - analyze structure and get recommendations
    study = pb.formulate(
        response=pb.cols.Measurement,
        factors=[pb.cols.Operator],
        time=pb.cols.ProductionTime
    )

    # Step 2: execute() - run the chart
    result = study.execute()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from .exceptions import ColumnNotFoundError, ValidationError
from .formulation_spec import FormulationSpec
from .sds_detector import SDSRegistry, SDSResult

if TYPE_CHECKING:
    from .study import Study

logger = logging.getLogger(__name__)

# Threshold for auto-converting object columns to numeric.
# If >= this fraction of non-NA values convert successfully, apply the conversion.
_NUMERIC_CONVERSION_THRESHOLD = 0.8


def _try_clean_numeric_strings(series: pd.Series) -> pd.Series | None:
    """Try to clean formatted numeric strings (currency, thousands sep, etc.).

    Returns a numeric Series if cleaning succeeds for >= 80% of non-NA values,
    or None if the column doesn't look numeric after cleaning.

    Handles: $, EUR, GBP, JPY, unicode currency symbols, thousands commas,
    accounting negatives like (1,234.56), percentage signs, and whitespace.
    """
    from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

    # Don't "clean" already-typed columns. Crucially, datetime64 must be guarded:
    # pd.to_numeric(datetime64) silently succeeds (int64 nanoseconds), which would
    # otherwise replace a date column with huge integers. (Mirrors the datetime
    # guard in DataPreparation._detect_and_convert_type.)
    if is_numeric_dtype(series) or is_datetime64_any_dtype(series) or isinstance(series.dtype, pd.PeriodDtype):
        return None

    # Only process object columns with actual data
    non_na = series.dropna()
    if len(non_na) == 0:
        return None

    # Fast path: try direct pd.to_numeric first (handles plain numeric strings)
    direct = pd.to_numeric(series, errors='coerce')
    direct_success = direct.notna().sum()
    original_non_na = non_na.shape[0]
    if direct_success >= original_non_na * _NUMERIC_CONVERSION_THRESHOLD:
        return direct

    # Convert to string for .str operations (handles mixed float/str columns)
    s = series.astype(str)
    # Preserve original NAs
    original_na_mask = series.isna()
    # Also treat the string 'nan' (from astype) as NA
    str_nan_mask = s == 'nan'

    # Strip whitespace
    cleaned = s.str.strip()
    # Accounting negatives: (xxx) -> -xxx
    cleaned = cleaned.str.replace(r'^\((.*)\)$', r'-\1', regex=True)
    # Remove currency symbols and optional adjacent whitespace
    # Use literal Unicode chars (€£¥) — raw string \u escapes aren't valid in pyarrow regex
    cleaned = cleaned.str.replace(r'[$€£¥]\s*', '', regex=True)
    # Remove thousands separators (commas)
    cleaned = cleaned.str.replace(',', '', regex=False)
    # Remove percentage signs
    cleaned = cleaned.str.replace('%', '', regex=False)
    # Final strip (catches residual whitespace after symbol removal)
    cleaned = cleaned.str.strip()

    result = pd.to_numeric(cleaned, errors='coerce')

    # Restore original NAs (don't count astype('str') artifacts as conversions)
    result[original_na_mask | str_nan_mask] = pd.NA

    # Check success rate against non-NA values only
    converted_count = result.notna().sum()
    if converted_count >= original_non_na * _NUMERIC_CONVERSION_THRESHOLD:
        return result

    return None


@dataclass
class ColumnRef:
    """
    Column reference with level awareness for IDE discoverability.

    NOT a str subclass to avoid pandas/numpy quirks, serialization issues,
    and hashing/equality surprises. Implements __hash__ and __eq__ for
    dict key usage. Compares equal to strings for flexibility.

    Attributes
    ----------
    name : str
        The column name in the DataFrame
    levels : list
        Sorted unique values from the data (property)
    count : int
        Number of distinct levels (property)

    Examples
    --------
    >>> pb = ProcessBehavior(df)
    >>> pb.cols.Lane           # Lane (4): [1, 2, 3, 4]
    >>> pb.cols.Lane.levels    # [1, 2, 3, 4]
    >>> pb.cols.Lane.count     # 4

    Use in plan:

    >>> study = pb.formulate(
    ...     response=pb.cols.Weight,
    ...     plan={pb.cols.Lane: [1, 2, 3, 4]}
    ... )
    """

    name: str
    _df: pd.DataFrame = field(repr=False, compare=False)

    @property
    def levels(self) -> list:
        """Sorted unique values from the data."""
        values = self._df[self.name].dropna().unique()
        try:
            return sorted(values.tolist())
        except TypeError:
            # Mixed types can't be sorted
            return list(values)

    @property
    def count(self) -> int:
        """Number of distinct levels."""
        return len(self.levels)

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ColumnRef):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def __repr__(self) -> str:
        lvls = self.levels
        if len(lvls) <= 6:
            return f'{self.name} ({len(lvls)}): {lvls}'
        return f'{self.name} ({len(lvls)}): [{lvls[0]}..{lvls[-1]}]'


class ColumnAccessor:
    """
    Provides IDE auto-completion for DataFrame column names with level awareness.

    Usage:
        pb = ProcessBehavior(df)
        pb.cols.Height         # ColumnRef with auto-completion
        pb.cols.Height.levels  # [1.0, 1.5, 2.0, ...]
        pb.cols.Height.count   # Number of unique levels

    This class dynamically creates ColumnRef attributes for each column in the
    DataFrame, enabling IDE auto-completion, preventing typos, and providing
    level discoverability for sampling plans. Columns are sorted alphabetically
    for consistent tab-completion ordering.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize accessor with DataFrame columns.

        Args:
            df: The DataFrame whose columns will be accessible
        """
        self._df = df
        # Sort by string representation to handle mixed-type column names
        self._columns = sorted(df.columns, key=str)
        self._attr_to_col = {}  # Track sanitized_name → original_column

        # Dynamically add each column as a ColumnRef attribute
        for col in self._columns:
            # Convert column name to valid Python identifier if needed
            attr_name = self._sanitize_column_name(col)

            if attr_name in self._attr_to_col:
                # Collision detected - warn and skip
                existing_col = self._attr_to_col[attr_name]
                logger.warning(
                    f"Column name collision: '{col}' and '{existing_col}' "
                    f"both sanitize to '{attr_name}'. "
                    f"'{col}' will only be accessible via pb.cols['{col}']."
                )
            else:
                # Avoid overwriting internal attributes (e.g., _df, _columns)
                if hasattr(self, attr_name):
                    logger.warning(
                        f"Column '{col}' sanitizes to '{attr_name}' which conflicts "
                        f"with an internal attribute. Use pb.cols['{col}'] to access."
                    )
                    continue
                self._attr_to_col[attr_name] = col
                setattr(self, attr_name, ColumnRef(col, df))

    def _sanitize_column_name(self, col_name: str) -> str:
        """
        Convert column name to valid Python identifier.

        Handles spaces, special characters, etc.

        Args:
            col_name: Original column name

        Returns:
            Sanitized name safe for use as Python attribute
        """
        # Normalize non-string column names (e.g., integers)
        col_name = str(col_name)

        # Replace spaces and special chars with underscores
        safe_name = col_name.replace(' ', '_').replace('-', '_')

        # Remove other special characters
        safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in safe_name)

        # Handle empty names
        if not safe_name:
            return '_empty'

        # Ensure doesn't start with number
        if safe_name[0].isdigit():
            safe_name = f'col_{safe_name}'

        return safe_name

    def __repr__(self) -> str:
        """Display available columns."""
        return f'ColumnAccessor({self._columns})'

    def __getitem__(self, col_name: str) -> ColumnRef:
        """
        Access column by original name (dict-style).

        Useful for columns with names that can't be valid Python identifiers
        or when collisions occur during sanitization.

        Args:
            col_name: Original column name

        Returns:
            ColumnRef for the column (for use in formulate())

        Raises:
            ColumnNotFoundError: If column doesn't exist
        """
        resolved_name = col_name

        # Support string lookup for non-string column labels (e.g., 123 -> "123")
        if resolved_name not in self._df.columns and isinstance(resolved_name, str):
            matching = [c for c in self._df.columns if str(c) == resolved_name]
            if len(matching) == 1:
                resolved_name = matching[0]

        if resolved_name not in self._df.columns:
            available = list(self._df.columns)
            raise ColumnNotFoundError(
                f"Column '{col_name}' not found. Available: {available}", column=col_name, available=available
            )
        return ColumnRef(resolved_name, self._df)

    def __dir__(self):
        """Support for tab-completion in IPython/Jupyter."""
        return list(self._attr_to_col.keys())


# ---------------------------------------------------------------------------
# formulate() helpers (module-level so they're easy to test in isolation)
# ---------------------------------------------------------------------------


def _validate_factors_or_plan_args(
    factors: list | None, plan: dict | None
) -> None:
    """Enforce the factors XOR plan contract for formulate().

    Both None is rejected (Bishop's design states require a factor × time
    grid). Both provided is rejected (ambiguous source of factor structure).
    """
    if factors is not None and plan is not None:
        raise ValidationError(
            "Cannot specify both 'factors' and 'plan'. Use either:\n"
            '  • factors=[...] to infer structure from observed data (complete designs)\n'
            '  • plan={col: [levels], ...} to specify expected structure (complete + incomplete designs)'
        )
    if factors is None and plan is None:
        raise ValidationError(
            'Cannot analyze response-only data without grouping structure.\n\n'
            "Bishop's design states (codes 1-6) require a factor × time grid.\n"
            'Please specify:\n'
            '  - factors: categorical variables defining subgroups (e.g., Machine, Operator)\n'
            '  - plan: expected factor levels for detecting incomplete designs\n\n'
            'See documentation for examples of proper study formulation.'
        )


def _compute_pds(
    detector: SDSRegistry,
    sampling_plan: dict[str, list] | None,
    T_planned: int | None,
    N_planned: int | None,
) -> SDSResult | None:
    """Compute the Plan Design State (PDS) when a sampling plan was supplied.

    PDS is `None` when no plan exists or any of T/N are missing — in that
    case the study has only an ODS (observed) and ADS (analytical) state.
    """
    if sampling_plan is None or T_planned is None or N_planned is None:
        return None
    from math import prod

    K = prod(len(v) for v in sampling_plan.values())
    pds = detector.classify_from_plan(K, T_planned, N_planned)
    logger.debug('PDS: SDS %s (%s)', pds.sds, pds.reason)
    return pds


class ProcessBehavior:
    """
    Main entry point for process behavior analysis with auto-completion.

    This class makes analysis frictionless by:
    1. Providing IDE auto-completion for column names
    2. Auto-detecting the design-state lineage (PDS / ODS / ADS)
    3. Showing valid chart types based on the analytical design state (ADS)
    4. Recommending the best chart for the detected ADS
    5. Two-step workflow: formulate() then execute()

    Usage:
        # Basic usage with auto-completion
        pb = ProcessBehavior(df)

        # Step 1: formulate() - analyze structure and get recommendations
        study = pb.formulate(
            response=pb.cols.Measurement,
            time=pb.cols.Time,
            factors=[pb.cols.Operator, pb.cols.Machine]
        )

        # Inspect the study
        print(study.observed_design_state.sds)  # Detected ODS
        print(study.valid_charts)  # What's available
        print(study.recommended_chart)  # Best choice

        # Step 2: execute() - run the chart
        result = study.execute()  # Uses recommended chart
        result = study.execute(chart='Xbar')  # Or explicit chart

    Attributes:
        cols: ColumnAccessor for IDE auto-completion of column names
        data: The underlying pandas DataFrame
    """

    def __init__(self, df: pd.DataFrame, na_values: list[str] | None = None):
        """
        Initialize ProcessBehavior with data and optional NA value handling.

        Args:
            df: pandas DataFrame containing process data
            na_values: Additional values to treat as NA/missing (beyond pandas defaults).
                      Common garbage characters are handled automatically.
                      Examples: ['*', '?', '--', 'ND', 'BDL', '<LOD']

        Examples:
            # Basic usage - automatic garbage character handling
            >>> pb = ProcessBehavior(df)

            # Custom NA indicators (combined with defaults)
            >>> pb = ProcessBehavior(df, na_values=['-999', '9999', 'MISSING'])
        """
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(f'Expected pandas DataFrame, got {type(df).__name__}')

        # Default garbage characters commonly found in real-world data
        # These are NOT recognized by pandas by default
        default_na = [
            '*',  # Common in lab data for missing/invalid
            '?',  # Question mark for unknown
            '--',  # Double dash for missing
            'ND',  # Not Detected
            'BDL',  # Below Detection Limit
            'BQL',  # Below Quantification Limit
            '<LOD',  # Below Limit of Detection
            '>ULQ',  # Above Upper Limit of Quantification
            'N/D',  # Not Detected (variant)
            'n/d',  # Not detected (lowercase)
            'MISSING',
            'missing',
        ]

        # Combine default with user-specified NA values
        all_na_values = default_na + (na_values or [])

        # Clean the data - replace garbage characters with pd.NA
        cleaned_df = df.copy()

        # Track which columns had NA values for informative warning
        columns_with_na = []
        na_counts = {}

        for col in cleaned_df.columns:
            # Count how many garbage values we find
            na_mask = cleaned_df[col].isin(all_na_values)
            na_count = na_mask.sum()

            if na_count > 0:
                columns_with_na.append(col)
                na_counts[col] = na_count
                # Replace with pd.NA
                cleaned_df.loc[na_mask, col] = pd.NA

                # Try to convert to numeric if it was originally numeric
                # This handles cases like ['235.5', '*', '237.2'] -> [235.5, NaN, 237.2]
                try:
                    # Try conversion - if it fails, keep original dtype
                    numeric_col = pd.to_numeric(cleaned_df[col])
                    cleaned_df[col] = numeric_col
                except (ValueError, TypeError):
                    # Keep as-is if conversion fails (likely string data)
                    pass

        # Warn user if we found and cleaned garbage characters
        if columns_with_na:
            total_na = sum(na_counts.values())
            logger.warning(
                f'Found {total_na} garbage/NA values across {len(columns_with_na)} column(s):\n'
                + '\n'.join([f'  • {col}: {count} values' for col, count in na_counts.items()])
                + '\n\nThese values were converted to NA and will be excluded from analysis.'
            )

        # Phase 2: Clean numeric formatting (currency symbols, thousands
        # separators, accounting negatives, percentages) in object columns
        formatting_cleaned = {}
        for col in cleaned_df.columns:
            result = _try_clean_numeric_strings(cleaned_df[col])
            if result is not None:
                formatting_cleaned[col] = int(result.notna().sum())
                cleaned_df[col] = result

        if formatting_cleaned:
            logger.warning(
                f'Cleaned numeric formatting in {len(formatting_cleaned)} column(s):\n'
                + '\n'.join([f'  • {col}: {count} values converted' for col, count in formatting_cleaned.items()])
                + '\n\nCurrency symbols, thousands separators, and '
                'accounting negatives were removed.'
            )

        self.data = cleaned_df
        self.cols = ColumnAccessor(self.data)

        logger.info(f'ProcessBehavior: {len(df)} rows, {len(df.columns)} columns')

    # =========================================================================
    # Factory Methods: Read from files
    # =========================================================================

    @classmethod
    def read_csv(cls, path: str, na_values: list[str] | None = None, **kwargs) -> ProcessBehavior:
        """
        Read data from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file.
        na_values : list of str, optional
            Additional values to treat as NA/missing.
        **kwargs
            Additional arguments passed to pandas.read_csv().

        Returns
        -------
        ProcessBehavior
            ProcessBehavior instance with loaded data.

        Examples
        --------
        >>> pb = ProcessBehavior.read_csv('fillweight_data.csv')
        >>> pb = ProcessBehavior.read_csv('data.csv', encoding='latin-1')
        """
        df = pd.read_csv(path, **kwargs)
        return cls(df, na_values=na_values)

    @classmethod
    def read_excel(
        cls, path: str, sheet_name: str | int = 0, na_values: list[str] | None = None, **kwargs
    ) -> ProcessBehavior:
        """
        Read data from an Excel file.

        Parameters
        ----------
        path : str
            Path to the Excel file (.xlsx, .xls).
        sheet_name : str or int, default 0
            Sheet name or index to read.
        na_values : list of str, optional
            Additional values to treat as NA/missing.
        **kwargs
            Additional arguments passed to pandas.read_excel().

        Returns
        -------
        ProcessBehavior
            ProcessBehavior instance with loaded data.

        Examples
        --------
        >>> pb = ProcessBehavior.read_excel('data.xlsx')
        >>> pb = ProcessBehavior.read_excel('data.xlsx', sheet_name='Sheet2')
        """
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        return cls(df, na_values=na_values)

    @classmethod
    def read_parquet(cls, path: str, na_values: list[str] | None = None, **kwargs) -> ProcessBehavior:
        """
        Read data from a Parquet file.

        Parameters
        ----------
        path : str
            Path to the Parquet file.
        na_values : list of str, optional
            Additional values to treat as NA/missing.
        **kwargs
            Additional arguments passed to pandas.read_parquet().

        Returns
        -------
        ProcessBehavior
            ProcessBehavior instance with loaded data.

        Examples
        --------
        >>> pb = ProcessBehavior.read_parquet('data.parquet')
        """
        try:
            df = pd.read_parquet(path, **kwargs)
        except ImportError:
            raise ImportError(
                'Reading Parquet files requires pyarrow or fastparquet. Install with: pip install pyarrow'
            ) from None
        return cls(df, na_values=na_values)

    @classmethod
    def read_clipboard(cls, na_values: list[str] | None = None, **kwargs) -> ProcessBehavior:
        """
        Read data from the system clipboard.

        Useful for quickly pasting data from Excel or Google Sheets.

        Parameters
        ----------
        na_values : list of str, optional
            Additional values to treat as NA/missing.
        **kwargs
            Additional arguments passed to pandas.read_clipboard().

        Returns
        -------
        ProcessBehavior
            ProcessBehavior instance with clipboard data.

        Examples
        --------
        Copy data from Excel, then:

        >>> pb = ProcessBehavior.read_clipboard()
        """
        df = pd.read_clipboard(**kwargs)
        return cls(df, na_values=na_values)

    @staticmethod
    def _to_column_name(col: str | ColumnRef) -> str:
        """Extract column name from str or ColumnRef."""
        return col.name if isinstance(col, ColumnRef) else col

    def _validate_plan(self, plan: dict) -> tuple[dict[str, list], list[str], int, int]:
        """
        Validate and normalize sampling plan.

        Parameters
        ----------
        plan : dict
            Sampling plan with required 'factors' key and optional 'T', 'N'.
            Example: {'factors': {'Lane': [1,2,3,4], 'Phase': [1,2,3]}, 'T': 10, 'N': 2}

        Returns
        -------
        tuple[dict[str, list], list[str], int | None, int | None]
            (normalized_factors, factor_order, T_planned, N_planned)

        Raises
        ------
        ValidationError
            If 'factors' key is missing
        ColumnNotFoundError
            If a plan column doesn't exist in the data
        """
        # Require 'factors' key
        if 'factors' not in plan:
            raise ValidationError(
                "Sampling plan must have 'factors' key.\n"
                "Example: plan={'factors': {'Lane': [1,2,3,4], 'Phase': [1,2,3]}, 'T': 10, 'N': 2}"
            )

        plan_factors = plan['factors']

        # Require at least one factor
        if not plan_factors:
            raise ValidationError(
                "Sampling plan 'factors' must contain at least one factor.\n"
                "Example: plan={'factors': {'Lane': [1,2,3,4]}, 'T': 10}"
            )
        T_planned = plan.get('T')
        N_planned = plan.get('N')

        if T_planned is None:
            raise ValidationError(
                "Sampling plan must include 'T' (planned number of time periods).\n"
                "Example: plan={'factors': {'Lane': [1,2,3,4], 'Phase': [1,2,3]}, 'T': 10, 'N': 2}"
            )
        if N_planned is None:
            raise ValidationError(
                "Sampling plan must include 'N' (planned observations per cell).\n"
                "Example: plan={'factors': {'Lane': [1,2,3,4], 'Phase': [1,2,3]}, 'T': 10, 'N': 2}"
            )

        normalized: dict[str, list] = {}
        factor_order: list[str] = []

        for col, levels in plan_factors.items():
            col_name = self._to_column_name(col)

            # Validate column exists
            if col_name not in self.data.columns:
                available = list(self.data.columns)
                raise ColumnNotFoundError(
                    f"Plan column '{col_name}' not found in data. Available: {available}",
                    column=col_name,
                    available=available,
                )

            # Validate levels is non-empty
            if not levels:
                raise ValidationError(
                    f"Factor '{col_name}' has empty level list in plan.\n"
                    f'Each factor must have at least one planned level.'
                )

            normalized[col_name] = list(levels)
            factor_order.append(col_name)

            # Check for extra observed levels (warn, don't error)
            observed = set(self.data[col_name].dropna().unique())
            planned = set(levels)
            extra = observed - planned

            if extra:
                extra_list = sorted(extra, key=lambda x: (type(x).__name__, x))
                observed_sorted = sorted(observed, key=lambda x: (type(x).__name__, x))
                logger.warning(
                    f"Factor '{col_name}' has observed levels not in plan: {extra_list}\n"
                    f'  Your plan: {levels}\n'
                    f'  Observed:  {observed_sorted}\n'
                    f'\n'
                    f'  To update your plan:\n'
                    f"    plan['factors']['{col_name}'] = pb.cols['{col_name}'].levels  # Use observed\n"
                    f'    # or\n'
                    f"    plan['factors']['{col_name}'] = {observed_sorted}  # Add manually"
                )

        return normalized, factor_order, T_planned, N_planned

    def formulate(
        self,
        response: str | ColumnRef,
        factors: list[str | ColumnRef] | None = None,
        time: str | ColumnRef | None = None,
        plan: dict | None = None,
        precision: int = 3,
        unit_of_analysis: str | None = None,
    ) -> Study:
        """
        Formulate a study for process behavior analysis.

        This method prepares and enriches the dataset (including residuals and
        effects as applicable) and returns a Study describing what analyses are
        valid. Call study.execute() to perform chart-specific calculations and
        produce an AnalysisResult.

        Parameters
        ----------
        response : str or ColumnRef
            The response variable (measurement) to analyze.
            Use pb.cols for IDE auto-completion.
        factors : list of str or ColumnRef, optional
            Grouping factors defining rational subgroups (e.g., ['Lane', 'Operator']).
            If provided, enables Xbar/S analysis. Cannot be used with `plan`.
        time : str or ColumnRef, optional
            Time/sequence variable for ordering observations.
        plan : dict, optional
            Sampling plan specifying expected factor levels. Must contain a
            ``'factors'`` key mapping column names (or ColumnRefs) to lists of
            expected levels. Optionally include ``'T'`` (planned time points)
            and ``'N'`` (planned observations per cell). Enables ODS 4-6
            detection by comparing observed structure to planned structure.
            Cannot be used with ``factors``.

            Example::

                plan={
                    'factors': {pb.cols.Lane: [1,2,3,4], pb.cols.Phase: [1,2,3]},
                    'T': 10,
                    'N': 2
                }
        precision : int, default 3
            Decimal places for output values.
        unit_of_analysis : str, optional
            The fundamental entity being measured. For example, in a manufacturing
            process producing cups filled with yogurt, the unit of analysis is
            'filled cup'. In a loan collection process, it would be 'loan contract'.
            This is informational metadata and does not affect calculations.

        Returns
        -------
        Study
            A Study object with:
            - PDS / ODS / ADS design-state lineage
            - Valid and recommended chart types (routed by ADS)
            - Pre-calculated residuals and effects (via study.dataset)
            - study.execute() to run chart-specific analysis
            - study.design() to compare plan vs observed (when plan provided)

        Examples
        --------
        Basic formulation (infer factors from data):

        >>> pb = ProcessBehavior(df)
        >>> study = pb.formulate(response='weight')
        >>> print(study)  # Shows PDS / ODS / ADS, valid charts, next steps

        With factors (complete designs, ADS 1-3):

        >>> study = pb.formulate(
        ...     response='fill_weight',
        ...     factors=['lane', 'phase'],
        ...     time='pull'
        ... )

        With sampling plan (enables SDS 4-6):

        >>> study = pb.formulate(
        ...     response=pb.cols.Weight,
        ...     time=pb.cols.Pull,
        ...     plan={
        ...         'factors': {
        ...             pb.cols.Lane: [1, 2, 3, 4],
        ...             pb.cols.Phase: [1, 2, 3]  # Even if Phase 3 not in data
        ...         },
        ...         'T': 10
        ...     }
        ... )
        >>> study.design()  # Shows planned vs observed structure

        Notes
        -----
        **Implicit Time Ordering**

        When no ``time`` parameter is specified, the system treats observation
        order as implicit time. This design decision is intentional:

        1. Wheeler's XmR chart fundamentally assumes temporal ordering - moving
           ranges between consecutive observations only make sense in sequence.

        2. The ``obs_id`` column (assigned during data preparation) serves as
           the implicit time dimension, preserving the order in which data was
           provided.

        If your observations are NOT in temporal order, you MUST specify the
        ``time`` parameter to ensure correct analysis.

        **Factors Required**

        At least one grouping factor (via ``factors`` or ``plan``) is required.
        Bishop's SDS classification assumes a factor × time grid structure.

        **SDS Detection**

        SDS detection runs on raw data (before NA rows are dropped) to preserve
        attempted-but-invalid cells. This ensures cells with all-NA responses
        are counted as empty (Nₖₜ=0) rather than vanishing from structure analysis.

        See Also
        --------
        Study : The returned Study object
        ColumnRef : Column reference with .levels property for discoverability
        """
        from .analysis_dataset import AnalysisDataSet
        from .data_preparation import DataPreparation
        from .study import Study

        # 1. Validate the factors/plan combination + presence.
        _validate_factors_or_plan_args(factors, plan)

        # 2. Normalize inputs into (response_str, time_str, factors_str,
        #    sampling_plan, factor_order, T_planned, N_planned).
        response_str = self._to_column_name(response)
        time_str = self._to_column_name(time) if time is not None else None
        factors_str, sampling_plan, factor_order, T_planned, N_planned = (
            self._resolve_factors_and_plan(factors, plan)
        )

        # 3. Build the FormulationSpec and validate columns (fail fast).
        spec = FormulationSpec(
            response_var=response_str,
            rsg_vars=tuple(factors_str) if factors_str else None,
            time_var=time_str,
            round_to=precision,
            unit_of_analysis=unit_of_analysis,
        )
        DataPreparation().validate_columns(self.data, spec)

        # 4. SDS detection on raw data (NA response rows preserved so
        #    cells with all-NA responses still count as attempted —
        #    matches Bishop's Minitab approach).
        detector = SDSRegistry()
        sds_result = detector.detect_sds_from_structure(
            self.data,
            spec,
            response_col=response_str,
            plan=sampling_plan,
            T_planned=T_planned,
        )

        # 5. Plan Design State (PDS) — only when a plan was supplied.
        pds_result = _compute_pds(detector, sampling_plan, T_planned, N_planned)

        # 6. Calculate the full AnalysisDataSet (residuals R1-R5, RCR1-RCR5).
        #    ADS is chart-agnostic — chart-specific params live in ChartRequest.
        ads = AnalysisDataSet(self.data, spec, observed_sds=sds_result.sds)

        # 7. ADS drives analysis: build analysis_plan from ADS, not ODS.
        analysis_plan = SDSRegistry.get_analysis_plan(
            ads.analytical_design_state.sds,
            min_cell_size=ads.analytical_design_state.min_cell_size,
        )

        return Study(
            _pdf=self,
            _spec=spec,
            _plan=analysis_plan,
            _ads=ads,
            _sampling_plan=sampling_plan,
            _factor_order=factor_order,
            _T=T_planned,
            _N=N_planned,
            _sds_result=sds_result,
            _pds_result=pds_result,
        )

    def _resolve_factors_and_plan(
        self,
        factors: list[str | ColumnRef] | None,
        plan: dict | None,
    ) -> tuple[list[str] | None, dict[str, list] | None, list[str] | None, int | None, int | None]:
        """Normalize factors/plan inputs to the tuple formulate() consumes.

        Exactly one of `factors` or `plan` is non-None (enforced by the
        caller). Returns `(factors_str, sampling_plan, factor_order,
        T_planned, N_planned)`.

        - `plan` path: delegates to `_validate_plan` for normalization and
          extracts T_planned/N_planned.
        - `factors` path: simple ColumnRef → str normalization; the
          sampling-plan slots stay None.
        """
        if plan is not None:
            sampling_plan, factor_order, T_planned, N_planned = self._validate_plan(plan)
            return factor_order, sampling_plan, factor_order, T_planned, N_planned
        factors_str = [self._to_column_name(f) for f in factors] if factors else None
        return factors_str, None, None, None, None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f'ProcessBehavior({len(self.data)} rows × {len(self.data.columns)} columns)\n'
            f'Columns: {list(self.data.columns)}'
        )

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.data)
