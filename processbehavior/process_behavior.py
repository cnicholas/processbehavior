"""
ProcessBehavior - Main entry point for process behavior analysis.

This module provides a user-friendly interface with IDE auto-completion for column names
and SDS-driven automatic analysis selection.

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

from .analysis_specification import DataPrepConfig
from .exceptions import ColumnNotFoundError, ValidationError
from .sds_detector import SDSRegistry

if TYPE_CHECKING:
    from .study import Study

logger = logging.getLogger(__name__)


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
            return f"{self.name} ({len(lvls)}): {lvls}"
        return f"{self.name} ({len(lvls)}): [{lvls[0]}..{lvls[-1]}]"


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
        self._columns = sorted(df.columns)  # Sort alphabetically for consistent ordering
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
        # Replace spaces and special chars with underscores
        safe_name = col_name.replace(' ', '_').replace('-', '_')

        # Remove other special characters
        safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in safe_name)

        # Ensure doesn't start with number
        if safe_name[0].isdigit():
            safe_name = f'col_{safe_name}'

        return safe_name

    def __repr__(self) -> str:
        """Display available columns."""
        return f"ColumnAccessor({self._columns})"

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
        if col_name not in self._df.columns:
            available = list(self._df.columns)
            raise ColumnNotFoundError(
                f"Column '{col_name}' not found. "
                f"Available: {available}",
                column=col_name,
                available=available
            )
        return ColumnRef(col_name, self._df)

    def __dir__(self):
        """Support for tab-completion in IPython/Jupyter."""
        return list(self._attr_to_col.keys())


class ProcessBehavior:
    """
    Main entry point for process behavior analysis with auto-completion.

    This class makes analysis frictionless by:
    1. Providing IDE auto-completion for column names
    2. Auto-detecting Sampling Design State (SDS)
    3. Showing valid chart types for the detected SDS
    4. Recommending the best chart for the detected SDS
    5. Two-step workflow: formulate() then analyze()

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
        print(study.sds)  # Detected SDS
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
            raise ValidationError(f"Expected pandas DataFrame, got {type(df).__name__}")

        # Default garbage characters commonly found in real-world data
        # These are NOT recognized by pandas by default
        default_na = [
            '*',      # Common in lab data for missing/invalid
            '?',      # Question mark for unknown
            '--',     # Double dash for missing
            'ND',     # Not Detected
            'BDL',    # Below Detection Limit
            'BQL',    # Below Quantification Limit
            '<LOD',   # Below Limit of Detection
            '>ULQ',   # Above Upper Limit of Quantification
            'N/D',    # Not Detected (variant)
            'n/d',    # Not detected (lowercase)
            'MISSING',
            'missing'
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
                f"Found {total_na} garbage/NA values across {len(columns_with_na)} column(s):\n"
                + "\n".join([f"  • {col}: {count} values" for col, count in na_counts.items()])
                + "\n\nThese values were converted to NA and will be excluded from analysis."
            )

        self.data = cleaned_df
        self.cols = ColumnAccessor(self.data)

        logger.info(f"ProcessBehavior: {len(df)} rows, {len(df.columns)} columns")

    @staticmethod
    def _to_column_name(col: str | ColumnRef) -> str:
        """Extract column name from str or ColumnRef."""
        return col.name if isinstance(col, ColumnRef) else col

    def _validate_plan(
        self,
        plan: dict
    ) -> tuple[dict[str, list], list[str], int | None, int | None]:
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
        T_planned = plan.get('T')
        N_planned = plan.get('N')

        normalized: dict[str, list] = {}
        factor_order: list[str] = []

        for col, levels in plan_factors.items():
            col_name = self._to_column_name(col)

            # Validate column exists
            if col_name not in self.data.columns:
                available = list(self.data.columns)
                raise ColumnNotFoundError(
                    f"Plan column '{col_name}' not found in data. "
                    f"Available: {available}",
                    column=col_name,
                    available=available
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
                    f"  Your plan: {levels}\n"
                    f"  Observed:  {observed_sorted}\n"
                    f"\n"
                    f"  To update your plan:\n"
                    f"    plan['factors']['{col_name}'] = pb.cols['{col_name}'].levels  # Use observed\n"
                    f"    # or\n"
                    f"    plan['factors']['{col_name}'] = {observed_sorted}  # Add manually"
                )

        return normalized, factor_order, T_planned, N_planned

    def formulate(
        self,
        response: str | ColumnRef,
        factors: list[str | ColumnRef] | None = None,
        time: str | ColumnRef | None = None,
        plan: dict[str | ColumnRef, list] | None = None,
        precision: int = 3,
        unit_of_analysis: str | None = None
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
            Sampling plan specifying expected factor levels. Keys are column names
            or ColumnRefs, values are lists of expected levels. Enables SDS 4-6
            detection by comparing observed structure to planned structure.
            Cannot be used with `factors`.

            Example: plan={pb.cols.Lane: [1,2,3,4], pb.cols.Phase: [1,2,3]}
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
            - SDS detection results
            - Valid and recommended chart types
            - Pre-calculated residuals and effects (via study.dataset)
            - study.execute() to run chart-specific analysis
            - study.design() to compare plan vs observed (when plan provided)

        Examples
        --------
        Basic formulation (infer factors from data):

        >>> pb = ProcessBehavior(df)
        >>> study = pb.formulate(response='weight')
        >>> print(study)  # Shows SDS, valid charts, next steps

        With factors (SDS 1-3):

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
        ...         pb.cols.Lane: [1, 2, 3, 4],
        ...         pb.cols.Phase: [1, 2, 3]  # Even if Phase 3 not in data
        ...     }
        ... )
        >>> study.design()  # Shows planned vs observed structure

        Notes
        -----
        **Implicit Time Ordering**

        When no ``time`` parameter is specified, the system treats observation
        order as implicit time. This design decision is intentional:

        1. Wheeler's IMR chart fundamentally assumes temporal ordering - moving
           ranges between consecutive observations only make sense in sequence.

        2. The ``obs_id`` column (assigned during data preparation) serves as
           the implicit time dimension, preserving the order in which data was
           provided.

        3. This enables "response-only" analysis where users can analyze a
           simple series without explicitly defining time structure. Such data
           is classified as SDS 4 (Single Condition Over Time) with implicit
           single condition.

        If your observations are NOT in temporal order, you MUST specify the
        ``time`` parameter to ensure correct analysis.

        See Also
        --------
        Study : The returned Study object
        ColumnRef : Column reference with .levels property for discoverability
        """
        from .study import Study

        # Mutual exclusion: factors OR plan, not both
        if factors is not None and plan is not None:
            raise ValidationError(
                "Cannot specify both 'factors' and 'plan'. Use either:\n"
                "  • factors=[...] to infer structure from observed data (SDS 1-3)\n"
                "  • plan={col: [levels], ...} to specify expected structure (SDS 1-6)"
            )

        # Normalize column names from ColumnRef to str
        response_str = self._to_column_name(response)
        time_str = self._to_column_name(time) if time is not None else None

        # Process plan or factors
        sampling_plan: dict[str, list] | None = None
        factor_order: list[str] | None = None
        factors_str: list[str] | None = None
        T_planned: int | None = None
        N_planned: int | None = None

        if plan is not None:
            # Validate and normalize the plan
            sampling_plan, factor_order, T_planned, N_planned = self._validate_plan(plan)
            # Extract factors from plan keys
            factors_str = factor_order
        elif factors is not None:
            # Normalize factors list
            factors_str = [self._to_column_name(f) for f in factors]

        # Build spec dict with user-friendly parameter names mapped to internal names
        spec_dict = {
            'response_var': response_str,   # response → response_var
            'rsg_vars': factors_str,        # factors → rsg_vars
            'time_var': time_str,           # time → time_var
            'round_to': precision,          # precision → round_to
            'rsg_var_name': 'rsg',          # Auto-generated (hidden from user)
            'rsg_var_delim': '_',           # Auto-generated (hidden from user)
            'zero_center': False,           # Default
            'unit_of_analysis': unit_of_analysis
        }

        # Create config for data preparation (no analysis_type needed yet)
        config = DataPrepConfig(spec_dict)

        # Prepare data (adds 'rsg' column if needed)
        from .data_preparation import DataPreparation
        prep = DataPreparation()
        prep.validate_columns(self.data, config)
        prepared_df = prep.prepare_dataset(self.data, config)

        # Detect SDS on prepared data
        # Pass sampling_plan to enable SDS 4-6 detection
        detector = SDSRegistry()
        sds_result = detector.detect_sds(
            prepared_df, config, plan=sampling_plan
        )

        # Get SDS analysis plan with all metadata
        analysis_plan = SDSRegistry.get_analysis_plan(
            sds_result.sds, min_cell_size=sds_result.min_cell_size
        )

        # Calculate full dataset with residuals (R1-R5, RCR1-RCR5)
        # Use AnalysisDataSet with the recommended chart type to trigger calculation
        # Pass SDS to avoid redundant detection (SDS is the driver of the system)
        from .analysis_dataset import AnalysisDataSet
        from .analysis_specification import AnalysisSpecification

        full_spec_dict = {
            **spec_dict,
            'analysis_type': analysis_plan.recommended_chart
        }
        full_spec = AnalysisSpecification(full_spec_dict)
        ads = AnalysisDataSet(self.data, full_spec, sds=sds_result.sds)

        # Create and return Study object with pre-calculated AnalysisDataSet
        # This enables execute() to reuse the expensive calculation
        return Study(
            _pdf=self,
            _spec=config,
            _plan=analysis_plan,
            _ads=ads,
            _sampling_plan=sampling_plan,
            _factor_order=factor_order,
            _T=T_planned,
            _N=N_planned,
            _sds_result=sds_result
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProcessBehavior({len(self.data)} rows × {len(self.data.columns)} columns)\n"
            f"Columns: {list(self.data.columns)}"
        )

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.data)
