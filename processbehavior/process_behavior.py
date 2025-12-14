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
from typing import TYPE_CHECKING

import pandas as pd

from .analysis_specification import DataPrepConfig
from .sds_detector import SamplingDesignDetector

if TYPE_CHECKING:
    from .study import Study

logger = logging.getLogger(__name__)


class ColumnAccessor:
    """
    Provides IDE auto-completion for DataFrame column names.

    Usage:
        pb = ProcessBehavior(df)
        pb.cols.Height  # Auto-completes to column name string

    This class dynamically creates attributes for each column in the DataFrame,
    enabling IDE auto-completion and preventing typos. Columns are sorted
    alphabetically for consistent tab-completion ordering.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize accessor with DataFrame columns.

        Args:
            df: The DataFrame whose columns will be accessible
        """
        self._df = df
        self._columns = sorted(df.columns)  # Sort alphabetically for consistent ordering

        # Dynamically add each column as an attribute
        for col in self._columns:
            # Convert column name to valid Python identifier if needed
            attr_name = self._sanitize_column_name(col)
            setattr(self, attr_name, col)

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

    def __dir__(self):
        """Support for tab-completion in IPython/Jupyter."""
        return [self._sanitize_column_name(col) for col in self._columns]


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
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

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

    def formulate(
        self,
        response: str,
        factors: list[str] | None = None,
        time: str | None = None,
        precision: int = 3
    ) -> Study:
        """
        Formulate a study for process behavior analysis.

        This creates a Study object that describes your data structure and
        guides you toward correct analysis. Unlike analyze(), this method
        does not run calculations immediately - it helps you understand
        what's possible first.

        Parameters
        ----------
        response : str
            The response variable (measurement) to analyze.
            Use pb.cols for IDE auto-completion.
        factors : list of str, optional
            Grouping factors defining rational subgroups (e.g., ['Lane', 'Operator']).
            If provided, enables Xbar/S analysis.
        time : str, optional
            Time/sequence variable for ordering observations.
        precision : int, default 3
            Decimal places for output values.

        Returns
        -------
        Study
            A Study object with:
            - SDS detection results
            - Valid and recommended chart types
            - guidance methods like study.why_not()
            - study.execute() to run calculations

        Examples
        --------
        Basic formulation:

        >>> pb = ProcessBehavior(df)
        >>> study = pb.formulate(response='weight')
        >>> print(study)  # Shows SDS, valid charts, next steps

        With factors and time:

        >>> study = pb.formulate(
        ...     response='fill_weight',
        ...     factors=['lane', 'phase'],
        ...     time='pull'
        ... )
        >>> study.sds  # Check detected SDS
        >>> study.valid_charts  # See what's available
        >>> study.charts.Xbar  # IDE auto-complete

        Run the analysis:

        >>> result = study.execute()  # Uses recommended chart
        >>> result = study.execute(chart='Xbar')  # Explicit chart

        See Also
        --------
        Study : The returned Study object
        analyze : Alternative that runs immediately
        """
        from .study import Study

        # Build spec dict with user-friendly parameter names mapped to internal names
        spec_dict = {
            'response_var': response,       # response → response_var
            'rsg_vars': factors,            # factors → rsg_vars
            'time_var': time,               # time → time_var
            'round_to': precision,          # precision → round_to
            'rsg_var_name': 'rsg',          # Auto-generated (hidden from user)
            'rsg_var_delim': '_',           # Auto-generated (hidden from user)
            'zero_center': False            # Default
        }

        # Create config for data preparation (no analysis_type needed yet)
        config = DataPrepConfig(spec_dict)

        # Prepare data (adds 'rsg' column if needed)
        from .data_preparation import DataPreparation
        prep = DataPreparation()
        prep.validate_columns(self.data, config)
        prepared_df = prep.prepare_dataset(self.data, config)

        # Detect SDS on prepared data
        detector = SamplingDesignDetector()
        sds, min_cell_size = detector.detect_sds(prepared_df, config)

        # Get SDS analysis plan with all metadata
        plan = SamplingDesignDetector.get_analysis_plan(sds, min_cell_size=min_cell_size)

        # Calculate full dataset with residuals (R1-R5, RCR1-RCR5)
        # Use AnalysisDataSet with the recommended chart type to trigger calculation
        # Pass SDS to avoid redundant detection (SDS is the driver of the system)
        from .analysis_dataset import AnalysisDataSet
        from .analysis_specification import AnalysisSpecification

        full_spec_dict = {
            **spec_dict,
            'analysis_type': plan.recommended_chart
        }
        full_spec = AnalysisSpecification(full_spec_dict)
        ads = AnalysisDataSet(self.data, full_spec, sds=sds)

        # Create and return Study object with pre-calculated AnalysisDataSet
        # This enables execute() to reuse the expensive calculation
        return Study(
            _pdf=self,
            _spec=config,
            _plan=plan,
            _ads=ads
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
