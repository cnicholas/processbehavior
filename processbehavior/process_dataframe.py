"""
ProcessDataFrame - Intelligent wrapper for process behavior analysis.

This module provides a user-friendly interface with IDE auto-completion for column names
and SDS-driven automatic analysis selection.

Usage:
    from processbehavior import ProcessDataFrame

    data = ProcessDataFrame(raw_df)

    # Auto-completion for column names!
    analysis = data.analyze(
        response_vars=[data.columns.Height, data.columns.Width],
        time_var=data.columns.ProductionTime,
        grouping_vars=[data.columns.Operator]
    )
"""

from __future__ import annotations
import logging
from typing import List, Union, Optional
import pandas as pd

from .sds_detector import SamplingDesignDetector
from .analysis_specification import AnalysisSpecification
from .analysis_dataset import Analysis

logger = logging.getLogger(__name__)


class ColumnAccessor:
    """
    Provides IDE auto-completion for DataFrame column names.

    Usage:
        data = ProcessDataFrame(df)
        data.columns.Height  # Auto-completes to column name string

    This class dynamically creates attributes for each column in the DataFrame,
    enabling IDE auto-completion and preventing typos.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize accessor with DataFrame columns.

        Args:
            df: The DataFrame whose columns will be accessible
        """
        self._df = df
        self._columns = list(df.columns)

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


class ChartTypeAccessor:
    """
    Provides IDE auto-completion for valid chart types based on detected SDS.

    This class dynamically creates attributes for each valid chart type,
    enabling IDE auto-completion and preventing invalid chart selections.

    Usage:
        data = ProcessDataFrame(df)

        # After first analyze(), data.charts is populated
        result = data.analyze(response_var=data.columns.Height)

        # Now you can use auto-completion for valid charts
        result2 = data.analyze(
            response_var=data.columns.Height,
            chart_type=data.charts.Xbar  # IDE auto-completes valid options!
        )

    Attributes are set dynamically based on SDS-specific valid charts.
    """

    def __init__(self, valid_charts: List[str]):
        """
        Initialize accessor with valid chart types for the detected SDS.

        Args:
            valid_charts: List of valid chart type names for the current SDS
        """
        self._valid_charts = valid_charts

        # Dynamically add each valid chart as an attribute
        for chart in valid_charts:
            setattr(self, chart, chart)

    def __repr__(self) -> str:
        """Display available chart types."""
        return f"Available charts: {', '.join(self._valid_charts)}"

    def __dir__(self):
        """Support for tab-completion in IPython/Jupyter."""
        return self._valid_charts


class ProcessDataFrame:
    """
    Intelligent wrapper for process behavior analysis with auto-completion.

    This class makes analysis frictionless by:
    1. Providing IDE auto-completion for column names
    2. Auto-detecting Sampling Design State (SDS)
    3. Showing valid chart types for the detected SDS
    4. Running the best analysis for the detected SDS
    5. Explaining what analysis is being run and why

    Usage:
        # Basic usage with auto-completion
        data = ProcessDataFrame(raw_df)

        # Auto-detect best chart (frictionless)
        analysis = data.analyze(
            response_var=data.columns.Measurement,
            time_var=data.columns.Time,
            grouping_vars=[data.columns.Operator, data.columns.Machine]
        )

        # Explicit chart selection with auto-completion (power users)
        analysis = data.analyze(
            response_var=data.columns.Measurement,
            chart_type=data.charts.S  # Auto-completes only valid charts!
        )

    Attributes:
        columns: ColumnAccessor for IDE auto-completion of column names
        charts: ChartTypeAccessor for valid chart types (set after first analyze())
        data: The underlying pandas DataFrame
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize ProcessDataFrame with data.

        Args:
            df: pandas DataFrame containing process data
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

        self.data = df.copy()
        self.columns = ColumnAccessor(self.data)
        self.charts = None  # Will be populated after first analyze() call

        logger.info(f"ProcessDataFrame created with {len(df)} rows, {len(df.columns)} columns")

    def analyze(
        self,
        response_var: Optional[str] = None,
        response_vars: Optional[List[str]] = None,
        time_var: Optional[str] = None,
        grouping_vars: Optional[List[str]] = None,
        chart_type: Optional[str] = None,
        stratify: Union[bool, str, List[str]] = False,
        rsg_var_name: str = 'rsg',
        rsg_var_delim: str = '_',
        round_to: int = 3,
        zero_center: bool = False
    ) -> Analysis:
        """
        Auto-detect SDS and run the best analysis for your data.

        This method:
        1. Builds an AnalysisSpecification from your parameters
        2. Detects the Sampling Design State (SDS)
        3. Determines valid chart types for that SDS
        4. Runs the specified or recommended chart analysis
        5. Explains what analysis is being run and why

        Args:
            response_var: Single response variable (for simple series)
            response_vars: List of response variables (for multivariate)
            time_var: Time/sequence variable
            grouping_vars: Grouping variables for rational subgrouping
            chart_type: Explicit chart type ('Xbar', 'S', 'Imr', 'R'). If None,
                       uses recommended chart for detected SDS. After first analyze(),
                       use data.charts for auto-completion of valid options.
            stratify: Create separate charts for each level of grouping variable(s).
                     - False (default): Single combined chart
                     - True: Stratify by all grouping_vars (creates separate chart per stratum)
                     - 'VarName': Stratify by specific variable
                     - ['Var1', 'Var2']: Stratify by combination of variables
                     Most commonly used with chart_type='Imr' to create separate IMR
                     charts for each subgroup, enabling drill-down analysis.
            rsg_var_name: Name for rational subgroup column (default: 'rsg')
            rsg_var_delim: Delimiter for multi-variable groups (default: '_')
            round_to: Decimal places for rounding (default: 3)
            zero_center: Whether to center data at zero (default: False)

        Returns:
            Analysis object with results

        Raises:
            ValueError: If neither response_var nor response_vars is provided

        Examples:
            # Simple series - Auto-detects IMR chart
            analysis = data.analyze(response_var=data.columns.Measurement)

            # Grouped data - Auto-detects Xbar chart (recommended)
            analysis = data.analyze(
                response_var=data.columns.Height,
                time_var=data.columns.Time,
                grouping_vars=[data.columns.Operator]
            )

            # Explicit chart type selection (with auto-completion)
            analysis = data.analyze(
                response_var=data.columns.Height,
                grouping_vars=[data.columns.Operator],
                chart_type=data.charts.S  # IDE auto-completes valid options!
            )

            # Stratified analysis - separate IMR for each Operator
            analysis = data.analyze(
                response_var=data.columns.Height,
                grouping_vars=[data.columns.Operator],
                chart_type='Imr',
                stratify=True  # Creates separate IMR chart per operator
            )
            # Access: analysis.get_stratified_chart('Operator_A')
        """
        # Handle response variable specification
        if response_var is None and response_vars is None:
            raise ValueError("Must provide either response_var or response_vars")

        if response_var and response_vars:
            raise ValueError("Provide either response_var OR response_vars, not both")

        # Normalize to response_var for specification
        final_response_var = response_var if response_var else response_vars[0]

        # Build specification dict
        spec_dict = {
            'response_var': final_response_var,
            'time_var': time_var,
            'rsg_vars': grouping_vars,
            'rsg_var_name': rsg_var_name,
            'rsg_var_delim': rsg_var_delim,
            'round_to': round_to,
            'zero-center': zero_center  # Note: hyphen not underscore (legacy API)
        }

        # Create a temporary spec to detect SDS
        # We'll use 'Imr' as default since we don't know SDS yet
        temp_spec = AnalysisSpecification('Imr', spec_dict)

        # Prepare data first (adds 'rsg' column if needed)
        from .data_preparation import DataPreparation
        prep = DataPreparation()
        prep.validate_columns(self.data, temp_spec)
        prepared_df = prep.prepare_dataset(self.data, temp_spec)

        # Detect SDS on prepared data
        detector = SamplingDesignDetector()
        sds = detector.detect_sds(prepared_df, temp_spec)
        sds_info = detector.get_sds_characteristics(sds)

        # Get SDS analysis plan to determine valid charts
        plan = SamplingDesignDetector.get_analysis_plan(sds)

        # Set up charts accessor for IDE auto-completion
        self.charts = ChartTypeAccessor(plan.valid_charts)

        # Determine which chart to run
        if chart_type is None:
            # Auto-detect: use recommended chart from SDS plan
            analysis_type = plan.recommended_chart
            logger.info(
                f"Auto-selected '{analysis_type}' chart "
                f"(recommended for SDS {sds}: {plan.name})"
            )
        else:
            # User specified: validate it's valid for this SDS
            if chart_type not in plan.valid_charts:
                invalid_chart_reasons = {
                    invalid.split(' (')[0]: invalid.split('(')[1].rstrip(')')
                    for invalid in plan.invalid_charts
                    if '(' in invalid
                }
                reason = invalid_chart_reasons.get(chart_type, "not supported for this data structure")

                raise ValueError(
                    f"Chart type '{chart_type}' is not valid for SDS {sds} ({plan.name}).\n"
                    f"Reason: {reason}\n"
                    f"Valid options: {plan.valid_charts}\n"
                    f"Recommended: '{plan.recommended_chart}'\n"
                    f"Hint: Use data.charts for auto-completion of valid chart types."
                )
            analysis_type = chart_type
            logger.info(f"Using user-specified chart: '{analysis_type}'")

        # Handle stratification
        stratify_vars = self._process_stratify_parameter(stratify, grouping_vars, analysis_type)

        # Log what we're doing and why
        self._explain_analysis(sds, sds_info, analysis_type, spec_dict, plan, stratify_vars)

        # Update spec with correct analysis type and stratification info
        spec_dict['analysis_type'] = analysis_type
        spec_dict['stratify'] = stratify_vars
        final_spec = AnalysisSpecification(analysis_type, spec_dict)

        # Run the analysis
        analysis = Analysis(self.data, spec_dict)

        return analysis

    def _determine_analysis_type(
        self,
        sds: int,
        grouping_vars: Optional[List[str]]
    ) -> str:
        """
        Determine the best analysis type for the detected SDS.

        Decision logic:
        - SDS 0 (simple series): IMR chart
        - SDS 1-6 with grouping: Xbar and S charts
        - SDS with no grouping but has time: IMR chart

        Args:
            sds: Detected Sampling Design State
            grouping_vars: User-specified grouping variables

        Returns:
            Analysis type string ('Imr', 'Xbar', 'S', or 'R')
        """
        # SDS 0: Simple series → IMR chart
        if sds == 0:
            return 'Imr'

        # If user provided grouping variables → Xbar/S charts
        if grouping_vars and len(grouping_vars) > 0:
            return 'Xbar'  # Will also calculate S chart

        # Otherwise → IMR chart (individuals)
        return 'Imr'

    def _process_stratify_parameter(
        self,
        stratify: Union[bool, str, List[str]],
        grouping_vars: Optional[List[str]],
        analysis_type: str
    ) -> Optional[List[str]]:
        """
        Process and validate the stratify parameter.

        Args:
            stratify: User-specified stratification request
            grouping_vars: Grouping variables from analyze()
            analysis_type: Selected analysis type

        Returns:
            List of variables to stratify by, or None if no stratification

        Raises:
            ValueError: If stratify is invalid or incompatible with analysis
        """
        # No stratification requested
        if stratify is False or stratify is None:
            return None

        # Validate that grouping variables exist
        if not grouping_vars:
            raise ValueError(
                "Stratification requires grouping_vars to be specified.\n"
                "Cannot stratify data with no grouping structure."
            )

        # Determine which variables to stratify by
        if stratify is True:
            # Stratify by all grouping variables
            stratify_vars = grouping_vars.copy()
            logger.info(f"Stratifying by all grouping variables: {stratify_vars}")
        elif isinstance(stratify, str):
            # Stratify by single variable
            if stratify not in grouping_vars:
                raise ValueError(
                    f"Stratify variable '{stratify}' not in grouping_vars.\n"
                    f"Available: {grouping_vars}"
                )
            stratify_vars = [stratify]
            logger.info(f"Stratifying by: {stratify}")
        elif isinstance(stratify, list):
            # Stratify by list of variables
            invalid = [v for v in stratify if v not in grouping_vars]
            if invalid:
                raise ValueError(
                    f"Stratify variables {invalid} not in grouping_vars.\n"
                    f"Available: {grouping_vars}"
                )
            stratify_vars = stratify
            logger.info(f"Stratifying by: {stratify_vars}")
        else:
            raise TypeError(
                f"stratify must be bool, str, or list[str], got {type(stratify)}"
            )

        # Warn if stratifying with non-IMR charts (less common use case)
        if analysis_type not in ['Imr', 'R']:
            logger.warning(
                f"Stratifying {analysis_type} charts is uncommon. "
                f"Stratification is most useful with IMR charts for drill-down analysis."
            )

        return stratify_vars

    def _explain_analysis(
        self,
        sds: int,
        sds_info: dict,
        analysis_type: str,
        spec: dict,
        plan: 'SDSAnalysisPlan',
        stratify_vars: Optional[List[str]] = None
    ):
        """
        Print user-friendly explanation of what analysis is running and why.

        Args:
            sds: Detected Sampling Design State
            sds_info: SDS characteristics dict
            analysis_type: Selected analysis type
            spec: Analysis specification dict
            plan: SDS analysis plan with valid/invalid charts
            stratify_vars: Variables to stratify by, if any
        """
        print("\n" + "="*70)
        print("PROCESS BEHAVIOR ANALYSIS")
        print("="*70)

        print(f"\n📊 Detected SDS {sds}: {plan.name}")
        print(f"   {plan.description}")
        print(f"   Replication: {plan.has_replication.capitalize()}")

        # Show available charts
        print(f"\n📈 Available charts: {', '.join(plan.valid_charts)}")
        is_recommended = analysis_type == plan.recommended_chart
        selection_note = " (recommended)" if is_recommended else " (user-specified)"
        print(f"   Selected: {self._get_analysis_description(analysis_type)}{selection_note}")

        # Show stratification info
        if stratify_vars:
            num_strata = "multiple" if len(stratify_vars) > 1 else "per level"
            print(f"   Stratification: Enabled - creating separate charts {num_strata} of {', '.join(stratify_vars)}")

        # Show what's not available (if any)
        if plan.invalid_charts:
            print(f"\n⚠️  Not available for this SDS:")
            for invalid in plan.invalid_charts:
                print(f"   • {invalid}")

        # Show data configuration
        print(f"\n📋 Data Configuration:")
        print(f"   Response: {spec['response_var']}")
        if spec.get('time_var'):
            print(f"   Time: {spec['time_var']}")
        if spec.get('rsg_vars'):
            print(f"   Grouping: {', '.join(spec['rsg_vars'])}")

        # Show capabilities
        print(f"\n✨ Analysis Capabilities:")
        if plan.vas_residuals_supported:
            print(f"   • VAS residuals: {', '.join(plan.residuals_available)} (R2 method: {plan.residual_calculation_method})")
        else:
            print(f"   • VAS residuals: Not available")
        print(f"   • Main effects: {'Yes' if plan.main_effects_supported else 'No'}")
        print(f"   • Interactions: {'Yes' if plan.interaction_effects_supported else 'No'}")

        print("\n" + "="*70 + "\n")

    def _get_analysis_description(self, analysis_type: str) -> str:
        """Get friendly description of analysis type."""
        descriptions = {
            'Imr': 'IMR Chart (Individual Moving Range)',
            'Xbar': 'Xbar and S Charts (Subgroup Mean and Variation)',
            'S': 'S Chart (Subgroup Standard Deviation)',
            'R': 'R Chart (Subgroup Range)'
        }
        return descriptions.get(analysis_type, analysis_type)

    def _get_analysis_rationale(self, sds: int, analysis_type: str, spec: dict) -> str:
        """Explain why this analysis was chosen."""
        if sds == 0:
            return "Simple series with no grouping → IMR chart for individual measurements"

        if spec.get('rsg_vars'):
            return (
                f"Data has rational subgroups ({', '.join(spec['rsg_vars'])}) → "
                "Xbar/S charts to track subgroup means and variation"
            )

        return "No rational subgroups detected → IMR chart for individual measurements"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProcessDataFrame(rows={len(self.data)}, "
            f"columns={len(self.data.columns)})\n"
            f"Columns: {list(self.data.columns)}"
        )

    def __len__(self) -> int:
        """Return number of rows."""
        return len(self.data)
