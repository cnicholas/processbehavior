"""
AnalysisResult - Unified container for all analysis outputs.

This module provides a comprehensive result object that makes all analysis data
easily accessible in one place:
- Chart data (Xbar, S, IMR, stratified charts)
- Residuals (R1-R5)
- Effects (main effects, interactions)
- Summary metadata (SDS, statistics, capabilities)

Usage:
    analysis = Analysis(df, spec)
    result = analysis.calculate()  # Returns AnalysisResult

    # Access charts
    xbar_data = result.get_chart('Xbar')
    stats = result.get_statistics('Xbar')

    # Access residuals
    residuals = result.residuals  # DataFrame with R1-R5

    # Access effects
    main_effects = result.effects
    interactions = result.interactions

    # Get summary
    print(result.summary)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .analysis_dataset import AnalysisDataSet

logger = logging.getLogger(__name__)


class AnalysisResult:
    """
    Comprehensive analysis result container.

    This class unifies all analysis outputs into a single, easily accessible object.
    It provides:
    - Chart data and statistics (Xbar, S, IMR, stratified)
    - VAS residuals (R1-R5)
    - Main effects and interactions
    - Sampling Design State (SDS) information
    - Summary metadata

    Attributes
    ----------
    charts : dict
        Dictionary of chart data in format:
        {'chart_name': {'data': DataFrame, 'statistics': dict}}
    dataset : pd.DataFrame
        Full analysis dataset with all calculations
    residuals : pd.DataFrame or None
        VAS residuals (R1-R5) if calculated
    effects : dict or None
        Main effects if calculated
    interactions : dict or None
        Interaction effects if calculated
    summary : dict
        Comprehensive metadata about the analysis
    sds : int
        Sampling Design State (0-6)
    sds_info : dict
        Detailed SDS characteristics

    Examples
    --------
    >>> result = analysis.calculate()
    >>> xbar = result.get_chart('Xbar')
    >>> print(result.summary)
    >>> if result.has_residuals:
    ...     residuals = result.residuals
    """

    def __init__(
        self,
        charts: dict[str, dict[str, Any]],
        analysis_dataset_obj: AnalysisDataSet
    ):
        """
        Initialize AnalysisResult from chart data and AnalysisDataSet.

        Parameters
        ----------
        charts : dict
            Chart data in nested dict format
        analysis_dataset_obj : AnalysisDataSet
            The underlying AnalysisDataSet with all calculations
        """
        # Store chart data (backward compatible)
        self.charts = charts

        # Store reference to full dataset
        self._ads = analysis_dataset_obj
        self.dataset = analysis_dataset_obj.analysis_dataset

        # Extract SDS information
        self.sds = analysis_dataset_obj.sampling_design_state
        self.sds_info = analysis_dataset_obj.sds_characteristics

        # Extract residuals if calculated
        self._residuals = None
        if analysis_dataset_obj.has_vas_residuals:
            residual_cols = ['R1', 'R2', 'R3', 'R4', 'R5']
            available_cols = [c for c in residual_cols if c in self.dataset.columns]
            if available_cols:
                self._residuals = self.dataset[available_cols].copy()

        # Extract effects and interactions
        self._effects = (
            analysis_dataset_obj.effects if analysis_dataset_obj.effects else None
        )
        self._interactions = (
            analysis_dataset_obj.interactions
            if analysis_dataset_obj.interactions
            else None
        )

        # Build comprehensive summary
        self._summary = self._build_summary()

    def _build_summary(self) -> dict:
        """
        Build comprehensive summary of analysis.

        Returns
        -------
        dict
            Summary with SDS info, capabilities, and statistics
        """
        # Count total signals across all charts
        n_signals = 0

        # Handle case where charts might not be a dict
        if not isinstance(self.charts, dict):
            logger.warning(f"Charts is not a dict, it's a {type(self.charts)}")
            chart_values = []
        else:
            chart_values = self.charts.values()

        for chart_info in chart_values:
            if 'data' in chart_info and 'beyond_limits' in chart_info['data'].columns:
                n_signals += (chart_info['data']['beyond_limits'] != 0).sum()

        summary = {
            # SDS information
            'sds': self.sds,
            'sds_description': self.sds_info.get('description', 'Unknown'),
            'sds_capabilities': self.sds_info.get('capabilities', []),
            'replication_type': self.sds_info.get('replication_type', 'unknown'),

            # Analysis configuration
            'analysis_type': self._ads.spec.analysis_type,
            'response_var': self._ads.spec.response_var,
            'grouping_vars': self._ads.spec.rsg_vars,
            'time_var': self._ads.spec.time_var,

            # Data dimensions
            'n_observations': len(self.dataset),
            'n_charts': len(self.charts),
            'chart_types': list(self.charts.keys()),

            # Capabilities
            'has_residuals': self.has_residuals,
            'has_effects': self.has_effects,
            'has_interactions': self.has_interactions,
            'variance_decomposition': self.sds_info.get('variance_decomposition', False),
            'interaction_analysis': self.sds_info.get('interaction_analysis', False),

            # Signals
            'n_signals_total': int(n_signals),

            # Stratification
            'is_stratified': self._is_stratified(),
        }

        return summary

    def _is_stratified(self) -> bool:
        """
        Check if this is a stratified analysis (separate charts per group).

        Returns
        -------
        bool
            True if multiple charts exist and none are named 'Xbar', 'Sbar', 'R', 'all'
        """
        standard_chart_names = {'Xbar', 'Sbar', 'R', 'all'}
        chart_names = set(self.charts.keys())

        # If we have charts that aren't standard names, it's stratified
        non_standard = chart_names - standard_chart_names

        return len(non_standard) > 0 and len(self.charts) > 1

    # =========================================================================
    # Properties for easy access
    # =========================================================================

    @property
    def residuals(self) -> pd.DataFrame | None:
        """
        Get VAS residuals (R1-R5) if calculated.

        Returns
        -------
        DataFrame or None
            DataFrame with columns [R1, R2, R3, R4, R5] if residuals were
            calculated, None otherwise
        """
        return self._residuals

    @property
    def effects(self) -> dict | None:
        """
        Get main effects if calculated.

        Returns
        -------
        dict or None
            Dictionary with main effects:
            - 'k_effects': Factor effects (Series)
            - 't_effects': Time effects (Series)
        """
        return self._effects

    @property
    def interactions(self) -> dict | None:
        """
        Get interaction effects if calculated.

        Returns
        -------
        dict or None
            Dictionary with interaction terms (varies by SDS)
        """
        return self._interactions

    @property
    def summary(self) -> dict:
        """
        Get comprehensive summary of analysis.

        Returns
        -------
        dict
            Summary with SDS info, capabilities, dimensions, and statistics
        """
        return self._summary

    @property
    def has_residuals(self) -> bool:
        """Check if VAS residuals were calculated."""
        return self._residuals is not None

    @property
    def has_effects(self) -> bool:
        """Check if main effects were calculated."""
        return self._effects is not None and len(self._effects) > 0

    @property
    def has_interactions(self) -> bool:
        """Check if interaction effects were calculated."""
        return self._interactions is not None and len(self._interactions) > 0

    @property
    def all_charts(self) -> list[str]:
        """Get list of all available chart names."""
        return list(self.charts.keys())

    # =========================================================================
    # Convenience methods for accessing data
    # =========================================================================

    def get_chart(self, name: str) -> pd.DataFrame:
        """
        Get chart data by name.

        Parameters
        ----------
        name : str
            Chart name (e.g., 'Xbar', 'Sbar', 'GroupA', 'all')

        Returns
        -------
        DataFrame
            Chart data

        Raises
        ------
        KeyError
            If chart name not found

        Examples
        --------
        >>> xbar = result.get_chart('Xbar')
        >>> alice = result.get_chart('Alice')  # For stratified IMR
        """
        if name not in self.charts:
            raise KeyError(
                f"Chart '{name}' not found. Available charts: {self.all_charts}"
            )
        return self.charts[name]['data']

    def get_statistics(self, name: str) -> dict:
        """
        Get chart statistics by name.

        Parameters
        ----------
        name : str
            Chart name (e.g., 'Xbar', 'Sbar', 'GroupA')

        Returns
        -------
        dict
            Statistics dictionary with keys like 'mean', 'lcl', 'ucl', 'n'

        Raises
        ------
        KeyError
            If chart name not found

        Examples
        --------
        >>> stats = result.get_statistics('Xbar')
        >>> print(f"Mean: {stats['Mean']}, UCL: {stats['ucl']}")
        """
        if name not in self.charts:
            raise KeyError(
                f"Chart '{name}' not found. Available charts: {self.all_charts}"
            )
        return self.charts[name]['statistics']

    def get_residual(self, residual_type: str) -> pd.Series | None:
        """
        Get specific residual (R1, R2, R3, R4, or R5).

        Parameters
        ----------
        residual_type : str
            Residual type ('R1', 'R2', 'R3', 'R4', or 'R5')

        Returns
        -------
        Series or None
            Residual values if calculated, None otherwise

        Examples
        --------
        >>> r1 = result.get_residual('R1')
        >>> r2 = result.get_residual('R2')
        """
        if not self.has_residuals:
            return None

        if residual_type not in self._residuals.columns:
            available = list(self._residuals.columns)
            logger.warning(
                f"Residual '{residual_type}' not found. "
                f"Available: {available}"
            )
            return None

        return self._residuals[residual_type]

    def get_stratified_charts(self) -> dict[str, dict[str, Any]]:
        """
        Get all stratified charts (if stratification was used).

        Returns
        -------
        dict
            Dictionary of stratified charts with full chart names as keys

        Examples
        --------
        >>> if result.summary['is_stratified']:
        ...     strat_charts = result.get_stratified_charts()
        ...     for name, chart in strat_charts.items():
        ...         print(f"{name}: {len(chart['data'])} observations")
        """
        if not self.summary['is_stratified']:
            return {}

        # Filter out standard chart names
        standard_chart_names = {'Xbar', 'Sbar', 'R', 'all'}
        stratified = {
            name: chart
            for name, chart in self.charts.items()
            if name not in standard_chart_names
        }

        return stratified

    def get_stratified_chart(self, stratum: str) -> pd.DataFrame:
        """
        Get chart data for a specific stratum.

        Convenience method that searches for chart names containing the stratum.

        Parameters
        ----------
        stratum : str
            Stratum identifier (e.g., 'Operator_A', 'A', etc.)

        Returns
        -------
        DataFrame
            Chart data for the stratum

        Raises
        ------
        KeyError
            If no chart found for stratum

        Examples
        --------
        >>> # If stratified by Operator with levels A, B, C
        >>> chart_a = result.get_stratified_chart('A')
        >>> chart_b = result.get_stratified_chart('Operator_B')
        """
        # Find charts containing the stratum identifier
        matching_charts = [
            name for name in self.charts
            if stratum in name
        ]

        if not matching_charts:
            raise KeyError(
                f"No chart found for stratum '{stratum}'. "
                f"Available charts: {self.all_charts}"
            )

        if len(matching_charts) > 1:
            logger.warning(
                f"Multiple charts match '{stratum}': {matching_charts}. "
                f"Returning first match: {matching_charts[0]}"
            )

        chart_name = matching_charts[0]
        return self.charts[chart_name]['data']

    def list_strata(self) -> list[str]:
        """
        List all strata in stratified analysis.

        Returns
        -------
        list
            List of stratum identifiers

        Examples
        --------
        >>> strata = result.list_strata()
        >>> print(f"Strata: {strata}")
        ['Operator_A', 'Operator_B', 'Operator_C']
        """
        if not self.summary['is_stratified']:
            return []

        # Extract stratum names from chart names
        # Format: "Imr_Operator_A" -> "Operator_A"
        strata = []
        for chart_name in self.get_stratified_charts():
            # Split by underscore and take last 2 parts (variable_level)
            parts = chart_name.split('_')
            if len(parts) >= 2:
                stratum = '_'.join(parts[1:])  # Skip chart type
                strata.append(stratum)

        return sorted(set(strata))

    def iter_charts(self):
        """
        Iterate over all charts.

        Yields
        ------
        tuple of (name, data, statistics)

        Examples
        --------
        >>> for name, data, stats in result.iter_charts():
        ...     print(f"{name}: {len(data)} points, mean={stats.get('mean')}")
        """
        for name, chart_info in self.charts.items():
            yield name, chart_info['data'], chart_info['statistics']

    def get_signals(self, chart_name: str | None = None) -> pd.DataFrame:
        """
        Get points beyond control limits.

        Parameters
        ----------
        chart_name : str, optional
            Chart name to check. If None, checks all charts.

        Returns
        -------
        DataFrame
            Rows where beyond_limits != 0

        Examples
        --------
        >>> signals = result.get_signals('Xbar')
        >>> all_signals = result.get_signals()  # All charts
        """
        if chart_name:
            data = self.get_chart(chart_name)
            if 'beyond_limits' in data.columns:
                return data[data['beyond_limits'] != 0]
            return pd.DataFrame()

        # Get signals from all charts
        all_signals = []
        for name, data, _ in self.iter_charts():
            if 'beyond_limits' in data.columns:
                signals = data[data['beyond_limits'] != 0].copy()
                signals['chart'] = name
                all_signals.append(signals)

        if all_signals:
            return pd.concat(all_signals, ignore_index=True)
        return pd.DataFrame()

    # =========================================================================
    # String representation
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        charts_str = ', '.join(self.all_charts)
        return (
            f"AnalysisResult(\n"
            f"  sds={self.sds} ({self.sds_info['description']}),\n"
            f"  charts=[{charts_str}],\n"
            f"  n_obs={len(self.dataset)},\n"
            f"  has_residuals={self.has_residuals},\n"
            f"  has_effects={self.has_effects},\n"
            f"  n_signals={self.summary['n_signals_total']}\n"
            f")"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [
            "="*70,
            "ANALYSIS RESULT SUMMARY",
            "="*70,
            f"\nSampling Design State: SDS {self.sds}",
            f"Description: {self.sds_info['description']}",
            f"\nAnalysis Type: {self.summary['analysis_type']}",
            f"Response Variable: {self.summary['response_var']}",
        ]

        if self.summary['grouping_vars']:
            lines.append(f"Grouping: {', '.join(self.summary['grouping_vars'])}")

        if self.summary['time_var']:
            lines.append(f"Time Variable: {self.summary['time_var']}")

        lines.extend([
            f"\nObservations: {self.summary['n_observations']}",
            f"Charts: {', '.join(self.all_charts)}",
        ])

        if self.summary['is_stratified']:
            lines.append(f"Stratified: Yes ({len(self.charts)} groups)")

        lines.append("\nCapabilities:")
        lines.append(f"  Residuals: {'✓' if self.has_residuals else '✗'}")
        lines.append(f"  Effects: {'✓' if self.has_effects else '✗'}")
        lines.append(f"  Interactions: {'✓' if self.has_interactions else '✗'}")

        if self.summary['n_signals_total'] > 0:
            lines.append(f"\n⚠️  Signals: {self.summary['n_signals_total']} points beyond limits")

        lines.append("="*70)

        return '\n'.join(lines)

    # =========================================================================
    # Dictionary-like access (for backward compatibility)
    # =========================================================================

    def __getitem__(self, key):
        """Allow dict-like access to charts for backward compatibility."""
        return self.charts[key]

    def __contains__(self, key):
        """Check if chart exists."""
        return key in self.charts

    def keys(self):
        """Get chart names (for backward compatibility)."""
        return self.charts.keys()

    def values(self):
        """Get chart info (for backward compatibility)."""
        return self.charts.values()

    def items(self):
        """Get chart items (for backward compatibility)."""
        return self.charts.items()

    def __len__(self):
        """Return number of charts (for backward compatibility)."""
        return len(self.charts)

    def __iter__(self):
        """Iterate over chart names (for backward compatibility)."""
        return iter(self.charts)

    def get(self, key, default=None):
        """Get chart by name with default (for backward compatibility)."""
        return self.charts.get(key, default)

    # =========================================================================
    # Excel Export
    # =========================================================================

    def to_excel(
        self,
        filepath: str,
        include_summary: bool = True,
        include_charts: bool = True,
        include_residuals: bool = True,
        include_effects: bool = True,
        include_interactions: bool = True,
        include_full_dataset: bool = False,
        format_cells: bool = True
    ) -> None:
        """
        Export analysis results to Excel with each component on a separate tab.

        Creates a multi-sheet Excel workbook with organized analysis results:
        - Summary: Analysis metadata, SDS info, signal counts
        - Charts: One tab per chart (Xbar, Sbar, stratified IMR, etc.)
        - Residuals: R1-R5 variance decomposition (if available)
        - Effects: Main effects (if calculated)
        - Interactions: Interaction terms (if calculated)
        - Full_Dataset: Complete analysis dataset (optional)

        Parameters
        ----------
        filepath : str
            Output Excel file path (e.g., 'analysis.xlsx')
        include_summary : bool, default=True
            Include summary tab with analysis metadata
        include_charts : bool, default=True
            Include tabs for each chart
        include_residuals : bool, default=True
            Include residuals tab if available
        include_effects : bool, default=True
            Include effects tab if available
        include_interactions : bool, default=True
            Include interactions tab if available
        include_full_dataset : bool, default=False
            Include complete analysis dataset (can be large)
        format_cells : bool, default=True
            Apply formatting (bold headers, auto-width, freeze panes)

        Returns
        -------
        None
            File is written to disk

        Examples
        --------
        Export with default settings (all available data except full dataset):

        >>> result = analysis.calculate()
        >>> result.to_excel('my_analysis.xlsx')

        Export with full dataset included:

        >>> result.to_excel('complete_report.xlsx', include_full_dataset=True)

        Export only charts and summary (minimal):

        >>> result.to_excel('charts_only.xlsx',
        ...                 include_residuals=False,
        ...                 include_effects=False,
        ...                 include_interactions=False)

        Notes
        -----
        - Tab names are limited to 31 characters (Excel limitation)
        - Chart tabs are prefixed with 'Chart_' for clarity
        - Stratified charts use format 'IMR_{group_name}'
        - Summary tab includes SDS info and signal counts
        - Formatting includes frozen headers and auto-sized columns
        """
        try:
            import openpyxl  # noqa: F401
            from openpyxl.styles import Alignment, Font  # noqa: F401
            from openpyxl.utils.dataframe import dataframe_to_rows  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Excel export requires openpyxl. Install it with: "
                "pip install openpyxl"
            ) from e

        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

            # 1. Summary Tab
            if include_summary:
                self._write_summary_tab(writer, format_cells)

            # 2. Chart Tabs
            if include_charts:
                self._write_chart_tabs(writer, format_cells)

            # 3. Residuals Tab
            if include_residuals and self.has_residuals:
                self._write_residuals_tab(writer, format_cells)

            # 4. Effects Tab
            if include_effects and self.has_effects:
                self._write_effects_tab(writer, format_cells)

            # 5. Interactions Tab
            if include_interactions and self.has_interactions:
                self._write_interactions_tab(writer, format_cells)

            # 6. Full Dataset Tab (optional - can be large)
            if include_full_dataset:
                self._write_full_dataset_tab(writer, format_cells)

        logger.info(f"Analysis results exported to: {filepath}")

    def _write_summary_tab(self, writer: pd.ExcelWriter, format_cells: bool) -> None:
        """Write summary tab with analysis metadata."""
        # Convert summary dict to DataFrame for better Excel formatting
        summary_items = []

        # SDS Information
        summary_items.append(('Category', 'Sampling Design State'))
        summary_items.append(('SDS', self.summary['sds']))
        summary_items.append(('Description', self.summary['sds_description']))
        summary_items.append(('Replication Type', self.summary['replication_type']))
        summary_items.append(('', ''))

        # Analysis Configuration
        summary_items.append(('Category', 'Analysis Configuration'))
        summary_items.append(('Analysis Type', self.summary['analysis_type']))
        summary_items.append(('Response Variable', self.summary['response_var']))
        summary_items.append(('Grouping Variables', str(self.summary['grouping_vars'])))
        summary_items.append(('Time Variable', self.summary['time_var']))
        summary_items.append(('', ''))

        # Data Dimensions
        summary_items.append(('Category', 'Data Dimensions'))
        summary_items.append(('Total Observations', self.summary['n_observations']))
        summary_items.append(('Number of Charts', self.summary['n_charts']))
        summary_items.append(('Chart Types', ', '.join(self.summary['chart_types'])))
        summary_items.append(('Is Stratified', self.summary['is_stratified']))
        summary_items.append(('', ''))

        # Capabilities
        summary_items.append(('Category', 'Analysis Capabilities'))
        summary_items.append(('Has Residuals', self.summary['has_residuals']))
        summary_items.append(('Has Effects', self.summary['has_effects']))
        summary_items.append(('Has Interactions', self.summary['has_interactions']))
        summary_items.append(('Variance Decomposition', self.summary['variance_decomposition']))
        summary_items.append(('Interaction Analysis', self.summary['interaction_analysis']))
        summary_items.append(('', ''))

        # Signals
        summary_items.append(('Category', 'Process Signals'))
        summary_items.append(('Total Signals Detected', self.summary['n_signals_total']))

        # Add SDS capabilities as a list
        if self.summary['sds_capabilities']:
            summary_items.append(('', ''))
            summary_items.append(('Category', 'SDS Capabilities'))
            for capability in self.summary['sds_capabilities']:
                summary_items.append(('', capability))

        # Create DataFrame
        summary_df = pd.DataFrame(summary_items, columns=['Attribute', 'Value'])

        # Write to Excel
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Apply formatting if requested
        if format_cells:
            ws = writer.sheets['Summary']
            self._apply_formatting(ws)

    def _write_chart_tabs(self, writer: pd.ExcelWriter, format_cells: bool) -> None:
        """Write tabs for each chart."""

        # Check if this is a stratified analysis
        stratify_vars = self._ads.spec.spec.get('stratify')
        if self.summary['is_stratified'] and stratify_vars:
            # Stratified analysis: combine all stratified charts into single tab
            self._write_stratified_chart_tab(writer, format_cells, stratify_vars)

            # For stratified IMR/I charts, also create a summary tab for quick comparison
            chart_type = self._ads.spec.analysis_type
            if chart_type in ['Imr', 'I']:
                self._write_stratified_summary_tab(writer, format_cells, stratify_vars)
        else:
            # Standard analysis: each chart gets its own tab
            for chart_name, chart_info in self.charts.items():
                # Get chart data
                chart_data = chart_info.get('data')
                if chart_data is None or not isinstance(chart_data, pd.DataFrame):
                    continue

                # Create tab name (Excel limit: 31 chars)
                tab_name = f"Chart_{chart_name}"
                if len(tab_name) > 31:
                    tab_name = tab_name[:31]

                # Write chart data
                chart_data.to_excel(writer, sheet_name=tab_name, index=False)

                # Apply formatting if requested
                if format_cells:
                    ws = writer.sheets[tab_name]
                    self._apply_formatting(ws)

    def _write_stratified_chart_tab(
        self,
        writer: pd.ExcelWriter,
        format_cells: bool,
        stratify_vars: list
    ) -> None:
        """
        Write all stratified charts to a single combined tab.

        Combines all strata into one worksheet with a column identifying the stratum,
        making it easy to compare and filter in Excel.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer object
        format_cells : bool
            Whether to apply cell formatting
        stratify_vars : list
            Variables used for stratification
        """
        # Determine stratification column name
        strat_col = stratify_vars[0] if len(stratify_vars) == 1 else '_'.join(stratify_vars)

        # Collect all stratified charts
        combined_data = []

        standard_chart_names = {'Xbar', 'Sbar', 'R', 'all'}

        for chart_name, chart_info in self.charts.items():
            # Skip standard charts (will be written separately)
            if chart_name in standard_chart_names:
                continue

            chart_data = chart_info.get('data')
            if chart_data is None or not isinstance(chart_data, pd.DataFrame):
                continue

            # Extract stratum value from chart name
            # Pattern: "{StratumValue}_{StratCol}_{StratumValue}"
            # e.g., "Day_Shift_Day" -> stratum = "Day"
            # The first part before first underscore is the stratum value
            parts = chart_name.split('_')
            stratum_value = parts[0] if len(parts) > 0 else chart_name

            # Add stratum column to data
            chart_data_copy = chart_data.copy()
            chart_data_copy.insert(0, strat_col, stratum_value)

            combined_data.append(chart_data_copy)

        # Combine all stratified charts
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)

            # Create descriptive tab name
            chart_type = self._ads.spec.analysis_type
            if len(stratify_vars) == 1:
                tab_name = f"Chart_{chart_type}_by_{strat_col}"
            else:
                tab_name = f"Chart_{chart_type}_Stratified"

            # Excel tab name limit: 31 chars
            if len(tab_name) > 31:
                tab_name = tab_name[:31]

            # Write combined data
            combined_df.to_excel(writer, sheet_name=tab_name, index=False)

            # Apply formatting if requested
            if format_cells:
                ws = writer.sheets[tab_name]
                self._apply_formatting(ws)

        # Also write any standard charts (Xbar, Sbar, etc.)
        # These are separate analyses, not stratified
        standard_chart_names = {'Xbar', 'Sbar', 'R', 'all'}
        for chart_name in standard_chart_names:
            if chart_name in self.charts:
                chart_info = self.charts[chart_name]
                chart_data = chart_info.get('data')
                if chart_data is not None and isinstance(chart_data, pd.DataFrame):
                    tab_name = f"Chart_{chart_name}"
                    chart_data.to_excel(writer, sheet_name=tab_name, index=False)

                    if format_cells:
                        ws = writer.sheets[tab_name]
                        self._apply_formatting(ws)

    def _write_stratified_summary_tab(
        self,
        writer: pd.ExcelWriter,
        format_cells: bool,
        stratify_vars: list
    ) -> None:
        """
        Create a summary tab for stratified IMR charts.

        Provides a high-level comparison across all strata showing:
        - Stratum identifier (RSG)
        - Number of observations per stratum
        - Mean, LCL, UCL for each stratum
        - Total signals (beyond_limits != 0)
        - Signal rate (% of observations with signals)

        This enables quick triage: identify which strata have the most signals,
        then drill into the detailed chart tab for time-series investigation.

        Parameters
        ----------
        writer : pd.ExcelWriter
            Excel writer object
        format_cells : bool
            Whether to apply cell formatting
        stratify_vars : list
            Variables used for stratification
        """
        # Determine stratification column name
        stratify_vars[0] if len(stratify_vars) == 1 else '_'.join(stratify_vars)

        # Collect data from all stratified charts
        all_chart_data = []

        for chart_name, chart_info in self.charts.items():
            # Skip standard charts
            if chart_name in {'Xbar', 'Sbar', 'R', 'all'}:
                continue

            chart_data = chart_info.get('data')
            if chart_data is not None and isinstance(chart_data, pd.DataFrame):
                all_chart_data.append(chart_data)

        # Combine all chart data
        if not all_chart_data:
            return

        combined_data = pd.concat(all_chart_data, ignore_index=True)

        # Group by stratum (rsg column) and calculate summary statistics
        if 'rsg' not in combined_data.columns:
            return

        summary_rows = []

        for stratum, group in combined_data.groupby('rsg', sort=False):
            n_obs = len(group)

            # Get mean, lcl, ucl (should be constant within stratum)
            mean_val = group['mean'].iloc[0] if 'mean' in group.columns else None
            lcl_val = group['lcl'].iloc[0] if 'lcl' in group.columns else None
            ucl_val = group['ucl'].iloc[0] if 'ucl' in group.columns else None

            # Count signals
            if 'beyond_limits' in group.columns:
                n_signals = (group['beyond_limits'] != 0).sum()
                signal_rate = (n_signals / n_obs * 100) if n_obs > 0 else 0
            else:
                n_signals = None
                signal_rate = None

            summary_rows.append({
                'Stratum': stratum,
                'Observations': n_obs,
                'Mean': mean_val,
                'LCL': lcl_val,
                'UCL': ucl_val,
                'Signals': n_signals,
                'Signal_Rate_%': signal_rate
            })

        if summary_rows:
            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_rows)

            # Sort by signal count (descending) so worst strata appear first
            if 'Signals' in summary_df.columns:
                summary_df = summary_df.sort_values('Signals', ascending=False)

            # Write to Excel
            tab_name = 'Stratified_Summary'
            summary_df.to_excel(writer, sheet_name=tab_name, index=False)

            if format_cells:
                ws = writer.sheets[tab_name]
                self._apply_formatting(ws)

    def _write_residuals_tab(self, writer: pd.ExcelWriter, format_cells: bool) -> None:
        """Write residuals tab."""
        if self.residuals is not None:
            self.residuals.to_excel(writer, sheet_name='Residuals', index=False)

            if format_cells:
                ws = writer.sheets['Residuals']
                self._apply_formatting(ws)

    def _write_effects_tab(self, writer: pd.ExcelWriter, format_cells: bool) -> None:
        """Write effects tab."""
        # Convert effects dict to DataFrame
        effects_data = []

        for effect_name, effect_values in self.effects.items():
            if isinstance(effect_values, pd.DataFrame):
                # For DataFrames, extract all rows
                for _idx, row in effect_values.iterrows():
                    # Get the value column - look for specific patterns
                    # Exclude first column (grouping variable) and match effect/ME columns
                    value_col = [
                        c for c in effect_values.columns[1:]
                        if 'effect' in c.lower()
                        or c.upper().endswith('_ME')
                        or c.upper() == 'PT_ME'
                    ]
                    # Use first match or last column as value
                    val = row[value_col[0]] if value_col else row.iloc[-1]

                    # Get the level/category
                    level_col = effect_values.columns[0]
                    level = row[level_col]

                    effects_data.append({
                        'Effect_Type': effect_name,
                        'Level': str(level),
                        'Value': val
                    })
            elif isinstance(effect_values, pd.Series):
                for idx, val in effect_values.items():
                    effects_data.append({
                        'Effect_Type': effect_name,
                        'Level': str(idx),
                        'Value': val
                    })
            elif isinstance(effect_values, (int, float)):
                effects_data.append({
                    'Effect_Type': effect_name,
                    'Level': '',
                    'Value': effect_values
                })

        if effects_data:
            effects_df = pd.DataFrame(effects_data)
            effects_df.to_excel(writer, sheet_name='Effects', index=False)

            if format_cells:
                ws = writer.sheets['Effects']
                self._apply_formatting(ws)

    def _write_interactions_tab(self, writer: pd.ExcelWriter, format_cells: bool) -> None:
        """Write interactions tab."""
        # Convert interactions dict to DataFrame
        interactions_data = []

        for interaction_name, interaction_values in self.interactions.items():
            if isinstance(interaction_values, pd.Series):
                # Handle MultiIndex Series (cell-level interactions)
                if isinstance(interaction_values.index, pd.MultiIndex):
                    # Extract index level names and values
                    for idx_tuple, val in interaction_values.items():
                        # Build a dict with index level values plus the interaction value
                        row_dict = {'Interaction': interaction_name}

                        # Add each index level as a separate column
                        for level_name, level_val in zip(interaction_values.index.names, idx_tuple):
                            row_dict[level_name] = level_val

                        row_dict['Value'] = val
                        interactions_data.append(row_dict)
                else:
                    # Regular index - use as Combination
                    for idx, val in interaction_values.items():
                        interactions_data.append({
                            'Interaction': interaction_name,
                            'Combination': str(idx),
                            'Value': val
                        })
            elif isinstance(interaction_values, pd.DataFrame):
                # For DataFrames, flatten them
                for row_idx, row in interaction_values.iterrows():
                    for col_name, val in row.items():
                        interactions_data.append({
                            'Interaction': interaction_name,
                            'Combination': f"{row_idx} x {col_name}",
                            'Value': val
                        })

        if interactions_data:
            interactions_df = pd.DataFrame(interactions_data)
            interactions_df.to_excel(writer, sheet_name='Interactions', index=False)

            if format_cells:
                ws = writer.sheets['Interactions']
                self._apply_formatting(ws)

    def _write_full_dataset_tab(self, writer: pd.ExcelWriter, format_cells: bool) -> None:
        """Write full dataset tab."""
        self.dataset.to_excel(writer, sheet_name='Full_Dataset', index=False)

        if format_cells:
            ws = writer.sheets['Full_Dataset']
            self._apply_formatting(ws)

    def detect_signals(
        self,
        chart: str | None = None,
        rules: str | list[str] | None = None,
        config: Any | None = None,
        **kwargs
    ):
        """
        Detect Western Electric rule violations in control charts.

        This method applies configurable pattern detection rules to identify
        non-random patterns in control chart data.

        Parameters
        ----------
        chart : str, optional
            Specific chart to analyze. If None, analyzes all charts.
        rules : str, list, or RuleSet, optional
            Rules to apply:
            - 'standard': Rules 1-4 (default)
            - 'extended': Rules 1-8
            - 'all': All available rules
            - List of rule names: ['rule_1', 'rule_2', ...]
            - RuleSet: Fluent API configuration
        config : SignalConfig, optional
            Advanced configuration object
        **kwargs
            Additional parameters passed to SignalConfig

        Returns
        -------
        SignalResult or dict of SignalResult
            If chart specified: single SignalResult
            If no chart: dict mapping chart names to SignalResults

        Examples
        --------
        Simple usage (standard rules):

        >>> signals = result.detect_signals()
        >>> print(signals.summary)

        Specific chart and rules:

        >>> signals = result.detect_signals(
        ...     chart='Xbar',
        ...     rules=['rule_1', 'rule_2', 'rule_5']
        ... )

        Using fluent API:

        >>> from processbehavior.signals import RuleSet
        >>> signals = result.detect_signals(
        ...     rules=RuleSet()
        ...         .beyond_limits()
        ...         .zone_a(consecutive=2, window=3)
        ...         .trend(length=6)
        ... )

        Full configuration:

        >>> from processbehavior.signals import SignalConfig
        >>> config = SignalConfig(
        ...     enabled_rules=['rule_1', 'rule_2'],
        ...     min_observations=30,
        ...     ignore_first_n=5
        ... )
        >>> signals = result.detect_signals(config=config)

        Access violations:

        >>> signals.by_rule['rule_2']  # Rule 2 violations
        >>> signals.flagged_observations  # Set of obs_ids
        >>> signals.to_excel('violations.xlsx')
        """
        from .signals import RuleSet, SignalConfig, SignalDetector

        # Build configuration
        if config is None:
            config = SignalConfig()

            # Handle rules parameter
            if rules is not None:
                if isinstance(rules, RuleSet):
                    config.enabled_rules = rules.get_rules()
                elif isinstance(rules, str):
                    config.enabled_rules = rules
                elif isinstance(rules, list):
                    config.enabled_rules = rules

            # Apply kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Initialize detector
        detector = SignalDetector()

        # Detect on specific chart or all charts
        if chart:
            if chart not in self.charts:
                raise ValueError(
                    f"Chart '{chart}' not found.\n"
                    f"Available: {self.all_charts}"
                )

            chart_info = self.charts[chart]
            return detector.detect(
                data=chart_info['data'],
                stats=chart_info['statistics'],
                config=config,
                chart_name=chart
            )

        else:
            # Detect on all charts
            results = {}
            for chart_name, chart_info in self.charts.items():
                results[chart_name] = detector.detect(
                    data=chart_info['data'],
                    stats=chart_info['statistics'],
                    config=config,
                    chart_name=chart_name
                )

            return results

    def _apply_formatting(self, worksheet) -> None:
        """
        Apply standard formatting to worksheet.

        - Bold headers
        - Freeze top row
        - Auto-size columns
        """
        from openpyxl.styles import Font

        # Bold headers (first row)
        for cell in worksheet[1]:
            cell.font = Font(bold=True)

        # Freeze top row
        worksheet.freeze_panes = 'A2'

        # Auto-size columns
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if cell.value:
                        max_length = max(max_length, len(str(cell.value)))
                except Exception:  # noqa: S110
                    pass

            adjusted_width = min(max_length + 2, 50)  # Cap at 50
            worksheet.column_dimensions[column_letter].width = adjusted_width
