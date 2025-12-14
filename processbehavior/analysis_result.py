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
    from .plotting.control_chart import ControlChartFigure
    from .signals.result import SignalResult

logger = logging.getLogger(__name__)

# Standard SPC chart type names
STANDARD_CHART_NAMES = {'Xbar', 'S', 'Imr', 'R'}


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
            True if multiple charts exist and none are named 'Xbar', 'S', 'R', 'all'
        """
        standard_chart_names = STANDARD_CHART_NAMES
        chart_names = set(self.charts.keys())

        # If we have charts that aren't standard names, it's stratified
        non_standard = chart_names - standard_chart_names

        return len(non_standard) > 0 and len(self.charts) > 1

    # =========================================================================
    # Properties for easy access
    # =========================================================================

    @property
    def residuals(self) -> pd.DataFrame:
        """
        Get VAS residuals (R1-R5) if calculated.

        Returns
        -------
        DataFrame
            DataFrame with columns [R1, R2, R3, R4, R5] if residuals were
            calculated, empty DataFrame with same schema otherwise
        """
        if self._residuals is None:
            return pd.DataFrame(columns=['R1', 'R2', 'R3', 'R4', 'R5'])
        return self._residuals.copy()

    @property
    def effects(self) -> dict:
        """
        Get main effects if calculated.

        Returns a dict keyed by effect type rather than a single DataFrame.
        This is intentional: each effect type has different semantics and
        structure, and users typically work with one effect type at a time.

        Returns
        -------
        dict[str, DataFrame]
            Dictionary with main effects:
            - 'k_effects': Factor effects DataFrame
            - 't_effects': Time effects DataFrame
            Empty dict if not calculated.

        Examples
        --------
        >>> effects = result.effects
        >>> factor_effects = effects['k_effects']
        >>> time_effects = effects['t_effects']
        """
        return self._effects.copy() if self._effects else {}

    @property
    def interactions(self) -> dict:
        """
        Get interaction effects if calculated.

        Returns a dict keyed by interaction type rather than a single DataFrame.
        This is intentional: interaction structures vary by SDS, and the dict
        keys provide meaningful organization for different interaction terms.

        Returns
        -------
        dict[str, DataFrame | Series]
            Dictionary with interaction terms (structure varies by SDS).
            Empty dict if not calculated.

        Examples
        --------
        >>> interactions = result.interactions
        >>> for name, data in interactions.items():
        ...     print(f"{name}: {data.shape}")
        """
        return self._interactions.copy() if self._interactions else {}

    @property
    def summary(self) -> dict:
        """
        Get comprehensive summary of analysis.

        Returns
        -------
        dict
            Summary with SDS info, capabilities, dimensions, and statistics
        """
        return self._summary.copy()

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
            Chart name (e.g., 'Xbar', 'S', 'GroupA', 'all')

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
        return self.charts[name]['data'].copy()

    def get_statistics(self, name: str) -> dict:
        """
        Get chart statistics by name.

        Parameters
        ----------
        name : str
            Chart name (e.g., 'Xbar', 'S', 'GroupA')

        Returns
        -------
        dict
            Statistics dictionary with keys like 'mean', 'lpl', 'upl', 'n'

        Raises
        ------
        KeyError
            If chart name not found

        Examples
        --------
        >>> stats = result.get_statistics('Xbar')
        >>> print(f"Mean: {stats['Mean']}, UPL: {stats['upl']}")
        """
        if name not in self.charts:
            raise KeyError(
                f"Chart '{name}' not found. Available charts: {self.all_charts}"
            )
        return self.charts[name]['statistics'].copy()

    def get_residual(self, residual_type: str) -> pd.Series:
        """
        Get specific residual (R1, R2, R3, R4, or R5).

        Parameters
        ----------
        residual_type : str
            Residual type ('R1', 'R2', 'R3', 'R4', or 'R5')

        Returns
        -------
        Series
            Residual values if calculated, empty Series with proper name otherwise

        Examples
        --------
        >>> r1 = result.get_residual('R1')
        >>> r2 = result.get_residual('R2')
        """
        if not self.has_residuals:
            return pd.Series([], name=residual_type, dtype=float)

        if residual_type not in self._residuals.columns:
            available = list(self._residuals.columns)
            logger.warning(
                f"Residual '{residual_type}' not found. "
                f"Available: {available}"
            )
            return pd.Series([], name=residual_type, dtype=float)

        return self._residuals[residual_type].copy()

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
        standard_chart_names = STANDARD_CHART_NAMES
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
        return self.charts[chart_name]['data'].copy()

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

    def chart_table(  # noqa: C901
        self,
        chart: str | None = None,
        include_signal_col: bool = True,
        signal_symbols: bool = True
    ) -> pd.DataFrame:
        """
        Get a summary table of chart data with subgroup sizes.

        Returns a clean DataFrame suitable for display in notebooks,
        including the subgroup size (n) which affects control limit
        calculations when n varies across subgroups.

        Parameters
        ----------
        chart : str, optional
            Chart name (e.g., 'Xbar', 'S'). If None, uses first chart.
        include_signal_col : bool, default True
            Whether to include a signal indicator column
        signal_symbols : bool, default True
            If True, use symbols (↑/↓) for signals; if False, use -1/0/1

        Returns
        -------
        pd.DataFrame
            Summary table with columns:
            - subgroup: Subgroup identifier (RSG)
            - n: Number of observations in subgroup
            - value: The statistic value (xbar, s, etc.)
            - center: Center line value
            - lpl: Lower process limit
            - upl: Upper process limit
            - signal: Signal indicator (if include_signal_col=True)

        Examples
        --------
        Display subgroup summary in notebook:

        >>> result = study.execute()
        >>> result.chart_table('S')
          subgroup   n  value  center    lpl    upl signal
        0      1_1  99  1.241   1.289  1.012  1.565
        1      1_2  98  0.977   1.289  1.011  1.567      ↓
        2      2_1 100  0.902   1.289  1.014  1.564      ↓
        ...

        Get table for Xbar chart:

        >>> result.chart_table('Xbar')

        Use numeric signal values:

        >>> result.chart_table('S', signal_symbols=False)
        """
        # Determine which chart to use
        if chart is None:
            chart = self.all_charts[0]

        if chart not in self.charts:
            raise KeyError(
                f"Chart '{chart}' not found. Available charts: {self.all_charts}"
            )

        # Get chart data
        chart_data = self.charts[chart]['data'].copy()

        # Identify the value column - try response variable first, then infer
        value_col = None
        if self._ads is not None:
            response_var = self._ads.spec.response_var
            if response_var in chart_data.columns:
                value_col = response_var

        # If not found, infer by excluding known non-value columns
        if value_col is None:
            meta_cols = {
                'rsg', 'center', 'lpl', 'upl', 'beyond_limits', 'n', 'N',
                'obs_id', 'x', 'pull', 'time', 'date', 'datetime',
                'rsg_key', 'cell_key'
            }
            # Also exclude any time variable from spec
            if self._ads is not None and self._ads.spec.time_var:
                meta_cols.add(self._ads.spec.time_var)

            value_cols = [c for c in chart_data.columns if c not in meta_cols]
            # Prefer known statistic columns
            for preferred in ['xbar', 's', 'mr', 'r']:
                if preferred in value_cols:
                    value_col = preferred
                    break
            else:
                value_col = value_cols[0] if value_cols else None

        # Try to add n from analysis dataset if available
        if 'n' not in chart_data.columns and self._ads is not None:
            ads = self._ads.analysis_dataset
            if 'n' in ads.columns and 'rsg' in ads.columns and 'rsg' in chart_data.columns:
                n_per_rsg = ads.groupby('rsg', observed=True)['n'].first()
                chart_data['n'] = chart_data['rsg'].map(n_per_rsg)

        # Build output columns in logical order
        output_cols = []
        col_renames = {}

        # Subgroup column
        if 'rsg' in chart_data.columns:
            output_cols.append('rsg')
            col_renames['rsg'] = 'subgroup'

        # n column
        if 'n' in chart_data.columns:
            output_cols.append('n')

        # Value column
        if value_col:
            output_cols.append(value_col)
            col_renames[value_col] = 'value'

        # Control chart columns
        for col in ['center', 'lpl', 'upl']:
            if col in chart_data.columns:
                output_cols.append(col)

        # Signal column
        if include_signal_col and 'beyond_limits' in chart_data.columns:
            output_cols.append('beyond_limits')
            col_renames['beyond_limits'] = 'signal'

        # Select and rename columns
        result = chart_data[output_cols].copy()
        result = result.rename(columns=col_renames)

        # Format signal column
        if include_signal_col and 'signal' in result.columns and signal_symbols:
            # Convert -1/0/1 to ↓/blank/↑
            signal_map = {-1: '↓', 0: '', 1: '↑'}
            result['signal'] = result['signal'].map(signal_map)
            # Keep as numeric if signal_symbols=False

        return result.reset_index(drop=True)

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

    def to_excel(self, filepath: str, **kwargs) -> None:
        """
        Export analysis results to Excel with each component on a separate tab.

        Creates a multi-sheet Excel workbook with organized analysis results.
        See ExcelExporter for full documentation and options.

        Parameters
        ----------
        filepath : str
            Output Excel file path (e.g., 'analysis.xlsx')
        **kwargs
            Additional options passed to ExcelExporter.export():
            - include_summary : bool, default=True
            - include_charts : bool, default=True
            - include_residuals : bool, default=True
            - include_effects : bool, default=True
            - include_interactions : bool, default=True
            - include_full_dataset : bool, default=False
            - format_cells : bool, default=True
            - include_chart_images : bool, default=True
            - export_html : bool, default=True

        Examples
        --------
        >>> result.to_excel('analysis.xlsx')
        >>> result.to_excel('full.xlsx', include_full_dataset=True)
        """
        from .exporters import ExcelExporter
        return ExcelExporter(self).export(filepath, **kwargs)

    def detect_signals(
        self,
        chart: str | None = None,
        rules: str | list[str] | None = None,
        config: Any | None = None,
        **kwargs
    ) -> SignalResult | dict[str, SignalResult]:
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
                elif isinstance(rules, (str, list)):
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

            # Extract value column from metadata (required)
            if 'metadata' not in chart_info:
                raise ValueError(
                    f"Chart '{chart}' missing metadata. "
                    f"This indicates a bug in chart calculation."
                )
            value_col = chart_info['metadata']['value_col']

            # Get chart type from metadata (preferred) or extract from name
            chart_type = chart_info['metadata'].get(
                'chart_type', self._extract_chart_type(chart)
            )

            return detector.detect(
                data=chart_info['data'],
                stats=chart_info['statistics'],
                config=config,
                value_col=value_col,
                chart_name=chart,
                chart_type=chart_type
            )

        else:
            # Detect on all charts
            results = {}
            for chart_name, chart_info in self.charts.items():
                # Extract value column from metadata (required)
                if 'metadata' not in chart_info:
                    raise ValueError(
                        f"Chart '{chart_name}' missing metadata. "
                        f"This indicates a bug in chart calculation."
                    )
                value_col = chart_info['metadata']['value_col']

                # Get chart type from metadata (preferred) or extract from name
                chart_type = chart_info['metadata'].get(
                    'chart_type', self._extract_chart_type(chart_name)
                )

                results[chart_name] = detector.detect(
                    data=chart_info['data'],
                    stats=chart_info['statistics'],
                    config=config,
                    value_col=value_col,
                    chart_name=chart_name,
                    chart_type=chart_type
                )

            return results

    def _extract_chart_type(self, chart_name: str) -> str:
        """
        Extract base chart type from chart name.

        Handles stratified chart names like 'Imr_lane_1' -> 'Imr'.

        Parameters
        ----------
        chart_name : str
            Full chart name

        Returns
        -------
        str
            Base chart type ('Xbar', 'S', 'Imr', 'R')
        """
        # Common chart type mapping
        type_mapping = {
            'Xbar': 'Xbar',
            'S': 'S',
            'Imr': 'Imr',
            'R': 'R'
        }

        # Check if chart name starts with a known type
        for prefix, chart_type in type_mapping.items():
            if chart_name.startswith(prefix):
                return chart_type

        # Default to Xbar if unknown
        return 'Xbar'

    def plot(
        self,
        chart: str | None = None,
        facet: bool = False,
        facet_by: str | None = None,
        ncols: int = 2,
        highlight_signals: bool = True,
        show_limits: bool = True,
        show_zones: bool = False,
        show_rules: bool = False,
        show_stats: bool = False,
        template: str = 'processbehavior',
        width: int = 1000,
        height: int | None = None,
        title: str | None = None,
        **kwargs
    ) -> ControlChartFigure:
        """
        Create interactive control chart visualization.

        This is the main plotting method. It automatically determines
        the best visualization for your data and chart type.

        Parameters
        ----------
        chart : str, optional
            Specific chart to plot ('Xbar', 'S', 'Imr', etc.)
            If None, plots all available charts
        facet : bool, default False
            Whether to create faceted plot for stratified data
        facet_by : str, optional
            Variable to facet by (overrides auto-detection)
        ncols : int, default 2
            Number of columns in faceted layout
        highlight_signals : bool, default True
            Whether to highlight points beyond control limits
        show_limits : bool, default True
            Whether to show control limit lines
        show_zones : bool, default False
            Whether to show zone shading (A/B/C zones at 1-2-3 sigma)
        show_rules : bool, default False
            Whether to show additional run rules (WECO Rules 2-8)
        show_stats : bool, default False
            Whether to show statistics box with CL, UPL, LPL values
        template : str, default 'processbehavior'
            Visual theme ('processbehavior', 'minimal', 'dark', 'ggplot')
        width : int, default 1000
            Figure width in pixels
        height : int, optional
            Figure height in pixels (auto-calculated if None)
        title : str, optional
            Custom title for the figure
        **kwargs
            Additional parameters passed to Plotter.plot()

        Returns
        -------
        ControlChartFigure
            Interactive figure object with .show(), .save_html(), .save_image()

        Examples
        --------
        Simple plotting:

        >>> result = study.execute()
        >>> fig = result.plot()
        >>> fig.show()

        Specific chart with zones:

        >>> fig = result.plot(chart='Xbar', show_zones=True, show_stats=True)

        Faceted by operator:

        >>> fig = result.plot(facet_by='Operator', ncols=3)

        Custom styling:

        >>> fig = result.plot(
        ...     template='dark',
        ...     highlight_signals=True,
        ...     width=1200
        ... )

        Save as HTML:

        >>> fig = result.plot()
        >>> fig.save_html('report.html')

        Save as image (requires kaleido):

        >>> fig = result.plot()
        >>> fig.save_image('chart.png', width=1200, height=800)
        """
        from .plotting import Plotter

        plotter = Plotter(self)
        return plotter.plot(
            chart=chart,
            facet=facet,
            facet_by=facet_by,
            ncols=ncols,
            highlight_signals=highlight_signals,
            show_limits=show_limits,
            show_zones=show_zones,
            show_rules=show_rules,
            show_stats=show_stats,
            template=template,
            width=width,
            height=height,
            title=title,
            **kwargs
        )
