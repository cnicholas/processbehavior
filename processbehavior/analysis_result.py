"""
AnalysisResult - Unified container for all analysis outputs.

This module provides a comprehensive result object that makes all analysis data
easily accessible in one place:
- Chart data (Xbar, S, X, mR) with chart type as primary key
- Stratified chart support with strata property and focus() for drill-down
- Residuals (R1-R5)
- Effects (main effects, interactions)
- Summary metadata (design-state lineage, statistics, capabilities)

Chart Structure
---------------
Charts are always keyed by chart type (e.g., 'Xbar', 'S', 'X', 'mR').
X and mR charts are bundled together, similar to Xbar and S.

For stratified X/mR charts, the structure includes:
- Combined DataFrame with all strata (rsg column identifies them)
- Statistics nested by stratum: {'Machine1': {...}, 'Machine2': {...}}
- 'strata' list for discovering available subgroups

Usage
-----
Basic access::

    result = study.execute()
    xbar_data = result.get_chart('Xbar')
    stats = result.get_statistics('Xbar')

Stratified chart drill-down::

    result.strata           # ['Machine1_F2_1', 'Machine2_F2_2', ...]
    result.is_stratified    # True

    focused = result.focus('Machine1_F2_1')
    focused.plot()
    focused.to_excel('machine1.xlsx')

Access residuals and effects::

    residuals = result.residuals      # DataFrame with R1-R5
    main_effects = result.effects
    interactions = result.interactions
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from .data_preparation import encode_rsg
from .exceptions import ChartNotAvailableError, ProcessBehaviorError
from .sds_detector import SDSRegistry
from .spc_constants import normalize_chart_name
from .types import Charts

if TYPE_CHECKING:
    from .analysis_dataset import AnalysisDataSet
    from .plotting.control_chart import ControlChartFigure
    from .signals.result import SignalResult

logger = logging.getLogger(__name__)

# Built-in theme names (Literal for IDE autocomplete on .plot(theme=...)).
# Custom themes registered via ``register_theme`` work too — the bare
# ``str`` union on the parameter type allows them through.
ThemeName = Literal['processbehavior', 'ggplot', 'minimal', 'dark', 'publication']

# Standard SPC chart type names
STANDARD_CHART_NAMES = {'Xbar', 'S', 'X', 'mR'}


class AnalysisResult:
    """
    Comprehensive analysis result container.

    This class unifies all analysis outputs into a single, easily accessible object.
    It provides:
    - Chart data and statistics (Xbar, S, X, mR)
    - VAS residuals (R1-R5)
    - Main effects and interactions
    - Design-state lineage information (PDS / ODS / ADS)
    - Summary metadata
    - Stratified chart support with focus() for drill-down

    Chart Structure
    ---------------
    Charts are keyed by chart type (e.g., 'Xbar', 'S', 'X', 'mR'):

    - For standard charts:
      ``{'Xbar': {'data': DataFrame, 'statistics': dict, 'metadata': dict}}``

    - For stratified X/mR charts (multiple subgroups):
      ``{'X': {'data': DataFrame, 'statistics': {stratum: dict}, 'strata': list}}``

    X and mR charts are always bundled together, similar to Xbar and S.

    Attributes
    ----------
    charts : dict
        Dictionary of chart data keyed by chart type.
    dataset : pd.DataFrame
        Full analysis dataset with all calculations.
    residuals : pd.DataFrame or None
        VAS residuals (R1-R5) if calculated.
    effects : dict or None
        Main effects if calculated.
    interactions : dict or None
        Interaction effects if calculated.
    summary : dict
        Comprehensive metadata about the analysis.
    sds : int
        Sampling Design State (0-6).
    sds_info : dict
        Detailed SDS characteristics.
    strata : list[str]
        List of subgroup names for stratified charts (empty if not stratified).
    is_stratified : bool
        True if result contains stratified charts with multiple subgroups.

    Examples
    --------
    Basic usage:

    >>> result = study.execute()
    >>> xbar = result.get_chart('Xbar')
    >>> print(result.summary)

    Stratified chart access:

    >>> result.strata  # ['Machine1_F2_1', 'Machine2_F2_2', ...]
    >>> focused = result.focus('Machine1_F2_1')
    >>> focused.plot()
    >>> focused.to_excel('machine1.xlsx')
    """

    def __init__(self, charts: Charts, analysis_dataset_obj: AnalysisDataSet, analysis_type: str | None = None):
        """
        Initialize AnalysisResult from chart data and AnalysisDataSet.

        Parameters
        ----------
        charts : Charts (dict[str, ChartPayload])
            Chart-name → :class:`ChartPayload` map. Each payload carries
            ``data`` (DataFrame), ``statistics`` (flat or by-stratum),
            ``metadata``, and optional ``strata``.
        analysis_dataset_obj : AnalysisDataSet
            The underlying AnalysisDataSet with all calculations
        analysis_type : str, optional
            The executed chart type ('Xbar', 'S', 'X', 'mR').
            Passed from Analysis at execute() time so result.summary
            reports the executed chart, not the recommended one.
        """
        # Store chart data
        self.charts: Charts = charts

        # Store reference to full dataset
        self._ads = analysis_dataset_obj
        self.dataset = analysis_dataset_obj.analysis_dataset

        # Store the executed analysis type (passed from Analysis)
        self._analysis_type = analysis_type

        # Design state information
        self.observed_sds: int = analysis_dataset_obj.observed_design_state
        self.analytical_sds: int = analysis_dataset_obj.analytical_design_state.sds
        self.observed_sds_info: dict = analysis_dataset_obj.raw_sds_characteristics
        self.analytical_sds_info: dict = SDSRegistry().get_sds_characteristics(self.analytical_sds)

        # Extract residuals if calculated
        self._residuals = None
        if analysis_dataset_obj.has_vas_residuals:
            residual_cols = ['R1', 'R2', 'R3', 'R4', 'R5']
            available_cols = [c for c in residual_cols if c in self.dataset.columns]
            if available_cols:
                self._residuals = self.dataset[available_cols].copy()

        # Extract effects and interactions
        self._effects = analysis_dataset_obj.effects if analysis_dataset_obj.effects else None
        self._interactions = analysis_dataset_obj.interactions if analysis_dataset_obj.interactions else None

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
            'observed_sds': self.observed_sds,
            'analytical_sds': self.analytical_sds,
            'analytical_sds_description': self.analytical_sds_info.get('description', 'Unknown'),
            'analytical_sds_capabilities': self.analytical_sds_info.get('capabilities', []),
            'replication_type': self.analytical_sds_info.get('replication_type', 'unknown'),
            # Analysis configuration
            'analysis_type': self._analysis_type,
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
            'variance_decomposition': self.analytical_sds_info.get('variance_decomposition', False),
            'interaction_analysis': self.analytical_sds_info.get('interaction_analysis', False),
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
            True if any chart has a non-empty 'strata' key
        """
        # Handle case where charts might not be a dict
        if not isinstance(self.charts, dict):
            return False

        for chart_info in self.charts.values():
            if isinstance(chart_info, dict) and 'strata' in chart_info and chart_info['strata']:
                return True
        return False

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

    @property
    def strata(self) -> list[str]:
        """
        Get list of available subgroups focusable across every chart.

        For stratified analysis, returns the order-preserving intersection
        of every chart's strata list. A stratum present in one chart but
        missing data in another (e.g. an X+mR pair where mR dropped a
        single-observation cell) is excluded so that the returned values
        are valid inputs to ``focus()`` for the full result.

        Returns
        -------
        list[str]
            List of stratum names focusable across all charts. Empty list
            if no chart is stratified.

        Examples
        --------
        >>> result = study.execute(chart='X', by=['factor 1'])
        >>> result.strata
        ['level_1', 'level_2', ...]

        >>> if result.strata:
        ...     for stratum in result.strata:
        ...         focused = result.focus(stratum)
        ...         focused.plot()
        """
        strata_lists = [
            list(chart_info['strata'])
            for chart_info in self.charts.values()
            if 'strata' in chart_info and chart_info['strata']
        ]
        if not strata_lists:
            return []
        if len(strata_lists) == 1:
            return strata_lists[0]
        common = set(strata_lists[0])
        for sl in strata_lists[1:]:
            common &= set(sl)
        return [s for s in strata_lists[0] if s in common]

    @property
    def is_stratified(self) -> bool:
        """
        Check if this result contains stratified charts.

        Returns
        -------
        bool
            True if charts have multiple strata, False otherwise.
        """
        return len(self.strata) > 0

    def focus(self, stratum: str) -> AnalysisResult:
        """
        Return new AnalysisResult focused on a single stratum.

        For stratified X/mR analysis, this allows drilling down to
        a specific subgroup. The returned result is immutable - the
        original result is unchanged.

        Parameters
        ----------
        stratum : str
            Name of the stratum to focus on (from result.strata)

        Returns
        -------
        AnalysisResult
            New AnalysisResult containing only data for the specified stratum

        Raises
        ------
        ValidationError
            If stratum is not in result.strata or focus produces empty data.
        ProcessBehaviorError
            If required stratification metadata/columns are inconsistent.

        Examples
        --------
        >>> result = study.execute(chart='X', by=['factor 1'])
        >>> result.strata
        ['level_1', 'level_2', ...]
        >>> focused = result.focus('level_1')
        >>> focused.plot()

        >>> # Chaining works
        >>> result.focus('level_1').plot()
        """
        from .exceptions import ValidationError

        if not self.strata:
            raise ValidationError(
                'Cannot focus: this result is not stratified. Use result.strata to check available subgroups.'
            )

        if stratum not in self.strata:
            raise ValidationError(f"Stratum '{stratum}' not found. Available strata: {self.strata}")

        # Build focused charts dict
        focused_charts = {}

        for chart_name, chart_info in self.charts.items():
            if 'strata' not in chart_info or not chart_info['strata']:
                # Non-stratified chart - include as-is
                focused_charts[chart_name] = chart_info.copy()
                continue

            # Filter data to this stratum
            data = chart_info['data']
            metadata = chart_info.get('metadata', {})
            stratify_col = metadata.get('stratify_col')

            # Preferred path: filter by explicit stratification column from metadata
            if stratify_col and stratify_col in data.columns:
                mask = data[stratify_col] == stratum
            else:
                # Legacy fallback: find an rsg-like column
                rsg_col = None
                for col in data.columns:
                    if col in ['rsg', 'RSG'] or 'rsg' in col.lower():
                        rsg_col = col
                        break

                if rsg_col is None:
                    raise ProcessBehaviorError(
                        f"Cannot focus stratified chart '{chart_name}': "
                        'missing stratification column metadata and no rsg column found. '
                        'This indicates a bug in chart construction.'
                    )

                # Note: stratum identity assumes canonical factor ordering defined upstream
                mask = data[rsg_col].astype(str) == encode_rsg(stratum)
            # Reset_index so the focused data's row positions are 0..n-1.
            # The renderer falls back to `data.index` for trace x when there's
            # no explicit x_col (integer-position axis); without the reset, that
            # index inherits 376..755 from the unfiltered data and the tick
            # positions computed by x_axis_layout (which use iloc 0..n-1) land
            # left of the rendered data — same hazard as _get_stratified_charts
            # already handles.
            focused_data = data[mask].copy().reset_index(drop=True)

            if focused_data.empty:
                raise ValidationError(
                    f"No data found for stratum '{stratum}' in chart '{chart_name}'. "
                    f'This may indicate an encoding mismatch between strata keys and rsg values.'
                )

            # Extract stratum-specific statistics
            nested_stats = chart_info.get('statistics', {})
            if isinstance(nested_stats, dict) and stratum in nested_stats:
                focused_stats = nested_stats[stratum]
            elif isinstance(nested_stats, dict) and encode_rsg(stratum) in nested_stats:
                focused_stats = nested_stats[encode_rsg(stratum)]
            elif chart_info.get('strata'):
                # Stratified chart but key doesn't match — don't silently return the nested dict
                raise ProcessBehaviorError(
                    f"Statistics key mismatch for stratum '{stratum}' in chart '{chart_name}'. "
                    f'Available keys: {list(nested_stats.keys())}'
                )
            else:
                focused_stats = nested_stats  # flat stats dict — OK

            # Build focused chart info. Unpack the parent's per-stratum
            # lane_boundaries to the focused stratum's flat list so downstream
            # consumers (plotter) see a single chart's positions, not a dict
            # they'd have to disambiguate. Without this, the plotter silently
            # used the first stratum's positions on every focused chart.
            parent_metadata = chart_info.get('metadata', {})
            raw_lb = parent_metadata.get('lane_boundaries')
            if isinstance(raw_lb, dict):
                focused_raw_lb = raw_lb.get(stratum) or raw_lb.get(encode_rsg(stratum))
                if focused_raw_lb is None and stratum in raw_lb:
                    focused_raw_lb = raw_lb[stratum]
            elif isinstance(raw_lb, list):
                focused_raw_lb = raw_lb
            else:
                focused_raw_lb = None

            focused_charts[chart_name] = {
                'data': focused_data,
                'statistics': focused_stats,
                'metadata': {
                    **parent_metadata,
                    'stratified': False,  # No longer stratified after focus
                    'focused_stratum': stratum,
                    'lane_boundaries': focused_raw_lb,
                },
            }

        # Create new AnalysisResult with focused data
        # We need to create a minimal AnalysisDataSet-like object
        return FocusedAnalysisResult(charts=focused_charts, original_result=self, focused_stratum=stratum)

    # =========================================================================
    # Convenience methods for accessing data
    # =========================================================================

    def _resolve_chart_name(self, name: str) -> str:
        """Resolve chart name with case-insensitive fallback for base charts."""
        if name in self.charts:
            return name
        normalized = normalize_chart_name(name)
        if normalized in self.charts:
            return normalized
        return name  # Return as-is; caller raises the appropriate error

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
        ChartNotAvailableError
            If chart name not found in this result.

        Examples
        --------
        >>> xbar = result.get_chart('Xbar')
        >>> alice = result.get_chart('Alice')  # For stratified X
        """
        name = self._resolve_chart_name(name)
        if name not in self.charts:
            raise ChartNotAvailableError(
                f"Chart '{name}' not found. Available charts: {self.all_charts}", chart=name, available=self.all_charts
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
            Statistics dictionary. Every chart guarantees the same four
            unified keys: ``{'N', 'center', 'lpl', 'upl'}``.

            When the control limits or subgroup size vary across
            subgroups (e.g. Xbar/S with ``n_mode='actual'`` on unbalanced
            data, phased charts, or any chart whose limits aren't a
            single scalar), ``N``, ``lpl``, and ``upl`` are ``None`` and
            an optional ``'limits_vary': True`` flag is added so callers
            can detect variable limits without a type check.

            The Histogram chart also includes ``{'mean', 'std', 'n'}``
            as additional fields (``mean`` is an alias for ``center``;
            ``std`` is the sample standard deviation; ``n`` is an alias
            for ``N``); its ``'lpl'`` and ``'upl'`` are ``None`` because
            a histogram has no control limits.

        Raises
        ------
        ChartNotAvailableError
            If chart name not found in this result.

        Examples
        --------
        >>> stats = result.get_statistics('Xbar')
        >>> print(f"Center: {stats['center']}, UPL: {stats['upl']}")
        """
        name = self._resolve_chart_name(name)
        if name not in self.charts:
            raise ChartNotAvailableError(
                f"Chart '{name}' not found. Available charts: {self.all_charts}", chart=name, available=self.all_charts
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
            logger.warning(f"Residual '{residual_type}' not found. Available: {available}")
            return pd.Series([], name=residual_type, dtype=float)

        return self._residuals[residual_type].copy()

    def list_strata(self) -> list[str]:
        """
        List all strata in stratified analysis.

        Equivalent to accessing the ``strata`` property. Values returned
        here are valid inputs to ``focus()``.

        Returns
        -------
        list
            List of stratum identifiers

        Examples
        --------
        >>> strata = result.list_strata()
        >>> for s in strata:
        ...     focused = result.focus(s)
        """
        return self.strata

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
        self, chart: str | None = None, include_signal_col: bool = True, signal_symbols: bool = True
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
        chart = self.all_charts[0] if chart is None else self._resolve_chart_name(chart)

        if chart not in self.charts:
            raise ChartNotAvailableError(
                f"Chart '{chart}' not found. Available charts: {self.all_charts}",
                chart=chart,
                available=self.all_charts,
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
                'rsg',
                'center',
                'lpl',
                'upl',
                'beyond_limits',
                'n',
                'N',
                'obs_id',
                'x',
                'pull',
                'time',
                'date',
                'datetime',
                'rsg_key',
                'cell_key',
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
        # Note: n is computed per kt (factor × time) cell, so we need to join
        # by kt columns, not just factor columns
        if 'n' not in chart_data.columns and self._ads is not None:
            ads = self._ads.analysis_dataset
            spec = self._ads.spec
            if 'n' in ads.columns:
                # Build kt join columns (must match columns in chart_data)
                kt_cols = []
                if spec.rsg_var_name and spec.rsg_var_name in ads.columns:
                    # Use rsg if it's in both
                    rsg_col = 'rsg' if 'rsg' in chart_data.columns else spec.rsg_var_name
                    if rsg_col in chart_data.columns:
                        kt_cols.append(spec.rsg_var_name)
                if spec.has_time and spec.time_var in ads.columns and spec.time_var in chart_data.columns:
                    kt_cols.append(spec.time_var)

                if kt_cols:
                    # Get unique n per kt cell
                    n_per_kt = ads.groupby(kt_cols, observed=True)['n'].first().reset_index()
                    # Coerce join-key dtypes — chart construction may
                    # stringify by-columns (e.g. 'PRODUCTION TIME' becomes
                    # object) while the ads keeps them numeric, which pandas
                    # refuses to merge. When the dtypes differ on either
                    # side, cast both sides of every kt column to string so
                    # values (not just dtypes) line up.
                    mismatched = any(chart_data[col].dtype != n_per_kt[col].dtype for col in kt_cols)
                    if mismatched:
                        for col in kt_cols:
                            chart_data[col] = chart_data[col].astype(str)
                            n_per_kt[col] = n_per_kt[col].astype(str)
                    # Merge to add n; if it still fails, skip the n-join
                    # rather than killing the entire table.
                    with contextlib.suppress(ValueError, TypeError):
                        chart_data = chart_data.merge(n_per_kt, on=kt_cols, how='left')

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
            f'AnalysisResult(\n'
            f'  analytical_sds={self.analytical_sds} ({self.analytical_sds_info["description"]}),\n'
            f'  charts=[{charts_str}],\n'
            f'  n_obs={len(self.dataset)},\n'
            f'  has_residuals={self.has_residuals},\n'
            f'  has_effects={self.has_effects},\n'
            f'  n_signals={self.summary["n_signals_total"]}\n'
            f')'
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        lines = [
            '=' * 70,
            'ANALYSIS RESULT SUMMARY',
            '=' * 70,
            f'\nAnalytical Design State (ADS): {self.analytical_sds}',
            f'Description: {self.analytical_sds_info["description"]}',
            f'\nAnalysis Type: {self.summary["analysis_type"]}',
            f'Response Variable: {self.summary["response_var"]}',
        ]

        if self.summary['grouping_vars']:
            lines.append(f'Grouping: {", ".join(self.summary["grouping_vars"])}')

        if self.summary['time_var']:
            lines.append(f'Time Variable: {self.summary["time_var"]}')

        lines.extend(
            [
                f'\nObservations: {self.summary["n_observations"]}',
                f'Charts: {", ".join(self.all_charts)}',
            ]
        )

        if self.summary['is_stratified']:
            lines.append(f'Stratified: Yes ({len(self.charts)} groups)')

        lines.append('\nCapabilities:')
        lines.append(f'  Residuals: {"✓" if self.has_residuals else "✗"}')
        lines.append(f'  Effects: {"✓" if self.has_effects else "✗"}')
        lines.append(f'  Interactions: {"✓" if self.has_interactions else "✗"}')

        if self.summary['n_signals_total'] > 0:
            lines.append(f'\n⚠️  Signals: {self.summary["n_signals_total"]} points beyond limits')

        lines.append('=' * 70)

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
        self, chart: str | None = None, rules: str | list[str] | None = None, config: Any | None = None, **kwargs
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
            chart = self._resolve_chart_name(chart)
            if chart not in self.charts:
                raise ChartNotAvailableError(
                    f"Chart '{chart}' not found.\nAvailable: {self.all_charts}", chart=chart, available=self.all_charts
                )

            chart_info = self.charts[chart]

            # Extract value column from metadata (required)
            if 'metadata' not in chart_info:
                raise ProcessBehaviorError(
                    f"Chart '{chart}' missing metadata. This indicates a bug in chart calculation."
                )
            value_col = chart_info['metadata']['value_col']

            # Get chart type from metadata (preferred) or extract from name
            chart_type = chart_info['metadata'].get('chart_type', self._extract_chart_type(chart))

            return detector.detect(
                data=chart_info['data'],
                stats=chart_info['statistics'],
                config=config,
                value_col=value_col,
                chart_name=chart,
                chart_type=chart_type,
            )

        else:
            # Detect on all charts
            results = {}
            for chart_name, chart_info in self.charts.items():
                # Extract value column from metadata (required)
                if 'metadata' not in chart_info:
                    raise ProcessBehaviorError(
                        f"Chart '{chart_name}' missing metadata. This indicates a bug in chart calculation."
                    )
                value_col = chart_info['metadata']['value_col']

                # Get chart type from metadata (preferred) or extract from name
                chart_type = chart_info['metadata'].get('chart_type', self._extract_chart_type(chart_name))

                results[chart_name] = detector.detect(
                    data=chart_info['data'],
                    stats=chart_info['statistics'],
                    config=config,
                    value_col=value_col,
                    chart_name=chart_name,
                    chart_type=chart_type,
                )

            return results

    def _extract_chart_type(self, chart_name: str) -> str:
        """
        Extract base chart type from chart name.

        Handles stratified chart names like 'X_lane_1' -> 'X'.

        Parameters
        ----------
        chart_name : str
            Full chart name

        Returns
        -------
        str
            Base chart type ('Xbar', 'S', 'X', 'mR')
        """
        # Common chart type mapping
        type_mapping = {'Xbar': 'Xbar', 'S': 'S', 'X': 'X', 'mR': 'mR'}

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
        ncols: int = 2,
        highlight_signals: bool = True,
        show_limits: bool = True,
        show_limit_values: bool = True,
        show_zones: bool = False,
        show_rules: bool = False,
        show_stats: bool = False,
        theme: ThemeName | str = 'processbehavior',
        width: int = 1000,
        height: int | None = None,
        aspect_ratio: float | None = None,
        title: str | None = None,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
        shared_yaxis: bool = True,
        yaxis_padding: float = 0.05,
        vertical_spacing: float = 0.15,
    ) -> ControlChartFigure:
        """Create interactive control chart visualisation.

        Parameters
        ----------
        chart : str, optional
            Specific chart to plot ('Xbar', 'S', 'X', etc.).
            Also accepts 'Effects', 'MainEffects', 'TimeEffects',
            'TimeInteraction', 'FactorInteraction'.
            If None, plots all available charts.
        facet : bool, default False
            Whether to create faceted plot for stratified data.
        ncols : int, default 2
            Number of columns in faceted layout.
        highlight_signals : bool, default True
            Highlight points beyond control limits.
        show_limits : bool, default True
            Show control limit lines.
        show_limit_values : bool, default True
            Show numeric values in limit labels.
        show_zones : bool, default False
            Show zone shading (A/B/C zones at 1-2-3 sigma).
        show_rules : bool, default False
            Show additional run rules (WECO Rules 2-8).
        show_stats : bool, default False
            Show statistics box with CL, UPL, LPL values.
        theme : str, default 'processbehavior'
            Visual theme ('processbehavior', 'minimal', 'dark', 'ggplot',
            'publication') or a ChartTheme instance.
        width : int, default 1000
            Figure width in pixels.
        height : int, optional
            Figure height in pixels (auto-calculated if None).
        aspect_ratio : float, optional
            Width-to-height ratio (overrides height if specified).
        title : str, optional
            Custom title for the figure.
        xaxis_title, yaxis_title : str, optional
            Custom axis labels.
        shared_yaxis : bool, default True
            Whether faceted charts share y-axis range.
        yaxis_padding : float, default 0.05
            Padding fraction for y-axis range.
        vertical_spacing : float, default 0.15
            Vertical spacing between rows in faceted layouts.

        Returns
        -------
        ControlChartFigure
            Interactive figure object with .show(), .save_html(), .save_image()

        Examples
        --------
        >>> fig = result.plot()
        >>> fig = result.plot(chart='Xbar', show_zones=True, show_stats=True)
        >>> fig = result.plot(facet=True, ncols=3)
        """
        from .plotting import Plotter

        if chart:
            chart = self._resolve_chart_name(chart)

        plotter = Plotter(self)
        return plotter.plot(
            chart=chart,
            facet=facet,
            ncols=ncols,
            highlight_signals=highlight_signals,
            show_limits=show_limits,
            show_limit_values=show_limit_values,
            show_zones=show_zones,
            show_rules=show_rules,
            show_stats=show_stats,
            theme=theme,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            shared_yaxis=shared_yaxis,
            yaxis_padding=yaxis_padding,
            vertical_spacing=vertical_spacing,
        )

    def plot_residuals(
        self,
        residual_type: str = 'R1',
        plot_type: str = 'all',
        theme: ThemeName | str = 'processbehavior',
        width: int = 1200,
        height: int = 400,
    ) -> ControlChartFigure:
        """Create residual diagnostic plots.

        Parameters
        ----------
        residual_type : str, default 'R1'
            Which residual ('R1'-'R5').
        plot_type : str, default 'all'
            'histogram', 'qq', 'sequence', or 'all'.
        theme : str, default 'processbehavior'
            Visual theme.
        width, height : int
            Figure dimensions.

        Returns
        -------
        ControlChartFigure
        """
        from .plotting.residuals import plot_residuals as _plot_residuals

        return _plot_residuals(
            self,
            residual_type=residual_type,
            plot_type=plot_type,
            theme=theme,
            width=width,
            height=height,
        )

    def plot_effects(
        self,
        effect_type: str = 'factor',
        theme: ThemeName | str = 'processbehavior',
        width: int = 800,
        height: int = 500,
    ) -> ControlChartFigure:
        """Create bar chart of main effects.

        Parameters
        ----------
        effect_type : str, default 'factor'
            'factor', 'time', or 'all'.
        theme : str, default 'processbehavior'
            Visual theme.
        width, height : int
            Figure dimensions.

        Returns
        -------
        ControlChartFigure
        """
        from .plotting import Plotter

        plotter = Plotter(self)
        return plotter.plot_effects(
            effect_type=effect_type,
            theme=theme,
            width=width,
            height=height,
        )

    def report(
        self,
        filepath: str,
        include_charts: bool = True,
        include_residuals: bool = True,
        include_effects: bool = True,
        include_summary: bool = True,
        theme: ThemeName | str = 'processbehavior',
        width: int = 1200,
        title: str | None = None,
    ) -> None:
        """Generate comprehensive HTML report.

        Parameters
        ----------
        filepath : str
            Output HTML file path.
        include_charts, include_residuals, include_effects, include_summary : bool
            Sections to include.
        theme : str, default 'processbehavior'
            Visual theme.
        width : int, default 1200
            Chart width in pixels.
        title : str, optional
            Report title.
        """
        from .plotting.report import generate_report

        generate_report(
            self,
            filepath=filepath,
            include_charts=include_charts,
            include_residuals=include_residuals,
            include_effects=include_effects,
            include_summary=include_summary,
            theme=theme,
            width=width,
            title=title,
        )


class FocusedAnalysisResult(AnalysisResult):
    """
    Lightweight AnalysisResult for focused (single-stratum) analysis.

    This class is returned by AnalysisResult.focus() and provides
    the same interface as AnalysisResult but without requiring a
    full AnalysisDataSet.

    Parameters
    ----------
    charts : dict
        Chart data in standard format
    original_result : AnalysisResult
        The parent result this was focused from
    focused_stratum : str
        The stratum this result is focused on
    """

    def __init__(self, charts: dict[str, dict[str, Any]], original_result: AnalysisResult, focused_stratum: str):
        # Store chart data
        self.charts = charts

        # Store reference to original result for metadata
        self._original = original_result
        self._focused_stratum = focused_stratum

        # Preserve _ads reference from original result (needed by chart_table, plot, to_excel)
        self._ads = original_result._ads

        # Preserve the executed analysis type
        self._analysis_type = original_result._analysis_type

        # Copy dataset reference (filtered)
        rsg_col = None
        for col in original_result.dataset.columns:
            if col in ['rsg', 'RSG'] or 'rsg' in col.lower():
                rsg_col = col
                break

        # Note: stratum identity assumes canonical factor ordering defined upstream
        stratum_id = encode_rsg(focused_stratum)
        if rsg_col:
            mask = original_result.dataset[rsg_col].astype(str) == stratum_id
            self.dataset = original_result.dataset[mask].copy()
        else:
            self.dataset = original_result.dataset.copy()

        # Copy design state information
        self.observed_sds = original_result.observed_sds
        self.analytical_sds = original_result.analytical_sds
        self.observed_sds_info = original_result.observed_sds_info.copy()
        self.analytical_sds_info = original_result.analytical_sds_info.copy()

        # Copy residuals/effects (filtered if possible)
        self._residuals = None
        if original_result._residuals is not None:
            if rsg_col and rsg_col in original_result._residuals.columns:
                mask = original_result._residuals[rsg_col].astype(str) == stratum_id
                self._residuals = original_result._residuals[mask].copy()
            else:
                # Can't filter - take subset based on index
                self._residuals = original_result._residuals.loc[self.dataset.index].copy()

        self._effects = original_result._effects
        self._interactions = original_result._interactions

        # Store original's spec info for summary
        self._original_summary = original_result._summary.copy()

        # Build summary
        self._summary = self._build_focused_summary()

    def _build_focused_summary(self) -> dict:
        """Build summary for focused result."""
        # Count signals in focused charts
        n_signals = 0
        for chart_info in self.charts.values():
            if 'data' in chart_info and 'beyond_limits' in chart_info['data'].columns:
                n_signals += (chart_info['data']['beyond_limits'] != 0).sum()

        # Copy from original and update
        summary = self._original_summary.copy()
        summary.update(
            {
                'n_observations': len(self.dataset),
                'n_charts': len(self.charts),
                'chart_types': list(self.charts.keys()),
                'is_stratified': False,
                'focused_stratum': self._focused_stratum,
                'n_signals_total': int(n_signals),
            }
        )
        return summary

    @property
    def strata(self) -> list[str]:
        """Focused result has no strata (single stratum)."""
        return []

    @property
    def is_stratified(self) -> bool:
        """Focused result is not stratified."""
        return False

    @property
    def focused_stratum(self) -> str:
        """Get the stratum this result is focused on."""
        return self._focused_stratum

    def focus(self, stratum: str) -> AnalysisResult:
        """Cannot focus further - already focused on single stratum."""
        from .exceptions import ValidationError

        raise ValidationError(
            f"Cannot focus: this result is already focused on '{self._focused_stratum}'. "
            'Use the original result to focus on a different stratum.'
        )

    def __repr__(self) -> str:
        """String representation."""
        charts_str = ', '.join(self.all_charts)
        return (
            f'FocusedAnalysisResult(\n'
            f"  stratum='{self._focused_stratum}',\n"
            f'  analytical_sds={self.analytical_sds} ({self.analytical_sds_info["description"]}),\n'
            f'  charts=[{charts_str}],\n'
            f'  n_obs={len(self.dataset)},\n'
            f'  has_residuals={self.has_residuals},\n'
            f'  has_effects={self.has_effects}\n'
            f')'
        )
