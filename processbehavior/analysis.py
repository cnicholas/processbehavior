"""
Analysis - Chart calculation strategies for process behavior analysis.

This module provides the Analysis class which executes chart calculations using
the strategy pattern. It supports:
- Xbar and S charts (subgroup mean and variation)
- IMR charts (individual moving range)
- R charts (range)

The Analysis class coordinates between:
- FormulationSpec (structural configuration)
- ChartRequest (per-execute chart parameters)
- AnalysisDataSet (data preparation and VAS calculations)
- Chart calculation strategies
- AnalysisResult (unified result container)

Usage:
    spec = FormulationSpec(response_var='Height', rsg_vars=('Operator', 'Machine'), time_var='Time')
    request = ChartRequest(chart='Xbar')
    analysis = Analysis(spec, request, analysis_dataset=ads)
    result = analysis.calculate()

    # Access charts
    xbar = result.get_chart('Xbar')
    s = result.get_chart('S')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .analysis_dataset import AnalysisDataSet
from .analysis_result import AnalysisResult
from .formulation_spec import ChartRequest, FormulationSpec
from .spc_constants import calculate_limits, detect_beyond_limits

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# MR-Family Chart Specification
# ============================================================================

@dataclass(frozen=True)
class _MRChartSpec:
    """Captures what makes an MR-family chart different from its sibling.

    IMR and R charts share >85% of their calculation pipeline. The differences
    are behavioral, not parametric — this dataclass encodes those differences
    so a single shared method can serve both chart types.
    """
    chart_type: str         # 'Imr' or 'R'
    limits_type: str        # "Imr" or "R" — passed to calculate_limits()
    plot_col: str           # Which column to plot: 'raw' (use value_col) or 'mr'
    center_source: str      # What the center line represents: 'mean' or 'mR'
    drops_first_mr: bool    # False for Imr, True for R
    lane_boundary_offset: int  # 0 for Imr, -1 for R


_IMR_SPEC = _MRChartSpec(
    chart_type='Imr', limits_type='Imr',
    plot_col='raw', center_source='mean',
    drops_first_mr=False, lane_boundary_offset=0,
)
_R_SPEC = _MRChartSpec(
    chart_type='R', limits_type='R',
    plot_col='mr', center_source='mR',
    drops_first_mr=True, lane_boundary_offset=-1,
)


def _join_limits(grouped: pd.DataFrame, lims_result, rsuffix: str = '') -> pd.DataFrame:
    """Join lpl/upl from a DataFrame.apply(calculate_limits) result.

    ``DataFrame.apply()`` with ``calculate_limits`` returns either:
    - A ``pd.DataFrame`` (when all Series share the same index), or
    - A ``pd.Series`` of ``pd.Series`` objects.

    This helper normalizes both cases and joins ``lpl``/``upl`` onto *grouped*.
    """
    if isinstance(lims_result, pd.DataFrame):
        return grouped.join(lims_result[['lpl', 'upl']], rsuffix=rsuffix)

    # Series of Series — extract lpl/upl manually
    lims_df = pd.DataFrame(
        [(row['lpl'], row['upl']) for row in lims_result],
        index=grouped.index,
        columns=['lpl', 'upl'],
    )
    return grouped.join(lims_df, rsuffix=rsuffix)


class Analysis:
    """
    Unified analysis class handling all chart types via strategy pattern.

    This class replaces the AbstractFactory pattern with a simpler, more maintainable
    approach. All analysis types (Xbar, S, Imr, R) are handled through internal
    strategy methods.

    Usage:
        # Standard usage (calculates AnalysisDataSet from scratch)
        analysis = Analysis(df, specification)
        result = analysis.calculate()

        # With pre-calculated AnalysisDataSet (for Study.execute())
        analysis = Analysis(df, specification, analysis_dataset=ads)
        result = analysis.calculate()  # Reuses pre-calculated data
    """

    def __init__(
        self,
        spec: FormulationSpec,
        request: ChartRequest,
        analysis_dataset: AnalysisDataSet | None = None,
        sds: int | None = None,
        df: pd.DataFrame | None = None
    ):
        """
        Initialize analysis with spec and chart request.

        Args:
            spec: Structural configuration (from formulate())
            request: Chart-specific request (from execute())
            analysis_dataset: Optional pre-calculated AnalysisDataSet.
                If provided, skips expensive residual calculation.
                Used by Study.execute() to reuse formulate() calculations.
            sds: Sampling Design State (0-6). Required if analysis_dataset is not
                provided. SDS should be detected at the entry point (ProcessBehavior)
                and passed through the system.
            df: Raw DataFrame. Required if analysis_dataset is not provided.

        Raises:
            ValueError: If neither analysis_dataset nor sds is provided.
        """
        self.spec = spec
        self.request = request
        self.analysis_type = request.chart

        # Use pre-calculated AnalysisDataSet if provided, otherwise calculate
        if analysis_dataset is not None:
            self.ads = analysis_dataset
        else:
            if sds is None:
                raise ValueError(
                    "sds is required when analysis_dataset is not provided. "
                    "SDS should be detected at the entry point (ProcessBehavior) "
                    "and passed to Analysis."
                )
            if df is None:
                raise ValueError(
                    "df is required when analysis_dataset is not provided."
                )
            self.ads = AnalysisDataSet(df, spec, sds=sds)

    def calculate(self) -> AnalysisResult:
        """
        Execute the appropriate analysis strategy and return comprehensive results.

        Returns
        -------
        AnalysisResult
            Unified result object containing:
            - charts: Chart data and statistics
            - residuals: VAS residuals (R1-R5) if calculated
            - effects: Main effects if calculated
            - interactions: Interaction effects if calculated
            - summary: Comprehensive metadata
            - dataset: Full analysis dataset

        Raises
        ------
        ValueError
            If analysis_type is not supported

        Examples
        --------
        >>> result = analysis.calculate()
        >>> xbar = result.get_chart('Xbar')
        >>> if result.has_residuals:
        ...     residuals = result.residuals
        """
        # Check if this is a residual chart request
        residual = self.request.residual
        if residual:
            # Residual chart analysis - inline logic (was _calculate_residual_chart)
            chart_type = self.request.residual_chart_type or 'Imr'
            recentered = self.request.recentered

            # Determine column name
            col_prefix = 'RCR' if recentered else 'R'
            residual_num = residual[1:]  # Extract number from 'R2', 'R3', 'R10', etc.
            col_name = f'{col_prefix}{residual_num}'

            # Validate column exists
            if col_name not in self.ads.analysis_dataset.columns:
                available = [c for c in self.ads.analysis_dataset.columns
                            if c.startswith('R') or c == self.spec.response_var]
                raise ValueError(
                    f"Residual column '{col_name}' not found.\n"
                    f"Available columns: {available}\n"
                    f"This may indicate the SDS doesn't support this residual type."
                )

            # Validate chart type
            valid_chart_types = {'Xbar', 'S', 'Imr', 'Histogram'}
            if chart_type not in valid_chart_types:
                raise ValueError(
                    f"Chart type '{chart_type}' not supported for residual charts.\n"
                    f"Valid types: {', '.join(sorted(valid_chart_types))}"
                )

            # Question answered by each residual
            questions = {
                'R2': 'Is within-subgroup variation stable?',
                'R3': 'Is there interaction between factors and time?',
                'R4': 'Does time have a significant effect?',
                'R5': 'Do factors have a significant effect?'
            }

            # Calculate using base method with value_col
            residual_strategies = {
                'Xbar': self._calculate_xbar,
                'S': self._calculate_s,
                'Imr': self._calculate_imr,
                'Histogram': self._calculate_histogram,
            }
            chart_data = residual_strategies[chart_type](value_col=col_name)
            chart_data = self._add_residual_metadata(
                chart_data, residual, recentered, questions
            )

        else:
            # Standard chart analysis
            # Check if paired charts requested (Xbar+S or Imr+R together)
            paired = self.request.paired

            if paired:
                # Paired mode: return both charts regardless of which was requested
                strategies = {
                    'Xbar': self._calculate_xbar_s,
                    'S': self._calculate_xbar_s,
                    'Imr': self._calculate_imr_r,  # Bundled Imr+R
                    'R': self._calculate_imr_r,
                    'Histogram': self._calculate_histogram  # No pairing for Histogram
                }
            else:
                # SRP-compliant mode: return only the requested chart
                strategies = {
                    'Xbar': self._calculate_xbar,
                    'S': self._calculate_s,
                    'Imr': self._calculate_imr,  # SRP: Imr only
                    'R': self._calculate_r,  # SRP: R only
                    'Histogram': self._calculate_histogram
                }

            if self.analysis_type not in strategies:
                raise ValueError(
                    f'Analysis type {self.analysis_type} not supported! '
                    f'Valid types: {list(strategies.keys())}'
                )

            # Execute analysis strategy
            chart_data = strategies[self.analysis_type]()

        # Wrap in AnalysisResult for unified access
        # Pass analysis_type so result.summary reports the executed chart type, not the recommended one
        return AnalysisResult(
            charts=chart_data,
            analysis_dataset_obj=self.ads,
            analysis_type=self.analysis_type
        )

    # =========================================================================
    # Helper Methods (DRY principle)
    # =========================================================================

    @staticmethod
    def _add_residual_metadata(chart_data, residual, recentered, questions):
        """Add residual context to chart results. Returns new dict (no mutation)."""
        result = {}
        for chart_key, data in chart_data.items():
            result[chart_key] = {
                **data,
                'metadata': {
                    **data.get('metadata', {}),
                    'residual_type': residual,
                    'recentered': recentered,
                    'question_answered': questions.get(residual, ''),
                },
            }
        return result

    def _add_beyond_limits_flag(
        self,
        df: pd.DataFrame,
        value_col: str,
        lpl_col: str = 'lpl',
        upl_col: str = 'upl'
    ) -> pd.DataFrame:
        """
        Add beyond_limits flag column.

        Returns -1 for below LPL, 1 for above UPL, 0 for in control.

        Pure function: doesn't modify input.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with control limits
        value_col : str
            Column name containing values to check
        lpl_col : str, default 'lpl'
            Column name for lower process limit
        upl_col : str, default 'upl'
            Column name for upper process limit

        Returns
        -------
        pd.DataFrame
            Data with added 'beyond_limits' column
        """
        result = df.copy()

        result['beyond_limits'] = result.apply(
            lambda row: detect_beyond_limits(
                x=row[value_col],
                upl=row[upl_col],
                lpl=row[lpl_col]
            ),
            axis=1
        )

        return result

    def _determine_n_to_use(self, df: pd.DataFrame, n_col: str = 'n') -> tuple[str, int]:
        """
        Determine if subgroup sizes are constant or variable.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with subgroup size column
        n_col : str, default 'n'
            Column name containing subgroup sizes

        Returns
        -------
        tuple[str, int]
            (n_to_use, max_n) where:
            - n_to_use: 'N' if constant sizes, 'n' if variable
            - max_n: maximum subgroup size

        Examples
        --------
        >>> n_to_use, max_n = self._determine_n_to_use(out)
        >>> # Use max_n for statistics, n_to_use for per-row calculations
        """
        n_max = df[n_col].max()
        n_to_use = "N" if df[n_col].eq(n_max).all() else "n"

        logger.debug(
            'Analysis using %s for calculations (Scenario: %s)',
            n_to_use,
            1 if n_to_use == "N" else 2
        )

        return n_to_use, n_max

    def _resolve_by_grouping(
        self,
        value_col: str
    ) -> tuple[list[str], str | None]:
        """
        Resolve groupby columns and pre-calculated Ybar column based on `by` parameter.

        The `by` parameter controls aggregation level for Xbar/S charts.
        When possible, we use pre-calculated Ybar columns from the analytic dataset.

        Parameters
        ----------
        value_col : str
            Column being charted (response_var or residual column)

        Returns
        -------
        tuple[list[str], str | None]
            (groupby_cols, ybar_col) where:
            - groupby_cols: columns to group by (empty list for collapse all)
            - ybar_col: pre-calculated Ybar column to use, or None if must aggregate

        Examples
        --------
        >>> groupby_cols, ybar_col = self._resolve_by_grouping('y')
        >>> # by=[] -> ([], 'Ybar') - collapse all, use grand mean
        >>> # by=['factor1','factor2'] -> (['rsg'], 'Ybar_k') - use factor means
        """
        spec = self.spec
        by = self.request.by
        # Normalize tuple to list for internal comparison
        by = list(by) if by is not None else None
        rsg_vars = spec.rsg_vars_list
        time_var = spec.time_var

        # Determine if we're charting response (can use pre-calculated) or residual
        is_response = value_col == spec.response_var

        # Xbar/S: by=[] normalizes to by=None (Kt-level aggregation).
        # Imr/R handles by=[] directly in _calculate_imr() as single-stream.
        # This split is intentional — `by` means "aggregate by" for Xbar/S
        # and "stratify by" for Imr/R, matching each chart type's semantics.
        if by == []:
            by = None

        # Default: by=None means use cell_key level (factors + time)
        if by is None:
            if time_var and rsg_vars:
                # Factors + time (Kt level) - matches cell_key
                groupby_cols = rsg_vars + [time_var]
                ybar_col = 'Ybar_kt' if is_response else None
            elif rsg_vars:
                # Factors only (no time)
                groupby_cols = [spec.rsg_var_name]
                ybar_col = 'Ybar_k' if is_response else None
            elif time_var:
                # Time only (no factors)
                groupby_cols = [time_var]
                ybar_col = 'Ybar_t' if is_response else None
            else:
                # No grouping
                groupby_cols = []
                ybar_col = None
            return groupby_cols, ybar_col

        # Check if by matches known aggregation levels for Ybar optimization
        by_set = set(by)
        rsg_set = set(rsg_vars)

        # by == all factors (rsg_key level) -> use Ybar_k optimization
        # Only if order matches rsg_vars; otherwise preserve user's order
        if by_set == rsg_set and list(by) == rsg_vars:
            groupby_cols = [spec.rsg_var_name]
            ybar_col = 'Ybar_k' if is_response else None
            return groupby_cols, ybar_col

        # by == [time_var] only -> use Ybar_t
        if by_set == {time_var}:
            groupby_cols = [time_var]
            ybar_col = 'Ybar_t' if is_response else None
            return groupby_cols, ybar_col

        # by == all factors + time (cell_key level) -> use Ybar_kt
        cell_key_vars = rsg_set | ({time_var} if time_var else set())
        if by_set == cell_key_vars:
            # Group by all factors + time
            groupby_cols = rsg_vars + [time_var] if time_var else rsg_vars
            ybar_col = 'Ybar_kt' if is_response else None
            return groupby_cols, ybar_col

        # Partial subset - must aggregate at runtime
        # Use the by columns directly
        groupby_cols = list(by)
        return groupby_cols, None

    def _calculate_lane_boundaries(
        self,
        df: pd.DataFrame,
        collapsed_vars: list[str]
    ) -> list[dict]:
        """
        Calculate lane boundaries where collapsed factors change.

        Lane boundaries are positions in the data where a factor that was
        "collapsed" (not in `by`) changes value. These are rendered as
        vertical dashed lines on IMR charts.

        Parameters
        ----------
        df : pd.DataFrame
            Data in display order (already sorted)
        collapsed_vars : list[str]
            Variables that were collapsed (not in `by`)

        Returns
        -------
        list[dict]
            List of boundary dicts with 'position' (x index) and 'label' (factor value)

        Examples
        --------
        >>> # If by=['factor1'] and data has factor1='A' with factor2 changing X→Y
        >>> boundaries = self._calculate_lane_boundaries(df, ['factor2'])
        >>> # Returns: [{'position': 5, 'label': 'Y', 'variable': 'factor2'}]
        """
        if not collapsed_vars:
            return []

        boundaries = []

        # Create a combined key for all collapsed variables
        if len(collapsed_vars) == 1:
            combined = df[collapsed_vars[0]].astype(str)
        else:
            combined = df[collapsed_vars].astype(str).agg('_'.join, axis=1)

        # Find where the combined key changes
        changes = combined != combined.shift(1)

        # Get positions (skip first row - it's always a "change" from NaN)
        change_positions = df.index[changes].tolist()
        if change_positions and change_positions[0] == df.index[0]:
            change_positions = change_positions[1:]

        # Build boundary info
        for pos in change_positions:
            idx = df.index.get_loc(pos)
            label = combined.loc[pos]
            boundaries.append({
                'position': idx,  # 0-based position in the sorted data
                'label': label,
                'variables': collapsed_vars
            })

        return boundaries

    def _build_output_columns(
        self,
        df: pd.DataFrame,
        value_cols: list[str]
    ) -> pd.DataFrame:
        """
        Select output columns with appropriate time/grouping columns.

        Handles the pattern:
        - If has_time and has_grouping: [time, rsg, ...values, obs_id]
        - If has_time only: [time, ...values, obs_id]
        - If has_grouping only: [x, rsg, ...values, obs_id] (with generated x)
        - Otherwise: [x, ...values, obs_id] (with generated x)

        Always includes obs_id (if available) for traceability of violations.

        Parameters
        ----------
        df : pd.DataFrame
            Input data
        value_cols : list[str]
            Columns to keep (e.g., ['mean', 'lpl', 'upl', 'beyond_limits'])

        Returns
        -------
        pd.DataFrame
            Data with selected columns, rounded

        Notes
        -----
        The obs_id column enables tracing violations back to original data:
        - Find violations: df[df['beyond_limits'] != 0]
        - Get obs_ids: violation_ids = df['obs_id'].tolist()
        - Trace to raw data: raw_df[raw_df['obs_id'].isin(violation_ids)]

        Examples
        --------
        >>> out = self._build_output_columns(
        ...     df=out,
        ...     value_cols=[spec.response_var, 'mean', 'lpl', 'upl', 'beyond_limits']
        ... )
        """
        result = df.copy()
        cols_to_keep = value_cols.copy()

        if self.spec.has_time:
            if self.spec.has_grouping:
                cols_to_keep.insert(0, self.spec.rsg_var_name)
                cols_to_keep.insert(0, self.spec.time_var)
            else:
                cols_to_keep.insert(0, self.spec.time_var)
        else:
            # No explicit time variable specified - use implicit ordering
            if self.spec.has_grouping:
                # Per-group sequence number for grouped data
                result['x'] = result.groupby(self.spec.rsg_var_name, observed=True).cumcount() + 1
                cols_to_keep.insert(0, self.spec.rsg_var_name)
                cols_to_keep.insert(0, 'x')
            else:
                # Use obs_id as implicit time for single condition over time (SDS 4)
                # Rationale: Wheeler's IMR assumes temporal ordering, and obs_id
                # provides that ordering from the original data sequence.
                # See: ProcessBehavior.formulate() docstring for full explanation.
                cols_to_keep.insert(0, 'obs_id')

        # Always include obs_id for traceability (if available)
        # This enables linking violations back to original data
        if 'obs_id' in result.columns and 'obs_id' not in cols_to_keep:
            cols_to_keep.append('obs_id')

        return result[cols_to_keep].round(self.spec.round_to)

    # =========================================================================
    # Chart Calculation Methods (Strategy Pattern)
    # =========================================================================

    def _calculate_xbar(
        self,
        value_col: str = None,
        _return_intermediates: bool = False
    ) -> dict:
        """
        Calculate Xbar (mean) chart statistics.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
            Pass a residual column (R2, R5, etc.) for residual charts.
        _return_intermediates : bool, optional
            If True, includes '_intermediates' key with values needed by
            _calculate_s() to avoid redundant computation. Used internally
            by _calculate_xbar_s() for paired chart calculation.

        Returns
        -------
        dict
            Chart data: {'Xbar': {'data': df, 'statistics': dict, 'metadata': dict}}
            If _return_intermediates=True, also includes '_intermediates' key.

        Logic moved from Xbar.calculate_statistics()
        """
        spec = self.spec
        if value_col is None:
            value_col = spec.response_var

        result = {}
        statistics = {}
        df = self.ads.analysis_dataset.copy()

        logger.debug('In calculate statistics XbarS')
        logger.debug('Dataframe has columns: %s', df.columns.to_list())
        logger.debug('Dataframe head:\n%s', df.head(10))

        # Resolve grouping based on `by` parameter
        groupby_cols, ybar_col = self._resolve_by_grouping(value_col)

        # Calculate grand mean (center line) - use pre-calculated if available
        _Ybar = df['Ybar'].iloc[0] if ybar_col == 'Ybar' and 'Ybar' in df.columns else df[value_col].mean()

        # Handle by=[] (collapse all) - single point chart
        if groupby_cols == []:
            # Single aggregation across all data
            _S = df[value_col].std()
            _N = len(df)
            out = pd.DataFrame({
                'group': ['All'],
                'xbar': [_Ybar],
                's': [_S],
                'n': [_N],
                'N': [_N]
            })
        else:
            # Group by specified columns
            out = df.groupby(groupby_cols, as_index=False, observed=True).agg(
                s=pd.NamedAgg(column=value_col, aggfunc="std"),
                mean=pd.NamedAgg(column=value_col, aggfunc="mean"),
                # Count on response_var (not value_col) to avoid NaN issues with residuals
                n=pd.NamedAgg(column=spec.response_var, aggfunc="count")
            )

            # If we have pre-calculated Ybar, use it for xbar values
            if ybar_col and ybar_col in df.columns:
                # Get unique Ybar values per group
                ybar_by_group = df.groupby(groupby_cols, observed=True)[ybar_col].first().reset_index()
                out = out.merge(ybar_by_group, on=groupby_cols, how='left')
                out['xbar'] = out[ybar_col]
                out = out.drop(columns=[ybar_col, 'mean'], errors='ignore')
            else:
                # Rename columns for consistency: mean→xbar (values)
                out = out.rename(columns={'mean': 'xbar'})

            _N = out['n'].max()
            out['N'] = _N

        # Filter out groups with n=1 (can't compute c4 for variance estimation)
        # Xbar charts require n >= 2 for within-group variance
        # Note: SDS 2 (all n=1) is already handled upstream - Xbar not in valid_charts
        mask_n1 = out['n'].eq(1)
        if mask_n1.any():
            n_filtered = mask_n1.sum()
            logger.info(f"Filtered {n_filtered} subgroup(s) with n=1 from Xbar calculation")
            out = out[~mask_n1].copy()

        # Handle case where no subgroups have >1 observation
        if out.shape[0] == 0:
            raise ValueError("All subgroups have 1 or less observations!")

        # Use grand mean as center line (not mean of subgroup means)
        _Xbar = _Ybar
        _S = out["s"].mean()
        _N = out['n'].max()
        if 'N' not in out.columns:
            out['N'] = _N

        # Determine if subgroup sizes are constant or variable
        n_to_use, n_max = self._determine_n_to_use(out)

        # CALCULATE XBAR
        xbar = out.copy()
        xbar['center'] = _Xbar  # Add center column for Xbar chart
        xbar[['lpl', 'upl']] = xbar.apply(
            lambda row: calculate_limits(
                mean=row['center'],
                sd=_S,
                N=row[n_to_use],
                limits_type='Xbar',
                round_to=spec.round_to
            ), axis=1
        )

        # Detect beyond limits signals
        xbar = self._add_beyond_limits_flag(xbar, value_col='xbar')
        xbar = xbar.round(spec.round_to)

        statistics['center'] = round(_Xbar, spec.round_to)
        if n_to_use == "N":
            statistics['N'] = _N
            statistics['upl'] = xbar['upl'].max()
            statistics['lpl'] = xbar['lpl'].max()
        else:
            variable_stats = 'Varies'
            statistics['N'] = variable_stats
            statistics['lpl'] = variable_stats
            statistics['upl'] = variable_stats

        # Determine the grouping column for output
        by = self.request.by
        if by is not None and len(by) > 0:
            # Explicit by list - create 'subgroup' column
            if len(groupby_cols) == 1:
                xbar['subgroup'] = xbar[groupby_cols[0]].astype(str)
            else:
                xbar['subgroup'] = xbar[groupby_cols].astype(str).agg('_'.join, axis=1)
            group_col = 'subgroup'
        elif len(groupby_cols) > 1:
            # by=None - create 'group' from multiple columns (original behavior)
            xbar['group'] = xbar[groupby_cols].astype(str).agg('_'.join, axis=1)
            group_col = 'group'
        elif groupby_cols:
            # Single groupby column
            group_col = groupby_cols[0]
        else:
            group_col = None

        cols_to_keep = [group_col, 'xbar', 'center', 'lpl', 'upl', 'beyond_limits']
        cols_to_keep = [c for c in cols_to_keep if c in xbar.columns]
        xbar = xbar[cols_to_keep]
        result['Xbar'] = {
            'data': xbar,
            'statistics': statistics,
            'metadata': {
                'chart_type': 'Xbar',
                'value_col': 'xbar',
                'center_col': 'center'
            }
        }

        # Optionally include intermediates for paired calculation
        if _return_intermediates:
            result['_intermediates'] = {
                '_S': _S,
                'n_to_use': n_to_use,
                'n_max': n_max,
                'groupby_cols': groupby_cols,
                'group_col': group_col,
                'out': out,  # Pre-aggregated DataFrame with s, n, mean columns
            }

        return result

    def _calculate_s(
        self,
        value_col: str = None,
        _precomputed: dict = None
    ) -> dict:
        """
        Calculate S (standard deviation) chart statistics.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
            Pass a residual column (R2, R5, etc.) for residual charts.
        _precomputed : dict, optional
            Pre-computed values from _calculate_xbar() to avoid redundant
            computation. Used internally by _calculate_xbar_s() for paired
            chart calculation. Keys: '_S', 'n_to_use', 'n_max', 'groupby_cols',
            'group_col', 'out'.

        Returns
        -------
        dict
            Chart data: {'S': {'data': df, 'statistics': dict, 'metadata': dict}}

        Logic moved from calculate_statistics_S()
        """
        spec = self.spec
        if value_col is None:
            value_col = spec.response_var

        # Use precomputed values if available (from _calculate_xbar_s)
        if _precomputed is not None:
            _S = _precomputed['_S']
            n_to_use = _precomputed['n_to_use']
            n_max = _precomputed['n_max']
            groupby_cols = _precomputed['groupby_cols']
            group_col = _precomputed['group_col']
            out = _precomputed['out'].copy()
        else:
            # Independent calculation (existing logic)
            df = self.ads.analysis_dataset.copy()

            # Resolve grouping based on `by` parameter
            groupby_cols, _ = self._resolve_by_grouping(value_col)

            # Handle by=[] (collapse all) - single point chart
            if groupby_cols == []:
                _S = df[value_col].std()
                _N = len(df)
                out = pd.DataFrame({
                    'group': ['All'],
                    's': [_S],
                    'n': [_N],
                    'N': [_N]
                })
            else:
                out = df.groupby(groupby_cols, as_index=False, observed=True).agg(
                    s=pd.NamedAgg(column=value_col, aggfunc="std"),
                    # Count on response_var (not value_col) to avoid NaN issues with residuals
                    n=pd.NamedAgg(column=spec.response_var, aggfunc="count"),
                )

                # remove groups with a single observation
                mask = out['n'].eq(1)
                out = out[~mask]

                out['N'] = out['n'].max()

            # Calculate center line (mean of subgroup std devs)
            _S = out["s"].mean()

            # Determine if subgroup sizes are constant or variable
            n_to_use, n_max = self._determine_n_to_use(out)

            # Determine group_col
            by = self.request.by
            if by is not None and len(by) > 0:
                if len(groupby_cols) == 1:
                    out['subgroup'] = out[groupby_cols[0]].astype(str)
                else:
                    out['subgroup'] = out[groupby_cols].astype(str).agg('_'.join, axis=1)
                group_col = 'subgroup'
            elif len(groupby_cols) > 1:
                out['group'] = out[groupby_cols].astype(str).agg('_'.join, axis=1)
                group_col = 'group'
            elif groupby_cols:
                group_col = groupby_cols[0]
            else:
                group_col = None

        # Calculate S chart (common path for both precomputed and independent)
        out['center'] = _S
        out['groups'] = out["n"].count()

        # Add limits columns
        out[['lpl', 'upl']] = out.apply(
            lambda row: calculate_limits(
                mean=0,
                sd=row['center'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        # Detect beyond limits signals
        out = self._add_beyond_limits_flag(out, value_col='s')

        # Apply grouping column transformation if needed and not already done
        if _precomputed is None:
            # Already handled in independent path above
            pass
        else:
            # For precomputed path, apply group_col transformation
            by = self.request.by
            if by is not None and len(by) > 0 and 'subgroup' not in out.columns:
                if len(groupby_cols) == 1:
                    out['subgroup'] = out[groupby_cols[0]].astype(str)
                else:
                    out['subgroup'] = out[groupby_cols].astype(str).agg('_'.join, axis=1)
            elif len(groupby_cols) > 1 and group_col == 'group' and 'group' not in out.columns:
                out['group'] = out[groupby_cols].astype(str).agg('_'.join, axis=1)

        cols_to_keep = [group_col, 's', 'center', 'lpl', 'upl', 'beyond_limits']
        cols_to_keep = [c for c in cols_to_keep if c in out.columns]
        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        # Build statistics
        _S_stat = out['center'].iloc[0] if len(out) > 0 else None

        statistics = {'center': round(_S_stat, spec.round_to) if _S_stat else None}
        if n_to_use == "N":
            statistics['N'] = n_max
            statistics['upl'] = out['upl'].max()
            statistics['lpl'] = out['lpl'].max()
        else:
            variable_stats = 'Varies'
            statistics['N'] = variable_stats
            statistics['lpl'] = variable_stats
            statistics['upl'] = variable_stats

        return {
            'S': {
                'data': out,
                'statistics': statistics,
                'metadata': {
                    'chart_type': 'S',
                    'value_col': 's',
                    'center_col': 'center'
                }
            }
        }

    def _calculate_xbar_s(self, value_col: str = None) -> dict:
        """
        Calculate Xbar and S charts together (Wheeler methodology).

        This method ensures that when paired=True, both charts are calculated
        efficiently using shared intermediate values. The S chart uses the
        same aggregated data computed for Xbar, avoiding redundant calculation.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.

        Returns
        -------
        dict
            Combined chart data: {'Xbar': {...}, 'S': {...}}
        """
        # Calculate Xbar with intermediates for S reuse
        xbar_result = self._calculate_xbar(value_col, _return_intermediates=True)

        # Extract intermediates and remove from result
        intermediates = xbar_result.pop('_intermediates')

        # Calculate S using precomputed values
        s_result = self._calculate_s(value_col, _precomputed=intermediates)

        # Combine results
        return {**xbar_result, **s_result}

    def _calculate_mr_chart(
        self,
        mr_spec: _MRChartSpec,
        value_col: str = None,
        _return_intermediates: bool = False,
        _precomputed: dict = None,
    ) -> dict:
        """
        Shared pipeline for MR-family charts (IMR and R).

        The IMR and R charts share >85% of their calculation logic. The behavioral
        differences are encoded in ``mr_spec`` — no boolean flags needed.

        Parameters
        ----------
        mr_spec : _MRChartSpec
            Chart behavior specification (_IMR_SPEC or _R_SPEC).
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
        _return_intermediates : bool, optional
            If True, includes '_intermediates' key for paired-mode reuse.
        _precomputed : dict, optional
            Pre-computed intermediates from a prior IMR call (paired mode).
            When provided, skips data preparation and uses these values directly.

        Returns
        -------
        dict
            Chart data: {chart_type: {'data': df, 'statistics': dict, 'metadata': dict}}
        """
        spec = self.spec
        if value_col is None:
            value_col = spec.response_var

        logger.debug('In calculate statistics %s', mr_spec.chart_type)

        # --- Determine stratification (shared logic) ---
        by = list(self.request.by) if self.request.by is not None else None
        rsg_vars = spec.rsg_vars_list

        if by is None:
            stratify_by = [spec.rsg_var_name] if rsg_vars else []
            collapsed_factors = []
        elif by == []:
            stratify_by = []
            collapsed_factors = list(rsg_vars)
        else:
            stratify_by = list(by)
            collapsed_factors = [v for v in rsg_vars if v not in by]

        is_stratified = len(stratify_by) > 0

        # --- Resolve plot column ---
        plot_col = value_col if mr_spec.plot_col == 'raw' else 'mr'

        if is_stratified:
            return self._calculate_mr_chart_stratified(
                mr_spec, value_col, plot_col,
                stratify_by, collapsed_factors,
                _return_intermediates, _precomputed,
            )
        else:
            return self._calculate_mr_chart_ungrouped(
                mr_spec, value_col, plot_col,
                collapsed_factors,
                _return_intermediates, _precomputed,
            )

    def _calculate_mr_chart_from_precomputed(
        self,
        mr_spec: _MRChartSpec,
        plot_col: str,
        precomputed: dict,
    ) -> dict:
        """Build an MR-family chart reusing intermediates from a paired IMR call."""
        spec = self.spec
        out = precomputed['out'].copy()
        grouped = precomputed['grouped'].copy()
        stratify_col = precomputed['stratify_col']
        stratify_by = precomputed['stratify_by']
        strata = precomputed['strata']
        imr_lane_boundaries = precomputed['lane_boundaries']
        extra_cols = precomputed['extra_cols']

        # For R chart: compute R-specific limits from IMR's grouped data
        # grouped has columns: stratify_col, center (=mean), mR, lpl, upl (IMR limits)
        r_grouped = grouped.rename(columns={'center': 'imr_center', 'mR': 'center'})

        r_lims = r_grouped.apply(
            lambda row: calculate_limits(
                mean=0, sd=0, N=0,
                mR=row['center'],
                limits_type=mr_spec.limits_type,
                round_to=spec.round_to,
            ),
            axis=1,
        )

        r_grouped = _join_limits(r_grouped, r_lims, rsuffix='_r')

        # Resolve lpl/upl column names (may have suffix if IMR lpl/upl exist)
        if 'lpl_r' in r_grouped.columns:
            r_grouped = r_grouped.rename(columns={'lpl_r': 'r_lpl', 'upl_r': 'r_upl'})
            r_lpl_col, r_upl_col = 'r_lpl', 'r_upl'
        else:
            r_lpl_col, r_upl_col = 'lpl', 'upl'

        # Merge R limits to the data (drop IMR-specific columns first)
        r_out = out.drop(columns=['center', 'lpl', 'upl', 'beyond_limits'], errors='ignore')
        r_merge_data = r_grouped[[stratify_col, 'center', r_lpl_col, r_upl_col]].rename(
            columns={r_lpl_col: 'lpl', r_upl_col: 'upl'}
        )
        r_out = r_out.merge(r_merge_data, on=stratify_col, how='left', validate='many_to_one')

        # R chart drops first observation per stratum (NaN moving range)
        if mr_spec.drops_first_mr:
            r_out = r_out.dropna(subset=['mr'])

        # Detect beyond limits
        r_out = self._add_beyond_limits_flag(r_out, value_col=plot_col)

        # Build statistics
        chart_statistics = {}
        for stratum in strata:
            row = r_grouped[r_grouped[stratify_col] == stratum].iloc[0]
            chart_statistics[stratum] = {
                'center': round(row['center'], spec.round_to),
                'lpl': round(row[r_lpl_col], spec.round_to),
                'upl': round(row[r_upl_col], spec.round_to),
            }

        # Format output
        chart_out = self._build_output_columns(
            df=r_out,
            value_cols=extra_cols + [plot_col, 'center', 'lpl', 'upl', 'beyond_limits'],
        )

        # Adjust lane boundaries for R chart (first row dropped per stratum)
        lane_boundaries = {}
        if imr_lane_boundaries:
            for stratum, boundaries in imr_lane_boundaries.items():
                adjusted = [
                    {**b, 'position': b['position'] + mr_spec.lane_boundary_offset}
                    for b in boundaries
                    if b['position'] + mr_spec.lane_boundary_offset >= 0
                ]
                if adjusted:
                    lane_boundaries[stratum] = adjusted

        return {
            mr_spec.chart_type: {
                'data': chart_out,
                'statistics': chart_statistics,
                'metadata': {
                    'chart_type': mr_spec.chart_type,
                    'value_col': plot_col,
                    'center_col': 'center',
                    'stratified': True,
                    'lane_boundaries': lane_boundaries if lane_boundaries else None,
                    'stratify_by': list(stratify_by),
                },
                'strata': strata,
            },
        }

    def _calculate_mr_chart_stratified(
        self,
        mr_spec: _MRChartSpec,
        value_col: str,
        plot_col: str,
        stratify_by: list[str],
        collapsed_factors: list[str],
        _return_intermediates: bool,
        _precomputed: dict | None,
    ) -> dict:
        """Stratified path for _calculate_mr_chart."""
        spec = self.spec

        if _precomputed is not None and _precomputed['is_stratified']:
            return self._calculate_mr_chart_from_precomputed(
                mr_spec, plot_col, _precomputed,
            )

        # --- Independent calculation (no precomputed values) ---
        result = {}
        out = self.ads.analysis_dataset.copy()
        logger.debug('Dataframe has columns: %s', out.columns.to_list())

        # Determine stratification column
        if stratify_by == [spec.rsg_var_name]:
            stratify_col = spec.rsg_var_name
        elif len(stratify_by) == 1:
            stratify_col = stratify_by[0]
        else:
            out['_stratify_key'] = out[stratify_by].apply(tuple, axis=1)
            stratify_col = '_stratify_key'

        # Canonical sort
        out = out.sort_values('sort_key', kind='stable')

        # Moving range per stratum
        out['mr'] = out.groupby(stratify_col, sort=False, observed=True)[value_col].diff().abs()

        # Aggregate per stratum
        agg = {}
        if value_col in out.columns:
            agg['mean'] = (value_col, 'mean')
        if 'mr' in out.columns:
            agg['mR'] = ('mr', 'mean')

        if not agg:
            raise RuntimeError(
                f"{mr_spec.chart_type} aggregation spec is empty. "
                f"Have columns: {out.columns.tolist()}, value_col={value_col!r}"
            )

        grouped = (
            out.groupby(stratify_col, sort=False, observed=True)
            .agg(**agg)
            .reset_index()
        )

        # Compute limits per group
        # For IMR: center=mean, limits based on mean ± E2*mR
        # For R:   center=mR, limits based on 0 to D4*mR
        if mr_spec.center_source == 'mean':
            lims = grouped.apply(
                lambda row: calculate_limits(
                    mean=row['mean'],
                    sd=0, N=0,
                    mR=(row['mR'] if pd.notna(row['mR']) else 0.0),
                    limits_type=mr_spec.limits_type,
                    round_to=spec.round_to,
                ),
                axis=1,
            )
        else:  # center_source == 'mR'
            lims = grouped.apply(
                lambda row: calculate_limits(
                    mean=0, sd=0, N=0,
                    mR=row['mR'],
                    limits_type=mr_spec.limits_type,
                    round_to=spec.round_to,
                ),
                axis=1,
            )

        # Normalize/join lpl/upl
        grouped = _join_limits(grouped, lims)

        # Set center column based on chart type
        if mr_spec.center_source == 'mean':
            grouped = grouped.rename(columns={'mean': 'center'})
        else:
            grouped['center'] = grouped['mR']

        # Merge limits back to row-level data
        out = out.merge(
            grouped[[stratify_col, 'center', 'mR', 'lpl', 'upl']],
            on=stratify_col, how='left', validate='many_to_one',
        )

        # R chart: drop first observation per stratum (NaN moving range)
        if mr_spec.drops_first_mr:
            out = out.dropna(subset=['mr'])

        # Detect beyond limits
        out = self._add_beyond_limits_flag(out, value_col=plot_col)

        # Strata list
        strata = grouped[stratify_col].tolist()

        # Lane boundaries per stratum
        lane_boundaries = {}
        if collapsed_factors:
            for stratum in strata:
                stratum_data = out[out[stratify_col] == stratum].reset_index(drop=True)
                boundaries = self._calculate_lane_boundaries(stratum_data, collapsed_factors)
                if boundaries:
                    lane_boundaries[stratum] = boundaries

        # Build statistics nested by stratum
        chart_statistics = {}
        for stratum in strata:
            row = grouped[grouped[stratify_col] == stratum].iloc[0]
            chart_statistics[stratum] = {
                'center': round(row['center'], spec.round_to),
                'lpl': round(row['lpl'], spec.round_to),
                'upl': round(row['upl'], spec.round_to),
            }

        # Format output
        extra_cols = [c for c in stratify_by
                      if c != spec.rsg_var_name and c != spec.time_var]
        chart_out = self._build_output_columns(
            df=out,
            value_cols=extra_cols + [plot_col, 'center', 'lpl', 'upl', 'beyond_limits'],
        )

        result[mr_spec.chart_type] = {
            'data': chart_out,
            'statistics': chart_statistics,
            'metadata': {
                'chart_type': mr_spec.chart_type,
                'value_col': plot_col,
                'center_col': 'center',
                'stratified': True,
                'lane_boundaries': lane_boundaries if lane_boundaries else None,
                'stratify_by': list(stratify_by),
            },
            'strata': strata,
        }

        # Optionally include intermediates for paired calculation
        if _return_intermediates:
            result['_intermediates'] = {
                'out': out,
                'grouped': grouped,
                'stratify_col': stratify_col,
                'stratify_by': stratify_by,
                'strata': strata,
                'lane_boundaries': lane_boundaries,
                'collapsed_factors': collapsed_factors,
                'extra_cols': extra_cols,
                'is_stratified': True,
            }

        return result

    def _calculate_mr_chart_ungrouped(
        self,
        mr_spec: _MRChartSpec,
        value_col: str,
        plot_col: str,
        collapsed_factors: list[str],
        _return_intermediates: bool,
        _precomputed: dict | None,
    ) -> dict:
        """Ungrouped (single-stream) path for _calculate_mr_chart."""
        spec = self.spec
        result = {}

        if _precomputed is not None and not _precomputed['is_stratified']:
            # Reuse intermediates from paired IMR call
            out = _precomputed['out'].copy()
            mR = _precomputed['mR']
            imr_lane_boundaries = _precomputed['lane_boundaries']

            # R chart: drop first observation (NaN moving range)
            if mr_spec.drops_first_mr:
                out = out.dropna(subset=['mr'])

            # Compute R-specific limits
            r_lims = calculate_limits(
                mean=0, sd=0, N=0, mR=mR,
                limits_type=mr_spec.limits_type, round_to=spec.round_to,
            )
            out['center'] = mR
            out['lpl'] = r_lims['lpl']
            out['upl'] = r_lims['upl']

            # Re-detect beyond limits for R chart
            out = out.drop(columns=['beyond_limits'], errors='ignore')
            out = self._add_beyond_limits_flag(out, value_col=plot_col)

            # Format output
            chart_out = self._build_output_columns(
                df=out,
                value_cols=[plot_col, 'center', 'lpl', 'upl', 'beyond_limits'],
            )

            chart_statistics = {
                'center': round(mR, spec.round_to),
                'lpl': round(r_lims['lpl'], spec.round_to),
                'upl': round(r_lims['upl'], spec.round_to),
            }

            # Adjust lane boundaries
            r_lane_boundaries = None
            if imr_lane_boundaries:
                r_lane_boundaries = []
                for b in imr_lane_boundaries:
                    new_pos = b['position'] + mr_spec.lane_boundary_offset
                    if new_pos >= 0:
                        r_lane_boundaries.append({**b, 'position': new_pos})
                if not r_lane_boundaries:
                    r_lane_boundaries = None

            result[mr_spec.chart_type] = {
                'data': chart_out,
                'statistics': chart_statistics,
                'metadata': {
                    'chart_type': mr_spec.chart_type,
                    'value_col': plot_col,
                    'center_col': 'center',
                    'lane_boundaries': r_lane_boundaries,
                },
            }

            return result

        # --- Independent calculation ---
        out = self.ads.analysis_dataset.copy()
        out = out.sort_values('sort_key', kind='stable')
        out = out.reset_index(drop=True)

        # Lane boundaries before any row drops
        lane_boundaries = None
        if collapsed_factors:
            lane_boundaries = self._calculate_lane_boundaries(out, collapsed_factors)

        # Moving range
        out['mr'] = out[value_col].diff().abs()
        mR = out['mr'].mean()
        mean_ = out[value_col].mean()

        # R chart: drop first observation
        if mr_spec.drops_first_mr:
            out = out.dropna(subset=['mr'])

        # Compute limits
        if mr_spec.center_source == 'mean':
            center_val = mean_
            lims = calculate_limits(
                mean=mean_, sd=0, N=0, mR=mR,
                limits_type=mr_spec.limits_type, round_to=spec.round_to,
            )
        else:
            center_val = mR
            lims = calculate_limits(
                mean=0, sd=0, N=0, mR=mR,
                limits_type=mr_spec.limits_type, round_to=spec.round_to,
            )

        out['center'] = center_val
        out['mR'] = mR
        out['lpl'] = lims['lpl']
        out['upl'] = lims['upl']

        # Detect beyond limits
        out = self._add_beyond_limits_flag(out, value_col=plot_col)

        # Format output
        chart_out = self._build_output_columns(
            df=out,
            value_cols=[plot_col, 'center', 'lpl', 'upl', 'beyond_limits'],
        )

        chart_statistics = {
            'center': round(center_val, spec.round_to),
            'lpl': round(lims['lpl'], spec.round_to),
            'upl': round(lims['upl'], spec.round_to),
        }

        # Adjust lane boundaries for chart type
        if lane_boundaries and mr_spec.lane_boundary_offset != 0:
            adjusted = []
            for b in lane_boundaries:
                new_pos = b['position'] + mr_spec.lane_boundary_offset
                if new_pos >= 0:
                    adjusted.append({**b, 'position': new_pos})
            lane_boundaries = adjusted if adjusted else None

        result[mr_spec.chart_type] = {
            'data': chart_out,
            'statistics': chart_statistics,
            'metadata': {
                'chart_type': mr_spec.chart_type,
                'value_col': plot_col,
                'center_col': 'center',
                'lane_boundaries': lane_boundaries,
            },
        }

        # Optionally include intermediates for paired calculation
        if _return_intermediates:
            result['_intermediates'] = {
                'out': out,
                'mR': mR,
                'lane_boundaries': lane_boundaries,
                'is_stratified': False,
            }

        return result

    def _calculate_imr(
        self,
        value_col: str = None,
        _return_intermediates: bool = False,
    ) -> dict:
        """
        Calculate IMR (Individual) chart statistics.

        Delegates to _calculate_mr_chart with _IMR_SPEC.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
        _return_intermediates : bool, optional
            If True, includes '_intermediates' key for paired-mode reuse.

        Returns
        -------
        dict
            Chart data: {'Imr': {'data': df, 'statistics': dict, 'metadata': dict}}
        """
        return self._calculate_mr_chart(
            _IMR_SPEC, value_col=value_col,
            _return_intermediates=_return_intermediates,
        )

    def _calculate_r(
        self,
        value_col: str = None,
        _precomputed: dict = None,
    ) -> dict:
        """
        Calculate R (Range) chart statistics.

        Delegates to _calculate_mr_chart with _R_SPEC.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
        _precomputed : dict, optional
            Pre-computed intermediates from _calculate_imr() for paired mode.

        Returns
        -------
        dict
            Chart data: {'R': {'data': df, 'statistics': dict, 'metadata': dict}}
        """
        return self._calculate_mr_chart(
            _R_SPEC, value_col=value_col,
            _precomputed=_precomputed,
        )

    def _calculate_imr_r(self, value_col: str = None) -> dict:
        """
        Calculate Imr and R charts together (Wheeler methodology).

        This method ensures that when paired=True, both charts are calculated
        efficiently using shared intermediate values. The R chart uses the
        same data computed for Imr, avoiding redundant calculation.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.

        Returns
        -------
        dict
            Combined chart data: {'Imr': {...}, 'R': {...}}
        """
        # Calculate Imr with intermediates for R reuse
        imr_result = self._calculate_imr(value_col, _return_intermediates=True)

        # Extract intermediates and remove from result
        intermediates = imr_result.pop('_intermediates')

        # Calculate R using precomputed values
        r_result = self._calculate_r(value_col, _precomputed=intermediates)

        # Combine results
        return {**imr_result, **r_result}

    def _calculate_histogram(self, value_col: str = None) -> dict:
        """
        Calculate histogram data with mean/std statistics.

        Parameters
        ----------
        value_col : str, optional
            Column to use for histogram. Defaults to spec.response_var.
            Pass a residual column (R2, R5, etc.) for residual histograms.

        Returns
        -------
        dict
            Histogram data:
            - For unstratified: {'Histogram': {'data': df, 'statistics': {...}, 'metadata': {...}}}
            - For stratified: {'Histogram': {'data': df, 'statistics': {stratum: {...}}, 'strata': [...]}}

        Notes
        -----
        - by=[] → single histogram of all observations (default for Histogram chart)
        - by=['factor1'] → one histogram per factor1 level
        - Multi-column `by` uses tuple keys to avoid collision from string joining
        """
        spec = self.spec
        if value_col is None:
            value_col = self.request.value_col if self.request.value_col else spec.response_var

        data = self.ads.analysis_dataset.copy()
        values = data[value_col].dropna()

        # Calculate global statistics
        n = len(values)
        mean = values.mean() if n > 0 else float('nan')
        std = values.std() if n >= 2 else float('nan')

        # Handle stratification via `by` parameter
        by = list(self.request.by) if self.request.by is not None else []

        if len(by) > 0:
            # Stratified histogram - one per stratum
            # Use pandas groupby with list of columns (no collision risk)
            grouped = data.groupby(by, observed=True)

            # Build strata list - tuples for multi-key, values for single-key
            # Use tuple keys for multi-key to avoid collision (('A_B','C') != ('A','B_C'))
            strata = data[by[0]].unique().tolist() if len(by) == 1 else list(grouped.groups.keys())

            # Calculate per-stratum statistics
            per_stratum_stats = {}
            for stratum, group_df in grouped:
                # Unwrap single-element tuple for single-column groupby
                # groupby(['col']) returns ('value',) tuples, but strata uses scalar values
                key = stratum[0] if len(by) == 1 else stratum
                stratum_vals = group_df[value_col].dropna()
                stratum_n = len(stratum_vals)
                stratum_mean = stratum_vals.mean() if stratum_n > 0 else float('nan')
                stratum_std = stratum_vals.std() if stratum_n >= 2 else float('nan')
                per_stratum_stats[key] = {
                    'mean': round(stratum_mean, spec.round_to) if pd.notna(stratum_mean) else None,
                    'std': round(stratum_std, spec.round_to) if pd.notna(stratum_std) else None,
                    'n': stratum_n
                }

            # Keep value column and by columns for plotting
            output_cols = [value_col] + list(by)
            output_data = data[output_cols].copy()

            return {
                'Histogram': {
                    'data': output_data,
                    'statistics': per_stratum_stats,
                    'metadata': {
                        'chart_type': 'Histogram',
                        'value_col': value_col,
                        'bins': self.request.bins,
                        'stratified': True,
                        'stratify_by': list(by)
                    },
                    'strata': strata
                }
            }

        # Unstratified (by=[]) - full distribution
        output_data = data[[value_col]].copy()

        return {
            'Histogram': {
                'data': output_data,
                'statistics': {
                    'mean': round(mean, spec.round_to) if pd.notna(mean) else None,
                    'std': round(std, spec.round_to) if n >= 2 and pd.notna(std) else None,
                    'n': n
                },
                'metadata': {
                    'chart_type': 'Histogram',
                    'value_col': value_col,
                    'bins': self.request.bins
                }
            }
        }
