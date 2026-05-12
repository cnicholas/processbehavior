"""
Analysis - Chart calculation strategies for process behavior analysis.

This module provides the Analysis class which executes chart calculations using
the strategy pattern. It supports:
- Xbar and S charts (subgroup mean and variation)
- X charts (individual values)
- mR charts (moving range)

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
from .data_preparation import encode_rsg
from .exceptions import ChartNotAvailableError, ValidationError
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

    X and mR charts share >85% of their calculation pipeline. The differences
    are behavioral, not parametric — this dataclass encodes those differences
    so a single shared method can serve both chart types.
    """
    chart_type: str         # 'X' or 'mR'
    limits_type: str        # "XmR" or "R" — passed to calculate_limits()
    plot_col: str           # Which column to plot: 'raw' (use value_col) or 'mr'
    center_source: str      # What the center line represents: 'mean' or 'mR'
    drops_first_mr: bool    # False for X, True for mR
    lane_boundary_offset: int  # 0 for X, -1 for mR


_XMR_SPEC = _MRChartSpec(
    chart_type='X', limits_type='XmR',
    plot_col='raw', center_source='mean',
    drops_first_mr=False, lane_boundary_offset=0,
)
_R_SPEC = _MRChartSpec(
    chart_type='mR', limits_type='R',
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
    approach. All analysis types (Xbar, S, X, mR) are handled through internal
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
                raise ValidationError(
                    "sds is required when analysis_dataset is not provided. "
                    "SDS should be detected at the entry point (ProcessBehavior) "
                    "and passed to Analysis."
                )
            if df is None:
                raise ValidationError(
                    "df is required when analysis_dataset is not provided."
                )
            self.ads = AnalysisDataSet(df, spec, observed_sds=sds)

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
            chart_type = self.request.residual_chart_type or 'X'
            recentered = self.request.recentered

            # Determine column name
            col_prefix = 'RCR' if recentered else 'R'
            residual_num = residual[1:]  # Extract number from 'R2', 'R3', 'R10', etc.
            col_name = f'{col_prefix}{residual_num}'

            # Validate column exists
            if col_name not in self.ads.analysis_dataset.columns:
                available = [c for c in self.ads.analysis_dataset.columns
                            if c.startswith('R') or c == self.spec.response_var]
                raise ChartNotAvailableError(
                    f"Residual column '{col_name}' not found.\n"
                    f"Available columns: {available}\n"
                    f"This may indicate the analytical design state (ADS) doesn't support this residual type.",
                    chart=col_name,
                    available=available
                )

            # Validate chart type
            valid_chart_types = {'Xbar', 'S', 'X', 'mR', 'Histogram'}
            if chart_type not in valid_chart_types:
                raise ChartNotAvailableError(
                    f"Chart type '{chart_type}' not supported for residual charts.\n"
                    f"Valid types: {', '.join(sorted(valid_chart_types))}",
                    chart=chart_type,
                    available=sorted(valid_chart_types)
                )

            # Question answered by each residual
            questions = {
                'R2': 'Is within-subgroup variation stable?',
                'R3': 'Is there interaction between factors and time?',
                'R4': 'Does time have a significant effect?',
                'R5': 'Do factors have a significant effect?'
            }

            # Calculate using base method with value_col
            companion = self.request.companion
            if companion:
                residual_strategies = {
                    'Xbar': self._calculate_xbar_s,
                    'S': self._calculate_xbar_s,
                    'X': self._calculate_xmr_r,
                    'mR': self._calculate_xmr_r,
                    'Histogram': self._calculate_histogram,
                }
            else:
                residual_strategies = {
                    'Xbar': self._calculate_xbar,
                    'S': self._calculate_s,
                    'X': self._calculate_xmr,
                    'Histogram': self._calculate_histogram,
                }
            chart_data = residual_strategies[chart_type](value_col=col_name)
            chart_data = self._add_residual_metadata(
                chart_data, residual, recentered, questions
            )

        else:
            # Standard chart analysis
            # Check if companion charts requested (Xbar+S or X+mR together)
            companion = self.request.companion

            if companion:
                # Companion mode: return both charts regardless of which was requested
                strategies = {
                    'Xbar': self._calculate_xbar_s,
                    'S': self._calculate_xbar_s,
                    'X': self._calculate_xmr_r,  # Bundled X+mR
                    'mR': self._calculate_xmr_r,
                    'Histogram': self._calculate_histogram  # No companion for Histogram
                }
            else:
                # SRP-compliant mode: return only the requested chart
                strategies = {
                    'Xbar': self._calculate_xbar,
                    'S': self._calculate_s,
                    'X': self._calculate_xmr,  # SRP: X only
                    'mR': self._calculate_r,  # SRP: mR only
                    'Histogram': self._calculate_histogram
                }

            if self.analysis_type not in strategies:
                raise ChartNotAvailableError(
                    f'Analysis type {self.analysis_type} not supported! '
                    f'Valid types: {list(strategies.keys())}',
                    chart=self.analysis_type,
                    available=list(strategies.keys())
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

    def _residual_grain(self, value_col: str) -> list[str]:
        """Return the cell-grid columns at which to compute Bishop's grand mean
        for an Xbar chart on this column.

        Each VAS residual is recentered around a baseline computed at a specific
        grain (see `analysis_dataset.py:364-371` and `study.py:2009`). The
        canonical grand mean of an RCRk chart averages over that grain — equal
        weight per cell — yielding Bishop's unweighted center line.

        Grain table (None for non-residuals; falls through to other logic):
          R1, R2, R3 -> [rsg_var, time_var]   (full cell grid)
          R4         -> [rsg_var]             (time effect removed)
          R5, R6     -> [rsg_var]             (factor effects live at rsg level)

        Returns an empty list when the column isn't a residual or the spec
        lacks the required factors/time — caller falls back to `df.mean()`.
        """
        if value_col is None:
            return []
        spec = self.spec
        base = value_col.upper()
        if base.startswith('RCR'):
            base = 'R' + base[3:]
        if not (base.startswith('R') and len(base) >= 2 and base[1].isdigit()):
            return []
        # R5/R6 are factor-effect residuals; R4 is the time-effect residual
        # (time component removed). For our SDS 3 validation only R3 and R6 are
        # exercised; R4/R5 grains are inferred from the recentering structure.
        if base in ('R5', 'R6'):
            return [spec.rsg_var_name] if spec.rsg_var_name else []
        if base == 'R4':
            return [spec.rsg_var_name] if spec.rsg_var_name else []
        # R1, R2, R3: full (rsg x time) cell grid
        grain = []
        if spec.rsg_var_name:
            grain.append(spec.rsg_var_name)
        if spec.time_var:
            grain.append(spec.time_var)
        return grain

    @staticmethod
    def _resolve_limits_column(value_col: str, df: pd.DataFrame) -> str:
        """Return the column to use for within-group std in limit calculations.

        For effect-carrying residuals (R1/R3/R4/R5 and RCR variants), use R2 instead.
        Non-R2 residuals retain structural effects (factor, time, interaction)
        whose within-group std inflates limits. R2 (within-cell noise) is the
        correct dispersion basis per Bishop.

        Always uses plain R2 (not RCR2) because recentered residuals add
        cell-specific offsets that inflate within-group std when groups span
        multiple cells.
        """
        _EFFECT_RESIDUALS = {'R1', 'R3', 'R4', 'R5', 'RCR1', 'RCR3', 'RCR4', 'RCR5'}
        if value_col is not None and value_col.upper() in _EFFECT_RESIDUALS and 'R2' in df.columns:
            return 'R2'
        return value_col

    def _resolve_by_grouping(
        self,
        value_col: str
    ) -> tuple[list[str], str | None, list[str]]:
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
        tuple[list[str], str | None, list[str]]
            (groupby_cols, ybar_col, stratify_by) where:
            - groupby_cols: columns to group by (empty list for collapse all)
            - ybar_col: pre-calculated Ybar column to use, or None if must aggregate
            - stratify_by: columns to stratify by (separate charts per combo),
              empty list if no stratification

        Examples
        --------
        >>> groupby_cols, ybar_col, stratify_by = self._resolve_by_grouping('y')
        >>> # by=[] (normalized to by=None) -> ([rsg, time_var], 'Ybar_kt', [])
        >>> #   - cell-level aggregation when factors+time exist
        >>> # by=['factor1','factor2'] -> (['rsg'], 'Ybar_k', []) - use factor means
        >>> # by=['time_var'] with factors -> ([time_var], 'Ybar_t', [rsg_var_name])
        >>> # NOTE: ybar_col is None for residual value_col (no pre-cached mean).
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
        # X/mR handles by=[] directly in _calculate_xmr() as single-stream.
        # This split is intentional — `by` means "aggregate by" for Xbar/S
        # and "stratify by" for X/mR, matching each chart type's semantics.
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
            return groupby_cols, ybar_col, []

        # Check if by matches known aggregation levels for Ybar optimization
        by_set = set(by)
        rsg_set = set(rsg_vars)

        # by == all factors (rsg_key level) -> use Ybar_k optimization
        # Only if order matches rsg_vars; otherwise preserve user's order
        if by_set == rsg_set and list(by) == rsg_vars:
            groupby_cols = [spec.rsg_var_name]
            ybar_col = 'Ybar_k' if is_response else None
            return groupby_cols, ybar_col, []

        # by == [time_var] only -> use Ybar_t
        # When factors exist and charting response, also stratify by factor combos
        if by_set == {time_var}:
            groupby_cols = [time_var]
            ybar_col = 'Ybar_t' if is_response else None
            stratify_by = [spec.rsg_var_name] if (rsg_vars and is_response) else []
            return groupby_cols, ybar_col, stratify_by

        # by == all factors + time (cell_key level) -> use Ybar_kt
        cell_key_vars = rsg_set | ({time_var} if time_var else set())
        if by_set == cell_key_vars:
            # Group by all factors + time
            groupby_cols = rsg_vars + [time_var] if time_var else rsg_vars
            ybar_col = 'Ybar_kt' if is_response else None
            return groupby_cols, ybar_col, []

        # Partial subset - must aggregate at runtime
        # Use the by columns directly
        groupby_cols = list(by)
        return groupby_cols, None, []

    def _calculate_lane_boundaries(
        self,
        df: pd.DataFrame,
        collapsed_vars: list[str]
    ) -> list[dict]:
        """
        Calculate lane boundaries where collapsed factors change.

        Lane boundaries are positions in the data where a factor that was
        "collapsed" (not in `by`) changes value. These are rendered as
        vertical dashed lines on X charts.

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
                # Rationale: Wheeler's X chart assumes temporal ordering, and obs_id
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

    def _calculate_xbar(  # noqa: C901
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
            by _calculate_xbar_s() for companion chart calculation.

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
        groupby_cols, ybar_col, stratify_by = self._resolve_by_grouping(value_col)

        # Stratified path: separate chart per factor combination
        if stratify_by:
            return self._calculate_xbar_stratified(
                value_col, groupby_cols, stratify_by,
                _return_intermediates=_return_intermediates,
            )

        _limits_col = self._resolve_limits_column(value_col, df)

        # Bishop VAS grand mean: equal weight per cell at the residual's natural
        # grain, regardless of cell N. The unweighted form is canonical; weighted
        # vs unweighted differ only on unbalanced designs, and Bishop's principle
        # is that the practical difference is negligible -- the methodology still
        # requires the unweighted form. Each residual is recentered at a specific
        # grain (analysis_dataset.py:364-371; study.py:2009 for R6), and the grand
        # mean averages over that grain:
        #   R3 (interaction)        -> (rsg, time) full cell grid
        #   R4 (time effect)        -> (rsg) factor grain (time component removed)
        #   R5, R6 (factor effects) -> (time) or (rsg) depending on what's removed
        # In practice the only residuals exercised by Bishop's validation are R3
        # (full cell grid) and R5/R6 (factor grain).
        if value_col == spec.response_var and 'Ybar' in df.columns:
            _Ybar = df['Ybar'].iloc[0]
        else:
            grain_cols = self._residual_grain(value_col)
            if grain_cols and value_col in df.columns:
                _Ybar = (
                    df.groupby(grain_cols, observed=True)[value_col]
                    .mean()
                    .mean()
                )
            else:
                _Ybar = df[value_col].mean()

        # Handle by=[] (collapse all) - single point chart
        if groupby_cols == []:
            # Single aggregation across all data
            _S = df[_limits_col].std()
            _N = len(df)
            out_dict = {
                'group': ['All'],
                'xbar': [_Ybar],
                's': [_S],
                'n': [_N],
                'N': [_N]
            }
            if _limits_col != value_col:
                out_dict['s_value'] = [df[value_col].std()]
            out = pd.DataFrame(out_dict)
        else:
            # Group by specified columns
            agg_dict = {
                's': pd.NamedAgg(column=_limits_col, aggfunc="std"),
                'mean': pd.NamedAgg(column=value_col, aggfunc="mean"),
                # Count on response_var (not value_col) to avoid NaN issues with residuals
                'n': pd.NamedAgg(column=spec.response_var, aggfunc="count"),
            }
            if _limits_col != value_col:
                agg_dict['s_value'] = pd.NamedAgg(column=value_col, aggfunc="std")
            out = df.groupby(groupby_cols, as_index=False, observed=True).agg(**agg_dict)

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
            sds = self.ads._ads_result.sds if self.ads._ads_result else '?'
            raise ValidationError(
                f"No subgroups with n > 1 found — Xbar chart requires replicated observations.\n"
                f"This data has Analytical Design State {sds}.\n"
                f"Use chart='X' for individual values, or chart='Xbar' with value='R6' "
                f"for effects analysis."
            )

        # Use Bishop VAS grand mean (mean of cell means on value_col) as center
        _Xbar = _Ybar
        _S = out["s"].mean()
        _N = out['n'].max()
        if 'N' not in out.columns:
            out['N'] = _N

        # Determine if subgroup sizes are constant or variable
        n_to_use, n_max = self._determine_n_to_use(out)

        # Override n_to_use if n_mode="average"
        n_mode = self.request.n_mode
        n_avg = None
        if n_mode == "average":
            n_avg = out['n'].mean()
            out['N'] = n_avg      # overwrite N column with average
            n_to_use = 'N'        # force constant-N path

        # CALCULATE XBAR
        xbar = out.copy()
        xbar['center'] = _Xbar  # Add center column for Xbar chart
        xbar[['lpl', 'upl']] = xbar.apply(
            lambda row: calculate_limits(
                mean=row['center'],
                sd=_S,
                N=row[n_to_use],
                limits_type='Xbar',
                round_to=spec.round_to,
                sigma_multiplier=self.request.n_sigma,
            ), axis=1
        )

        # Detect beyond limits signals
        xbar = self._add_beyond_limits_flag(xbar, value_col='xbar')
        xbar = xbar.round(spec.round_to)

        statistics['center'] = round(_Xbar, spec.round_to)
        if n_to_use == "N":
            statistics['N'] = round(n_avg, spec.round_to) if n_avg is not None else _N
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
            # Explicit by list
            if len(by) == 1:
                # Single factor: preserve original column name for axis labeling.
                # Cast to string so Plotly treats coded factors (e.g., 1,2,3) as categories.
                xbar[groupby_cols[0]] = xbar[groupby_cols[0]].astype(str)
                group_col = groupby_cols[0]
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
                'center_col': 'center',
                'n_sigma': self.request.n_sigma,
                'n_mode': n_mode,
                **({'n_avg': round(n_avg, spec.round_to)} if n_avg is not None else {}),
            }
        }

        # Optionally include intermediates for companion calculation
        if _return_intermediates:
            result['_intermediates'] = {
                '_S': _S,
                'n_to_use': n_to_use,
                'n_max': n_max,
                'groupby_cols': groupby_cols,
                'group_col': group_col,
                'out': out,  # Pre-aggregated DataFrame with s, n, mean columns
                'n_avg': n_avg,
            }

        return result

    def _calculate_xbar_stratified(
        self,
        value_col: str,
        groupby_cols: list[str],
        stratify_by: list[str],
        _return_intermediates: bool = False,
    ) -> dict:
        """
        Stratified Xbar chart: one chart per factor combination, time on x-axis.

        Each stratum computes its own Sbar and limits independently.

        Parameters
        ----------
        value_col : str
            Column to chart (response variable).
        groupby_cols : list[str]
            Columns to group by within each stratum (typically [time_var]).
        stratify_by : list[str]
            Columns defining strata (typically [rsg_var_name]).
        _return_intermediates : bool
            If True, include intermediates for companion S chart.

        Returns
        -------
        dict
            Chart data with 'strata' key for stratified output.
        """
        spec = self.spec
        df = self.ads.analysis_dataset.copy()
        result = {}

        # Determine stratification column
        if len(stratify_by) == 1:
            stratify_col = stratify_by[0]
        else:
            df['_stratify_key'] = df[stratify_by].apply(lambda row: encode_rsg(tuple(row)), axis=1)
            stratify_col = '_stratify_key'

        _limits_col = self._resolve_limits_column(value_col, df)

        strata = df[stratify_col].unique().tolist()
        all_xbar_frames = []
        chart_statistics = {}
        intermediates_per_stratum = {}

        for stratum in strata:
            sdf = df[df[stratify_col] == stratum].copy()

            # Group by time within this stratum
            # Note: we always compute mean directly from filtered data, NOT using
            # Ybar_t (which is the marginal time mean across ALL factors).
            agg_dict = {
                's': pd.NamedAgg(column=_limits_col, aggfunc="std"),
                'xbar': pd.NamedAgg(column=value_col, aggfunc="mean"),
                'n': pd.NamedAgg(column=spec.response_var, aggfunc="count"),
            }
            if _limits_col != value_col:
                agg_dict['s_value'] = pd.NamedAgg(column=value_col, aggfunc="std")
            out = sdf.groupby(groupby_cols, as_index=False, observed=True).agg(**agg_dict)

            out['N'] = out['n'].max()

            # Filter subgroups with n=1
            mask_n1 = out['n'].eq(1)
            if mask_n1.any():
                out = out[~mask_n1].copy()

            if out.shape[0] == 0:
                continue

            # Per-stratum statistics
            _Xbar = out['xbar'].mean()
            _S = out['s'].mean()
            n_to_use, n_max = self._determine_n_to_use(out)

            n_mode = self.request.n_mode
            n_avg = None
            if n_mode == "average":
                n_avg = out['n'].mean()
                out['N'] = n_avg
                n_to_use = 'N'

            out['center'] = _Xbar
            out[['lpl', 'upl']] = out.apply(
                lambda row, _sd=_S, _n_col=n_to_use: calculate_limits(
                    mean=row['center'],
                    sd=_sd,
                    N=row[_n_col],
                    limits_type='Xbar',
                    round_to=spec.round_to,
                    sigma_multiplier=self.request.n_sigma,
                ), axis=1
            )

            out = self._add_beyond_limits_flag(out, value_col='xbar')
            out = out.round(spec.round_to)

            # Add stratum identifier
            out[stratify_col] = stratum

            # Cast time column to string for categorical x-axis
            if groupby_cols:
                out[groupby_cols[0]] = out[groupby_cols[0]].astype(str)

            all_xbar_frames.append(out)

            chart_statistics[stratum] = {
                'center': round(_Xbar, spec.round_to),
                'N': round(n_avg, spec.round_to) if n_avg is not None else (
                    out['N'].iloc[0] if n_to_use == 'N' else 'Varies'
                ),
                'lpl': round(out['lpl'].iloc[0], spec.round_to) if n_to_use == 'N' else 'Varies',
                'upl': round(out['upl'].iloc[0], spec.round_to) if n_to_use == 'N' else 'Varies',
            }

            if _return_intermediates:
                intermediates_per_stratum[stratum] = {
                    '_S': _S,
                    'n_to_use': n_to_use,
                    'n_max': n_max,
                    'out': out,
                    'n_avg': n_avg,
                }

        if not all_xbar_frames:
            sds = self.ads._ads_result.sds if self.ads._ads_result else '?'
            raise ValidationError(
                f"No subgroups with n > 1 found — Xbar chart requires replicated observations.\n"
                f"This data has Analytical Design State {sds} "
                f"({'no replication' if sds == 2 else 'partial replication' if sds == 3 else ''}).\n"
                f"Use chart='X' for individual values, or chart='Xbar' with value='R6' "
                f"for effects analysis."
            )
        chart_out = pd.concat(all_xbar_frames, ignore_index=True)

        # Select output columns
        group_col = groupby_cols[0] if groupby_cols else None
        cols_to_keep = [stratify_col, group_col, 'xbar', 'center', 'lpl', 'upl', 'beyond_limits']
        cols_to_keep = [c for c in cols_to_keep if c is not None and c in chart_out.columns]
        chart_out = chart_out[cols_to_keep]

        result['Xbar'] = {
            'data': chart_out,
            'statistics': chart_statistics,
            'metadata': {
                'chart_type': 'Xbar',
                'value_col': 'xbar',
                'center_col': 'center',
                'stratified': True,
                'stratify_col': stratify_col,
                'stratify_by': list(stratify_by),
                'n_sigma': self.request.n_sigma,
                'n_mode': self.request.n_mode,
            },
            'strata': strata,
        }

        if _return_intermediates:
            result['_intermediates'] = {
                'stratify_col': stratify_col,
                'stratify_by': stratify_by,
                'strata': strata,
                'groupby_cols': groupby_cols,
                'per_stratum': intermediates_per_stratum,
                'is_stratified': True,
            }

        return result

    def _calculate_s_stratified(
        self,
        value_col: str,
        groupby_cols: list[str],
        stratify_by: list[str],
        _precomputed: dict | None = None,
    ) -> dict:
        """
        Stratified S chart: one chart per factor combination, time on x-axis.

        Parameters
        ----------
        value_col : str
            Column to chart.
        groupby_cols : list[str]
            Columns to group by within each stratum.
        stratify_by : list[str]
            Columns defining strata.
        _precomputed : dict, optional
            Pre-computed intermediates from _calculate_xbar_stratified().

        Returns
        -------
        dict
            Chart data with 'strata' key for stratified output.
        """
        spec = self.spec

        if _precomputed is not None:
            stratify_col = _precomputed['stratify_col']
            strata = _precomputed['strata']
            per_stratum = _precomputed['per_stratum']
        else:
            df = self.ads.analysis_dataset.copy()
            if len(stratify_by) == 1:
                stratify_col = stratify_by[0]
            else:
                df['_stratify_key'] = df[stratify_by].apply(lambda row: encode_rsg(tuple(row)), axis=1)
                stratify_col = '_stratify_key'
            strata = df[stratify_col].unique().tolist()
            per_stratum = None

        all_s_frames = []
        chart_statistics = {}

        for stratum in strata:
            if per_stratum is not None:
                # Use precomputed data from Xbar stratified
                inter = per_stratum[stratum]
                out = inter['out'].copy()
                # Always use Xbar's _S (R2-based for residuals) for CL and limits
                _S = inter['_S']
                # Drop s_value — S chart plots R2 std (same basis as CL/limits)
                if 's_value' in out.columns:
                    out = out.drop(columns=['s_value'])
                n_to_use = inter['n_to_use']
                n_max = inter['n_max']
                n_avg = inter['n_avg']
            else:
                sdf = df[df[stratify_col] == stratum].copy()
                _limits_col = self._resolve_limits_column(value_col, sdf)
                agg_dict = {
                    's': pd.NamedAgg(column=_limits_col, aggfunc="std"),
                    'n': pd.NamedAgg(column=spec.response_var, aggfunc="count"),
                }
                out = sdf.groupby(groupby_cols, as_index=False, observed=True).agg(**agg_dict)
                mask = out['n'].eq(1)
                out = out[~mask]
                if out.shape[0] == 0:
                    continue
                out['N'] = out['n'].max()
                # CL from R2-based std — S chart plots same basis
                _S = out['s'].mean()
                n_to_use, n_max = self._determine_n_to_use(out)
                n_avg = None
                if self.request.n_mode == "average":
                    n_avg = out['n'].mean()
                    out['N'] = n_avg
                    n_to_use = 'N'

                # Cast time column to string for categorical x-axis
                if groupby_cols:
                    out[groupby_cols[0]] = out[groupby_cols[0]].astype(str)

            out['center'] = _S
            out[['lpl', 'upl']] = out.apply(
                lambda row, _n_col=n_to_use: calculate_limits(
                    mean=0,
                    sd=row['center'],
                    N=row[_n_col],
                    limits_type='S',
                    round_to=spec.round_to,
                    sigma_multiplier=self.request.n_sigma,
                ), axis=1
            )

            out = self._add_beyond_limits_flag(out, value_col='s')
            out = out.round(spec.round_to)
            out[stratify_col] = stratum

            all_s_frames.append(out)

            chart_statistics[stratum] = {
                'center': round(_S, spec.round_to),
                'N': round(n_avg, spec.round_to) if n_avg is not None else (
                    out['N'].iloc[0] if n_to_use == 'N' else 'Varies'
                ),
                'lpl': round(out['lpl'].iloc[0], spec.round_to) if n_to_use == 'N' else 'Varies',
                'upl': round(out['upl'].iloc[0], spec.round_to) if n_to_use == 'N' else 'Varies',
            }

        chart_out = pd.concat(all_s_frames, ignore_index=True)

        group_col = groupby_cols[0] if groupby_cols else None
        cols_to_keep = [stratify_col, group_col, 's', 'center', 'lpl', 'upl', 'beyond_limits']
        cols_to_keep = [c for c in cols_to_keep if c is not None and c in chart_out.columns]
        chart_out = chart_out[cols_to_keep]

        return {
            'S': {
                'data': chart_out,
                'statistics': chart_statistics,
                'metadata': {
                    'chart_type': 'S',
                    'value_col': 's',
                    'center_col': 'center',
                    'stratified': True,
                    'stratify_col': stratify_col,
                    'stratify_by': list(stratify_by),
                    'n_sigma': self.request.n_sigma,
                    'n_mode': self.request.n_mode,
                },
                'strata': strata,
            }
        }

    def _calculate_s(  # noqa: C901
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
            computation. Used internally by _calculate_xbar_s() for companion
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
            n_to_use = _precomputed['n_to_use']
            n_max = _precomputed['n_max']
            groupby_cols = _precomputed['groupby_cols']
            group_col = _precomputed['group_col']
            out = _precomputed['out'].copy()
            n_avg = _precomputed.get('n_avg')
            # Always use Xbar's _S (R2-based for residuals) for CL and limits
            _S = _precomputed['_S']
            # Drop s_value — S chart plots R2 std (same basis as CL/limits)
            if 's_value' in out.columns:
                out = out.drop(columns=['s_value'])
        else:
            # Independent calculation (existing logic)
            df = self.ads.analysis_dataset.copy()

            # Resolve grouping based on `by` parameter
            groupby_cols, _, stratify_by = self._resolve_by_grouping(value_col)

            # Stratified path: separate chart per factor combination
            if stratify_by:
                return self._calculate_s_stratified(
                    value_col, groupby_cols, stratify_by,
                )


            _limits_col = self._resolve_limits_column(value_col, df)

            # Handle by=[] (collapse all) - single point chart
            if groupby_cols == []:
                _S = df[_limits_col].std()
                _N = len(df)
                out = pd.DataFrame({
                    'group': ['All'],
                    's': [_S],
                    'n': [_N],
                    'N': [_N]
                })
            else:
                agg_dict = {
                    's': pd.NamedAgg(column=_limits_col, aggfunc="std"),
                    # Count on response_var (not value_col) to avoid NaN issues with residuals
                    'n': pd.NamedAgg(column=spec.response_var, aggfunc="count"),
                }
                out = df.groupby(groupby_cols, as_index=False, observed=True).agg(**agg_dict)

                # remove groups with a single observation
                mask = out['n'].eq(1)
                out = out[~mask]

                # Handle case where no subgroups have >1 observation
                if out.shape[0] == 0:
                    sds = self.ads._ads_result.sds if self.ads._ads_result else '?'
                    raise ValidationError(
                        f"No subgroups with n > 1 found — S chart requires replicated observations.\n"
                        f"This data has Analytical Design State {sds}.\n"
                        f"Use chart='X' for individual values."
                    )

                out['N'] = out['n'].max()

            # CL from R2-based std — S chart plots same basis
            _S = out["s"].mean()

            # Determine if subgroup sizes are constant or variable
            n_to_use, n_max = self._determine_n_to_use(out)

            # Override n_to_use if n_mode="average"
            n_mode = self.request.n_mode
            n_avg = None
            if n_mode == "average":
                n_avg = out['n'].mean()
                out['N'] = n_avg
                n_to_use = 'N'

            # Determine group_col
            by = self.request.by
            if by is not None and len(by) > 0:
                if len(by) == 1:
                    out[groupby_cols[0]] = out[groupby_cols[0]].astype(str)
                    group_col = groupby_cols[0]
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
        n_mode = self.request.n_mode

        # Add limits columns
        out[['lpl', 'upl']] = out.apply(
            lambda row: calculate_limits(
                mean=0,
                sd=row['center'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to,
                sigma_multiplier=self.request.n_sigma,
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
                if len(by) == 1:
                    out[groupby_cols[0]] = out[groupby_cols[0]].astype(str)
                    group_col = groupby_cols[0]
                else:
                    out['subgroup'] = out[groupby_cols].astype(str).agg('_'.join, axis=1)
                    group_col = 'subgroup'
            elif len(groupby_cols) > 1 and group_col == 'group' and 'group' not in out.columns:
                out['group'] = out[groupby_cols].astype(str).agg('_'.join, axis=1)

        cols_to_keep = [group_col, 's', 'center', 'lpl', 'upl', 'beyond_limits']
        cols_to_keep = [c for c in cols_to_keep if c in out.columns]
        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        # Build statistics
        _S_stat = out['center'].iloc[0] if len(out) > 0 else None

        statistics = {'center': round(_S_stat, spec.round_to) if _S_stat is not None else None}
        if n_to_use == "N":
            statistics['N'] = round(n_avg, spec.round_to) if n_avg is not None else n_max
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
                    'center_col': 'center',
                    'n_sigma': self.request.n_sigma,
                    'n_mode': n_mode,
                    **({'n_avg': round(n_avg, spec.round_to)} if n_avg is not None else {}),
                }
            }
        }

    def _calculate_xbar_s(self, value_col: str = None) -> dict:
        """
        Calculate Xbar and S charts together (Bishop methodology).

        This method ensures that when companion=True, both charts are calculated
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

        # Stratified companion path
        if intermediates.get('is_stratified'):
            s_result = self._calculate_s_stratified(
                value_col,
                intermediates['groupby_cols'],
                intermediates['stratify_by'],
                _precomputed=intermediates,
            )
            return {**xbar_result, **s_result}

        # Calculate S using precomputed values
        s_result = self._calculate_s(value_col, _precomputed=intermediates)

        # Combine results
        return {**xbar_result, **s_result}

    @staticmethod
    def _resolve_mr_source_column(value_col: str) -> str:
        """Map recentered column (RCR*) to its non-recentered residual (R*).

        Moving ranges must be computed from non-recentered residuals to avoid
        inflating mR with structural jumps between cells. Returns value_col
        unchanged if it is not a recentered column.
        """
        if value_col.startswith('RCR'):
            return value_col[2:]  # RCR3 -> R3, RCR6 -> R6, etc.
        return value_col

    def _calculate_mr_chart(
        self,
        mr_spec: _MRChartSpec,
        value_col: str = None,
        _return_intermediates: bool = False,
        _precomputed: dict = None,
    ) -> dict:
        """
        Shared pipeline for MR-family charts (X and mR).

        The X and mR charts share >85% of their calculation logic. The behavioral
        differences are encoded in ``mr_spec`` — no boolean flags needed.

        Parameters
        ----------
        mr_spec : _MRChartSpec
            Chart behavior specification (_XMR_SPEC or _R_SPEC).
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
        _return_intermediates : bool, optional
            If True, includes '_intermediates' key for companion-mode reuse.
        _precomputed : dict, optional
            Pre-computed intermediates from a prior X call (companion mode).
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

        # --- Resolve mR source column ---
        # When recentered (value_col is RCR*), compute mR from the non-recentered
        # residual (R*) to avoid inflating moving ranges with structural jumps.
        mr_source_col = self._resolve_mr_source_column(value_col)

        if is_stratified:
            return self._calculate_mr_chart_stratified(
                mr_spec, value_col, plot_col, mr_source_col,
                stratify_by, collapsed_factors,
                _return_intermediates, _precomputed,
                phased=self.request.phased,
            )
        else:
            return self._calculate_mr_chart_ungrouped(
                mr_spec, value_col, plot_col, mr_source_col,
                collapsed_factors,
                _return_intermediates, _precomputed,
                phased=self.request.phased,
            )

    def _calculate_mr_chart_from_precomputed(
        self,
        mr_spec: _MRChartSpec,
        plot_col: str,
        precomputed: dict,
    ) -> dict:
        """Build an MR-family chart reusing intermediates from a companion X call."""
        spec = self.spec
        out = precomputed['out'].copy()
        stratify_col = precomputed['stratify_col']
        stratify_by = precomputed['stratify_by']
        strata = precomputed['strata']
        xmr_lane_boundaries = precomputed['lane_boundaries']
        extra_cols = precomputed['extra_cols']

        # --- Phased precomputed path ---
        if precomputed.get('phased'):
            phase_stats = precomputed['phase_stats']

            # Recompute R-specific limits per phase within each stratum
            r_phase_stats = phase_stats[[stratify_col, '_phase_id', 'mR']].copy()
            r_lims = r_phase_stats.apply(
                lambda row: calculate_limits(
                    mean=0, sd=0, N=0, mR=row['mR'],
                    limits_type=mr_spec.limits_type,
                    round_to=spec.round_to,
                ), axis=1,
            )
            r_phase_stats = _join_limits(r_phase_stats, r_lims)
            r_phase_stats['center'] = r_phase_stats['mR']

            # Replace X limits with mR limits
            out = out.drop(columns=['center', 'lpl', 'upl', 'beyond_limits'], errors='ignore')
            out = out.merge(
                r_phase_stats[[stratify_col, '_phase_id', 'center', 'lpl', 'upl']],
                on=[stratify_col, '_phase_id'], how='left', validate='many_to_one',
            )

            # Re-detect beyond limits
            assert plot_col in out.columns, f"plot_col {plot_col!r} not in DataFrame"
            out = self._add_beyond_limits_flag(out, value_col=plot_col)

            # Format output
            chart_out = self._build_output_columns(
                df=out,
                value_cols=extra_cols + [plot_col, 'center', 'lpl', 'upl', 'beyond_limits'],
            )

            chart_statistics = {}
            for stratum in strata:
                chart_statistics[stratum] = {
                    'center': 'Varies', 'lpl': 'Varies', 'upl': 'Varies',
                }

            # Phased R: no lane boundary offset (no rows dropped)
            return {
                mr_spec.chart_type: {
                    'data': chart_out,
                    'statistics': chart_statistics,
                    'metadata': {
                        'chart_type': mr_spec.chart_type,
                        'value_col': plot_col,
                        'center_col': 'center',
                        'stratified': True,
                        'lane_boundaries': xmr_lane_boundaries if xmr_lane_boundaries else None,
                        'stratify_col': stratify_col,
                        'stratify_by': list(stratify_by),
                        'phased': True,
                        'run_rules_applicable': False,
                    },
                    'strata': strata,
                },
            }

        # --- Non-phased precomputed path ---
        grouped = precomputed['grouped'].copy()

        # For mR chart: compute mR-specific limits from X's grouped data
        # grouped has columns: stratify_col, center (=mean), mR, lpl, upl (X limits)
        r_grouped = grouped.rename(columns={'center': 'xmr_center', 'mR': 'center'})

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

        # Resolve lpl/upl column names (may have suffix if X lpl/upl exist)
        if 'lpl_r' in r_grouped.columns:
            r_grouped = r_grouped.rename(columns={'lpl_r': 'r_lpl', 'upl_r': 'r_upl'})
            r_lpl_col, r_upl_col = 'r_lpl', 'r_upl'
        else:
            r_lpl_col, r_upl_col = 'lpl', 'upl'

        # Merge mR limits to the data (drop X-specific columns first)
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
        if xmr_lane_boundaries:
            for stratum, boundaries in xmr_lane_boundaries.items():
                adjusted = [
                    {**b, 'position': b['position'] + mr_spec.lane_boundary_offset}
                    for b in boundaries
                    if b['position'] + mr_spec.lane_boundary_offset >= 0
                ]
                if adjusted:
                    lane_boundaries[stratum] = adjusted

        # Identify strata with insufficient data (< 2 observations)
        insufficient_strata = [
            s for s in strata
            if len(r_out[r_out[stratify_col] == s]) < 2
        ]

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
                    'stratify_col': stratify_col,
                    'stratify_by': list(stratify_by),
                    'insufficient_strata': insufficient_strata if insufficient_strata else None,
                },
                'strata': strata,
            },
        }

    def _calculate_mr_chart_stratified(  # noqa: C901
        self,
        mr_spec: _MRChartSpec,
        value_col: str,
        plot_col: str,
        mr_source_col: str,
        stratify_by: list[str],
        collapsed_factors: list[str],
        _return_intermediates: bool,
        _precomputed: dict | None,
        phased: bool = False,
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
            out['_stratify_key'] = out[stratify_by].apply(lambda row: encode_rsg(tuple(row)), axis=1)
            stratify_col = '_stratify_key'

        # Canonical sort
        out = out.sort_values('sort_key', kind='stable')

        # Strata list (needed for both phased and non-phased paths)
        strata = out[stratify_col].unique().tolist()

        # === Phased limits path (per-phase limits within each stratum) ===
        if phased and collapsed_factors:
            # 1. Build phase key from collapsed factors only (vectorized)
            if len(collapsed_factors) == 1:
                out['_phase_key'] = out[collapsed_factors[0]]
            else:
                out['_phase_key'] = list(map(tuple, out[collapsed_factors].to_numpy()))

            # 2. Assign within-stratum phase IDs via transitions of _phase_key
            out['_phase_id'] = (
                out.groupby(stratify_col, sort=False)['_phase_key']
                .transform(lambda s: (s != s.shift()).cumsum())
                .astype('int64')
            )

            # 3. Per-phase mR (reset at phase boundaries within each stratum)
            out['mr'] = (
                out.groupby([stratify_col, '_phase_id'], sort=False)[mr_source_col]
                .diff().abs()
            )

            # 4. Per-phase aggregation
            phase_stats = (
                out.groupby([stratify_col, '_phase_id'], sort=False)
                .agg(_n=(value_col, 'size'), _mean=(value_col, 'mean'), mR=('mr', 'mean'))
                .reset_index()
            )
            phase_stats['mR'] = phase_stats['mR'].fillna(0.0)

            # 5. Compute per-phase limits
            if mr_spec.center_source == 'mean':
                phase_stats['center'] = phase_stats['_mean']
                phase_lims = phase_stats.apply(
                    lambda row: calculate_limits(
                        mean=row['_mean'], sd=0, N=0, mR=row['mR'],
                        limits_type=mr_spec.limits_type,
                        round_to=spec.round_to,
                    ), axis=1,
                )
            else:  # center_source == 'mR' (R chart)
                phase_stats['center'] = phase_stats['mR']
                phase_lims = phase_stats.apply(
                    lambda row: calculate_limits(
                        mean=0, sd=0, N=0, mR=row['mR'],
                        limits_type=mr_spec.limits_type,
                        round_to=spec.round_to,
                    ), axis=1,
                )
            phase_stats = _join_limits(phase_stats, phase_lims)

            # 6. Merge per-phase limits back to row-level data
            out = out.merge(
                phase_stats[[stratify_col, '_phase_id', 'center', 'mR', 'lpl', 'upl']],
                on=[stratify_col, '_phase_id'], how='left', validate='many_to_one',
            )

            # 7. Phased path: NEVER drop rows. NaN mR at phase boundaries preserved.

            # 8. Signal detection (per-row, uses row-level lpl/upl)
            assert plot_col in out.columns, f"plot_col {plot_col!r} not in DataFrame"
            out = self._add_beyond_limits_flag(out, value_col=plot_col)

            # 9. Lane boundaries per stratum
            lane_boundaries = {}
            if collapsed_factors:
                for stratum in strata:
                    stratum_data = out[out[stratify_col] == stratum].reset_index(drop=True)
                    boundaries = self._calculate_lane_boundaries(stratum_data, collapsed_factors)
                    if boundaries:
                        lane_boundaries[stratum] = boundaries

            # 10. Statistics per stratum = 'Varies'
            chart_statistics = {}
            for stratum in strata:
                chart_statistics[stratum] = {
                    'center': 'Varies', 'lpl': 'Varies', 'upl': 'Varies',
                }

            # 11. Track single-point phases
            n_single_point = int((phase_stats['_n'] < 2).sum())

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
                    'stratify_col': stratify_col,
                    'stratify_by': list(stratify_by),
                    'phased': True,
                    'single_point_phases': n_single_point,
                    'run_rules_applicable': False,
                },
                'strata': strata,
            }

            if _return_intermediates:
                result['_intermediates'] = {
                    'out': out,
                    'phase_stats': phase_stats,
                    'stratify_col': stratify_col,
                    'stratify_by': stratify_by,
                    'strata': strata,
                    'lane_boundaries': lane_boundaries,
                    'collapsed_factors': collapsed_factors,
                    'extra_cols': extra_cols,
                    'is_stratified': True,
                    'phased': True,
                }

            return result

        # === Global limits path (unchanged) ===

        # Moving range per stratum
        out['mr'] = out.groupby(stratify_col, sort=False, observed=True)[mr_source_col].diff().abs()

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
        # For X:  center=mean, limits based on mean ± E2*mR
        # For mR: center=mR, limits based on 0 to D4*mR
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

        # Use strata from grouped for non-phased path (preserves original order)
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

        # Identify strata with insufficient data (< 2 observations)
        insufficient_strata = [
            s for s in strata
            if len(out[out[stratify_col] == s]) < 2
        ]

        result[mr_spec.chart_type] = {
            'data': chart_out,
            'statistics': chart_statistics,
            'metadata': {
                'chart_type': mr_spec.chart_type,
                'value_col': plot_col,
                'center_col': 'center',
                'stratified': True,
                'lane_boundaries': lane_boundaries if lane_boundaries else None,
                'stratify_col': stratify_col,
                'stratify_by': list(stratify_by),
                'insufficient_strata': insufficient_strata if insufficient_strata else None,
            },
            'strata': strata,
        }

        # Optionally include intermediates for companion calculation
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

    def _calculate_mr_chart_ungrouped(  # noqa: C901
        self,
        mr_spec: _MRChartSpec,
        value_col: str,
        plot_col: str,
        mr_source_col: str,
        collapsed_factors: list[str],
        _return_intermediates: bool,
        _precomputed: dict | None,
        phased: bool = False,
    ) -> dict:
        """Ungrouped (single-stream) path for _calculate_mr_chart."""
        spec = self.spec
        result = {}

        if _precomputed is not None and not _precomputed['is_stratified']:
            # --- Phased precomputed path (for companion R chart) ---
            if _precomputed.get('phased'):
                out = _precomputed['out'].copy()
                phase_stats = _precomputed['phase_stats']
                xmr_lane_boundaries = _precomputed['lane_boundaries']

                # Recompute R-specific limits per phase
                r_phase_stats = phase_stats[['_phase_id', 'mR']].copy()
                r_lims = r_phase_stats.apply(
                    lambda row: calculate_limits(
                        mean=0, sd=0, N=0, mR=row['mR'],
                        limits_type=mr_spec.limits_type,
                        round_to=spec.round_to,
                    ), axis=1,
                )
                r_phase_stats = _join_limits(r_phase_stats, r_lims)
                r_phase_stats['center'] = r_phase_stats['mR']

                # Replace X limits with mR limits
                out = out.drop(columns=['center', 'lpl', 'upl'], errors='ignore')
                out = out.merge(
                    r_phase_stats[['_phase_id', 'center', 'lpl', 'upl']],
                    on='_phase_id', how='left', validate='many_to_one',
                )

                # Re-detect beyond limits
                out = out.drop(columns=['beyond_limits'], errors='ignore')
                assert plot_col in out.columns, f"plot_col {plot_col!r} not in DataFrame"
                out = self._add_beyond_limits_flag(out, value_col=plot_col)

                chart_out = self._build_output_columns(
                    df=out,
                    value_cols=[plot_col, 'center', 'lpl', 'upl', 'beyond_limits'],
                )

                chart_statistics = {
                    'center': 'Varies',
                    'lpl': 'Varies',
                    'upl': 'Varies',
                }

                # Phased R: no lane boundary offset needed (no rows dropped)
                r_lane_boundaries = xmr_lane_boundaries

                result[mr_spec.chart_type] = {
                    'data': chart_out,
                    'statistics': chart_statistics,
                    'metadata': {
                        'chart_type': mr_spec.chart_type,
                        'value_col': plot_col,
                        'center_col': 'center',
                        'lane_boundaries': r_lane_boundaries,
                        'phased': True,
                        'run_rules_applicable': False,
                    },
                }
                return result

            # --- Existing non-phased precomputed path ---
            # Reuse intermediates from companion X call
            out = _precomputed['out'].copy()
            mR = _precomputed['mR']
            xmr_lane_boundaries = _precomputed['lane_boundaries']

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
            if xmr_lane_boundaries:
                r_lane_boundaries = []
                for b in xmr_lane_boundaries:
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

        # === Phased limits path ===
        if phased and collapsed_factors:
            # 1. Assign phase_id: contiguous runs of the same rsg_key
            out['_phase_id'] = (
                (out['rsg_key'] != out['rsg_key'].shift())
                .cumsum()
                .astype('int64')
            )

            # 2. Per-phase moving range (reset at boundaries)
            out['mr'] = (
                out.groupby('_phase_id', sort=False)[mr_source_col]
                .diff().abs()
            )

            # 3. Per-phase aggregation
            phase_stats = (
                out.groupby('_phase_id', sort=False)
                .agg(
                    _n=(value_col, 'size'),
                    _mean=(value_col, 'mean'),
                    mR=('mr', 'mean'),
                )
                .reset_index()
            )

            # 4. Handle single-point phases (mR=NaN -> use 0)
            phase_stats['mR'] = phase_stats['mR'].fillna(0.0)

            # 5. Compute per-phase limits
            if mr_spec.center_source == 'mean':
                phase_stats['center'] = phase_stats['_mean']
                phase_lims = phase_stats.apply(
                    lambda row: calculate_limits(
                        mean=row['_mean'], sd=0, N=0, mR=row['mR'],
                        limits_type=mr_spec.limits_type,
                        round_to=spec.round_to,
                    ), axis=1,
                )
            else:  # center_source == 'mR' (R chart)
                phase_stats['center'] = phase_stats['mR']
                phase_lims = phase_stats.apply(
                    lambda row: calculate_limits(
                        mean=0, sd=0, N=0, mR=row['mR'],
                        limits_type=mr_spec.limits_type,
                        round_to=spec.round_to,
                    ), axis=1,
                )
            phase_stats = _join_limits(phase_stats, phase_lims)

            # 6. Merge per-phase limits back to row-level data
            out = out.merge(
                phase_stats[['_phase_id', 'center', 'mR', 'lpl', 'upl']],
                on='_phase_id', how='left', validate='many_to_one',
            )

            # 7. Phased path: NEVER drop rows. NaN mR values at phase boundaries
            #    stay in the DataFrame. Plotly skips NaN naturally.

            # 8. Signal detection (per-row, uses row-level lpl/upl)
            assert plot_col in out.columns, f"plot_col {plot_col!r} not in DataFrame"
            out = self._add_beyond_limits_flag(out, value_col=plot_col)

            # 9. Format output
            chart_out = self._build_output_columns(
                df=out,
                value_cols=[plot_col, 'center', 'lpl', 'upl', 'beyond_limits'],
            )

            # 10. Statistics: 'Varies' signals stepped rendering
            chart_statistics = {
                'center': 'Varies',
                'lpl': 'Varies',
                'upl': 'Varies',
            }

            # 11. Track single-point phases for visibility
            n_single_point = int((phase_stats['_n'] < 2).sum())

            result[mr_spec.chart_type] = {
                'data': chart_out,
                'statistics': chart_statistics,
                'metadata': {
                    'chart_type': mr_spec.chart_type,
                    'value_col': plot_col,
                    'center_col': 'center',
                    'lane_boundaries': lane_boundaries,
                    'phased': True,
                    'single_point_phases': n_single_point,
                    'run_rules_applicable': False,
                },
            }

            if _return_intermediates:
                result['_intermediates'] = {
                    'out': out,
                    'mR': None,
                    'lane_boundaries': lane_boundaries,
                    'is_stratified': False,
                    'phased': True,
                    'phase_stats': phase_stats,
                }

            return result

        # === Global limits path (unchanged) ===
        # Moving range
        out['mr'] = out[mr_source_col].diff().abs()
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

        # Optionally include intermediates for companion calculation
        if _return_intermediates:
            result['_intermediates'] = {
                'out': out,
                'mR': mR,
                'lane_boundaries': lane_boundaries,
                'is_stratified': False,
            }

        return result

    def _calculate_xmr(
        self,
        value_col: str = None,
        _return_intermediates: bool = False,
    ) -> dict:
        """
        Calculate X (Individual) chart statistics.

        Delegates to _calculate_mr_chart with _XMR_SPEC.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
        _return_intermediates : bool, optional
            If True, includes '_intermediates' key for companion-mode reuse.

        Returns
        -------
        dict
            Chart data: {'X': {'data': df, 'statistics': dict, 'metadata': dict}}
        """
        return self._calculate_mr_chart(
            _XMR_SPEC, value_col=value_col,
            _return_intermediates=_return_intermediates,
        )

    def _calculate_r(
        self,
        value_col: str = None,
        _precomputed: dict = None,
    ) -> dict:
        """
        Calculate mR (Moving Range) chart statistics.

        Delegates to _calculate_mr_chart with _R_SPEC.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
        _precomputed : dict, optional
            Pre-computed intermediates from _calculate_xmr() for companion mode.

        Returns
        -------
        dict
            Chart data: {'mR': {'data': df, 'statistics': dict, 'metadata': dict}}
        """
        return self._calculate_mr_chart(
            _R_SPEC, value_col=value_col,
            _precomputed=_precomputed,
        )

    def _calculate_xmr_r(self, value_col: str = None) -> dict:
        """
        Calculate X and mR charts together (Bishop methodology).

        This method ensures that when companion=True, both charts are calculated
        efficiently using shared intermediate values. The mR chart uses the
        same data computed for X, avoiding redundant calculation.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.

        Returns
        -------
        dict
            Combined chart data: {'X': {...}, 'mR': {...}}
        """
        # Calculate X with intermediates for mR reuse
        xmr_result = self._calculate_xmr(value_col, _return_intermediates=True)

        # Extract intermediates and remove from result
        intermediates = xmr_result.pop('_intermediates')

        # Calculate mR using precomputed values
        r_result = self._calculate_r(value_col, _precomputed=intermediates)

        # Combine results
        return {**xmr_result, **r_result}

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

            def _py(v):
                """Normalize numpy/pandas scalars to python natives."""
                return v.item() if hasattr(v, 'item') else v

            # Build strata list - strings for all cases via encode_rsg
            if len(by) == 1:
                strata = data[by[0]].unique().tolist()
            else:
                strata = [encode_rsg(tuple(_py(v) for v in k)) for k in grouped.groups]

            # Calculate per-stratum statistics
            per_stratum_stats = {}
            for stratum, group_df in grouped:
                # Unwrap single-element tuple for single-column groupby
                # groupby(['col']) returns ('value',) tuples, but strata uses scalar values
                key = stratum[0] if len(by) == 1 else encode_rsg(tuple(_py(v) for v in stratum))
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

            # Add rsg column so focus() can filter consistently with other chart types

            if len(by) == 1:
                output_data['rsg'] = output_data[by[0]].apply(
                    lambda v: encode_rsg(_py(v))
                )
            else:
                output_data['rsg'] = output_data[by].apply(
                    lambda row: encode_rsg(tuple(_py(v) for v in row)), axis=1
                )

            return {
                'Histogram': {
                    'data': output_data,
                    'statistics': per_stratum_stats,
                    'metadata': {
                        'chart_type': 'Histogram',
                        'value_col': value_col,
                        'bins': self.request.bins,
                        'stratified': True,
                        'stratify_col': by[0] if len(by) == 1 else '_stratify_key',
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
