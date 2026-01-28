"""
Analysis - Chart calculation strategies for process behavior analysis.

This module provides the Analysis class which executes chart calculations using
the strategy pattern. It supports:
- Xbar and S charts (subgroup mean and variation)
- IMR charts (individual moving range)
- R charts (range)

The Analysis class coordinates between:
- AnalysisSpecification (configuration)
- AnalysisDataSet (data preparation and VAS calculations)
- Chart calculation strategies
- AnalysisResult (unified result container)

Usage:
    spec = {
        'analysis_type': 'Xbar',
        'response_var': 'Height',
        'time_var': 'Time',
        'rsg_vars': ['Operator', 'Machine']
    }

    analysis = Analysis(df, spec)
    result = analysis.calculate()

    # Access charts
    xbar = result.get_chart('Xbar')
    s = result.get_chart('S')
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .analysis_dataset import AnalysisDataSet
from .analysis_result import AnalysisResult
from .analysis_specification import AnalysisSpecification
from .spc_constants import calculate_limits, detect_beyond_limits

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Module-level Helper Functions
# ============================================================================

def split_df_by_group(df: pd.DataFrame, grouping_var: str) -> dict:
    """
    Split DataFrame by grouping variable into dictionary of DataFrames.

    :param pandas.Dataframe df: data frame to split by group
    :param str grouping_var: group_variable to group dataframe by
    :return: dictionary of dataframes with grouping_var values as keys
    :rtype: dict
    :raises ValueError: if group_var is not in input dataframe

    For example: if df.groups = ['a','b','c'] the function will return:
            {
                'a': pandas.Dataframe for a,
                'b': pandas.Dataframe for b,
                'c': pandas.Dataframe for c
            }
    """
    # Make sure grouping_var column exists
    if grouping_var not in df.columns.tolist():
        raise ValueError(
            f'The group_var: {grouping_var} is not in the data set!'
        )

    out = {}
    grouped = df.groupby(grouping_var, observed=True)

    # package_results
    for g in grouped.groups:
        criteria = df[grouping_var].eq(g)
        out[g] = df[criteria]

    return out


def gather_analysis_statistics(
    df: pd.DataFrame,
    statistics_to_collect: list,
    grouping_var: str = None
) -> dict:
    """
    Gather summary statistics from analysis results.

    Returns a dictionary of statistics (contained in stats_to_package)
    for each analytic result passed.

    :param pandas.Dataframe df: a grouped dataframe of analysis results,
        i.e., output from R or Imr
    :param list statistics_to_collect: list of variables/columns to summarize
        (dataframes currently contain columns for mean, moving range, and N)
    :param str grouping_var: optional grouping variable

    This function will take the max for each value specified in stats to package
    and put in dictionary with key equal to the value of list item,
    i.e., "mean" will be returned in a dictionary
    {statistics:{group_name: "abc", mean:1.0, etc...}}

    :return: dictionary of statistics with grouping_var values as keys
    :rtype: dict
    :raises ValueError: if variables specified in list are not in input dataframe
    """
    logger.debug('In call gather_statistics')
    stats = {}

    out = df.copy()
    out_cols = df.columns.to_list()

    is_valid = all(cols in out_cols for cols in statistics_to_collect)

    if is_valid:
        if grouping_var is not None:
            statistics_to_collect.append("n")
            N = out.groupby(grouping_var, as_index=False, observed=True).size()
            N.reset_index()
            N.rename(columns={"size": "n"}, inplace=True)

            summarized = df.groupby([grouping_var], observed=True).max()
            summarized = pd.merge(N, summarized, how='left', on=grouping_var)

            for _index, row in summarized.iterrows():
                stats[row[grouping_var]] = row[statistics_to_collect].to_dict()

        else:
            n = len(out)

            summarized = out[statistics_to_collect].max().to_dict()
            summarized["n"] = n
            stats['Imr'] = summarized

    else:
        raise ValueError(f'Statistics: {statistics_to_collect} are not in {df.columns.to_list()}')

    return stats


def package_analysis(analysis_output: dict, summary_statistics_output: dict):
    """
    Combine analysis results with summary statistics.

    Combines two dictionaries into one with the rational subgroup name as the key.
    Returns a dictionary of statistics (contained in stats_to_package)
    for each analytic result passed.

    :param dict analysis_output: dictionary of dataframes with
        a key matching the name of the rational subgroup name for grouped
        individuals analyses, R or Imr
    :param dict summary_statistics_output: dictionary of collected
        statistics for each grouped individuals analysis. key is expected
        to be the name of the rational subgroup.

    :return: dictionary with combined data and statistics
    :rtype: dict
    :raises ValueError: if keys don't match between the two dictionaries
    """
    logger.debug('In call package_analysis')
    out = {}

    output_keys = analysis_output.keys()
    stats_keys = summary_statistics_output.keys()

    is_valid = all(keys in output_keys for keys in stats_keys)

    if is_valid:
        for key in analysis_output:
            out[key] = {
                'data': analysis_output[key],
                'statistics': summary_statistics_output[key]
            }
    else:
        msg = (
            f'Call: package_analysis: The rational subgroups do not match '
            f'for statistics being collected {list(stats_keys)}, '
            f'data: {list(output_keys)}'
        )
        raise ValueError(msg)

    return out


def _limits_to_pair(x):
    """
    Extract (lpl, upl) pair from various limit representations.

    Handles different return types from calculate_limits:
    - dict with 'lpl'/'upl' keys
    - object with lpl/upl attributes
    - list/tuple with positional values
    """
    if isinstance(x, dict):
        return x.get('lpl', np.nan), x.get('upl', np.nan)
    try:
        return x['lpl'], x['upl']
    except Exception:
        pass
    if hasattr(x, 'lpl') and hasattr(x, 'upl'):
        return getattr(x, 'lpl', np.nan), getattr(x, 'upl', np.nan)
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        return x[0], x[1]
    return np.nan, np.nan


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
        df: pd.DataFrame,
        specification: dict,
        analysis_dataset: AnalysisDataSet | None = None,
        sds: int | None = None
    ):
        """
        Initialize analysis with data and specification.

        Args:
            df: Input DataFrame with raw data
            specification: Dictionary containing analysis configuration including 'analysis_type'
            analysis_dataset: Optional pre-calculated AnalysisDataSet.
                If provided, skips expensive residual calculation.
                Used by Study.execute() to reuse formulate() calculations.
            sds: Sampling Design State (0-6). Required if analysis_dataset is not
                provided. SDS should be detected at the entry point (ProcessBehavior)
                and passed through the system.

        Raises:
            ValueError: If neither analysis_dataset nor sds is provided.
        """
        # Import here to avoid circular dependency
        from .analysis_dataset import AnalysisDataSet

        self.raw_df = df
        self.spec = AnalysisSpecification(specification)
        self.analysis_type = self.spec.analysis_type

        # Use pre-calculated AnalysisDataSet if provided, otherwise calculate
        if analysis_dataset is not None:
            self.ads = analysis_dataset
        else:
            if sds is None:
                raise ValueError(
                    "sds is required when analysis_dataset is not provided. "
                    "SDS should be detected at the entry point (ProcessBehavior)"
                    "and passed to Analysis."
                )
            self.ads = AnalysisDataSet(df, self.spec, sds=sds)

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
        residual = getattr(self.spec, 'residual', None)
        if residual:
            # Residual chart analysis - inline logic (was _calculate_residual_chart)
            chart_type = getattr(self.spec, 'residual_chart_type', 'Imr')
            recentered = getattr(self.spec, 'recentered', False)

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
            if chart_type == 'Histogram':
                # Histogram of residual values
                chart_data = self._calculate_histogram(value_col=col_name)

                # Add residual metadata
                hist_data = chart_data['Histogram']
                hist_data['metadata']['residual_type'] = residual
                hist_data['metadata']['recentered'] = recentered
                hist_data['metadata']['question_answered'] = questions.get(residual, '')

            elif chart_type == 'S':
                s_result = self._calculate_s(value_col=col_name)
                chart_name = 'S'
                s_data = s_result['S']

                chart_data = {
                    chart_name: {
                        'data': s_data['data'],
                        'statistics': s_data['statistics'],
                        'metadata': {
                            **s_data.get('metadata', {}),
                            'residual_type': residual,
                            'recentered': recentered,
                            'question_answered': questions.get(residual, '')
                        }
                    }
                }

            elif chart_type == 'Xbar':
                xbar_result = self._calculate_xbar(value_col=col_name)
                chart_name = 'Xbar'
                xbar_data = xbar_result['Xbar']

                chart_data = {
                    chart_name: {
                        'data': xbar_data['data'],
                        'statistics': xbar_data['statistics'],
                        'metadata': {
                            **xbar_data.get('metadata', {}),
                            'residual_type': residual,
                            'recentered': recentered,
                            'question_answered': questions.get(residual, '')
                        }
                    }
                }

            else:  # Imr
                chart_data = self._calculate_imr(value_col=col_name)

                # Add residual metadata and rename keys
                renamed_result = {}
                for chart_key, data in chart_data.items():
                    if 'metadata' not in data:
                        data['metadata'] = {}
                    data['metadata']['residual_type'] = residual
                    data['metadata']['recentered'] = recentered
                    data['metadata']['question_answered'] = questions.get(residual, '')
                    renamed_result[chart_key] = data

                chart_data = renamed_result

        else:
            # Standard chart analysis
            # Note: Imr and R are bundled together (both call _calculate_imr)
            # just like Xbar and S are bundled together
            strategies = {
                'Xbar': self._calculate_xbar,
                'S': self._calculate_s,
                'Imr': self._calculate_imr,
                'R': self._calculate_imr,  # Bundled with Imr
                'Histogram': self._calculate_histogram
            }

            if self.analysis_type not in strategies:
                raise ValueError(
                    f'Analysis type {self.analysis_type} not supported! '
                    f'Valid types: {list(strategies.keys())}'
                )

            # Execute analysis strategy
            chart_data = strategies[self.spec.analysis_type]()

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
        by = spec.by
        rsg_vars = spec.rsg_vars or []
        time_var = spec.time_var

        # Determine if we're charting response (can use pre-calculated) or residual
        is_response = value_col == spec.response_var

        # by=[] is equivalent to by=None (both mean Kt level)
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

    def _package_stratified_results(
        self,
        df: pd.DataFrame,
        statistics_cols: list[str],
        chart_type: str,
        value_col: str
    ) -> dict:
        """
        Package analysis results with statistics for stratified charts.

        Used for IMR and R charts with optional grouping.

        Parameters
        ----------
        df : pd.DataFrame
            Analysis results with chart data
        statistics_cols : list[str]
            Columns to collect statistics for (e.g., ['mean', 'lpl', 'upl'])
        chart_type : str
            Type of chart ('Imr' or 'R')
        value_col : str
            Name of the value column to plot

        Returns
        -------
        dict
            Packaged results: {group: {'data': df, 'statistics': dict, 'metadata': dict}}
            For ungrouped data: {'all': {'data': df, 'statistics': dict, 'metadata': dict}}

        Examples
        --------
        >>> result = self._package_stratified_results(
        ...     df=out,
        ...     statistics_cols=['center', 'lpl', 'upl'],
        ...     chart_type='Imr',
        ...     value_col='measurement'
        ... )
        """
        # Functions are now in the same module - no import needed
        if self.spec.has_grouping:
            statistics = gather_analysis_statistics(
                df=df,
                statistics_to_collect=statistics_cols,
                grouping_var=self.spec.rsg_var_name
            )
            split_dict = split_df_by_group(
                df=df,
                grouping_var=self.spec.rsg_var_name
            )
        else:
            statistics = gather_analysis_statistics(
                df=df,
                statistics_to_collect=statistics_cols
            )
            split_dict = {'Imr': df}

        result = package_analysis(
            analysis_output=split_dict,
            summary_statistics_output=statistics
        )

        # Add metadata to each group's result
        for group_key in result:
            result[group_key]['metadata'] = {
                'chart_type': chart_type,
                'value_col': value_col,
                'center_col': 'center'
            }

        return result

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

    def _calculate_xbar(self, value_col: str = None) -> pd.DataFrame:
        """
        Calculate Xbar (mean) chart statistics.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
            Pass a residual column (R2, R5, etc.) for residual charts.

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
        if ybar_col == 'Ybar' and 'Ybar' in df.columns:
            _Ybar = df['Ybar'].iloc[0]  # Constant across all rows
        else:
            _Ybar = df[value_col].mean()

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
        if spec.by is not None and spec.by != []:
            # Explicit by list - create 'subgroup' column (THE BUG FIX)
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

        # CALCULATE S
        statistics = {}
        statistics['center'] = round(_S, spec.round_to)

        sbar = out.copy()
        sbar['center'] = _S  # Add center column for S chart
        sbar[['lpl', 'upl']] = sbar.apply(
            lambda row: calculate_limits(
                mean=0,
                sd=row['center'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        # Detect beyond limits signals
        # FIX: Should use 's' (varying values) not 'S'/'center' (constant)
        sbar = self._add_beyond_limits_flag(sbar, value_col='s')
        sbar = sbar.round(spec.round_to)

        if n_to_use == "N":
            statistics['N'] = _N
            statistics['upl'] = sbar['upl'].max()
            statistics['lpl'] = sbar['lpl'].max()
        else:
            variable_stats = 'Varies'
            statistics['N'] = variable_stats
            statistics['lpl'] = variable_stats
            statistics['upl'] = variable_stats

        # Apply same grouping transformation as xbar
        if len(groupby_cols) > 1 and group_col == 'group':
            sbar['group'] = sbar[groupby_cols].astype(str).agg('_'.join, axis=1)

        cols_to_keep = [group_col, 's', 'center', 'lpl', 'upl', 'beyond_limits']
        cols_to_keep = [c for c in cols_to_keep if c in sbar.columns]
        sbar = sbar[cols_to_keep]
        result['S'] = {
            'data': sbar,
            'statistics': statistics,
            'metadata': {
                'chart_type': 'S',
                'value_col': 's',
                'center_col': 'center'
            }
        }

        return result

    def _calculate_s(self, value_col: str = None) -> pd.DataFrame:
        """
        Calculate S (standard deviation) chart statistics.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
            Pass a residual column (R2, R5, etc.) for residual charts.

        Logic moved from calculate_statistics_S()
        """
        spec = self.spec
        if value_col is None:
            value_col = spec.response_var

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
        out['center'] = out["s"].mean()
        out['groups'] = out["n"].count()

        # Determine if subgroup sizes are constant or variable
        n_to_use, n_max = self._determine_n_to_use(out)

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
        # FIX: Should use 's' (varying values) not 'center' (constant)
        out = self._add_beyond_limits_flag(out, value_col='s')

        # Determine the grouping column for output
        if spec.by is not None and spec.by != []:
            # Explicit by list - create 'subgroup' column (THE BUG FIX)
            if len(groupby_cols) == 1:
                out['subgroup'] = out[groupby_cols[0]].astype(str)
            else:
                out['subgroup'] = out[groupby_cols].astype(str).agg('_'.join, axis=1)
            group_col = 'subgroup'
        elif len(groupby_cols) > 1:
            # by=None - create 'group' from multiple columns (original behavior)
            out['group'] = out[groupby_cols].astype(str).agg('_'.join, axis=1)
            group_col = 'group'
        elif groupby_cols:
            # Single groupby column
            group_col = groupby_cols[0]
        else:
            group_col = None

        cols_to_keep = [group_col, 's', 'center', 'lpl', 'upl', 'beyond_limits']
        cols_to_keep = [c for c in cols_to_keep if c in out.columns]
        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        # Build statistics
        _S = out['center'].iloc[0] if len(out) > 0 else None
        _N = n_max

        statistics = {'center': round(_S, spec.round_to) if _S else None}
        if n_to_use == "N":
            statistics['N'] = _N
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

    def _calculate_imr(self, value_col: str = None) -> dict:
        """
        Calculate IMR (Individual Moving Range) and R (Range) chart statistics.

        These charts are bundled together following Wheeler's convention of
        always analyzing individuals and moving range charts as a pair.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.
            Pass a residual column (R2, R5, etc.) for residual charts.

        Returns
        -------
        dict
            Chart data with chart type as key:
            - For ungrouped: {'Imr': {...}, 'R': {...}}
            - For stratified: {'Imr': {..., 'strata': [...]}, 'R': {..., 'strata': [...]}}

        Notes
        -----
        Imr and R are bundled together, similar to how Xbar and S are bundled.
        For stratified data, the DataFrame is kept combined (with rsg column)
        and statistics are nested by stratum.

        The `by` parameter controls stratification:
        - by=[] → single chart with all observations, lane boundaries for all factors
        - by=['factor1'] → one chart per factor1, lane boundaries for remaining factors
        - by=['factor1','factor2'] or by=None → one chart per factor combo (current)
        """
        spec = self.spec
        if value_col is None:
            value_col = spec.response_var

        out = self.ads.analysis_dataset.copy()
        result = {}

        logger.debug('In calculate statistics IMR')
        logger.debug('Dataframe has columns: %s', out.columns.to_list())

        # Determine stratification based on `by` parameter
        by = spec.by
        rsg_vars = spec.rsg_vars or []
        time_var = spec.time_var

        # Determine stratify_by and collapsed_factors
        # collapsed_factors = factor variables not in `by` (for lane boundaries)
        # Note: time is NOT included in collapsed_factors - it's expected to change
        if by is None:
            # Default: stratify by all factors (current behavior)
            stratify_by = [spec.rsg_var_name] if rsg_vars else []
            collapsed_factors = []  # No collapsed factors when stratifying by all
        elif by == []:
            # Collapse all: single chart with all observations
            stratify_by = []
            # All factors are collapsed (for lane boundaries)
            collapsed_factors = list(rsg_vars)
        else:
            # Partial: stratify by specified factors
            # Use the by columns directly for stratification
            stratify_by = list(by)
            # Collapsed factors = rsg_vars not in by (time is NOT included)
            collapsed_factors = [v for v in rsg_vars if v not in by]

        # Determine if we're doing stratified or single-stream analysis
        is_stratified = len(stratify_by) > 0

        if is_stratified:
            # Determine the stratification column
            # If stratifying by rsg_var_name equivalent, use it directly
            # Otherwise create/use the specified columns
            if stratify_by == [spec.rsg_var_name]:
                stratify_col = spec.rsg_var_name
            elif len(stratify_by) == 1:
                stratify_col = stratify_by[0]
            else:
                # Multiple stratify columns - create combined key using tuples
                # Tuples avoid collision risk: ('A_B', 'C') != ('A', 'B_C')
                out['_stratify_key'] = out[stratify_by].apply(tuple, axis=1)
                stratify_col = '_stratify_key'

            # Use canonical sort_key for consistent ordering
            out = out.sort_values('sort_key', kind='stable')

            # Moving range per stratum (must exist BEFORE agg)
            out['mr'] = out.groupby(stratify_col, sort=False, observed=True)[value_col].diff().abs()

            # Build a SAFE aggregation spec (only include existing cols)
            agg = {}
            if value_col in out.columns:
                agg['mean'] = (value_col, 'mean')
            if 'mr' in out.columns:
                agg['mR'] = ('mr', 'mean')

            if not agg:
                raise RuntimeError(
                    f"IMR aggregation spec is empty. "
                    f"Have columns: {out.columns.tolist()}, value_col={value_col!r}"
                )

            grouped = (
                out.groupby(stratify_col, sort=False, observed=True)
                .agg(**agg)
                .reset_index()
            )

            # Compute IMR limits per group
            lims = grouped.apply(
                lambda row: calculate_limits(
                    mean=row['mean'],
                    sd=0, N=0,
                    mR=(row['mR'] if pd.notna(row['mR']) else 0.0),
                    limits_type="Imr",
                    round_to=spec.round_to
                ),
                axis=1
            )

            # --- Normalize/join lpl/upl regardless of return type ---
            if isinstance(lims, pd.DataFrame):
                grouped = grouped.join(lims[['lpl','upl']])
            else:
                lims_df = pd.DataFrame(lims.map(_limits_to_pair).tolist(),
                                    index=grouped.index,
                                    columns=['lpl','upl'])
                grouped = grouped.join(lims_df)

            # Rename mean to center for consistency
            grouped = grouped.rename(columns={'mean': 'center'})

            # Attach IMR limits back to rows
            out = out.merge(
                grouped[[stratify_col, 'center', 'mR', 'lpl', 'upl']],
                on=stratify_col, how='left', validate='many_to_one'
            )

            # Detect beyond limits signals for IMR
            out = self._add_beyond_limits_flag(out, value_col=value_col)

            # Get strata list
            strata = grouped[stratify_col].tolist()

            # Calculate lane boundaries for each stratum (where collapsed factors change)
            lane_boundaries = {}
            if collapsed_factors:
                for stratum in strata:
                    stratum_data = out[out[stratify_col] == stratum].reset_index(drop=True)
                    boundaries = self._calculate_lane_boundaries(stratum_data, collapsed_factors)
                    if boundaries:
                        lane_boundaries[stratum] = boundaries

            # Build IMR statistics nested by stratum
            imr_statistics = {}
            for stratum in strata:
                row = grouped[grouped[stratify_col] == stratum].iloc[0]
                imr_statistics[stratum] = {
                    'center': round(row['center'], spec.round_to),
                    'lpl': round(row['lpl'], spec.round_to),
                    'upl': round(row['upl'], spec.round_to)
                }

            # Format IMR output with appropriate columns
            # Include stratify_by columns so plotter can split data correctly
            # Filter out columns already handled by _build_output_columns (rsg_var_name, time_var)
            extra_cols = [c for c in stratify_by
                          if c != spec.rsg_var_name and c != spec.time_var]
            imr_out = self._build_output_columns(
                df=out,
                value_cols=extra_cols + [value_col, 'center', 'lpl', 'upl', 'beyond_limits']
            )

            result['Imr'] = {
                'data': imr_out,
                'statistics': imr_statistics,
                'metadata': {
                    'chart_type': 'Imr',
                    'value_col': value_col,
                    'center_col': 'center',
                    'stratified': True,
                    'lane_boundaries': lane_boundaries if lane_boundaries else None,
                    'stratify_by': list(stratify_by)
                },
                'strata': strata
            }

            # CALCULATE R CHART (bundled with IMR)
            # Compute R limits per group
            r_grouped = grouped.copy()
            r_grouped = r_grouped.rename(columns={'center': 'imr_center', 'mR': 'center'})

            r_lims = r_grouped.apply(
                lambda row: calculate_limits(
                    mean=0,
                    sd=0,
                    N=0,
                    mR=row['center'],
                    limits_type="R",
                    round_to=spec.round_to
                ),
                axis=1
            )

            if isinstance(r_lims, pd.DataFrame):
                r_grouped = r_grouped.join(r_lims[['lpl', 'upl']], rsuffix='_r')
            else:
                r_lims_df = pd.DataFrame(r_lims.map(_limits_to_pair).tolist(),
                                         index=r_grouped.index,
                                         columns=['lpl', 'upl'])
                r_grouped = r_grouped.join(r_lims_df, rsuffix='_r')

            # Ensure lpl/upl columns for R chart
            if 'lpl_r' in r_grouped.columns:
                r_grouped = r_grouped.rename(columns={'lpl_r': 'r_lpl', 'upl_r': 'r_upl'})
                r_lpl_col, r_upl_col = 'r_lpl', 'r_upl'
            else:
                r_lpl_col, r_upl_col = 'lpl', 'upl'

            # Merge R limits to the data
            r_out = out.copy()
            # Drop IMR limits columns
            r_out = r_out.drop(columns=['center', 'lpl', 'upl', 'beyond_limits'], errors='ignore')
            # Merge R limits
            r_merge_data = r_grouped[[stratify_col, 'center', r_lpl_col, r_upl_col]].rename(
                columns={r_lpl_col: 'lpl', r_upl_col: 'upl'}
            )
            r_out = r_out.merge(r_merge_data, on=stratify_col, how='left', validate='many_to_one')

            # Drop NA rows (first observation in each group has no moving range)
            r_out = r_out.dropna(subset=['mr'])

            # Detect beyond limits signals for R
            r_out = self._add_beyond_limits_flag(r_out, value_col='mr')

            # Build R statistics nested by stratum
            r_statistics = {}
            for stratum in strata:
                row = r_grouped[r_grouped[stratify_col] == stratum].iloc[0]
                r_statistics[stratum] = {
                    'center': round(row['center'], spec.round_to),
                    'lpl': round(row[r_lpl_col], spec.round_to),
                    'upl': round(row[r_upl_col], spec.round_to)
                }

            # Format R output with appropriate columns
            # Include stratify_by columns so plotter can split data correctly
            # Filter out columns already handled by _build_output_columns
            r_out = self._build_output_columns(
                df=r_out,
                value_cols=extra_cols + ['mr', 'center', 'lpl', 'upl', 'beyond_limits']
            )

            # Calculate R lane boundaries (adjust for dropped first row per stratum)
            r_lane_boundaries = {}
            if lane_boundaries:
                for stratum, boundaries in lane_boundaries.items():
                    adjusted = []
                    for b in boundaries:
                        # Adjust position since row 0 of each stratum is removed
                        new_pos = b['position'] - 1
                        if new_pos >= 0:
                            adjusted.append({**b, 'position': new_pos})
                    if adjusted:
                        r_lane_boundaries[stratum] = adjusted

            result['R'] = {
                'data': r_out,
                'statistics': r_statistics,
                'metadata': {
                    'chart_type': 'R',
                    'value_col': 'mr',
                    'center_col': 'center',
                    'stratified': True,
                    'lane_boundaries': r_lane_boundaries if r_lane_boundaries else None,
                    'stratify_by': list(stratify_by)
                },
                'strata': strata
            }

        else:
            # Ungrouped path - single stream
            # Use canonical sort_key for consistent ordering
            out = out.sort_values('sort_key', kind='stable')
            out = out.reset_index(drop=True)

            # Calculate lane boundaries before any calculations
            lane_boundaries = None
            if collapsed_factors:
                lane_boundaries = self._calculate_lane_boundaries(out, collapsed_factors)

            out['mr'] = out[value_col].diff().abs()
            mR = out['mr'].mean()
            mean_ = out[value_col].mean()

            # Calculate IMR limits
            imr_lims = calculate_limits(
                mean=mean_, sd=0, N=0, mR=mR,
                limits_type="Imr", round_to=spec.round_to
            )
            out['center'] = mean_
            out['mR'] = mR
            out['lpl'] = imr_lims['lpl']
            out['upl'] = imr_lims['upl']

            # Detect beyond limits signals for IMR
            out = self._add_beyond_limits_flag(out, value_col=value_col)

            # Format IMR output
            imr_out = self._build_output_columns(
                df=out,
                value_cols=[value_col, 'center', 'lpl', 'upl', 'beyond_limits']
            )

            imr_statistics = {
                'center': round(mean_, spec.round_to),
                'lpl': round(imr_lims['lpl'], spec.round_to),
                'upl': round(imr_lims['upl'], spec.round_to)
            }

            result['Imr'] = {
                'data': imr_out,
                'statistics': imr_statistics,
                'metadata': {
                    'chart_type': 'Imr',
                    'value_col': value_col,
                    'center_col': 'center',
                    'lane_boundaries': lane_boundaries
                }
            }

            # CALCULATE R CHART (bundled with IMR)
            r_out = out.copy()
            # Drop NA rows (first observation has no moving range)
            r_out = r_out.dropna(subset=['mr'])

            # Calculate R limits
            r_lims = calculate_limits(
                mean=0, sd=0, N=0, mR=mR,
                limits_type="R", round_to=spec.round_to
            )
            r_out['center'] = mR
            r_out['lpl'] = r_lims['lpl']
            r_out['upl'] = r_lims['upl']

            # Detect beyond limits signals for R (using 'mr' as value column)
            # Need to reset beyond_limits since we're checking a different value
            r_out = r_out.drop(columns=['beyond_limits'], errors='ignore')
            r_out = self._add_beyond_limits_flag(r_out, value_col='mr')

            # Format R output
            r_out = self._build_output_columns(
                df=r_out,
                value_cols=['mr', 'center', 'lpl', 'upl', 'beyond_limits']
            )

            r_statistics = {
                'center': round(mR, spec.round_to),
                'lpl': round(r_lims['lpl'], spec.round_to),
                'upl': round(r_lims['upl'], spec.round_to)
            }

            # Adjust lane boundaries for R chart (first row dropped)
            r_lane_boundaries = None
            if lane_boundaries:
                r_lane_boundaries = []
                for b in lane_boundaries:
                    # Adjust position since row 0 is removed
                    new_pos = b['position'] - 1
                    if new_pos >= 0:
                        r_lane_boundaries.append({
                            **b,
                            'position': new_pos
                        })
                if not r_lane_boundaries:
                    r_lane_boundaries = None

            result['R'] = {
                'data': r_out,
                'statistics': r_statistics,
                'metadata': {
                    'chart_type': 'R',
                    'value_col': 'mr',
                    'center_col': 'center',
                    'lane_boundaries': r_lane_boundaries
                }
            }

        return result

    def _calculate_r(self, value_col: str = None) -> pd.DataFrame:
        """
        Calculate R (Range) chart statistics.

        Parameters
        ----------
        value_col : str, optional
            Column to use for calculations. Defaults to spec.response_var.

        Logic moved from calculate_statistics_R()
        """
        spec = self.spec
        if value_col is None:
            value_col = spec.response_var

        out = self.ads.analysis_dataset.copy()

        logger.debug('In calculate statistics R')
        logger.debug('Dataframe has columns: %s', out.columns.to_list())

        if spec.has_grouping:
            out['mr'] = abs(out.groupby(spec.rsg_var_name, observed=True)[value_col].diff())
            grouped = out.groupby(spec.rsg_var_name, as_index=False, observed=True)
            # Rename 'mR' to 'center' for consistency
            grouped = grouped.agg(center=pd.NamedAgg('mr', 'mean'))

            limits = grouped.apply(
                lambda row: calculate_limits(
                    mean=0,
                    sd=0,
                    N=0,
                    mR=row.center,
                    limits_type="R",
                    round_to=spec.round_to
                ), axis=1
            )

            grouped = pd.merge(grouped, limits, left_index=True, right_index=True)
            out = pd.merge(out, grouped, how='left', on=spec.rsg_var_name)
        else:
            out['mr'] = abs(out[value_col].diff())
            # Rename 'mR' to 'center' for consistency
            out['center'] = out['mr'].mean()
            center_val = out['center'].max()
            limits = calculate_limits(
                mean=0,
                sd=0,
                N=0,
                mR=center_val,
                limits_type="R",
                round_to=spec.round_to
            )
            out['lpl'] = limits['lpl']
            out['upl'] = limits['upl']

        # Drop NAs
        out = out.dropna()

        # Detect beyond limits signals
        out = self._add_beyond_limits_flag(out, value_col='mr')

        # Format output with appropriate columns
        out = self._build_output_columns(
            df=out,
            value_cols=['mr', 'center', 'lpl', 'upl', 'beyond_limits']
        )

        # Package results with statistics and metadata
        return self._package_stratified_results(
            df=out,
            statistics_cols=['center', 'lpl', 'upl'],
            chart_type='R',
            value_col='mr'
        )

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
            value_col = spec.value_col if spec.value_col else spec.response_var

        data = self.ads.analysis_dataset.copy()
        values = data[value_col].dropna()

        # Calculate global statistics
        n = len(values)
        mean = values.mean() if n > 0 else float('nan')
        std = values.std() if n >= 2 else float('nan')

        # Handle stratification via `by` parameter
        by = spec.by if spec.by is not None else []

        if len(by) > 0:
            # Stratified histogram - one per stratum
            # Use pandas groupby with list of columns (no collision risk)
            grouped = data.groupby(by, observed=True)

            # Build strata list - tuples for multi-key, values for single-key
            if len(by) == 1:
                strata = data[by[0]].unique().tolist()
            else:
                # Use tuple keys to avoid collision (('A_B','C') != ('A','B_C'))
                strata = list(grouped.groups.keys())

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
                        'bins': spec.bins,
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
                    'bins': spec.bins
                }
            }
        }
