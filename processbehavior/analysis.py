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
            # Residual chart analysis
            chart_type = getattr(self.spec, 'residual_chart_type', 'Imr')
            recentered = getattr(self.spec, 'recentered', False)

            chart_data = self._calculate_residual_chart(
                residual=residual,
                chart_type=chart_type,
                recentered=recentered
            )
        else:
            # Standard chart analysis
            # Note: Imr and R are bundled together (both call _calculate_imr)
            # just like Xbar and S are bundled together
            strategies = {
                'Xbar': self._calculate_xbar,
                'S': self._calculate_s,
                'Imr': self._calculate_imr,
                'R': self._calculate_imr  # Bundled with Imr
            }

            if self.analysis_type not in strategies:
                raise ValueError(
                    f'Analysis type {self.analysis_type} not supported! '
                    f'Valid types: {list(strategies.keys())}'
                )

            # Execute analysis strategy
            chart_data = strategies[self.spec.analysis_type]()

        # Wrap in AnalysisResult for unified access
        return AnalysisResult(
            charts=chart_data,
            analysis_dataset_obj=self.ads
        )

    # =========================================================================
    # Helper Methods (DRY principle)
    # =========================================================================

    def _apply_zero_centering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zero-center response variable if specified.

        Pure function approach: doesn't modify input, returns new DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with response variable

        Returns
        -------
        pd.DataFrame
            Data with zero-centered response variable (if spec.zero_center is True)
        """
        if not self.spec.zero_center:
            return df

        logger.debug('Zero-centering data')
        result = df.copy()
        zero_mean = result[self.spec.response_var].mean()
        logger.debug('Zero-mean: %s', zero_mean)
        result[self.spec.response_var] = result[self.spec.response_var] - zero_mean

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

    def _calculate_xbar(self) -> pd.DataFrame:
        """
        Calculate Xbar (mean) chart statistics.

        Logic moved from Xbar.calculate_statistics()
        """
        #df = self.ads.analysis_dataset
        spec = self.spec
        result = {}
        statistics = {}
        out = self.ads.analysis_dataset.copy()

        logger.debug('In calculate statistics XbarS')
        logger.debug('Dataframe has columns: %s', out.columns.to_list())
        logger.debug('Dataframe head:\n%s', out.head(10))
        logger.debug('n.max=%s', out["n"].max())

        # Apply zero-centering if requested
        out = self._apply_zero_centering(out)

        out = out.groupby(spec.rsg_var_name, as_index=False, observed=True).agg(
            s=pd.NamedAgg(column=spec.response_var, aggfunc="std"),
            mean=pd.NamedAgg(column=spec.response_var, aggfunc="mean"),
            n=pd.NamedAgg(column='n', aggfunc="max")
        )

        # Handle case where no subgroups have >1 observation
        if out.shape[0] == 0:
            raise ValueError("All subgroups have 1 or less observations!")

        # Rename columns for consistency: mean→xbar (values)
        out = out.rename(columns={'mean': 'xbar'})

        _Xbar = out["xbar"].mean()
        _S = out["s"].mean()
        _N = out['n'].max()
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

        cols_to_keep = ['rsg', 'xbar', 'center', 'lpl', 'upl', 'beyond_limits']
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

        cols_to_keep = ['rsg', 's', 'center', 'lpl', 'upl', 'beyond_limits']
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

    def _calculate_s(self) -> pd.DataFrame:
        """
        Calculate S (standard deviation) chart statistics.

        Logic moved from calculate_statistics_S()
        """
        spec = self.spec
        out = self.ads.analysis_dataset.copy()

        out = out.groupby(spec.rsg_var_name, as_index=False, observed=True).agg(
            s=pd.NamedAgg(column=spec.response_var, aggfunc="std"),
            n=pd.NamedAgg(column=spec.rsg_var_name, aggfunc="count"),
        )

        # remove RSGs with a single observation
        mask = out['n'].eq(1)
        out = out[~mask]

        # Rename 'S' to 'center' for consistency
        out['center'] = out["s"].mean()
        out['groups'] = out["n"].count()
        out['N'] = out['n'].max()

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

        cols_to_keep = ['rsg', 's', 'center', 'lpl', 'upl', 'beyond_limits']
        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        return out

    def _calculate_xbar_by_time(self) -> dict:
        """
        Calculate Xbar/S charts with TIME as the rational subgroup.

        Used for R4 residuals per Wheeler/Bishop Section 20.6.3.
        Subgroups are time points, aggregating across all factor levels.
        Sample size per subgroup: N_.t = Σ_k N_kt

        Returns dict with 'Xbar' and 'S' chart data.
        """
        spec = self.spec
        result = {}
        out = self.ads.analysis_dataset.copy()

        # Group by TIME instead of factor (rsg_var_name)
        out = out.groupby(spec.time_var, as_index=False, observed=True).agg(
            s=pd.NamedAgg(column=spec.response_var, aggfunc="std"),
            mean=pd.NamedAgg(column=spec.response_var, aggfunc="mean"),
            n=pd.NamedAgg(column=spec.response_var, aggfunc="count")  # Count observations per time
        )

        if out.shape[0] == 0:
            raise ValueError("No time subgroups with observations!")

        # Rename for consistency: time_var→rsg (to match chart column naming)
        out = out.rename(columns={spec.time_var: 'rsg', 'mean': 'xbar'})

        _Xbar = out["xbar"].mean()
        _S = out["s"].mean()
        _N = out['n'].max()
        out['N'] = _N

        # Determine if subgroup sizes are constant or variable
        n_to_use, n_max = self._determine_n_to_use(out)

        # CALCULATE XBAR
        statistics = {}
        xbar = out.copy()
        xbar['center'] = _Xbar
        xbar[['lpl', 'upl']] = xbar.apply(
            lambda row: calculate_limits(
                mean=row['center'],
                sd=_S,
                N=row[n_to_use],
                limits_type='Xbar',
                round_to=spec.round_to
            ), axis=1
        )

        xbar = self._add_beyond_limits_flag(xbar, value_col='xbar')
        xbar = xbar.round(spec.round_to)

        statistics['center'] = round(_Xbar, spec.round_to)
        if n_to_use == "N":
            statistics['N'] = _N
            statistics['upl'] = xbar['upl'].max()
            statistics['lpl'] = xbar['lpl'].max()
        else:
            statistics['N'] = 'Varies'
            statistics['lpl'] = 'Varies'
            statistics['upl'] = 'Varies'

        cols_to_keep = ['rsg', 'xbar', 'center', 'lpl', 'upl', 'beyond_limits']
        xbar = xbar[cols_to_keep]
        result['Xbar'] = {
            'data': xbar,
            'statistics': statistics,
            'metadata': {
                'chart_type': 'Xbar',
                'value_col': 'xbar',
                'center_col': 'center',
                'subgroup_type': 'time'
            }
        }

        # CALCULATE S
        statistics = {}
        statistics['center'] = round(_S, spec.round_to)

        sbar = out.copy()
        sbar['center'] = _S
        sbar[['lpl', 'upl']] = sbar.apply(
            lambda row: calculate_limits(
                mean=0,
                sd=row['center'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        sbar = self._add_beyond_limits_flag(sbar, value_col='s')
        sbar = sbar.round(spec.round_to)

        if n_to_use == "N":
            statistics['N'] = _N
            statistics['upl'] = sbar['upl'].max()
            statistics['lpl'] = sbar['lpl'].max()
        else:
            statistics['N'] = 'Varies'
            statistics['lpl'] = 'Varies'
            statistics['upl'] = 'Varies'

        cols_to_keep = ['rsg', 's', 'center', 'lpl', 'upl', 'beyond_limits']
        sbar = sbar[cols_to_keep]
        result['S'] = {
            'data': sbar,
            'statistics': statistics,
            'metadata': {
                'chart_type': 'S',
                'value_col': 's',
                'center_col': 'center',
                'subgroup_type': 'time'
            }
        }

        return result

    def _calculate_s_by_time(self) -> pd.DataFrame:
        """
        Calculate S chart with TIME as the rational subgroup.

        Used for R4 residuals per Wheeler/Bishop Section 20.6.3.
        Subgroups are time points, aggregating across all factor levels.
        """
        spec = self.spec
        out = self.ads.analysis_dataset.copy()

        # Group by TIME instead of factor
        out = out.groupby(spec.time_var, as_index=False, observed=True).agg(
            s=pd.NamedAgg(column=spec.response_var, aggfunc="std"),
            n=pd.NamedAgg(column=spec.response_var, aggfunc="count"),
        )

        # Rename time_var to rsg for consistency
        out = out.rename(columns={spec.time_var: 'rsg'})

        # Remove subgroups with single observation (can't calculate std)
        mask = out['n'].eq(1)
        out = out[~mask]

        out['center'] = out["s"].mean()
        out['groups'] = out["n"].count()
        out['N'] = out['n'].max()

        n_to_use, n_max = self._determine_n_to_use(out)

        out[['lpl', 'upl']] = out.apply(
            lambda row: calculate_limits(
                mean=0,
                sd=row['center'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        out = self._add_beyond_limits_flag(out, value_col='s')

        cols_to_keep = ['rsg', 's', 'center', 'lpl', 'upl', 'beyond_limits']
        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        return out

    def _calculate_imr(self) -> dict:
        """
        Calculate IMR (Individual Moving Range) and R (Range) chart statistics.

        These charts are bundled together following Wheeler's convention of
        always analyzing individuals and moving range charts as a pair.

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
        """
        y = self.spec.response_var
        spec = self.spec
        out = self.ads.analysis_dataset.copy()
        result = {}

        # Apply zero-centering if requested
        out = self._apply_zero_centering(out)

        logger.debug('In calculate statistics IMR')
        logger.debug('Dataframe has columns: %s', out.columns.to_list())

        if spec.has_grouping:
            # Stable order before diff
            sort_cols = [spec.rsg_var_name]
            if spec.has_time:
                sort_cols.append(spec.time_var)
            if sort_cols:
                out = out.sort_values(sort_cols, kind='stable')

            # Moving range per subgroup (must exist BEFORE agg)
            out['mr'] = out.groupby(spec.rsg_var_name, sort=False, observed=True)[y].diff().abs()

            # Build a SAFE aggregation spec (only include existing cols)
            agg = {}
            if y in out.columns:
                agg['mean'] = (y, 'mean')
            if 'mr' in out.columns:
                agg['mR'] = ('mr', 'mean')

            if not agg:
                raise RuntimeError(
                    f"IMR aggregation spec is empty. "
                    f"Have columns: {out.columns.tolist()}, y={y!r}"
                )

            grouped = (
                out.groupby(spec.rsg_var_name, sort=False, observed=True)
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
                def _to_pair(x):
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

                lims_df = pd.DataFrame(lims.map(_to_pair).tolist(),
                                    index=grouped.index,
                                    columns=['lpl','upl'])
                grouped = grouped.join(lims_df)

            # Rename mean to center for consistency
            grouped = grouped.rename(columns={'mean': 'center'})

            # Attach IMR limits back to rows
            out = out.merge(
                grouped[[spec.rsg_var_name, 'center', 'mR', 'lpl', 'upl']],
                on=spec.rsg_var_name, how='left', validate='many_to_one'
            )

            # Detect beyond limits signals for IMR
            out = self._add_beyond_limits_flag(out, value_col=spec.response_var)

            # Get strata list
            strata = grouped[spec.rsg_var_name].tolist()

            # Build IMR statistics nested by stratum
            imr_statistics = {}
            for stratum in strata:
                row = grouped[grouped[spec.rsg_var_name] == stratum].iloc[0]
                imr_statistics[stratum] = {
                    'center': round(row['center'], spec.round_to),
                    'lpl': round(row['lpl'], spec.round_to),
                    'upl': round(row['upl'], spec.round_to)
                }

            # Format IMR output with appropriate columns
            imr_out = self._build_output_columns(
                df=out,
                value_cols=[spec.response_var, 'center', 'lpl', 'upl', 'beyond_limits']
            )

            result['Imr'] = {
                'data': imr_out,
                'statistics': imr_statistics,
                'metadata': {
                    'chart_type': 'Imr',
                    'value_col': spec.response_var,
                    'center_col': 'center',
                    'stratified': True
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
                r_lims_df = pd.DataFrame(r_lims.map(_to_pair).tolist(),
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
            r_merge_cols = [spec.rsg_var_name, 'center', r_lpl_col, r_upl_col]
            r_merge_data = r_grouped[[spec.rsg_var_name, 'center', r_lpl_col, r_upl_col]].rename(
                columns={r_lpl_col: 'lpl', r_upl_col: 'upl'}
            )
            r_out = r_out.merge(r_merge_data, on=spec.rsg_var_name, how='left', validate='many_to_one')

            # Drop NA rows (first observation in each group has no moving range)
            r_out = r_out.dropna(subset=['mr'])

            # Detect beyond limits signals for R
            r_out = self._add_beyond_limits_flag(r_out, value_col='mr')

            # Build R statistics nested by stratum
            r_statistics = {}
            for stratum in strata:
                row = r_grouped[r_grouped[spec.rsg_var_name] == stratum].iloc[0]
                r_statistics[stratum] = {
                    'center': round(row['center'], spec.round_to),
                    'lpl': round(row[r_lpl_col], spec.round_to),
                    'upl': round(row[r_upl_col], spec.round_to)
                }

            # Format R output with appropriate columns
            r_out = self._build_output_columns(
                df=r_out,
                value_cols=['mr', 'center', 'lpl', 'upl', 'beyond_limits']
            )

            result['R'] = {
                'data': r_out,
                'statistics': r_statistics,
                'metadata': {
                    'chart_type': 'R',
                    'value_col': 'mr',
                    'center_col': 'center',
                    'stratified': True
                },
                'strata': strata
            }

        else:
            # Ungrouped path - single stream
            if spec.has_time:
                out = out.sort_values([spec.time_var], kind='stable')
            out['mr'] = out[y].diff().abs()
            mR = out['mr'].mean()
            mean_ = out[y].mean()

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
            out = self._add_beyond_limits_flag(out, value_col=spec.response_var)

            # Format IMR output
            imr_out = self._build_output_columns(
                df=out,
                value_cols=[spec.response_var, 'center', 'lpl', 'upl', 'beyond_limits']
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
                    'value_col': spec.response_var,
                    'center_col': 'center'
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

            result['R'] = {
                'data': r_out,
                'statistics': r_statistics,
                'metadata': {
                    'chart_type': 'R',
                    'value_col': 'mr',
                    'center_col': 'center'
                }
            }

        return result

    def _calculate_residual_chart(
        self,
        residual: str,
        chart_type: str,
        recentered: bool = False
    ) -> dict:
        """
        Calculate control chart from residual column.

        This method creates control charts from VAS residuals (R2-R5) to answer
        specific questions about variance sources:

        - R2_S or R2_Imr: Is unexplained within-cell variation stable?
        - R3_Imr: Is there significant interaction between factors and time?
        - R4_Xbar/R4_S/R4_Imr: Does time have a significant effect?
        - R5_Xbar/R5_S/R5_Imr: Does the factor have a significant effect?

        R4 and R5 use different rational subgrouping per Wheeler/Bishop:
        - R4: Subgroups by TIME (aggregate across factors), N_.t = Σ_k N_kt
        - R5: Subgroups by FACTOR (aggregate across time), N_k. = Σ_t N_kt

        Parameters
        ----------
        residual : str
            Residual type ('R2', 'R3', 'R4', 'R5')
        chart_type : str
            Base chart type ('S', 'Xbar', or 'Imr')
        recentered : bool, default False
            If True, use re-centered residuals (RCR2, RCR3, etc.)
            Re-centered residuals add back the appropriate mean for interpretation.

        Returns
        -------
        dict
            Chart data in standard format:
            {'ChartName': {'data': DataFrame, 'statistics': dict, 'metadata': dict}}

        Raises
        ------
        ValueError
            If residual column not found in dataset
            If chart_type is not supported for the residual

        Notes
        -----
        Re-centered residuals (Tom Bishop Equation 80):
            RCR = R + Ȳ (appropriate mean for each residual type)

        The question each residual answers:
            R2: Within-subgroup variation (measurement noise)
            R3: Interaction effects (factor × time)
            R4: Time effects (trends, shifts over time)
            R5: Factor effects (differences between levels)
        """
        # Determine column name
        col_prefix = 'RCR' if recentered else 'R'
        residual_num = residual[1]  # Extract number from 'R2', 'R3', etc.
        col_name = f'{col_prefix}{residual_num}'

        # Check column exists
        if col_name not in self.ads.analysis_dataset.columns:
            available = [c for c in self.ads.analysis_dataset.columns
                        if c.startswith('R') and len(c) == 2]
            raise ValueError(
                f"Residual column '{col_name}' not found.\n"
                f"Available residuals: {available}\n"
                f"This may indicate the SDS doesn't support this residual type."
            )

        # Question answered by each residual
        questions = {
            'R2': 'Is within-subgroup variation stable?',
            'R3': 'Is there interaction between factors and time?',
            'R4': 'Does time have a significant effect?',
            'R5': 'Do factors have a significant effect?'
        }

        # Create a modified spec for residual analysis
        # We temporarily treat the residual column as the response variable
        original_response = self.spec.response_var

        # Store original values and temporarily modify
        self.spec._response_var = col_name

        try:
            if chart_type == 'S':
                # S chart - R4 uses time-based subgrouping, others use factor-based
                s_result = self._calculate_s_by_time() if residual == 'R4' else self._calculate_s()

                # Wrap in dict format if needed
                if isinstance(s_result, pd.DataFrame):
                    chart_name = f'{residual}_S'

                    # Check if limits vary (different values across rows)
                    lpl_varies = s_result['lpl'].nunique() > 1 if 'lpl' in s_result.columns else False
                    upl_varies = s_result['upl'].nunique() > 1 if 'upl' in s_result.columns else False
                    limits_vary = lpl_varies or upl_varies

                    if limits_vary:
                        statistics = {
                            'center': s_result['center'].iloc[0] if 'center' in s_result.columns else None,
                            'lpl': 'Varies',
                            'upl': 'Varies',
                        }
                    else:
                        statistics = {
                            'center': s_result['center'].iloc[0] if 'center' in s_result.columns else None,
                            'lpl': s_result['lpl'].iloc[0] if 'lpl' in s_result.columns else None,
                            'upl': s_result['upl'].iloc[0] if 'upl' in s_result.columns else None,
                        }

                    result = {
                        chart_name: {
                            'data': s_result,
                            'statistics': statistics,
                            'metadata': {
                                'chart_type': 'S',
                                'value_col': 's',
                                'center_col': 'center',
                                'residual_type': residual,
                                'recentered': recentered,
                                'question_answered': questions.get(residual, '')
                            }
                        }
                    }
            elif chart_type == 'Xbar':
                # Xbar chart for R3/R4/R5 with different subgrouping
                # R4: subgroups by TIME (aggregate across factors)
                # R3/R5: subgroups by FACTOR (aggregate across time) - uses standard _calculate_xbar
                xbar_result = self._calculate_xbar_by_time() if residual == 'R4' else self._calculate_xbar()

                # Extract just the Xbar portion (not S)
                chart_name = f'{residual}_Xbar'
                xbar_data = xbar_result['Xbar']

                result = {
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

            elif chart_type == 'Imr':
                # IMR chart for R3, R4, R5 (and R2 when no replication)
                result = self._calculate_imr()

                # Add residual metadata to each chart
                for chart_key in result:
                    if 'metadata' not in result[chart_key]:
                        result[chart_key]['metadata'] = {}
                    result[chart_key]['metadata']['residual_type'] = residual
                    result[chart_key]['metadata']['recentered'] = recentered
                    result[chart_key]['metadata']['question_answered'] = questions.get(residual, '')

                # Rename chart keys to include residual prefix
                renamed_result = {}
                for chart_key, chart_data in result.items():
                    new_key = f'{residual}_{chart_key}'
                    renamed_result[new_key] = chart_data

                result = renamed_result
            else:
                raise ValueError(
                    f"Chart type '{chart_type}' not supported for residual charts.\n"
                    f"Valid types: 'S', 'Xbar' (for R4/R5), 'Imr' (for R2-R5)"
                )
        finally:
            # Restore original response variable
            self.spec._response_var = original_response

        return result

    def _calculate_r(self) -> pd.DataFrame:
        """
        Calculate R (Range) chart statistics.

        Logic moved from calculate_statistics_R()
        """
        spec = self.spec
        out = self.ads.analysis_dataset.copy()

        # Apply zero-centering if requested
        out = self._apply_zero_centering(out)

        logger.debug('In calculate statistics R')
        logger.debug('Dataframe has columns: %s', out.columns.to_list())

        if spec.has_grouping:
            out['mr'] = abs(out.groupby(spec.rsg_var_name, observed=True)[spec.response_var].diff())
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
            out['mr'] = abs(out[spec.response_var].diff())
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
