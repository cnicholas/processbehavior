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
    s = result.get_chart('Sbar')
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

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
            stats['all'] = summarized

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
        analysis = Analysis(df, specification)
        result = analysis.calculate()
    """

    def __init__(self, df: pd.DataFrame, specification: dict):
        """
        Initialize analysis with data and specification.

        Args:
            df: Input DataFrame with raw data
            specification: Dictionary containing analysis configuration including 'analysis_type'
        """
        # Import here to avoid circular dependency
        from .analysis_dataset import AnalysisDataSet

        self.raw_df = df
        self.spec = AnalysisSpecification(specification)
        self.analysis_type = self.spec.analysis_type
        self.ads = AnalysisDataSet(df, self.spec)

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
        strategies = {
            'Xbar': self._calculate_xbar,
            'S': self._calculate_s,
            'Imr': self._calculate_imr,
            'R': self._calculate_r
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

        logger.info('Zero-centering data')
        result = df.copy()
        zero_mean = result[self.spec.response_var].mean()
        logger.debug('Zero-mean: %s', zero_mean)
        result[self.spec.response_var] = result[self.spec.response_var] - zero_mean

        return result

    def _add_beyond_limits_flag(
        self,
        df: pd.DataFrame,
        value_col: str,
        lcl_col: str = 'lcl',
        ucl_col: str = 'ucl'
    ) -> pd.DataFrame:
        """
        Add beyond_limits flag column.

        Returns -1 for below LCL, 1 for above UCL, 0 for in control.

        Pure function: doesn't modify input.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with control limits
        value_col : str
            Column name containing values to check
        lcl_col : str, default 'lcl'
            Column name for lower control limit
        ucl_col : str, default 'ucl'
            Column name for upper control limit

        Returns
        -------
        pd.DataFrame
            Data with added 'beyond_limits' column
        """
        result = df.copy()

        result['beyond_limits'] = result.apply(
            lambda row: detect_beyond_limits(
                x=row[value_col],
                ucl=row[ucl_col],
                lcl=row[lcl_col]
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

        logger.info(
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
            Columns to collect statistics for (e.g., ['mean', 'lcl', 'ucl'])
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
        ...     statistics_cols=['center', 'lcl', 'ucl'],
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
            split_dict = {'all': df}

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
            Columns to keep (e.g., ['mean', 'lcl', 'ucl', 'beyond_limits'])

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
        ...     value_cols=[spec.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']
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
            if self.spec.has_grouping:
                result['x'] = result.groupby(self.spec.rsg_var_name, observed=True).cumcount() + 1
                cols_to_keep.insert(0, self.spec.rsg_var_name)
                cols_to_keep.insert(0, 'x')
            else:
                result['x'] = result.index + 1
                cols_to_keep.insert(0, 'x')

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
        xbar[['lcl', 'ucl']] = xbar.apply(
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
            statistics['ucl'] = xbar['ucl'].max()
            statistics['lcl'] = xbar['lcl'].max()
        else:
            variable_stats = 'Varies'
            statistics['N'] = variable_stats
            statistics['lcl'] = variable_stats
            statistics['ucl'] = variable_stats

        cols_to_keep = ['rsg', 'xbar', 'center', 'lcl', 'ucl', 'beyond_limits']
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
        sbar[['lcl', 'ucl']] = sbar.apply(
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
            statistics['ucl'] = sbar['ucl'].max()
            statistics['lcl'] = sbar['lcl'].max()
        else:
            variable_stats = 'Varies'
            statistics['N'] = variable_stats
            statistics['lcl'] = variable_stats
            statistics['ucl'] = variable_stats

        cols_to_keep = ['rsg', 's', 'center', 'lcl', 'ucl', 'beyond_limits']
        sbar = sbar[cols_to_keep]
        result['Sbar'] = {
            'data': sbar,
            'statistics': statistics,
            'metadata': {
                'chart_type': 'Sbar',
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
        out[['lcl', 'ucl']] = out.apply(
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

        cols_to_keep = ['rsg', 's', 'center', 'lcl', 'ucl', 'beyond_limits']
        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        return out

    def _calculate_imr(self) -> pd.DataFrame:
        """
        Calculate IMR (Individual Moving Range) chart statistics.

        Logic moved from calculate_statistics_Imr()
        """
        y = self.spec.response_var
        spec = self.spec
        out = self.ads.analysis_dataset.copy()

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

            # Compute limits per group
            # grouped has columns: [spec.rsg_var_name, 'mean', 'mR']
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

            # --- Normalize/join lcl/ucl regardless of return type ---
            if isinstance(lims, pd.DataFrame):
                # Your case: lims already has columns ['lcl','ucl'], index aligned to grouped
                grouped = grouped.join(lims[['lcl','ucl']])
            else:
                # lims is a Series; each element could be dict/Series/tuple
                def _to_pair(x):
                    if isinstance(x, dict):
                        return x.get('lcl', np.nan), x.get('ucl', np.nan)
                    try:
                        return x['lcl'], x['ucl']         # pandas Series-like
                    except Exception:
                        pass
                    if hasattr(x, 'lcl') and hasattr(x, 'ucl'):
                        return getattr(x, 'lcl', np.nan), getattr(x, 'ucl', np.nan)
                    if isinstance(x, (list, tuple)) and len(x) >= 2:
                        return x[0], x[1]
                    return np.nan, np.nan

                lims_df = pd.DataFrame(lims.map(_to_pair).tolist(),
                                    index=grouped.index,
                                    columns=['lcl','ucl'])
                grouped = grouped.join(lims_df)


            # Rename mean to center for consistency
            grouped = grouped.rename(columns={'mean': 'center'})

            # Attach back to rows
            out = out.merge(
                grouped[[spec.rsg_var_name, 'center', 'mR', 'lcl', 'ucl']],
                on=spec.rsg_var_name, how='left', validate='many_to_one'
            )
        else:
            # (unchanged single-stream path, but keep same style)
            if spec.has_time:
                out = out.sort_values([spec.time_var], kind='stable')
            out['mr'] = out[y].diff().abs()
            mR = out['mr'].mean()
            mean_ = out[y].mean()
            lims = calculate_limits(
                mean=mean_, sd=0, N=0, mR=mR,
                limits_type="Imr", round_to=spec.round_to
            )
            out['center'] = mean_  # Renamed from 'mean' to 'center'
            out['mR']   = mR
            out['lcl']  = lims['lcl']
            out['ucl']  = lims['ucl']

        # Detect beyond limits signals
        out = self._add_beyond_limits_flag(out, value_col=spec.response_var)

        # Format output with appropriate columns
        out = self._build_output_columns(
            df=out,
            value_cols=[spec.response_var, 'center', 'lcl', 'ucl', 'beyond_limits']
        )

        # Package results with statistics and metadata
        return self._package_stratified_results(
            df=out,
            statistics_cols=['center', 'lcl', 'ucl'],
            chart_type='Imr',
            value_col=spec.response_var
        )

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
            out['lcl'] = limits['lcl']
            out['ucl'] = limits["ucl"]

        # Drop NAs
        out = out.dropna()

        # Detect beyond limits signals
        out = self._add_beyond_limits_flag(out, value_col='mr')

        # Format output with appropriate columns
        out = self._build_output_columns(
            df=out,
            value_cols=['mr', 'center', 'lcl', 'ucl', 'beyond_limits']
        )

        # Package results with statistics and metadata
        return self._package_stratified_results(
            df=out,
            statistics_cols=['center', 'lcl', 'ucl'],
            chart_type='R',
            value_col='mr'
        )
