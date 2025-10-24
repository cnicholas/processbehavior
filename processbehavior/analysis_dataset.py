
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .analysis_result import AnalysisResult
from .analysis_specification import AnalysisSpecification
from .data_preparation import DataPreparation
from .effects_calculator import EffectsCalculator
from .residual_calculator import ResidualCalculator
from .sds_detector import SamplingDesignDetector
from .spc_constants import calculate_limits, detect_beyond_limits

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# NEW: Refactored Analysis Class (replacing factory pattern)
# ============================================================================

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
        self.raw_df = df
        self.analysis_type = specification['analysis_type']
        self.spec = AnalysisSpecification(self.analysis_type, specification)
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

        # Check if stratification is requested
        stratify_vars = self.spec.spec.get('stratify')

        if stratify_vars:
            # Stratified analysis: create separate charts for each stratum
            strategy = strategies[self.spec.analysis_type]
            chart_data = self._calculate_stratified(strategy, stratify_vars)
        else:
            # Standard analysis: single combined chart
            chart_data = strategies[self.spec.analysis_type]()

        # Wrap in AnalysisResult for unified access
        return AnalysisResult(
            charts=chart_data,
            analysis_dataset_obj=self.ads
        )

    def _calculate_stratified(self, chart_method, stratify_vars: list) -> dict:
        """
        Execute stratified analysis: separate charts for each level of stratify variables.

        Args:
            chart_method: The chart calculation method to run for each stratum
            stratify_vars: List of variables to stratify by

        Returns:
            dict: Chart data with stratified results

        Example:
            If stratify_vars=['Operator'] with levels A, B, C:
            Returns {'Imr_Operator_A': {...}, 'Imr_Operator_B': {...}, 'Imr_Operator_C': {...}}
        """
        logger.info(f"Executing stratified analysis by: {stratify_vars}")

        # Get prepared data
        df = self.ads.analysis_dataset.copy()

        # Create stratification column (combination of stratify vars)
        if len(stratify_vars) == 1:
            strat_col = stratify_vars[0]
        else:
            # Combine multiple variables into single stratification key
            strat_col = '_'.join(stratify_vars)
            df[strat_col] = df[stratify_vars].apply(lambda x: '_'.join(x.astype(str)), axis=1)

        # Get unique strata
        strata = df[strat_col].unique()
        logger.info(f"Found {len(strata)} strata: {list(strata)}")

        # Validate minimum data per stratum
        for stratum in strata:
            stratum_df = df[df[strat_col] == stratum]
            if len(stratum_df) < 2:
                logger.warning(
                    f"Stratum '{stratum}' has only {len(stratum_df)} observation(s). "
                    f"IMR charts require at least 2 observations."
                )

        # Calculate chart for each stratum
        all_charts = {}

        for stratum in strata:
            logger.info(f"Calculating chart for stratum: {stratum}")

            # Filter data for this stratum
            stratum_df = df[df[strat_col] == stratum].copy()

            # Temporarily swap out the dataset for this stratum
            original_dataset = self.ads.analysis_dataset
            self.ads.analysis_dataset = stratum_df

            try:
                # Run the chart calculation for this stratum
                stratum_charts = chart_method()

                # Add stratum prefix to chart names
                for chart_name, chart_data in stratum_charts.items():
                    stratified_name = f"{chart_name}_{strat_col}_{stratum}"
                    all_charts[stratified_name] = chart_data

            finally:
                # Restore original dataset
                self.ads.analysis_dataset = original_dataset

        logger.info(f"Stratified analysis complete: {len(all_charts)} charts created")
        return all_charts

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
        statistics_cols: list[str]
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

        Returns
        -------
        dict
            Packaged results: {group: {'data': df, 'statistics': dict}}
            For ungrouped data: {'all': {'data': df, 'statistics': dict}}

        Examples
        --------
        >>> result = self._package_stratified_results(
        ...     df=out,
        ...     statistics_cols=['mean', 'lcl', 'ucl']
        ... )
        """
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

        return package_analysis(
            analysis_output=split_dict,
            summary_statistics_output=statistics
        )

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
                result['x'] = result.groupby(self.spec.rsg_var_name).cumcount() + 1
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
        logger.debug('Dataframe head:\\n%s', out.head(10))
        logger.debug('n.max=%s', out["n"].max())

        # Apply zero-centering if requested
        out = self._apply_zero_centering(out)

        out = out.groupby(spec.rsg_var_name, as_index=False).agg(
            s=pd.NamedAgg(column=spec.response_var, aggfunc="std"),
            mean=pd.NamedAgg(column=spec.response_var, aggfunc="mean"),
            n=pd.NamedAgg(column='n', aggfunc="max")
        )

        # Handle case where no subgroups have >1 observation
        if out.shape[0] == 0:
            raise ValueError("All subgroups have 1 or less observations!")

        _Xbar = out["mean"].mean()
        out['Xbar'] = _Xbar
        _S = out["s"].mean()
        out['S'] = _S
        _N = out['n'].max()
        out['N'] = _N

        # Determine if subgroup sizes are constant or variable
        n_to_use, n_max = self._determine_n_to_use(out)

        # CALCULATE XBAR
        xbar = out.copy()
        xbar[['lcl', 'ucl']] = xbar.apply(
            lambda row: calculate_limits(
                mean=row['Xbar'],
                sd=row['S'],
                N=row[n_to_use],
                limits_type='Xbar',
                round_to=spec.round_to
            ), axis=1
        )

        # Detect beyond limits signals
        xbar = self._add_beyond_limits_flag(xbar, value_col='mean')
        xbar = xbar.round(spec.round_to)

        statistics['Mean'] = round(_Xbar, spec.round_to)
        if n_to_use == "N":
            statistics['N'] = _N
            statistics['ucl'] = xbar['ucl'].max()
            statistics['lcl'] = xbar['lcl'].max()
        else:
            variable_stats = 'Varies'
            statistics['N'] = variable_stats
            statistics['lcl'] = variable_stats
            statistics['ucl'] = variable_stats

        cols_to_keep = ['rsg', 'mean', 'Xbar', 'lcl', 'ucl', 'beyond_limits']
        xbar = xbar[cols_to_keep]
        result['Xbar'] = {'data': xbar, 'statistics': statistics}

        # CALCULATE S
        statistics = {}
        statistics['S'] = round(_S, spec.round_to)

        sbar = out.copy()
        sbar[['lcl', 'ucl']] = sbar.apply(
            lambda row: calculate_limits(
                mean=0,
                sd=row['S'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        # Detect beyond limits signals
        sbar = self._add_beyond_limits_flag(sbar, value_col='S')
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

        cols_to_keep = ['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']
        sbar = sbar[cols_to_keep]
        result['Sbar'] = {'data': sbar, 'statistics': statistics}

        return result

    def _calculate_s(self) -> pd.DataFrame:
        """
        Calculate S (standard deviation) chart statistics.

        Logic moved from calculate_statistics_S()
        """
        spec = self.spec
        out = self.ads.analysis_dataset.copy()

        out = out.groupby(spec.rsg_var_name, as_index=False).agg(
            s=pd.NamedAgg(column=spec.response_var, aggfunc="std"),
            n=pd.NamedAgg(column=spec.rsg_var_name, aggfunc="count"),
        )

        # remove RSGs with a single observation
        mask = out['n'].eq(1)
        out = out[~mask]

        out['S'] = out["s"].mean()
        out['groups'] = out["n"].count()
        out['N'] = out['n'].max()

        # Determine if subgroup sizes are constant or variable
        n_to_use, n_max = self._determine_n_to_use(out)

        # Add limits columns
        out[['lcl', 'ucl']] = out.apply(
            lambda row: calculate_limits(
                mean=0,
                sd=row['S'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        # Detect beyond limits signals
        out = self._add_beyond_limits_flag(out, value_col='S')

        cols_to_keep = ['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']
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
            out['mr'] = out.groupby(spec.rsg_var_name, sort=False)[y].diff().abs()

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
                out.groupby(spec.rsg_var_name, sort=False)
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


            # Attach back to rows
            out = out.merge(
                grouped[[spec.rsg_var_name, 'mean', 'mR', 'lcl', 'ucl']],
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
            out['mean'] = mean_
            out['mR']   = mR
            out['lcl']  = lims['lcl']
            out['ucl']  = lims['ucl']

        # Detect beyond limits signals
        out = self._add_beyond_limits_flag(out, value_col=spec.response_var)

        # Format output with appropriate columns
        out = self._build_output_columns(
            df=out,
            value_cols=[spec.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']
        )

        # Package results with statistics
        return self._package_stratified_results(
            df=out,
            statistics_cols=['mean', 'lcl', 'ucl']
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
            out['mr'] = abs(out.groupby(spec.rsg_var_name)[spec.response_var].diff())
            grouped = out.groupby(spec.rsg_var_name, as_index=False)
            grouped = grouped.agg(mR=pd.NamedAgg('mr', 'mean'))

            limits = grouped.apply(
                lambda row: calculate_limits(
                    mean=0,
                    sd=0,
                    N=0,
                    mR=row.mR,
                    limits_type="R",
                    round_to=spec.round_to
                ), axis=1
            )

            grouped = pd.merge(grouped, limits, left_index=True, right_index=True)
            out = pd.merge(out, grouped, how='left', on=spec.rsg_var_name)
        else:
            out['mr'] = abs(out[spec.response_var].diff())
            out['mR'] = out['mr'].mean()
            mR = out['mR'].max()
            limits = calculate_limits(
                mean=0,
                sd=0,
                N=0,
                mR=mR,
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
            value_cols=['mr', 'mR', 'lcl', 'ucl', 'beyond_limits']
        )

        # Package results with statistics
        return self._package_stratified_results(
            df=out,
            statistics_cols=['mR', 'lcl', 'ucl']
        )


# ============================================================================
# Module-level Functions
# ============================================================================

def perform_analysis(df: pd.DataFrame, specification: dict):
    """
    Main entry point for performing statistical process control analysis.

    Args:
        df: Input DataFrame with raw data
        specification: Dictionary containing analysis configuration including 'analysis_type'

    Returns:
        DataFrame or dict containing analysis results

    Raises:
        ValueError: If analysis_type is not supported
    """
    # Use new unified Analysis class
    analysis = Analysis(df, specification)
    return analysis.calculate()

def split_df_by_group(df: pd.DataFrame, grouping_var: str) -> dict:
    """
    Function split_df_by_group returns a dictionary with one item per group in grouping_var. 
    :param pandas.Dataframe df: data frame to split by group
    :param str grouping_var: group_variable to group dataframe by    
    :return: dictionary of dataframes with grouping_var values as keys
    :rtype: dict
    :raises ValueError: if group_var is not in input dataframe 
    For example: if df.groups = ['a','b','c'] the function will return a dictionary with three items
    as follows: 
            {
                'a': pandas.Dataframe for a,
                'b': pandas.Dataframe for b,
                'c': pandas.Dataframe for c
            }
    able to collaborate among themf products may have several
    variants, but the products of one variant are incompatible with products of
    another.
    """
    # Make sure grouping_var column exists
    if grouping_var not in df.columns.tolist():
        raise ValueError(
            f'The group_var: {grouping_var} is not in the data set!'
        )

    out = {}

    grouped = df.groupby(grouping_var)

    # package_results
    for g in grouped.groups:
        criteria = df[grouping_var].eq(g)

        out[g] = df[criteria]

    return (out)


def gather_analysis_statistics(
    df: pd.DataFrame,
    statistics_to_collect: list,
    grouping_var: str = None
) -> dict:
    """
    Function gather_analysis_summary_statistics returns a dictionary of
    statistics (contained in stats_to_package) for each analytic result passed.

    :param pandas.Dataframe df: a grouped dataframe of analysis results,
        i.e., output from R or Imr
    :param list stats_to_package: list of variables/columns to summarize
        (dataframes currently contain columns for mean, moving range, and N)

    This function will take the max for each value specified in stats to package
    and put in dictionary with key equal to the value of list item,
    i.e., "mean" will be returned in a dictionary
    {statistics:{group_name: "abc", mean:1.0, etc...}}   
    
    :return: dictionary of dataframes with grouping_var values as keys
    
    :rtype: dict
    
    :raises ValueError: if variables specified in list are in input dataframe 
    """
    logger.debug('In call gather_statistics')
    stats = {}

    out = df.copy()

    out_cols = df.columns.to_list()

    is_valid = all(cols in out_cols for cols in statistics_to_collect)

    if is_valid:
        if grouping_var is not None:
            statistics_to_collect.append("n")
            N = out.groupby(grouping_var, as_index=False).size()
            N.reset_index()
            N.rename(columns={"size": "n"}, inplace=True)

            summarized = df.groupby([grouping_var]).max()
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

    return (stats)


def package_analysis(analysis_output: dict, summary_statistics_output: dict):
    """
    Function package_analysis combines the results of an analysis with
    the collected summary statistics from the analysis, i.e., combines
    two dictionaries into one with the rational subgroup name as the key.
    Returns a dictionary of statistics (contained in stats_to_package)
    for each analytic result passed.

    :param pandas.Dataframe analysis_output: dictionary of dataframes with
        a key matching the name of the rational subgroup name for grouped
        individuals analyses, R or Imr
    :param list summary_statistics_output: dictionary of collected
        statistics for each grouped individuals analysis. key is expected
        to be the name of the rational subgroup. 
    
    :return: dictionary of dataframes with grouping_var values as keys
    
    :rtype: dict
    
    :raises ValueError: if variables specified in list are in input dataframe 
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
            f'for statistics being collected {stats_keys.to_list()}, '
            f'data: {output_keys.to_list()}'
        )
        raise ValueError(msg)
    
    return (out)

class AnalysisDataSet:
    """
    Orchestrates statistical process control analysis using Wheeler/Bishop methodology.

    This class coordinates the workflow:
    1. Data preparation and validation
    2. Sampling Design State (SDS) detection
    3. VAS residual calculation (R1-R5)
    4. Effects and interactions analysis
    5. Control chart frame building

    Uses composition pattern - delegates to focused classes for each concern.
    """

    def __init__(self, df: pd.DataFrame, analysis_specification: AnalysisSpecification):
        """
        Initialize analysis with data and specification.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data
        analysis_specification : AnalysisSpecification
            Configuration for the analysis
        """
        # Store inputs
        self.raw_dataset = df
        self.spec = analysis_specification

        # Initialize output containers (for backward compatibility)
        self.obs_df = None
        self.cell_df = None
        self.k_df = None
        self.t_df = None
        self.statistics = {}
        self.residuals = {}
        self.interactions = {}
        self.effects = {}
        self.Rbar = 0

        # Composition - each component has one job (Single Responsibility Principle)
        self.prep = DataPreparation()
        self.sds_detector = SamplingDesignDetector()
        self.residual_calc = ResidualCalculator()
        self.effects_calc = EffectsCalculator()

        # Run the analysis workflow
        self._initialize()

    def _initialize(self):
        """
        Execute the analysis workflow.

        Clear orchestration that reads like a recipe:
        1. Validate and prepare data
        2. Build frames for charting
        3. Detect sampling design state
        4. Calculate VAS residuals (if appropriate)
        5. Calculate effects and interactions (if appropriate)
        """
        # Step 1: Validate and prepare data
        logger.info("Preparing dataset")
        self.prep.validate_columns(self.raw_dataset, self.spec)
        self.analysis_dataset = self.prep.prepare_dataset(self.raw_dataset, self.spec)
        self.analysis_dataset = self.prep.build_keys(self.analysis_dataset, self.spec)

        # Step 2: Build frames for charting
        self._build_frames()

        # Step 3: Detect SDS
        logger.info("Detecting sampling design state")
        self.sampling_design_state = self.sds_detector.detect_sds(
            self.analysis_dataset, self.spec
        )
        self.sds_characteristics = self.sds_detector.get_sds_characteristics(
            self.sampling_design_state
        )

        # Log analysis summary and SDS
        logger.info(self.analysis_summary)
        logger.info(
            f"Detected: SDS {self.sampling_design_state} - "
            f"{self.sds_characteristics['description']}"
        )

        # Step 4: Validate compatibility (fail fast if incompatible)
        self.sds_detector.validate_sds_for_analysis(
            self.sampling_design_state, self.spec.analysis_type
        )

        # Step 5: Calculate VAS residuals only when appropriate
        if self.sds_detector.should_calculate_vas_residuals(
            self.sampling_design_state, self.spec.analysis_type
        ):
            logger.info("Calculating VAS residuals (R1-R5)")
            self.analysis_dataset = self.residual_calc.calculate_residuals(
                self.analysis_dataset, self.spec, self.sampling_design_state
            )

            # Calculate centered residuals (legacy support)
            self._calculate_centered_residuals()

            # Step 6: Calculate effects and interactions
            logger.info("Calculating effects and interactions")
            self.effects = self.effects_calc.calculate_all_effects(
                self.analysis_dataset, self.spec
            )
            self.interactions = self.effects_calc.calculate_interactions(
                self.analysis_dataset, self.spec, self.sampling_design_state
            )
        else:
            logger.debug(
                f"Skipping VAS residuals for analysis_type={self.spec.analysis_type}, "
                f"SDS={self.sampling_design_state}"
            )

    # =========================================================================
    # Properties (for backward compatibility and convenience)
    # =========================================================================

    @property
    def has_vas_residuals(self) -> bool:
        """Check if VAS residuals were calculated."""
        return 'R1' in self.analysis_dataset.columns

    @property
    def analysis_summary(self) -> dict:
        """
        Get comprehensive summary of the analysis dataset.

        Returns
        -------
        dict
            Dictionary with analysis metadata including:
            - sds: Detected sampling design state
            - sds_info: Full SDS characteristics
            - has_vas: Whether VAS residuals calculated
            - n_observations: Total observations
            - analysis_type: Type of analysis being performed
        """
        summary = {
            'sds': self.sampling_design_state,
            'sds_info': self.sds_characteristics,
            'has_vas_residuals': self.has_vas_residuals,
            'n_observations': len(self.analysis_dataset),
            'analysis_type': self.spec.analysis_type
        }
        return summary

    # =========================================================================
    # Centered Residuals (legacy support - kept for backward compatibility)
    # =========================================================================

    def _calculate_centered_residuals(self):
        """
        Calculate centered residuals (Rbar and RCR values).

        These are legacy calculations that center residuals by their means.
        Kept for backward compatibility with existing code and tests.

        Calculates:
        - Rbar_kt: Mean of R1 per cell (factor × time)
        - Rbar_k: Mean of R1 per factor level
        - Rbar_t: Mean of R1 per time point
        - RCR1-RCR5: Reconstructed Y values from centered residuals
        """
        if not self.spec.has_grouping:
            return

        df = self.analysis_dataset

        # Calculate centered residual means
        rsg_time_groups = df.groupby([self.spec.rsg_var_name, self.spec.time_var])
        df["Rbar_kt"] = rsg_time_groups["R1"].transform("mean")
        df['Rbar_k'] = df.groupby([self.spec.rsg_var_name])["R1"].transform('mean')
        df['Rbar_t'] = df.groupby([self.spec.time_var])["R1"].transform("mean")

        # Calculate RCR (Reconstructed Centered Residuals)
        # These verify that Y can be reconstructed from components
        df['RCR1'] = df['Ybar'] + df['R1']  # Y = Ybar + R1
        df['RCR2'] = df['Ybar_kt'] + df['R2']  # Y = Ybar_kt + R2
        # Y = (Ybar_k + Ybar_t - Ybar) + R3
        df['RCR3'] = (df['Ybar_k'] + df['Ybar_t'] - df['Ybar']) + df['R3']
        # Y = (Ybar + Ybar_kt - Ybar_t) + R4
        df['RCR4'] = (df['Ybar'] + df['Ybar_kt'] - df['Ybar_t']) + df['R4']
        # Y = (Ybar + Ybar_kt - Ybar_k) + R5
        df['RCR5'] = (df['Ybar'] + df['Ybar_kt'] - df['Ybar_k']) + df['R5']

    # =========================================================================
    # Frame Building (kept as-is for backward compatibility)
    # =========================================================================

    def _build_frames(self) -> None:
        """
        Materialize canonical frames by grain:
        - obs_df : one row per observation
        - cell_df: one row per (k_vars + time)
        - k_df   : one row per k_vars combination
        - t_df   : one row per time point
        """
        df = self.__ensure_keys(self.analysis_dataset)

        # Build each frame using extracted helper methods
        self.obs_df = self._build_obs_df(df)
        self.k_df = self._build_k_df(df)
        self.t_df = self._build_t_df(df)
        self.cell_df = self._build_cell_df(df)

    def _build_obs_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build observation-level frame (one row per observation)."""
        spec = self.spec
        y = spec.response_var
        k_vars = list(spec.rsg_vars or [])
        t = spec.time_var

        base_cols = [
            c for c in [*k_vars, t, spec.rsg_var_name, 'obs_id', 'n']
            if c in df.columns
        ]
        means = [c for c in ['Ybar','Ybar_k','Ybar_t','Ybar_kt'] if c in df.columns]
        residuals = [c for c in ['R1','R2','R3','R4','R5'] if c in df.columns]
        rcrs = [c for c in ['RCR1','RCR2','RCR3','RCR4','RCR5'] if c in df.columns]
        centered = [c for c in ['Rbar_k','Rbar_t','Rbar_kt'] if c in df.columns]
        inter_row = [
            c for c in ['pdc_by_pt','interaction_cell','factor_interaction_effects']
            if c in df.columns
        ]

        obs_keep = [
            c for c in [y, *base_cols, *means, *residuals, *rcrs, *centered, *inter_row]
            if c in df.columns
        ]
        return (df[obs_keep]
                .sort_values('obs_id', kind='stable')
                .reset_index(drop=True))

    def _build_k_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build factor-level frame (one row per k_vars combination)."""
        spec = self.spec
        k_vars = list(spec.rsg_vars or [])

        if spec.has_grouping and k_vars:
            # counts by factor combo
            k_counts = (
                df.groupby(k_vars, sort=False)
                .size()
                .rename('n_k')
                .reset_index()
            )
            k_df = k_counts

            # add factor-level means if present
            if 'Ybar_k' in df.columns:
                k_first = self.__safe_first(df, k_vars, 'Ybar_k')
                k_df = k_df.merge(k_first, on=k_vars, how='left', validate='one_to_one')

            # join per-factor Main_Effect tables
            for factor in k_vars:
                me = self.effects.get(factor)
                required_cols = {factor, 'Main_Effect'}
                if isinstance(me, pd.DataFrame) and required_cols <= set(me.columns):
                    me_renamed = me[[factor, 'Main_Effect']].rename(
                        columns={'Main_Effect': f'{factor}_Main_Effect'}
                    )
                    k_df = k_df.merge(
                        me_renamed,
                        on=factor, how='left', validate='many_to_one'
                    )

            # single-factor convenience alias
            if len(k_vars) == 1 and f"{k_vars[0]}_Main_Effect" in k_df.columns:
                k_df['Main_Effect_k'] = k_df[f"{k_vars[0]}_Main_Effect"]

            return k_df.sort_values(k_vars, kind='stable').reset_index(drop=True)
        else:
            cols = k_vars + ['n_k']
            if 'Ybar_k' in df.columns:
                cols += ['Ybar_k']
            return pd.DataFrame(columns=cols)

    def _build_t_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build time-level frame (one row per time point)."""
        spec = self.spec
        t = spec.time_var

        if spec.has_time and t:
            t_counts = (
                df.groupby([t], sort=False)
                .size()
                .rename('n_t')
                .reset_index()
            )
            t_df = t_counts

            if 'Ybar_t' in df.columns:
                t_first = self.__safe_first(df, [t], 'Ybar_t')
                t_df = t_df.merge(t_first, on=[t], how='left', validate='one_to_one')

            # time main effect
            pt_me = self.effects.get('pt_me')
            if isinstance(pt_me, pd.DataFrame):
                # normalize shape: either index=t or column=t
                if t not in pt_me.columns and pt_me.index.name == t:
                    pt_me = pt_me.reset_index()
                if {'PT_ME'} <= set(pt_me.columns) and t in pt_me.columns:
                    t_df = t_df.merge(pt_me[[t, 'PT_ME']], on=t, how='left', validate='many_to_one')

            return t_df.sort_values([t], kind='stable').reset_index(drop=True)
        else:
            cols = ([t] if t else []) + ['n_t']
            if 'Ybar_t' in df.columns:
                cols += ['Ybar_t']
            cols += ['PT_ME']
            return pd.DataFrame(columns=cols)

    def _build_cell_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cell-level frame (one row per k_vars × time combination)."""
        spec = self.spec
        k_vars = list(spec.rsg_vars or [])
        t = spec.time_var

        if spec.has_grouping and spec.has_time and k_vars and t:
            keys = k_vars + [t]

            # n per cell
            cdf = (
                df.groupby(keys, sort=False)
                .size()
                .rename('n_cell')
                .reset_index()
            )

            # firsts of broadcast means (only if present)
            for col in ['Ybar_kt', 'Ybar_k', 'Ybar_t']:
                if col in df.columns:
                    c_first = self.__safe_first(df, keys, col)
                    cdf = cdf.merge(c_first, on=keys, how='left', validate='one_to_one')

            # interaction per cell: prefer explicit column else reconstruct
            if 'interaction_cell' in df.columns:
                ic = self.__safe_first(df, keys, 'interaction_cell')
                cdf = cdf.merge(ic, on=keys, how='left', validate='one_to_one')
            else:
                required_cols = ['Ybar_kt', 'Ybar_k', 'Ybar_t']
                has_cols = all(c in cdf.columns for c in required_cols)
                if has_cols and 'Ybar' in self.statistics:
                    ybar = float(self.statistics['Ybar'])
                    cdf['interaction_cell'] = (
                        cdf['Ybar_kt'] - cdf['Ybar_k'] - cdf['Ybar_t'] + ybar
                    )

            # centered residual cell means if present
            for col in ['Rbar_kt']:
                if col in df.columns:
                    c_first = self.__safe_first(df, keys, col)
                    cdf = cdf.merge(c_first, on=keys, how='left', validate='one_to_one')

            return cdf.sort_values(keys, kind='stable').reset_index(drop=True)
        else:
            # empty shell with predictable columns
            cols = k_vars + ([t] if t else [])
            cols += ['n_cell', 'Ybar_kt', 'Ybar_k', 'Ybar_t', 'interaction_cell', 'Rbar_kt']
            return pd.DataFrame(columns=cols)


    # --- helpers (put inside the class) ------------------------------------------
    def __ensure_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make sure key columns exist; create a deterministic obs_id if missing."""
        spec = self.spec
        out = df.copy()

        # ensure obs_id (stable) for row-grain sorting/debug
        if 'obs_id' not in out.columns:
            sort_cols = []
            if spec.has_grouping:
                sort_cols += [spec.rsg_var_name]
            if spec.has_time:
                sort_cols += [spec.time_var]
            if sort_cols:
                out = out.sort_values(sort_cols, kind='stable')
            out = out.reset_index(drop=True)
            out['obs_id'] = np.arange(len(out), dtype=int)

        return out

    def __safe_first(self, df: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame:
        """Return one row per keys with the first value of col, if col exists; else empty."""
        if col not in df.columns:
            return pd.DataFrame(columns=keys + [col])
        return (
            df.groupby(keys, sort=False)[col]
            .first()
            .reset_index()
        )
    # -----------------------------------------------------------------------------
