
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import scipy.special
from pandas.api.types import is_numeric_dtype

from .spc_constants import calculate_limits, detect_beyond_limits
from .data_preparation import DataPreparation
from .sds_detector import SamplingDesignDetector
from .residual_calculator import ResidualCalculator
from .effects_calculator import EffectsCalculator
from .analysis_result import AnalysisResult
from .analysis_specification import AnalysisSpecification

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

        # Execute strategy to get chart data
        chart_data = strategies[self.spec.analysis_type]()

        # Wrap in AnalysisResult for unified access
        return AnalysisResult(
            charts=chart_data,
            analysis_dataset_obj=self.ads
        )

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

        if spec.zero_center:
            logger.info('Zero-centering data')
            zero_mean = out[spec.response_var].mean()
            out[spec.response_var] = out[spec.response_var] - zero_mean

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

        # if subgroup sizes are equal use N (limits will be same for all groups)
        n_max = _N
        n_to_use = "N" if out['n'].eq(n_max).all() else "n"
        logger.info('Analysis using %s for calculations (Scenario: %s)', n_to_use, 1 if n_to_use=="N" else 2)

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

        xbar['beyond_limits'] = xbar.apply(
            lambda row: detect_beyond_limits(
                x=row['mean'],
                ucl=row['ucl'],
                lcl=row['lcl']
            ), axis=1
        )

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

        sbar['beyond_limits'] = sbar.apply(
            lambda row: detect_beyond_limits(
                x=row['S'],
                ucl=row['ucl'],
                lcl=row['lcl']
            ), axis=1
        )

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
        df = self.raw_df
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

        # if subgroup sizes are equal use N (limits will be same for all groups)
        n_max = out['n'].max()
        n_to_use = "N" if (out['n'].eq(n_max).all()) else "n"

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

        out['beyond_limits'] = out.apply(
            lambda row: detect_beyond_limits(
                x=row['S'],
                ucl=row['ucl'],
                lcl=row['lcl']
            ), axis=1
        )

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
        df = self.raw_df
        spec = self.spec
        out = self.ads.analysis_dataset.copy()#prepare_dataset(df=df, analysis_specification=spec)
        print (out)
        if spec.zero_center:
            logger.info('Zero-centering data')
            zero_mean = out[spec.response_var].mean()
            logger.debug('Zero-mean: %s', zero_mean)
            out[spec.response_var] = out[spec.response_var] - zero_mean

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
            lims = calculate_limits(mean=mean_, sd=0, N=0, mR=mR, limits_type="Imr", round_to=spec.round_to)
            out['mean'] = mean_
            out['mR']   = mR
            out['lcl']  = lims['lcl']
            out['ucl']  = lims['ucl']

        out['beyond_limits'] = np.where(out[spec.response_var] < out['lcl'], -1, 0)
        out['beyond_limits'] = np.where(out[spec.response_var] > out['ucl'], 1, 0)

        cols_to_keep = [spec.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']

        if spec.has_time:
            if spec.has_grouping:
                cols_to_keep.insert(0, spec.rsg_var_name)
                cols_to_keep.insert(0, spec.time_var)
            else:
                cols_to_keep.insert(0, spec.time_var)
        else:
            if spec.has_grouping:
                out['x'] = out.groupby(spec.rsg_var_name).cumcount() + 1
                cols_to_keep.insert(0, spec.rsg_var_name)
                cols_to_keep.insert(0, 'x')
            else:
                out['x'] = out.index + 1
                cols_to_keep.insert(0, 'x')

        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        if spec.has_grouping:
            statistics = gather_analysis_statistics(
                df=out,
                statistics_to_collect=['mean', 'lcl', 'ucl'],
                grouping_var=spec.rsg_var_name
            )
            split_dict = split_df_by_group(df=out, grouping_var=spec.rsg_var_name)
            out = package_analysis(
                analysis_output=split_dict,
                summary_statistics_output=statistics
            )
        else:
            statistics = gather_analysis_statistics(
                df=out,
                statistics_to_collect=['mean', 'lcl', 'ucl']
            )
            _out = {'all': out}
            out = package_analysis(
                analysis_output=_out,
                summary_statistics_output=statistics
            )

        return out

    def _calculate_r(self) -> pd.DataFrame:
        """
        Calculate R (Range) chart statistics.

        Logic moved from calculate_statistics_R()
        """
        df = self.raw_df
        spec = self.spec
        out = self.ads.analysis_dataset.copy()

        if spec.zero_center:
            logger.info('Zero-centering data')
            zero_mean = out[spec.response_var].mean()
            out[spec.response_var] = out[spec.response_var] - zero_mean

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

        # Calculate Beyond Limits
        out['beyond_limits'] = np.where(out[spec.response_var] > out['lcl'], -1, 0)
        out['beyond_limits'] = np.where(out[spec.response_var] > out['ucl'], 1, 0)

        cols_to_keep = ['mr', 'mR', 'lcl', 'ucl', 'beyond_limits']

        if spec.has_time:
            if spec.has_grouping:
                cols_to_keep.insert(0, 'rsg')
                cols_to_keep.insert(0, spec.time_var)
            else:
                cols_to_keep.insert(0, spec.time_var)
        else:
            if spec.has_grouping:
                out['x'] = out.groupby(spec.rsg_var_name).cumcount() + 1
                cols_to_keep.insert(0, 'rsg')
                cols_to_keep.insert(0, 'x')
            else:
                out['x'] = out.index
                cols_to_keep.insert(0, 'x')

        out = out[cols_to_keep]
        out = out.round(spec.round_to)

        if spec.has_grouping:
            statistics = gather_analysis_statistics(
                df=out,
                statistics_to_collect=['mR', 'lcl', 'ucl'],
                grouping_var=spec.rsg_var_name
            )
            split_dict = split_df_by_group(df=out, grouping_var=spec.rsg_var_name)
            out = package_analysis(
                analysis_output=split_dict,
                summary_statistics_output=statistics
            )
        else:
            statistics = gather_analysis_statistics(
                df=out,
                statistics_to_collect=['mR', 'lcl', 'ucl']
            )
            _out = {'all': out}
            out = package_analysis(
                analysis_output=_out,
                summary_statistics_output=statistics
            )

        return out


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
    if not grouping_var in df.columns.tolist(): raise ValueError(
        f'The group_var: {grouping_var} is not in the data set!')

    out = {}

    grouped = df.groupby(grouping_var)

    # package_results
    for g in grouped.groups:
        criteria = df[grouping_var].eq(g)

        out[g] = df[criteria]

    return (out)


def gather_analysis_statistics(df: pd.DataFrame, statistics_to_collect: list, grouping_var: str = None) -> dict:
    """
    Function gather_analysis_summary_statistics returns a dictionary of statistics (contained in stats_to_package) for each analytic result passed. 
    
    :param pandas.Dataframe df: a grouped dataframe of analysis results, i.e., output from R or Imr
    :param list stats_to_package: list of variables/columns to summarize (dataframes currently contain columns for mean, moving range, and N) 
    
    This function will take the max for each value specified in stats to package an put in dictionary with key equal to the value of list item, 
    i.e., "mean" will be returned in a dictionary {statistics:{group_name: "abc", mean:1.0, etc...}}   
    
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

            for index, row in summarized.iterrows():
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
    Function package_analysis combines the results of an analysis with the collected summary statistics from the analysis, i.e., 
    combines two dictionaries into one with the rational subgroup name as the key.returns a dictionary of statistics (contained in stats_to_package) for each analytic result passed. 
    
    :param pandas.Dataframe analysis_output: dictionary of dataframes with a key matching the name of the rational subgroup name for grouped individuals analyses, R or Imr
    :param list summary_statistics_output: dictionary of collected statistics for each grouped individuals analysis. key is expected to be the name of the rational subgroup. 
    
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
        for key in analysis_output.keys():
            out[key] = {'data': analysis_output[key], 'statistics': summary_statistics_output[key]}
    else:
      
        raise ValueError(f'Call: package_analysis: The rational subgroups do not match for statistics being collected {stats_keys.to_list()}, data: {output_keys.to_list()}')
    
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
        df["Rbar_kt"] = df.groupby([self.spec.rsg_var_name, self.spec.time_var])["R1"].transform("mean")
        df['Rbar_k'] = df.groupby([self.spec.rsg_var_name])["R1"].transform('mean')
        df['Rbar_t'] = df.groupby([self.spec.time_var])["R1"].transform("mean")

        # Calculate RCR (Reconstructed Centered Residuals)
        # These verify that Y can be reconstructed from components
        df['RCR1'] = df['Ybar'] + df['R1']  # Y = Ybar + R1
        df['RCR2'] = df['Ybar_kt'] + df['R2']  # Y = Ybar_kt + R2
        df['RCR3'] = (df['Ybar_k'] + df['Ybar_t'] - df['Ybar']) + df['R3']  # Y = (Ybar_k + Ybar_t - Ybar) + R3
        df['RCR4'] = (df['Ybar'] + df['Ybar_kt'] - df['Ybar_t']) + df['R4']  # Y = (Ybar + Ybar_kt - Ybar_t) + R4
        df['RCR5'] = (df['Ybar'] + df['Ybar_kt'] - df['Ybar_k']) + df['R5']  # Y = (Ybar + Ybar_kt - Ybar_k) + R5

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
        spec = self.spec
        y = spec.response_var
        k_vars = list(spec.rsg_vars or [])
        t = spec.time_var

        df = self.__ensure_keys(self.analysis_dataset)

        # ---------- obs_df (authoritative row-grain) ----------
        base_cols = [c for c in [*k_vars, t, spec.rsg_var_name, 'obs_id', 'n'] if c in df.columns]
        means     = [c for c in ['Ybar','Ybar_k','Ybar_t','Ybar_kt'] if c in df.columns]
        residuals = [c for c in ['R1','R2','R3','R4','R5'] if c in df.columns]
        rcrs      = [c for c in ['RCR1','RCR2','RCR3','RCR4','RCR5'] if c in df.columns]
        centered  = [c for c in ['Rbar_k','Rbar_t','Rbar_kt'] if c in df.columns]
        inter_row = [c for c in ['pdc_by_pt','interaction_cell','factor_interaction_effects'] if c in df.columns]

        obs_keep = [c for c in [y, *base_cols, *means, *residuals, *rcrs, *centered, *inter_row] if c in df.columns]
        self.obs_df = (df[obs_keep]
                    .sort_values('obs_id', kind='stable')
                    .reset_index(drop=True))

        # ---------- k_df (factor/main-effect grain) ----------
        if spec.has_grouping and k_vars:
            # counts by factor combo (no rename needed)
            k_counts = (
                df.groupby(k_vars, sort=False)
                .size()
                .rename('n_k')
                .reset_index()
            )
            k_df = k_counts

            # add factor-level means if you want them visible at k grain
            if 'Ybar_k' in df.columns:
                k_first = self.__safe_first(df, k_vars, 'Ybar_k')
                k_df = k_df.merge(k_first, on=k_vars, how='left', validate='one_to_one')

            # join per-factor Main_Effect tables you stored in self.effects[factor]
            for factor in k_vars:
                me = self.effects.get(factor)
                if isinstance(me, pd.DataFrame) and {factor, 'Main_Effect'} <= set(me.columns):
                    k_df = k_df.merge(
                        me[[factor, 'Main_Effect']].rename(columns={'Main_Effect': f'{factor}_Main_Effect'}),
                        on=factor, how='left', validate='many_to_one'
                    )

            # single-factor convenience alias
            if len(k_vars) == 1 and f"{k_vars[0]}_Main_Effect" in k_df.columns:
                k_df['Main_Effect_k'] = k_df[f"{k_vars[0]}_Main_Effect"]

            self.k_df = k_df.sort_values(k_vars, kind='stable').reset_index(drop=True)
        else:
            self.k_df = pd.DataFrame(columns=(k_vars + ['n_k'] + (['Ybar_k'] if 'Ybar_k' in df.columns else [])))

        # ---------- t_df (time/main-effect grain) ----------
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

            # time main effect: you store as self.effects['pt_me'] with column PT_ME
            pt_me = self.effects.get('pt_me')
            if isinstance(pt_me, pd.DataFrame):
                # normalize shape: either index=t or column=t
                if t not in pt_me.columns and pt_me.index.name == t:
                    pt_me = pt_me.reset_index()
                if {'PT_ME'} <= set(pt_me.columns) and t in pt_me.columns:
                    t_df = t_df.merge(pt_me[[t, 'PT_ME']], on=t, how='left', validate='many_to_one')

            self.t_df = t_df.sort_values([t], kind='stable').reset_index(drop=True)
        else:
            self.t_df = pd.DataFrame(columns=([t] if t else []) + ['n_t'] + (['Ybar_t'] if 'Ybar_t' in df.columns else []) + ['PT_ME'])

        # ---------- cell_df (cell-grain: k_vars × t) ----------
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
            elif all(c in cdf.columns for c in ['Ybar_kt', 'Ybar_k', 'Ybar_t']) and 'Ybar' in self.statistics:
                cdf['interaction_cell'] = cdf['Ybar_kt'] - cdf['Ybar_k'] - cdf['Ybar_t'] + float(self.statistics['Ybar'])

            # centered residual cell means if present
            for col in ['Rbar_kt']:
                if col in df.columns:
                    c_first = self.__safe_first(df, keys, col)
                    cdf = cdf.merge(c_first, on=keys, how='left', validate='one_to_one')

            self.cell_df = cdf.sort_values(keys, kind='stable').reset_index(drop=True)
        else:
            # empty shell with predictable columns
            self.cell_df = pd.DataFrame(columns=(k_vars + ([t] if t else []) + ['n_cell','Ybar_kt','Ybar_k','Ybar_t','interaction_cell','Rbar_kt']))


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
