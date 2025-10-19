
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import scipy.special
from pandas.api.types import is_numeric_dtype

import objects as obj

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

    def calculate(self) -> pd.DataFrame:
        """
        Execute the appropriate analysis strategy and return results.

        Returns:
            DataFrame containing analysis results

        Raises:
            ValueError: If analysis_type is not supported
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

        return strategies[self.analysis_type]()

    def _calculate_xbar(self) -> pd.DataFrame:
        """
        Calculate Xbar (mean) chart statistics.

        Logic moved from Xbar.calculate_statistics()
        """
        df = self.ads.analysis_dataset
        spec = self.spec
        result = {}
        statistics = {}
        out = df.copy()

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
            lambda row: obj.calculate_limits(
                mean=row['Xbar'],
                sd=row['S'],
                N=row[n_to_use],
                limits_type='Xbar',
                round_to=spec.round_to
            ), axis=1
        )

        xbar['beyond_limits'] = xbar.apply(
            lambda row: obj.detect_beyond_limits(
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
            lambda row: obj.calculate_limits(
                mean=0,
                sd=row['S'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        sbar['beyond_limits'] = sbar.apply(
            lambda row: obj.detect_beyond_limits(
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
        out = prepare_dataset(df=df, analysis_specification=spec)

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
            lambda row: obj.calculate_limits(
                mean=0,
                sd=row['S'],
                N=row[n_to_use],
                limits_type="S",
                round_to=spec.round_to
            ), axis=1
        )

        out['beyond_limits'] = out.apply(
            lambda row: obj.detect_beyond_limits(
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
        df = self.raw_df
        spec = self.spec
        out = prepare_dataset(df=df, analysis_specification=spec)

        if spec.zero_center:
            logger.info('Zero-centering data')
            zero_mean = out[spec.response_var].mean()
            logger.debug('Zero-mean: %s', zero_mean)
            out[spec.response_var] = out[spec.response_var] - zero_mean

        logger.debug('In calculate statistics IMR')
        logger.debug('Dataframe has columns: %s', out.columns.to_list())

        if spec.has_grouping:
            out['mr'] = abs(out.groupby(spec.rsg_var_name)[spec.response_var].diff())
            grouped = out.groupby(spec.rsg_var_name, as_index=False)
            grouped = grouped.agg(
                mean=pd.NamedAgg(spec.response_var, 'mean'),
                mR=pd.NamedAgg('mr', 'mean')
            )

            limits = grouped.apply(
                lambda row: obj.calculate_limits(
                    mean=row['mean'],
                    sd=0,
                    N=0,
                    mR=row.mR,
                    limits_type="Imr"
                ), axis=1
            )

            grouped = pd.merge(grouped, limits, left_index=True, right_index=True)
            out = pd.merge(out, grouped, how='left', on=spec.rsg_var_name)
        else:
            out['mR'] = abs(out[spec.response_var].diff()).mean()
            out['mean'] = out[spec.response_var].mean()
            mR = out['mR'].max()
            _mean = out['mean'].max()
            limits = obj.calculate_limits(
                mean=_mean,
                sd=0,
                N=0,
                mR=mR,
                limits_type="Imr",
                round_to=spec.round_to
            )
            out['lcl'] = limits['lcl']
            out['ucl'] = limits['ucl']

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
        out = prepare_dataset(df=df, analysis_specification=spec)

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
                lambda row: obj.calculate_limits(
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
            limits = obj.calculate_limits(
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

class AnalysisSpecification:
    """
    **AnalysisSpecification** is responsible for validating and processing analysis specifications
    currently implemented as a dictionary.

    Given a specification it determines how an analysis is executed and the results that are returned.
    :param str **analysis_type** type of analysis being defined, valide values are:*("Xbar","S","Imr", and "R")*
    :param dictionary analysis_specification: containing the parameters for the analysis: 
               
        rsg_vars:list rational subgrouping variables (optional) - if provided the input dataset 
                will be grouped by columns specified.  In the below example, a column of concatenated values will
                be returned and named "rsg" or the user specified: rsg_var_name.
        time_var: the time dimension or ordering variable for the data, if provided, data will be sorted 
                by rsg and time or time only.
        response_var: the response variabe being analyzed for the conditions specified by 
                rsg_vars and time_var
        rsg_var_name: user specified label for the rational subgroup column created by rsg_vars specified, defaults to rsg
        rsg_var_delim: user specified delimiter for multi-variable, *default value is '_'*, i.e., col_a, col_b 
                will return values col_a_col_b, labeled with rsg_var_name, valid values include any string: *['_','|','-', ...etc]*
        time_unit: enables aggregations for common intervals of time (Year, Quarter, Month, and Week) - using the 
                specified unit,the system will group and aggregate the response variable. *not implemented*
        round_to: digits to round analytic dataset numeric value to, default is 3

        Example: spec = ``{'rsg_vars':['lane','phase'], 'time_var':'pull', 
                         'response_var':'fill_weight','rsg_var_name':'rsg', 'time_unit':None, 'round_to':None}``

    :return: an instance of AnalysisSpecification for specified analysis
    :rtype: class
    :raises ValueError: if above listed variables are not propertly specified.     
    """

    VALID_TIME_UNITS = ['Year', 'Quarter', 'Month', 'Week']

    # s=['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']
    # xbar = ['rsg', 'mean', 'Xbar', 'lcl', 'ucl', 'beyond_limits']
    # spec = {'analysis_type':'S', 'rsg_vars':['a','b'], 'time_var':'d', 'response_var':'c','rsg_var_name':'rsg', 'time_unit':None}

    def __init__(self, analysis_type:str, analysis_specification: dict):
        SUPPORTED_ANALYSIS_TYPES = ['Xbar','S','Imr','R']
        GROUPED_ANALYSES = ['Xbar','S']       


        # Create output col list for each analyis type
        self.analysis_type = analysis_type
        self.spec = analysis_specification
        self.rsg_vars = self.spec.get('rsg_vars')
        self.rsg_var_name = self.spec.get('rsg_var_name', 'rsg')
        self.rsg_var_delim = self.spec.get('rsg_var_delim', '_')
        self.time_var = self.spec.get('time_var')
        self.response_var = self.spec.get('response_var')
        
        self.round_to = self.spec.get('round_to', 3)  # default round to 3 if none
        self.data_prep_output_cols = []
        self.sort_cols = []
        self.time_grouping_units = []
        self.time_grouping_cols = {}
        self.grouping_cols = []
        self.zero_center = self.spec.get('zero-center', False)

        if self.analysis_type not in SUPPORTED_ANALYSIS_TYPES:
            raise ValueError(
                f'Analyis type: {self.analysis_type} is not supported, specify one of: {SUPPORTED_ANALYSIS_TYPES}!')

        if self.analysis_type in GROUPED_ANALYSES and self.rsg_vars is None:
            raise ValueError(f'A grouping variable is required to produce a {self.analysis_type} analyis!')

        if self.response_var is None:
            raise ValueError(f'A response variable is required to produce a {self.analysis_type} analyis!')

        if self.zero_center not in [True, False]:
            raise ValueError(f'Supplied value for zero-centered needs to be True or False')

        #self.has_grouping = True if self.rsg_vars is not None else False
        self.has_time = True if self.time_var is not None else False
        self.grouping_cols = self.rsg_var_name if self.has_grouping else []

        # Need to raise exception in tidy function if time unit specified and time_var not a datetime
        # 293
        self.requires_sort = True if self.has_grouping and self.has_time or self.has_time else False

        
        self._build_data_prep_cols()
        # Initialize output cols
        self.analysis_output_cols = [self.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']

        self._build_sort_cols()
        self._build_output_cols()

    def data_prep_output_cols(self) -> list:
        self.data_prep_output_cols

    def analysis_output_cols(self) -> list:
        self.analysis_output_cols

    def sort_cols(self) -> list:
        self.sort_cols

    @property
    def has_grouping(self) -> bool:
        """Return True if rational subgrouping variables are defined."""
        return self.rsg_vars is not None

    def grouping_cols(self) -> list:
        self.grouping_cols

    def has_time(self) -> bool:
        self._has_time

    def requires_sort(self) -> bool:
        self.requires_sort

    def _build_output_cols(self):
        # Address Grouping var in output
        if self.has_grouping: self.analysis_output_cols.insert(0, self.rsg_var_name)
        # Address time var in output
        self.analysis_output_cols.insert(0, self.time_var) if self.has_time else self.analysis_output_cols.insert(0,
                                                                                                                  "x")
    def _build_sort_cols(self):
        # TODO: Need to factor in time unit
        # Both grouping var and time need to be provided to enable sorting
        if self.has_grouping and self.has_time:
            self.sort_cols = [self.rsg_var_name, self.time_var]
        elif self.has_time:
            self.sort_cols = [self.time_var]

    def _build_data_prep_cols(self) -> list:
        """
        _build_data_prep_cols builds a list of column names to keep 
        at the end of the data preparation step, including time units for aggregation
        """
        self.data_prep_output_cols.insert(0, self.response_var)  # this is the default value
        # Add time unit cols
        for col in self.time_grouping_cols:
            self.data_prep_output_cols.append(self.time_grouping_cols[col])

        if self.has_grouping:
            self.data_prep_output_cols.insert(0, 'n')
            self.data_prep_output_cols.insert(0, self.rsg_var_name)
            self.data_prep_output_cols.extend(self.rsg_vars)

        if self.has_time:
            self.data_prep_output_cols.insert(0, self.time_var)


def validate_columns(df: pd.DataFrame, analysis_specification: AnalysisSpecification) -> pd.DataFrame:
    spec = analysis_specification
    # TODO: Centralize constants 
    number_column_types = ['int64', 'float64', 'int32', 'Int64', 'Float64', 'Int32', 'category']
    # string_column_types = ['string', 'object', 'category', 'String', 'Object']
    date_column_types = ['int64', 'int32', 'Int64', 'Int32', 'datetime64[ns]', 'object']

    df_cols = df.columns.tolist()
    logger.info(f'Spec has grouping: {spec.has_grouping}')
    if spec.has_grouping:

        if not all(item in df_cols for item in spec.rsg_vars):
            raise ValueError(f'One or more of the rsg_vars: {spec.rsg_vars}: are not in the data set: {df_cols}!')

    if spec.has_time:

        if not spec.time_var in df_cols:
            raise ValueError(f'The time variable: {spec.time_var}: is not in the data set!')

        time_type = df[spec.time_var].dtype
        #TODO: Verify need to valid data type of time_var
        #if not pd.api.types.is_datetime64_any_dtype(time_type):
        #    raise ValueError(f'The time variable: {spec.time_var}: must be a datetime when a time unit is provided!')

    # Validate response variable
    if not spec.response_var in df_cols: raise ValueError(
        f'The response variable: {spec.response_var}: is not in the data set!')

    response_type = df[spec.response_var].dtype

    if not pd.api.types.is_numeric_dtype(response_type): raise ValueError(
        f'The response variable: {spec.response_var}: is not type numeric!')

    return df


def prepare_dataset(df: pd.DataFrame, analysis_specification: AnalysisSpecification) -> pd.DataFrame:
    logger.debug('Entering prepare_dataset')
    spec = analysis_specification
    out = df.copy()

    out = validate_columns(df=out, analysis_specification=spec)

    # 296 - Incorporate user specified delimiter into spec to delimit multiple grouping variables when specified, default will be underscore - "_"
    ##If specified add column for RSG - Rational Subgroup
    if spec.has_grouping:

        # Drop 'n' column if it exists in the input data to avoid conflicts
        if 'n' in out.columns:
            logger.debug('Dropping existing "n" column from input data (will be recalculated)')
            out = out.drop(columns=['n'])

        out = obj.add_grouping_variable_column(df=out, col_name=spec.rsg_var_name, cols_to_combine=spec.rsg_vars,
                                               col_delim=spec.rsg_var_delim)

        # Remove groups with one obs - solve for all grouped analyses
        grouped = out.groupby(spec.rsg_var_name).size()
        starting_count = grouped.count()
        logger.debug('Starting with %s groups', starting_count)
        grouped = grouped[grouped > 1]

        if grouped.shape[0] == 0:
            raise ValueError("All subgroups have 1 or less observations!")

        grouped = grouped.reset_index()
        grouped = grouped.rename(columns={0: 'n'})

        logger.debug('Pruning groups with 1 obs:\\n%s', grouped)
        ending_count = grouped.count()

        logger.debug('Groups remaining: %s', ending_count.iloc[0])
        logger.debug('Removed %s group(s)', starting_count - ending_count.iloc[0])
        # Empty print removed

        out = pd.merge(out, grouped, how='inner', on=spec.rsg_var_name)
    else:
        logger.debug('No groups specified')

    # 301 - when provided sort by time dimension
    if spec.requires_sort:
        out = out.sort_values(spec.sort_cols)  # These are predetermined by the spec

    out = out[spec.data_prep_output_cols]  # These are predetermined by the spec

    out = out.dropna()

    return out






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
    
    def __init__(self, df: pd.DataFrame, analysis_specification: AnalysisSpecification):
        self.raw_dataset = df
        self.spec = analysis_specification
        self.statistics = {}
        self.residuals = {}
        self.interactions = {}
        self.effects = {}
        self.Rbar=0
        # Validate and prepare data
        self.__validate_columns()
        self.analysis_dataset = self.__prepare_dataset()
        # Detect SDS
        self.sampling_design_state = self.__calculate_sampling_design_state()
        self.sds_characteristics = self.__get_sds_characteristics()
        logger.info(self.analysis_summary)

        # Get characteristics for logging/reporting
        logger.info(
            f"Detected: SDS {self.sampling_design_state} - "
            f"{self.sds_characteristics['description']}")

        # Validate BEFORE calculating (catch issues early)
        self.__validate_sds_for_analysis()
        # Calculate VAS residuals only when appropriate
        if self.__should_calculate_vas_residuals():
            logger.debug("Calculating VAS residuals (R1-R5)")
            self.__calculate_residuals()
            self.__calculate_centered_residuals()
            self.__calculate_interactions()   
            self.__calculate_effects()
        else:
            logger.debug(
                f"Skipping VAS residuals for analysis_type={self.spec.analysis_type}, "
                f"SDS={self.sampling_design_state}"
            )
  
    def __should_calculate_vas_residuals(self) -> bool:
        """
        Determine if VAS residual decomposition (R1-R5) should be calculated.
        
        VAS residuals are the foundation of Wheeler/Bishop's variance analysis
        system. They decompose total variation into components:
        - R1: Total deviation from grand mean
        - R2: Within-cell (unexplained) variation
        - R3: Interaction effects (factor × time)
        - R4: Time effects + unexplained
        - R5: Factor effects + unexplained
        
        Calculate VAS residuals when:
        1. User requests Xbar or S chart (cell-level analysis)
        2. AND we have proper (k,t) factorial structure (SDS 1, 2, 3)
        3. Purpose: Decompose variance, analyze interactions
        
        Do NOT calculate VAS residuals when:
        1. User requests IMR or R chart (individual-level analysis)
        2. OR no proper structure (SDS 0, 4, 6)
        3. Purpose: Monitor individual values over time (possibly stratified)
        
        Key Insight:
        -----------
        **Grouping variables serve different purposes:**
        
        For Xbar-S: Grouping defines CELLS for variance decomposition
            → Need R1-R5 to analyze factor/time/interaction effects
            → VAS residuals ARE the point
        
        For IMR/R: Grouping defines STRATA for separate charts
            → Each group gets own IMR chart with own limits
            → VAS residuals NOT needed (different methodology)
            → This is the "automatic stratification" killer feature
        
        Examples:
        --------
        >>> # VAS Analysis (needs residuals)
        >>> spec = {'analysis_type': 'Xbar', 'rsg_vars': ['lane'], 'time_var': 'pull'}
        >>> ads = AnalysisDataSet(df, spec)
        >>> ads.__should_calculate_vas_residuals()  # → True
        >>> # Gets: R1-R5, interactions, effects
        
        >>> # Stratified IMR (doesn't need residuals)
        >>> spec = {'analysis_type': 'Imr', 'rsg_vars': ['lane'], 'time_var': 'pull'}
        >>> ads = AnalysisDataSet(df, spec)  
        >>> ads.__should_calculate_vas_residuals()  # → False
        >>> # Gets: Separate IMR chart per lane with lane-specific limits
        
        Returns:
            True if VAS residuals should be calculated, False otherwise
        """
        
        # Quick rejections - clearly don't need VAS
        if self.sampling_design_state == 0:
            logger.debug("No VAS: SDS 0 (no structure)")
            return False
        
        if self.sampling_design_state == 4:
            logger.debug("No VAS: SDS 4 (single condition over time)")
            return False
        
        if self.sampling_design_state == 6:
            logger.debug("No VAS: SDS 6 (irregular grid - can't form proper cells)")
            return False
        
        # KEY DECISION: Analysis type determines if we need VAS
        # 
        # Xbar/S → Analyzing CELL statistics (means/std devs)
        #          Need VAS to decompose variance components
        #
        # IMR/R  → Analyzing INDIVIDUAL values over time
        #          Use moving ranges, don't need factorial decomposition
        #          Grouping just creates separate charts (stratification)
        
        if self.spec.analysis_type in ['Imr', 'R']:
            logger.debug(
                f"No VAS: {self.spec.analysis_type} analysis uses moving ranges, "
                f"not factorial decomposition. "
                f"Grouping creates stratified charts (separate chart per group)."
            )
            return False
        
        # At this point: Xbar or S chart with structure
        if self.sampling_design_state in [1, 2, 3]:
            logger.debug(
                f"Calculate VAS: SDS {self.sampling_design_state} with "
                f"{self.spec.analysis_type} analysis supports full decomposition"
            )
            return True
        
        # SDS 5 (nested) - special case
        if self.sampling_design_state == 5:
            logger.warning(
                "SDS 5 (nested design) detected with Xbar-S analysis. "
                "VAS decomposition for nested structures requires special handling. "
                "Proceeding with standard VAS - results may need interpretation."
            )
            return True
        
        # Shouldn't reach here, but be conservative
        logger.debug(f"No VAS: Unexpected case (SDS={self.sampling_design_state})")
        return False
    
    @property
    def has_vas_residuals(self) -> bool:
        """Check if VAS residuals were calculated."""
        return 'R1' in self.analysis_dataset.columns

    @property  
    def analysis_summary(self) -> dict:
        """
        Get comprehensive summary of the analysis dataset.
        
        Returns:
            Dictionary with analysis metadata including:
            - sds: Detected sampling design state
            - sds_info: Full SDS characteristics
            - has_vas: Whether VAS residuals calculated
            - n_observations: Total observations
            - n_groups: Number of unique groups
            - n_time_periods: Number of time periods
        """
        summary = {
            'sds': self.sampling_design_state,
            'sds_info': self.sds_characteristics,
            'has_vas_residuals': self.has_vas_residuals,
            'n_observations': len(self.analysis_dataset),
            'analysis_type': self.spec.analysis_type
        }
        return summary

    def sampling_design_state(self) -> int:
        self.sampling_design_state
        
    def raw_dataset(self) -> pd.DataFrame:
        self.raw_dataset
    
    def analysis_dataset(self) -> pd.DataFrame:
        self.analysis_dataset
        
    def statistics(self) -> dict:
        self.statistics
    
    def interactions(self) -> dict:
        self.interactions
        
    def effects(self) -> dict:
        self.effects
        
    def __validate_columns(self) -> bool:    

        # TODO: Centralize constants 
        number_column_types = ['int64', 'float64', 'int32', 'Int64', 'Float64', 'Int32']
        # string_column_types = ['string', 'object', 'category', 'String', 'Object']
        #TODO: Update to remove datatime types use this for numeric types https://pandas.pydata.org/docs/reference/api/pandas.api.types.is_numeric_dtype.html
        #TODO: add check for api types categorical types
        date_column_types = ['int64', 'int32', 'Int64', 'Int32', 'datetime64[ns]','object']

        df_cols = self.raw_dataset.columns.tolist()

        if self.spec.has_grouping:

            if not all(item in df_cols for item in self.spec.rsg_vars):
                raise ValueError(f'One or more of the rsg_vars: {self.spec.rsg_vars}: are not in the data set: {df_cols}!')
        
        # Validate response variable
        if not self.spec.response_var in df_cols: raise ValueError(
            f'The response variable: {self.spec.response_var}: is not in the data set!')

        response_type = self.raw_dataset[self.spec.response_var].dtype

        if not is_numeric_dtype(response_type): raise ValueError( #TODO: adjust to use api check
            f'The response variable: {self.spec.response_var}: is not type numeric!')

        return True

    def __add_column(self, df: pd.DataFrame, new_col_name: str, existing_column: str):
        # add a copy of an existing column for use with single variable groupings

        df_cols = df.columns.tolist()
        check = existing_column in df_cols

        if (check):

            kwargs = {new_col_name: df[existing_column]}

            out = df.assign(**kwargs)
            return (out)
        else:
            raise ValueError(str(existing_column) + ": is not in the data set!")


    def __add_composite_column(self, df: pd.DataFrame, cols_to_combine: list, col_name: str, col_delim: str='_') -> pd.DataFrame:
        df_cols = df.columns.tolist()
        check = all(item in df_cols for item in cols_to_combine)

        if (check):

            if (len(cols_to_combine) > 1):  # Only combine when list has more than one value
                len_col_delim = len(col_delim)
                # dynamically build columns from list of column names\n",
                combined = (df[cols_to_combine].astype(str) + col_delim).cumsum(1).iloc[:, -1].values
                # remove trailing \"_\" with comprehension\n",
                combined = [x[:-len_col_delim] for x in combined]

                # hack to get around assign not taking string col_name
                kwargs = {col_name: combined}
                out = df.assign(**kwargs)

                return (out)        
        else:

            raise ValueError("One or more of: " + str(cols_to_combine) + ": are not in the data set!")
    def __add_grouping_variable_column(self, df: pd.DataFrame, cols_to_combine: list, col_name: str, col_delim: str='_')->pd.DataFrame: 
        
        out = df.copy()
        
        if len(cols_to_combine)>1:
            out = self.__add_composite_column(df = out, cols_to_combine = cols_to_combine, col_name = col_name, col_delim = col_delim)
        else:        
            out = self.__add_column(df = df, new_col_name = col_name, existing_column = cols_to_combine[0])
        
        return(out)    
    def __prepare_dataset(self)-> pd.DataFrame:

        logger.debug('Entering prepare_dataset')
        out = self.raw_dataset.copy()

        if self.spec.has_grouping:

            # Drop 'n' column if it exists in the input data to avoid conflicts
            if 'n' in out.columns:
                logger.debug('Dropping existing "n" column from input data (will be recalculated)')
                out = out.drop(columns=['n'])

            out = self.__add_grouping_variable_column(df=self.raw_dataset, col_name=self.spec.rsg_var_name, cols_to_combine=self.spec.rsg_vars, col_delim=self.spec.rsg_var_delim)

            # Drop 'n' column again if it was added back by __add_grouping_variable_column
            if 'n' in out.columns:
                out = out.drop(columns=['n'])

            #Remove groups with one obs - solve for all grouped analyses
            grouped=out.groupby(self.spec.rsg_var_name).size()
            starting_count = grouped.count()
            logger.debug('Starting with %s groups', starting_count)
            grouped=grouped[grouped>1]

            if grouped.shape[0] == 0:
                raise ValueError("All subgroups have 1 or less observations!")

            grouped = grouped.reset_index()
            grouped = grouped.rename(columns = {0:'n'})

            logger.debug('Pruning groups with 1 obs:\\n%s', grouped)
            ending_count = grouped.count()

            logger.debug('Groups remaining: %s', ending_count.iloc[0])
            logger.debug('Removed %s group(s)', starting_count - ending_count.iloc[0])
            # Empty print removed

            out = pd.merge(out, grouped, how='inner', on=self.spec.rsg_var_name)

        # Perform sorting
        if self.spec.requires_sort:
            out = out.sort_values(self.spec.sort_cols) #These are predetermined by the spec

        out = out[self.spec.data_prep_output_cols] #These are predetermined by the spec

        out = out.dropna()

        return out
    
    
    def __calculate_sampling_design_state(self):
        """
        Detect Sampling Design State (SDS) from the current dataset and specification.

        This function implements the complete SDS classification system from the
        Variance Analysis System (VAS) framework by Wheeler and Bishop.

        Sampling Design States:
        ----------------------
        SDS 0: No structure
            - No grouping or time variables specified
            - Default/fallback state
            - Limited analysis capabilities

        SDS 1: Full replication
            - Every (k,t) cell has n≥2 observations
            - Enables direct within-cell variance estimation (R2)
            - Supports full interaction analysis
            - Most statistically powerful

        SDS 2: No replication  
            - Every (k,t) cell has exactly n=1 observation
            - Requires moving average for R2 estimation
            - Limited interaction analysis
            - Common in designed experiments

        SDS 3: Partial replication
            - Mix of n=1 and n≥2 cells
            - Requires hybrid R2 estimation
            - Most common in real-world data
            - Challenging to analyze correctly

        SDS 4: Single condition over time
            - Only one factor level (K=1)
            - Multiple time points (T>1)
            - Time series structure
            - Appropriate for IMR charts

        SDS 5: Nested design with asynchronous coverage
            - Hierarchical factor structure (Factor 2 nested in Factor 1)
            - Not all factor combinations present at all times
            - Irregular temporal patterns
            - Requires variance components

        SDS 6: Unstructured/regime changes
            - Cannot form regular (k,t) grid
            - Irregular sampling patterns
            - May have regime changes or process shifts
            - Complex time structure

        Detection Logic:
        ---------------
        1. Check for basic structure (grouping and/or time)
        2. If both present, check for nested structure (SDS 5)
        3. Examine cell size distribution:
        - All n≥2 → SDS 1
        - All n=1 → Check for complete grid
        - Mixed → SDS 3
        4. Check for single condition (SDS 4)
        5. Check for irregular grid (SDS 6)

        Returns:
        -------
        int
            0 = No structure (default)
            1 = Full replication
            2 = No replication (complete unreplicated factorial)
            3 = Partial replication
            4 = Single condition over time
            5 = Nested design
            6 = Unstructured/regime changes

        Notes:
        ------
        The detection is conservative - if unsure, defaults to a simpler SDS.
        SDS 3 (partial replication) is the most common in practice but also
        the most complex to handle correctly.

        References:
        ----------
        Wheeler, D. J. (1995). Advanced Topics in Statistical Process Control.
        """
    
        # ========================================================================
        # SDS 0: No structure - neither grouping nor time specified
        # ========================================================================
        if not (self.spec.has_grouping and self.spec.has_time):
            logger.info("SDS 0: No grouping or time structure")
            return 0
    
        # ========================================================================
        # From here on, we have both grouping and time variables
        # ========================================================================
        
        grouping_vars = [self.spec.rsg_var_name, self.spec.time_var]
        
        # Count observations per (k,t) cell
        cell_sizes = (self.analysis_dataset
                    .groupby(grouping_vars, dropna=False)[self.spec.response_var]
                    .count())
        
        n_cells = len(cell_sizes)
        min_n = cell_sizes.min()
        max_n = cell_sizes.max()
    
        # Calculate theoretical full grid size
        n_groups = self.analysis_dataset[self.spec.rsg_var_name].nunique()
        n_times = self.analysis_dataset[self.spec.time_var].nunique()
        full_grid_size = n_groups * n_times
        
        logger.debug(f"SDS Detection: {n_groups} groups × {n_times} times = {full_grid_size} possible cells")
        logger.debug(f"SDS Detection: {n_cells} cells observed, n range: [{min_n}, {max_n}]")
    
        # ========================================================================
        # SDS 5: Nested design - check for hierarchical structure
        # ========================================================================
        # In nested designs, one factor is nested within another
        # Example: heads nested within lanes (each head belongs to one lane only)
        
        if len(self.spec.rsg_vars) >= 2:
            # Check if second factor is nested within first
            # Each level of factor 2 should appear with only one level of factor 1
            
            factor1 = self.spec.rsg_vars[0]
            factor2 = self.spec.rsg_vars[1]
            
            # Count how many levels of factor1 each level of factor2 appears with
            nesting_check = (self.analysis_dataset
                            .groupby(factor2)[factor1]
                            .nunique())
        
            is_nested = (nesting_check == 1).all()
            
            if is_nested:
                # Additional check: incomplete temporal coverage (asynchronous)
                # In true SDS 5, not all factor combinations appear at all times
                coverage_ratio = n_cells / full_grid_size
                
                if coverage_ratio < 0.90:  # Less than 90% coverage
                    logger.info(
                        f"SDS 5: Nested design detected - {factor2} nested in {factor1}, "
                        f"{coverage_ratio:.1%} grid coverage"
                    )
                    return 5
                else:
                    logger.debug(
                        f"Nested structure detected but high coverage ({coverage_ratio:.1%}) - "
                        f"treating as crossed design"
                    )
        
        # ========================================================================
        # SDS 4: Single condition over time
        # ========================================================================
        # Only one factor level but multiple time points
        
        if n_groups == 1 and n_times > 1:
            logger.info(f"SDS 4: Single condition over time ({n_times} time points)")
            return 4
        
        # ========================================================================
        # SDS 6: Unstructured / Incomplete grid
        # ========================================================================
        # Check if we have an irregular (factor × time) grid
        # This indicates irregular sampling or regime changes
        
        coverage_ratio = n_cells / full_grid_size
        
        if coverage_ratio < 0.75:  # Less than 75% of cells have data
            logger.info(
                f"SDS 6: Unstructured/incomplete grid - "
                f"{n_cells}/{full_grid_size} cells present ({coverage_ratio:.1%})"
            )
            return 6
    
        # ========================================================================
        # SDS 1, 2, 3: Based on cell size distribution
        # ========================================================================
        # At this point we have a reasonably complete (k,t) grid
        
        # Count cells by size
        cells_with_n1 = (cell_sizes == 1).sum()
        cells_with_n2_plus = (cell_sizes >= 2).sum()
        
        logger.debug(f"SDS Detection: {cells_with_n1} cells with n=1, {cells_with_n2_plus} cells with n≥2")
        
        # --------------------------------------------------------------------
        # SDS 1: Full replication - ALL cells have n≥2
        # --------------------------------------------------------------------
        if min_n >= 2:
            logger.info(f"SDS 1: Full replication (all cells have n≥2, range: [{min_n}, {max_n}])")
            return 1
        
        # --------------------------------------------------------------------
        # SDS 2: No replication - ALL cells have n=1
        # --------------------------------------------------------------------
        if max_n == 1:
            # Verify this is truly a complete unreplicated factorial
            if coverage_ratio >= 0.95:  # At least 95% complete
                logger.info(f"SDS 2: No replication (all cells have n=1, {coverage_ratio:.1%} complete)")
                return 2
            else:
                # Incomplete grid with all n=1 → treat as SDS 6
                logger.info(
                    f"SDS 6: Incomplete grid with no replication "
                    f"({coverage_ratio:.1%} coverage)"
                )
                return 6
    
        # --------------------------------------------------------------------
        # SDS 3: Partial replication - Mix of n=1 and n≥2
        # --------------------------------------------------------------------
        if cells_with_n1 > 0 and cells_with_n2_plus > 0:
            pct_replicated = 100 * cells_with_n2_plus / n_cells
            logger.info(
                f"SDS 3: Partial replication - "
                f"{cells_with_n2_plus}/{n_cells} cells replicated ({pct_replicated:.1f}%), "
                f"n range: [{min_n}, {max_n}]"
            )
            return 3
    
        # ========================================================================
        # Fallback: Should not reach here, but default to SDS 0
        # ========================================================================
        logger.warning(
            f"SDS Detection: Unexpected case - defaulting to SDS 0. "
            f"n_groups={n_groups}, n_times={n_times}, n_cells={n_cells}, "
            f"min_n={min_n}, max_n={max_n}"
        )
        return 0


# ============================================================================
# Additional Helper Methods for SDS-Specific Logic
# ============================================================================

    def __get_sds_characteristics(self) -> dict:
        """
        Get detailed characteristics of the detected SDS.
        
        Returns dictionary with:
        - sds: The detected SDS number
        - description: Human-readable description
        - replication_type: 'full', 'none', 'partial', 'single', 'nested', 'irregular'
        - r2_method: How R2 should be calculated
        - capabilities: What analyses are supported
        
        This can be useful for logging, reporting, and determining analysis approach.
        
        Returns:
            Dictionary with SDS characteristics
        """
        sds = self.sampling_design_state
        
        characteristics = {
            0: {
                'description': 'No grouping or time structure',
                'replication_type': 'none',
                'r2_method': 'not_applicable',
                'capabilities': ['basic_statistics_only'],
                'interaction_analysis': False,
                'variance_decomposition': False
            },
            1: {
                'description': 'Full replication (all cells n≥2)',
                'replication_type': 'full',
                'r2_method': 'within_cell',
                'capabilities': ['full_vas', 'all_residuals', 'interactions', 'main_effects'],
                'interaction_analysis': True,
                'variance_decomposition': True
            },
            2: {
                'description': 'No replication (all cells n=1)',
                'replication_type': 'none',
                'r2_method': 'moving_average',
                'capabilities': ['all_residuals', 'limited_interactions', 'main_effects'],
                'interaction_analysis': 'limited',
                'variance_decomposition': True
            },
            3: {
                'description': 'Partial replication (mixed n=1 and n≥2)',
                'replication_type': 'partial',
                'r2_method': 'hybrid',
                'capabilities': ['all_residuals', 'partial_interactions', 'main_effects'],
                'interaction_analysis': 'partial',
                'variance_decomposition': True
            },
            4: {
                'description': 'Single condition over time (K=1)',
                'replication_type': 'single_stream',
                'r2_method': 'moving_range',
                'capabilities': ['time_series', 'imr_chart', 'trend_analysis'],
                'interaction_analysis': False,
                'variance_decomposition': False
            },
            5: {
                'description': 'Nested design with asynchronous coverage',
                'replication_type': 'nested',
                'r2_method': 'nested_variance_components',
                'capabilities': ['variance_components', 'nested_effects', 'hierarchical_analysis'],
                'interaction_analysis': 'hierarchical',
                'variance_decomposition': 'hierarchical'
            },
            6: {
                'description': 'Unstructured/irregular grid',
                'replication_type': 'irregular',
                'r2_method': 'adaptive',
                'capabilities': ['regime_detection', 'adaptive_limits', 'sparse_analysis'],
                'interaction_analysis': False,
                'variance_decomposition': 'limited'
            }
        }
        
        result = characteristics.get(sds, characteristics[0]).copy()
        result['sds'] = sds
        
        return result


    def __validate_sds_for_analysis(self) -> bool:
        """
        Validate that the detected SDS is appropriate for the requested analysis.
        
        Returns True if analysis can proceed, raises ValueError with helpful
        message if not.
        
        Examples of issues caught:
        - Requesting Xbar-S chart on SDS 2 (no within-cell variance)
        - Requesting full VAS on SDS 0 (no structure)
        - Insufficient data for requested analysis type
        """
        sds = self.sampling_design_state
        analysis_type = self.spec.analysis_type
        
        # SDS 0: Very limited capabilities
        if sds == 0:
            if analysis_type in ['Xbar', 'S']:
                raise ValueError(
                    f"Cannot perform {analysis_type} analysis without grouping structure. "
                    f"Detected SDS 0 (no grouping or time variables). "
                    f"Consider using 'Imr' analysis or specify grouping variables."
                )
        
        # SDS 2: No within-cell variance
        if sds == 2:
            if analysis_type in ['Xbar', 'S']:
                logger.warning(
                    f"SDS 2 detected: No replication (all cells n=1). "
                    f"{analysis_type} analysis will use moving average for variance estimation. "
                    f"Consider using 'Imr' analysis instead."
                )
        
        # SDS 4: Single stream - limited to IMR
        if sds == 4:
            if analysis_type not in ['Imr', 'R']:
                logger.warning(
                    f"SDS 4 detected: Single condition over time. "
                    f"{analysis_type} analysis may not be appropriate. "
                    f"Consider using 'Imr' analysis."
                )
        
        # SDS 6: Irregular - may have issues
        if sds == 6:
            logger.warning(
                f"SDS 6 detected: Unstructured/irregular grid. "
                f"Analysis results may be unreliable due to incomplete data coverage. "
                f"Consider checking for missing data or irregular sampling patterns."
            )
        
        return True


    # ============================================================================
    # Usage Example in AnalysisDataSet.__init__
    # ============================================================================
    
    def __calculate_Ybar(self):
        out = self.analysis_dataset[self.spec.response_var].mean()
        self.analysis_dataset['Ybar'] = out
        self.statistics['Ybar'] = out          
        
    def __calculate_Ybar_k(self):        
        
        #out=pd.merge(self.analysis_dataset, out, how='left', on =[self.spec.rsg_var_name])
        self.analysis_dataset['Ybar_k'] = self.analysis_dataset.groupby([self.spec.rsg_var_name])[self.spec.response_var].transform('mean')
       
    def __calculate_Ybar_kt(self):
        self.analysis_dataset["Ybar_kt"] = self.analysis_dataset.groupby([self.spec.rsg_var_name, self.spec.time_var])[self.spec.response_var].transform("mean")
    
    def __calculate_Ybar_t(self): 
               
        self.analysis_dataset['Ybar_t'] = self.analysis_dataset.groupby([self.spec.time_var])[self.spec.response_var].transform("mean")
                
    def __calculate_R1_residual(self):    # R1= Y-Ybar    
        
        self.analysis_dataset['R1'] = self.analysis_dataset[self.spec.response_var] \
                                    -  self.statistics['Ybar']
        
    def __calculate_R2_residual(self):
        """Calculate R2 residual based on SDS."""
        logger = logging.getLogger(__name__)

        if self.sampling_design_state == 1:
            # SDS1: standard within-(k,t) residual
            self.analysis_dataset['R2'] = (
                self.analysis_dataset[self.spec.response_var] - 
                self.analysis_dataset['Ybar_kt']
            )
            logger.debug("SDS1: R2 = Y - Ybar_kt")

        elif self.sampling_design_state == 2:
            # SDS2: moving-average residual
            df = self.analysis_dataset.sort_values(
                [self.spec.rsg_var_name, self.spec.time_var]
            ).copy()
            df['_MA2'] = (
                df.groupby(self.spec.rsg_var_name)[self.spec.response_var]
                .transform(lambda s: (s.shift(1) + s.shift(-1)) / 2.0)
            )
            fwd = df.groupby(self.spec.rsg_var_name)[self.spec.response_var].shift(-1)
            back = df.groupby(self.spec.rsg_var_name)[self.spec.response_var].shift(1)
            df['_MA2'] = df['_MA2'].where(df['_MA2'].notna(), fwd.where(fwd.notna(), back))
            df['R2'] = df[self.spec.response_var] - df['_MA2']
            self.analysis_dataset['R2'] = df['R2']
            logger.debug("SDS2: R2 ≈ Y - MA2_k(t)")
        
        elif self.sampling_design_state == 3:
            # SDS3: HYBRID approach
            # TODO: Implement hybrid R2 calculation
            # For now, use SDS1 approach (within-cell for replicated cells)
            logger.warning(
                "SDS3: Hybrid R2 calculation not yet fully implemented. "
                "Using within-cell method (may be suboptimal for n=1 cells)."
            )
            self.analysis_dataset['R2'] = (
                self.analysis_dataset[self.spec.response_var] - 
                self.analysis_dataset['Ybar_kt']
            )
        
        else:
            logger.error(f"Cannot calculate R2 for SDS {self.sampling_design_state}")
            raise ValueError(
                f"R2 calculation not defined for SDS {self.sampling_design_state}"
            )
            
    def __calculate_R3_residual(self):    #R3= Y(kt)-Ybar(k)-Ybar(.t)+YBAR 
        
        self.analysis_dataset['R3'] = self.analysis_dataset['Ybar_kt']\
                                    - self.analysis_dataset['Ybar_k']\
                                    - self.analysis_dataset['Ybar_t']\
                                    + self.analysis_dataset['Ybar']\
                                    + self.analysis_dataset['R2'] 
    
    def __calculate_R4_residual(self):
        # R4 = Ybar_t - Ybar + R2   (Eq. 72) for ALL SDS
        self.analysis_dataset['R4'] = (
            self.analysis_dataset['Ybar_t'] - self.analysis_dataset['Ybar'] + self.analysis_dataset['R2']
        )

    def __calculate_R5_residual(self):
        # R5 = Ybar_k - Ybar + R2   (Eq. 75) for ALL SDS
        self.analysis_dataset['R5'] = (
            self.analysis_dataset['Ybar_k'] - self.analysis_dataset['Ybar'] + self.analysis_dataset['R2']
        )
                                                   
    def __calculate_Rbar_kt(self):
        self.analysis_dataset["Rbar_kt"] = self.analysis_dataset.groupby([self.spec.rsg_var_name, self.spec.time_var])["R1"].transform("mean")
        
    def __calculate_Rbar_k(self):        
        self.analysis_dataset['Rbar_k'] = self.analysis_dataset.groupby([self.spec.rsg_var_name])["R1"].transform('mean')
        
    def __calculate_Rbar_t(self): 
               
        self.analysis_dataset['Rbar_t'] = self.analysis_dataset.groupby([self.spec.time_var])["R1"].transform("mean")
    
    def __calculate_RCR1(self): #Y-R1_bar
        self.analysis_dataset['RCR1'] = self.analysis_dataset["Ybar"] + self.analysis_dataset["R1"]
        
    def __calculate_RCR2(self): #Y-R1bar_kt
        
            self.analysis_dataset["RCR2"] = self.analysis_dataset["Ybar"]\
                                          + self.analysis_dataset["R2"]
                
    def __calculate_RCR3(self):    #R3= Y(kt)-Ybar(k)-Ybar(.t)+YBAR 
        
            self.analysis_dataset['RCR3'] = self.analysis_dataset['Ybar']\
                                          + self.analysis_dataset["R3"]                                                                             
            
    def __calculate_RCR4(self):     #R4= Y – YBAR – YBAR(kt)+Ybar(t)
        
            self.analysis_dataset['RCR4'] = self.analysis_dataset["Ybar"]\
                                          + self.analysis_dataset['R4']            
                                        
    def __calculate_RCR5(self):    #R5= Y-Ybar-YBar(kt)+YBar(k)
        
            self.analysis_dataset['RCR5'] = self.analysis_dataset["Ybar"]\
                                          + self.analysis_dataset['R5']
                                                   
    def __calculate_pdc_by_time(self):
        if self.sampling_design_state==1:
            # SDS1: Calculate mean R3 for each group-time cell and broadcast to all rows
            self.analysis_dataset['pdc_by_pt'] = self.analysis_dataset.groupby([self.spec.rsg_var_name, self.spec.time_var])['R3'].transform('mean')
            self.interactions["pdc_by_pt"] = self.analysis_dataset['pdc_by_pt']

        elif self.sampling_design_state==2:
            self.analysis_dataset['pdc_by_pt'] = self.analysis_dataset['Ybar_kt']\
                                                         - self.analysis_dataset['Ybar_k']\
                                                         - self.analysis_dataset['Ybar_t']\
                                                         + self.analysis_dataset['Ybar']
            
            self.interactions["pdc_by_pt"] = self.analysis_dataset['pdc_by_pt']
            #Two-factor interaction effect
            #Pg 77 Average R5ij-average R5i- average R5j 
            #there will be one main effect for the rational subgroup and one ME for each factor x level
           
            me  = self.analysis_dataset.groupby(self.spec.rsg_vars).agg(ME=pd.NamedAgg(column="R5",aggfunc="mean")).reset_index()
           
            if len(self.spec.rsg_vars)>2:
                logger.warning("There are more than 2 variables in the RSG - ME interactions being calculated for first 2")
                
            for factor in self.spec.rsg_vars:

                tmp=self.analysis_dataset.groupby(factor).agg(MEF=pd.NamedAgg(column="R5", aggfunc="mean"))
                me = me.merge(tmp, how='left', on=factor)
                me.rename(columns={"MEF":factor+"_ME"}, inplace=True)

            # Use the first two factor names from rsg_vars
            factor1_me_col = self.spec.rsg_vars[0] + "_ME"
            factor2_me_col = self.spec.rsg_vars[1] + "_ME" if len(self.spec.rsg_vars) > 1 else None

            if factor2_me_col:
                me["F1xF2"] = me["ME"] - me[factor1_me_col] - me[factor2_me_col]
            else:
                me["F1xF2"] = me["ME"] - me[factor1_me_col]
            self.interactions["F1xF2"] = me[self.spec.rsg_vars+["F1xF2"]]   
            
    def __calculate_main_effect(self):
            #Use RSG to perform the grouping
            #self.effects['main_effects'] = self.analysis_dataset.groupby(self.spec.rsg_var_name)['R2'].mean()
            self.effects['main_effect'] = self.analysis_dataset.groupby(self.spec.rsg_var_name).agg(
                                                                Main_Effect=pd.NamedAgg(column="R5", aggfunc="mean"))
            
            for factor in self.spec.rsg_vars:
                label=factor
                self.effects[label] = self.analysis_dataset.groupby([factor]).agg(
                                                               Main_Effect=pd.NamedAgg(column="R5", aggfunc="mean"))
    def __calculate_main_effects(self):
            #dataset Columns F1 MAIN EFFECTS & F2 MAIN EFFECTS - Note Effect versus Effects R2i+MEi
            for factor in self.spec.rsg_vars:
                
                me =self.effects[factor]
                
                df = pd.DataFrame()
                
                label=factor+"_MEs"               
                
                df= self.analysis_dataset[[factor,"R2"]].merge(me, how='left', on=factor)
                df[label] = df["R2"] + df["Main_Effect"]
               
                self.effects[label] = df[[factor,label]]
                
    def __calculate_factor_interaction_effects(self):
        #calculate average R5 for levels of RSG

        # Only calculate interaction effects if there are 2 or more factors
        if len(self.spec.rsg_vars) < 2:
            logger.info('Only %s factor(s) in RSG - factor interaction effects require at least 2 factors. Skipping.', len(self.spec.rsg_vars))
            return

        #df=self.analysis_dataset[self.spec.rsg_vars+["R2"]]
        rsg_effects = self.analysis_dataset.groupby(self.spec.rsg_vars).agg(
                                                                R5=pd.NamedAgg(column="R5", aggfunc="mean"))
        rsg_effects.reset_index(inplace=True)


        #get main effects for each factor and add to rsg_effects
        # for factor in self.spec.rsg_vars:
        #     df=pd.DataFrame()
        #     df = self.effects[factor]
        #     rsg_effects = rsg_effects.merge(df, how='left', on=factor)

        if len(self.spec.rsg_vars)>2:
            logger.warning('There are more than 2 variables in the RSG - ME interactions being calculated for first 2: %s', self.spec.rsg_vars)

        # Only process the minimum of 2 or the actual number of factors
        num_factors_to_process = min(2, len(self.spec.rsg_vars))
        i = 0

        while i < num_factors_to_process: #The system only handles 2 factor interactions

            df=pd.DataFrame()
            factor = self.spec.rsg_vars[i]
            logger.debug('Processing factor: %s', factor)
            df = self.effects[factor]
            rsg_effects = rsg_effects.merge(df, how='left', on=factor)
            i=i+1

        rsg_effects["Rx"] = rsg_effects["R5"] - rsg_effects["Main_Effect_x"] - rsg_effects["Main_Effect_y"]
        tmp = self.analysis_dataset[self.spec.rsg_vars+["R2"]]
        tmp = tmp.merge(rsg_effects, how='left', on=self.spec.rsg_vars)
        tmp["factor_interaction_effects"] = tmp["R2"] + tmp["Rx"]
        tmp = tmp[self.spec.rsg_vars+["factor_interaction_effects"]]
        self.effects["factor_interaction_effects"] = tmp

        #print(f'Factor Interaction Effects:\n {rsg_effects[self.spec.rsg_vars+["Rx"]]}/n Results:\n{tmp}') 
                                                              

    def __calculate_time_me(self):        
        
            self.effects['pt_me'] = self.analysis_dataset.groupby([self.spec.time_var]).agg(
                                                                PT_ME=pd.NamedAgg(column="R1", aggfunc="mean"))        

    def __calculate_residuals(self):
        
        if self.spec.has_grouping:
            
            self.__calculate_Ybar()
            self.__calculate_Ybar_k()
            self.__calculate_Ybar_kt()
            self.__calculate_Ybar_t()
            self.__calculate_R1_residual()
            self.__calculate_R2_residual()
            self.__calculate_R3_residual()
            self.__calculate_R4_residual()
            self.__calculate_R5_residual()
    
    def __calculate_centered_residuals(self):     
           
        if self.spec.has_grouping:
            
            self.__calculate_Rbar_kt()
            self.__calculate_Rbar_k()
            self.__calculate_Rbar_t()
            self.__calculate_RCR1()
            self.__calculate_RCR2()
            self.__calculate_RCR3()
            self.__calculate_RCR4()   
            self.__calculate_RCR5()    
            
    def __calculate_interactions(self):
        
        self.__calculate_pdc_by_time()
    
    def __calculate_effects(self):
        self.__calculate_main_effect() #order matters #1
        self.__calculate_time_me()
        self.__calculate_main_effects()
        self.__calculate_factor_interaction_effects()
        
