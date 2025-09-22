from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import prepare_dataset
from .limits import calculate_limits, detect_beyond_limits
from .spec import AnalysisSpecification


def _calculate_moving_ranges(df: pd.DataFrame, spec: AnalysisSpecification) -> pd.DataFrame:
    """Helper function to calculate moving ranges for Imr and R analyses."""
    out = df.copy()

    if spec.has_grouping:
        out['mr'] = abs(out.groupby(spec.rsg_var_name)[spec.response_var].diff())
    else:
        out['mr'] = abs(out[spec.response_var].diff())

    return out


def _prepare_output_columns(
    df: pd.DataFrame, spec: AnalysisSpecification, base_cols: list
) -> pd.DataFrame:
    """Helper function to prepare output columns based on time and grouping specs."""
    out = df.copy()
    cols_to_keep = base_cols.copy()

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

    return out[cols_to_keep].round(spec.round_to)


def _package_grouped_results(
    df: pd.DataFrame, spec: AnalysisSpecification, statistics_cols: list
) -> dict:
    """Helper function to package results for grouped vs non-grouped analyses."""
    if spec.has_grouping:
        statistics = gather_analysis_statistics(
            df=df, statistics_to_collect=statistics_cols, grouping_var=spec.rsg_var_name
        )
        split_dict = split_df_by_group(df=df, grouping_var=spec.rsg_var_name)
        return package_analysis(analysis_output=split_dict, summary_statistics_output=statistics)
    else:
        statistics = gather_analysis_statistics(df=df, statistics_to_collect=statistics_cols)
        _out = {'all': df}
        return package_analysis(analysis_output=_out, summary_statistics_output=statistics)


def perform_analysis(df: pd.DataFrame, specification: dict) -> pd.DataFrame:
    """Perform SPC analysis based on specification.

    Args:
        df: Input dataframe
        specification: Analysis specification dictionary

    Returns:
        DataFrame with analysis results

    Raises:
        ValueError: If analysis type is not supported
    """
    analysis_type = specification['analysis_type']

    # Create specification and prepare dataset
    spec = AnalysisSpecification(analysis_type=analysis_type, analysis_specification=specification)
    prepared_df = prepare_dataset(df=df, analysis_specification=spec)

    # Direct mapping to calculation functions
    if analysis_type == 'Xbar' or analysis_type == 'S':
        return calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)
    elif analysis_type == 'Imr':
        return calculate_statistics_Imr(df=prepared_df, analysis_specification=spec)
    elif analysis_type == 'R':
        return calculate_statistics_R(df=prepared_df, analysis_specification=spec)
    else:
        raise ValueError(
            f'Analysis type {analysis_type} not supported. '
            f'Available types: ["Xbar", "S", "Imr", "R"]'
        )


def calculate_statistics_Imr(
    df: pd.DataFrame, analysis_specification: AnalysisSpecification
) -> pd.DataFrame:
    spec = analysis_specification
    print('\nIn calculate statistics IMR...')
    print(f'\nDataframe has columns: {df.columns.to_list()}')

    # Calculate moving ranges using helper function
    out = _calculate_moving_ranges(df, spec)

    if spec.has_grouping:
        grouped = out.groupby(spec.rsg_var_name, as_index=False).agg(
            mean=pd.NamedAgg(spec.response_var, 'mean'), mR=pd.NamedAgg('mr', 'mean')
        )

        limits = grouped.apply(
            lambda row: calculate_limits(mean=row['mean'], sd=0, N=0, mR=row.mR, limits_type='Imr'),
            axis=1,
        )

        grouped = pd.merge(grouped, limits, left_index=True, right_index=True)
        out = pd.merge(out, grouped, how='left', on=spec.rsg_var_name)
    else:
        out['mR'] = out['mr'].mean()
        out['mean'] = out[spec.response_var].mean()

        mR = out['mR'].max()
        _mean = out['mean'].max()

        limits = calculate_limits(
            mean=_mean, sd=0, N=0, mR=mR, limits_type='Imr', round_to=spec.round_to
        )
        out['lcl'] = limits['lcl']
        out['ucl'] = limits['ucl']

    # Calculate beyond limits using vectorized operation
    out['beyond_limits'] = np.select(
        [out[spec.response_var] < out['lcl'], out[spec.response_var] > out['ucl']],
        [-1, 1],
        default=0,
    )

    # Prepare output columns using helper function
    base_cols = [spec.response_var, 'mean', 'lcl', 'ucl', 'beyond_limits']
    out = _prepare_output_columns(out, spec, base_cols)

    # Package results using helper function
    return _package_grouped_results(out, spec, ['mean', 'lcl', 'ucl'])


def calculate_statistics_R(
    df: pd.DataFrame, analysis_specification: AnalysisSpecification
) -> pd.DataFrame:
    spec = analysis_specification
    print('\nIn calculate statistics R...')
    print(f'\nDataframe has columns: {df.columns.to_list()}')

    # Calculate moving ranges using helper function
    out = _calculate_moving_ranges(df, spec)

    if spec.has_grouping:
        grouped = out.groupby(spec.rsg_var_name, as_index=False).agg(mR=pd.NamedAgg('mr', 'mean'))

        limits = grouped.apply(
            lambda row: calculate_limits(
                mean=0, sd=0, N=0, mR=row.mR, limits_type='R', round_to=spec.round_to
            ),
            axis=1,
        )

        grouped = pd.merge(grouped, limits, left_index=True, right_index=True)
        out = pd.merge(out, grouped, how='left', on=spec.rsg_var_name)
    else:
        out['mR'] = out['mr'].mean()
        mR = out['mR'].max()
        limits = calculate_limits(mean=0, sd=0, N=0, mR=mR, limits_type='R', round_to=spec.round_to)
        out['lcl'] = limits['lcl']
        out['ucl'] = limits['ucl']

    # Drop NAs
    out = out.dropna()

    # Calculate beyond limits using vectorized operation
    out['beyond_limits'] = np.select(
        [out['mr'] < out['lcl'], out['mr'] > out['ucl']], [-1, 1], default=0
    )

    # Prepare output columns using helper function
    base_cols = ['mr', 'mR', 'lcl', 'ucl', 'beyond_limits']

    # Prepare output columns using helper function
    out = _prepare_output_columns(out, spec, base_cols)

    # Package results using helper function
    return _package_grouped_results(out, spec, ['mR', 'lcl', 'ucl'])


def calculate_statistics_S(
    df: pd.DataFrame, analysis_specification: AnalysisSpecification
) -> pd.DataFrame:
    # TODO: update to use analysis spec to determine cals to do for S and xbar.
    spec = analysis_specification

    out = df.copy()

    out = out.groupby(spec.rsg_var_name, as_index=False).agg(
        s=pd.NamedAgg(column=spec.response_var, aggfunc='std'),
        n=pd.NamedAgg(column=spec.rsg_var_name, aggfunc='count'),
    )

    # remove RSGs with a single observation
    mask = out['n'].eq(1)
    out = out[~mask]

    out['S'] = out['s'].mean()
    out['groups'] = out['n'].count()
    out['N'] = out['n'].max()

    # if subgroup sizes are equal use N (limits will be same for all groups)
    n_max = out['n'].max()  # get max value for group size

    n_to_use = 'N' if (out['n'].eq(n_max).all()) else 'n'
    # Add limits columns
    out[['lcl', 'ucl']] = out.apply(
        lambda row: calculate_limits(
            mean=0, sd=row['S'], N=row[n_to_use], limits_type='S', round_to=spec.round_to
        ),
        axis=1,
    )

    out['beyond_limits'] = out.apply(
        lambda row: detect_beyond_limits(x=row['S'], ucl=row['ucl'], lcl=row['lcl']), axis=1
    )

    cols_to_keep = ['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']

    out = out[cols_to_keep]
    out = out.round(spec.round_to)
    return out


def calculate_statistics_XbarS(
    df: pd.DataFrame, analysis_specification: AnalysisSpecification
) -> pd.DataFrame:
    # TODO: update to use analysis spec to determine cals to do for S and xbar.
    spec = analysis_specification
    result = {}  # dictionary containing both Xbar and S analysis results
    statistics = {}  # dictionary to collect statistics about both Xbar and S
    out = df.copy()

    print('\nIn calculate statistics XbarS...')
    print(f'\nDataframe has columns: {out.columns.to_list()}')
    print(f'\n{out.head(10)}')
    print(f'\nn.max={out["n"].max()}')

    out = out.groupby(spec.rsg_var_name, as_index=False).agg(
        s=pd.NamedAgg(column=spec.response_var, aggfunc='std'),
        mean=pd.NamedAgg(column=spec.response_var, aggfunc='mean'),
        n=pd.NamedAgg(column='n', aggfunc='max'),
    )

    # 294 Handle case where no subgroups have >1 observation
    if out.shape[0] == 0:
        raise ValueError('All subgroups have 1 or less observations!')

    _Xbar = out['mean'].mean()
    out['Xbar'] = _Xbar  # out["mean"].mean()
    _S = out['s'].mean()
    out['S'] = _S  # out["s"].mean()
    _N = out['n'].max()
    out['N'] = _N  # out['n'].max()

    # if subgroup sizes are equal use N (limits will be same for all groups)
    # if they are different use n (limits will vary by group)
    n_max = _N  # out['n'].max()  # get max value for group size

    n_to_use = 'N' if out['n'].eq(n_max).all() else 'n'
    print(f'Analysis is using: {n_to_use} for calculations!\nScenario: {1 if n_to_use=="N" else 2}')

    # CALCULATE XBAR
    xbar = out.copy()

    xbar[['lcl', 'ucl']] = xbar.apply(
        lambda row: calculate_limits(
            mean=row['Xbar'],
            sd=row['S'],
            N=row[n_to_use],
            limits_type='Xbar',
            round_to=spec.round_to,
        ),
        axis=1,
    )

    xbar['beyond_limits'] = xbar.apply(
        lambda row: detect_beyond_limits(x=row['mean'], ucl=row['ucl'], lcl=row['lcl']), axis=1
    )

    # Round final output to specified decimal places
    xbar = xbar.round(spec.round_to)

    # TODO: Ugh! Refactor this mess...
    statistics['Mean'] = round(_Xbar, spec.round_to)
    if n_to_use == 'N':  # all RSGs are same size
        statistics['N'] = _N
        statistics['ucl'] = xbar['ucl'].max()
        statistics['lcl'] = xbar['lcl'].max()
    else:
        variable_stats = 'Varies'
        statistics['N'] = variable_stats
        statistics['lcl'] = variable_stats
        statistics['ucl'] = variable_stats

    # TODO: replace with spec
    cols_to_keep = ['rsg', 'mean', 'Xbar', 'lcl', 'ucl', 'beyond_limits']

    xbar = xbar[cols_to_keep]

    result['Xbar'] = {'data': xbar, 'statistics': statistics}
    # CALCULATE S - To make DRY Consider refactoring S function to take grouped dataset
    statistics = {}
    statistics['S'] = round(_S, spec.round_to)

    sbar = out.copy()
    sbar[['lcl', 'ucl']] = sbar.apply(
        lambda row: calculate_limits(
            mean=0, sd=row['S'], N=row[n_to_use], limits_type='S', round_to=spec.round_to
        ),
        axis=1,
    )

    sbar['beyond_limits'] = sbar.apply(
        lambda row: detect_beyond_limits(x=row['S'], ucl=row['ucl'], lcl=row['lcl']), axis=1
    )

    sbar = sbar.round(spec.round_to)

    if n_to_use == 'N':  # all RSGs are same size
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
        raise ValueError(f'The group_var: {grouping_var} is not in the data set!')

    out = {}

    grouped = df.groupby(grouping_var)

    # package_results
    for g in grouped.groups:
        criteria = df[grouping_var].eq(g)

        out[g] = df[criteria]

    return out


def gather_analysis_statistics(
    df: pd.DataFrame, statistics_to_collect: list, grouping_var: str = None
) -> dict:
    """
    Function gather_analysis_summary_statistics returns a dictionary of statistics
    (contained in stats_to_package) for each analytic result passed.

    :param pandas.Dataframe df: a grouped dataframe of analysis results, i.e., output from R or Imr
    :param list stats_to_package: list of variables/columns to summarize
        (dataframes currently contain columns for mean, moving range, and N)

    This function will take the max for each value specified in stats to package
    and put in dictionary with key equal to the value of list item,
    i.e., "mean" will be returned in a dictionary {statistics:{group_name: "abc", mean:1.0, etc...}}

    :return: dictionary of dataframes with grouping_var values as keys

    :rtype: dict

    :raises ValueError: if variables specified in list are in input dataframe
    """
    print('In call gather_statistics...')
    stats = {}

    out = df.copy()

    out_cols = df.columns.to_list()

    is_valid = all(cols in out_cols for cols in statistics_to_collect)

    if is_valid:
        if grouping_var is not None:
            statistics_to_collect.append('n')
            N = out.groupby(grouping_var, as_index=False).size()
            N.reset_index()
            N.rename(columns={'size': 'n'}, inplace=True)

            summarized = df.groupby([grouping_var]).max()
            summarized = pd.merge(N, summarized, how='left', on=grouping_var)

            for _index, row in summarized.iterrows():
                stats[row[grouping_var]] = row[statistics_to_collect].to_dict()

        else:
            n = len(out)

            summarized = out[statistics_to_collect].max().to_dict()
            summarized['n'] = n
            stats['all'] = summarized

    else:
        raise ValueError(f'Statistics: {statistics_to_collect} are not in {df.columns.to_list()}')

    return stats


def package_analysis(analysis_output: dict, summary_statistics_output: dict):
    """
    Function package_analysis combines the results of an analysis with the collected
    summary statistics from the analysis, i.e., combines two dictionaries into one
    with the rational subgroup name as the key.

    :param pandas.Dataframe analysis_output: dictionary of dataframes with a key matching
        the name of the rational subgroup name for grouped individuals analyses, R or Imr
    :param list summary_statistics_output: dictionary of collected statistics for each
        grouped individuals analysis. key is expected to be the name of the rational subgroup.

    :return: dictionary of dataframes with grouping_var values as keys

    :rtype: dict

    :raises ValueError: if variables specified in list are in input dataframe
    """
    print('In call package_analysis...')
    out = {}

    output_keys = analysis_output.keys()
    stats_keys = summary_statistics_output.keys()

    is_valid = all(keys in output_keys for keys in stats_keys)

    if is_valid:
        for key in analysis_output:
            out[key] = {'data': analysis_output[key], 'statistics': summary_statistics_output[key]}
    else:
        raise ValueError(
            f'Call: package_analysis: The rational subgroups do not match for '
            f'statistics being collected {list(stats_keys)}, data: {list(output_keys)}'
        )

    return out
