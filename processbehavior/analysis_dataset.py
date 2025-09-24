from __future__ import annotations

import pandas as pd

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
