"""
Data preparation utilities for SPC analysis.

This module contains functions for preparing and transforming data
for process behavior chart analysis.
"""

import pandas as pd

from .constants import DATE_COLUMN_TYPES, NUMBER_COLUMN_TYPES


def add_column(df: pd.DataFrame, new_col_name: str, existing_column: str) -> pd.DataFrame:
    """Add a copy of an existing column for use with single variable groupings.

    Args:
        df: Input dataframe
        new_col_name: Name for the new column
        existing_column: Name of existing column to copy

    Returns:
        DataFrame with new column added

    Raises:
        ValueError: If existing_column is not in the dataset
    """
    df_cols = df.columns.tolist()

    if existing_column not in df_cols:
        raise ValueError(f'{existing_column}: is not in the data set!')

    kwargs = {new_col_name: df[existing_column]}
    return df.assign(**kwargs)


def add_composite_column(
    df: pd.DataFrame, cols_to_combine: list, col_name: str, col_delim: str = '_'
) -> pd.DataFrame:
    """Combine multiple columns into a single composite column.

    Args:
        df: Input dataframe
        cols_to_combine: List of column names to combine
        col_name: Name for the new composite column
        col_delim: Delimiter to use when combining columns (default: '_')

    Returns:
        DataFrame with composite column added

    Raises:
        ValueError: If any column in cols_to_combine is not in the dataset
    """
    df_cols = df.columns.tolist()

    if not all(item in df_cols for item in cols_to_combine):
        raise ValueError(f'One or more of: {cols_to_combine}: are not in the data set!')

    if len(cols_to_combine) > 1:
        len_col_delim = len(col_delim)
        # Dynamically build columns from list of column names
        combined = (df[cols_to_combine].astype(str) + col_delim).cumsum(1).iloc[:, -1].values
        # Remove trailing delimiter
        combined = [x[:-len_col_delim] for x in combined]

        kwargs = {col_name: combined}
        return df.assign(**kwargs)

    return df


def add_grouping_variable_column(
    df: pd.DataFrame, cols_to_combine: list, col_name: str, col_delim: str = '_'
) -> pd.DataFrame:
    """Add a grouping variable column for rational subgrouping.

    This is the main function for creating grouping columns. It handles both
    single-column and multi-column grouping scenarios.

    Args:
        df: Input dataframe
        cols_to_combine: List of column names to use for grouping
        col_name: Name for the new grouping column
        col_delim: Delimiter to use when combining multiple columns (default: '_')

    Returns:
        DataFrame with grouping column added
    """
    out = df.copy()

    if len(cols_to_combine) > 1:
        out = add_composite_column(
            df=out, cols_to_combine=cols_to_combine, col_name=col_name, col_delim=col_delim
        )
    else:
        out = add_column(df=df, new_col_name=col_name, existing_column=cols_to_combine[0])

    return out


def validate_columns(df: pd.DataFrame, analysis_specification) -> pd.DataFrame:
    """Validate that required columns exist in the dataframe and have correct types."""
    spec = analysis_specification
    number_column_types = NUMBER_COLUMN_TYPES
    date_column_types = DATE_COLUMN_TYPES

    df_cols = df.columns.tolist()

    if spec.has_grouping and not all(item in df_cols for item in spec.rsg_vars):
        raise ValueError(
            f'One or more of the rsg_vars: {spec.rsg_vars}: are not in the data set: {df_cols}!'
        )

    if spec.has_time:
        if spec.time_var not in df_cols:
            raise ValueError(f'The time variable: {spec.time_var}: is not in the data set!')

        time_type = df[spec.time_var].dtype

        if spec.time_unit is not None and time_type != 'datetime64[ns]':
            raise ValueError(
                f'The time variable: {spec.time_var}: must be a datetime '
                f'when a time unit is provided!'
            )

        elif time_type not in date_column_types:
            raise ValueError(
                f'The time variable: {spec.time_var}: must be of type numeric, object or datetime!'
            )

    # Validate response variable
    if spec.response_var not in df_cols:
        raise ValueError(f'The response variable: {spec.response_var}: is not in the data set!')

    response_type = df[spec.response_var].dtype

    if response_type not in number_column_types:
        raise ValueError(f'The response variable: {spec.response_var}: is not type numeric!')

    return df


def prepare_dataset(df: pd.DataFrame, analysis_specification) -> pd.DataFrame:
    """Prepare dataset according to analysis specification."""
    print('\nEntering call to prepare_data...')
    spec = analysis_specification
    out = df.copy()

    out = validate_columns(df=out, analysis_specification=spec)

    # 296 - Incorporate user specified delimiter into spec to delimit multiple grouping variables
    # when specified, default will be underscore - "_"
    # If specified add column for RSG - Rational Subgroup
    if spec.has_grouping:
        out = add_grouping_variable_column(
            df=out,
            col_name=spec.rsg_var_name,
            cols_to_combine=spec.rsg_vars,
            col_delim=spec.rsg_var_delim,
        )

        # Remove groups with one obs - solve for all grouped analyses
        grouped = out.groupby(spec.rsg_var_name).size()
        starting_count = grouped.count()
        print(f'\nStarting with {starting_count} groups')
        grouped = grouped[grouped > 1]

        if grouped.shape[0] == 0:
            raise ValueError('All subgroups have 1 or less observations!')

        grouped = grouped.reset_index()
        grouped = grouped.rename(columns={0: 'n'})

        print(f'\nPruning groups with 1 obs:\n{grouped}\n')
        ending_count = grouped.count()

        print(f'\nGroups remaining: {ending_count[0]}')
        print(f'\nRemoved {starting_count-ending_count[0]} group(s)')
        print('\n')

        out = pd.merge(out, grouped, how='inner', on=spec.rsg_var_name)
    else:
        print('No Groups...\n')

    # Add date part to drive aggregations
    if len(spec.time_grouping_units) > 0:
        time_var = spec.time_var
        year_col = spec.time_grouping_cols['Year']

        if len(spec.time_grouping_units) == 1:  # We are grouping by year
            out[year_col] = out[time_var].dt.year

        elif (
            len(spec.time_grouping_units) > 1 and spec.time_grouping_units[1] == 'Week'
        ):  # special case - need to use ISO
            out[year_col] = out[time_var].dt.isocalendar().year
            out[spec.time_grouping_cols['Week']] = out[time_var].dt.isocalendar().week

        else:
            out[spec.time_grouping_cols['Year']] = out[time_var].dt.year

            if spec.time_grouping_units[1] == 'Month':
                out[spec.time_grouping_cols['Month']] = out[time_var].dt.month

            elif spec.time_grouping_units[1] == 'Quarter':
                out[spec.time_grouping_cols['Quarter']] = out[time_var].dt.quarter

            elif spec.time_grouping_units[1] == 'DOY':
                out[spec.time_grouping_cols['DOY']] = out[time_var].dt.day_of_year

    # Perform sorting
    if spec.requires_sort:
        out = out.sort_values(spec.sort_cols)  # These are predetermined by the spec

    out = out[spec.data_prep_output_cols]  # These are predetermined by the spec

    out = out.dropna()

    return out
