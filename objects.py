import math

import numpy as np
import pandas as pd
import scipy.special
from pandas.api.types import is_numeric_dtype

RSG_VARIABLE_NAME = "rsg"  # Need to refactor tests and eliminate these constants
TIME_VARIABLE_NAME = "time"
RESPONSE_VARIABLE_NAME = "response"

# ============================================================================
# Statistical Control Chart Constants
# ============================================================================
# These constants are derived from statistical process control theory.
# References: Wheeler, D. J. (1995). Advanced Topics in Statistical Process Control

# Control limit multiplier (3-sigma limits are standard in SPC)
SIGMA_MULTIPLIER = 3

# E2 constant for IMR charts (n=2, moving range of 2 consecutive observations)
# Used for calculating control limits on individual values
# E2 = d2 / d3 for n=2, where d2 = 1.128 and d3 = 0.8525
IMR_LIMIT_MULTIPLIER = 2.66  # E2 constant for IMR charts

# D4 constant for R charts (n=2, range of 2 consecutive observations)
# Used for upper control limit on moving range
# D4 = 1 + 3(d3/d2) for n=2
R_UPPER_LIMIT_MULTIPLIER = 3.268  # D4 constant for R charts (n=2)


#    Wrapper for input dataframe
# Takes raw data frame and validates core variables to support analytic methodology


class GoogleChart:
    def __init__(self, title: str, columns: list, rows: list, chartType: str):
        self.title = title
        self.columns = columns
        self.rows = rows
        self.chartType = chartType


######################################################################################
######################Analysis Code and Supporting Functions##########################
######################################################################################


def add_column(df: pd.DataFrame, new_col_name: str, existing_column: str):
    # add a copy of an existing column for use with single variable groupings

    df_cols = df.columns.tolist()
    check = existing_column in df_cols

    if (check):

        kwargs = {new_col_name: df[existing_column]}

        out = df.assign(**kwargs)
        return (out)
    else:
        raise ValueError(str(existing_column) + ": is not in the data set!")


def add_composite_column(df: pd.DataFrame, cols_to_combine: list, col_name: str, col_delim: str = '_') -> pd.DataFrame:
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


def add_grouping_variable_column(df: pd.DataFrame, cols_to_combine: list, col_name: str,
                                 col_delim: str = '_') -> pd.DataFrame:
    out = df.copy()

    if len(cols_to_combine) > 1:
        out = add_composite_column(df=out, cols_to_combine=cols_to_combine, col_name=col_name, col_delim=col_delim)
    else:
        out = add_column(df=df, new_col_name=col_name, existing_column=cols_to_combine[0])

    return (out)


def calculate_limits(limits_type: str, mean: float = None, sd: float = None, N: int = None, mR: float = None,
                     round_to: int = 3) -> dict:
    out = {}
    lcl = 0.0
    ucl = 0.0

    if (limits_type == "Xbar"):
        if (None in [sd, mean, N]):
            raise ValueError(f'The limits calculation for {limits_type}: requires (mean, sd, and N')
        # Wd = S / c4n from
        Wd = sd / c4(N)
        # Studentized Control Limits for sub-group means.
        # LCLx = X̿ – (3 ⋅ Wd) / √ n
        lcl = mean + (-1 * ((SIGMA_MULTIPLIER * Wd) / math.sqrt(N)))
        ucl = mean + ((SIGMA_MULTIPLIER * Wd) / math.sqrt(N))

        out = {'lcl': lcl, 'ucl': ucl}

    elif (limits_type == "S"):
        if (None in [sd, N]):
            raise ValueError(f'The limits calculation for {limits_type}: requires (sd, and N)')
        # lcl - use b3 - S*b3(N)
        lcl = sd * b3(N)
        # ucl -use b4
        ucl = sd * b4(N)

        out = {'lcl': lcl, 'ucl': ucl}

    elif (limits_type == "Imr"):
        if (None in [mean, mR]):
            raise ValueError(f'The limits calculation for {limits_type}: requires (mean, and mR)')

        lcl = mean + (-1.0 * (IMR_LIMIT_MULTIPLIER * mR))
        ucl = mean + (IMR_LIMIT_MULTIPLIER * mR)


    elif (limits_type == "R"):
        if (None in [mR]):
            raise ValueError(f'The limits calculation for {limits_type}: requires (mR)')
        lcl = 0
        ucl = mR * R_UPPER_LIMIT_MULTIPLIER


    else:
        raise ValueError(f'The limits type: {limits_type}: is not supported- provide (Xbar, S, IMR, or R')

    out = {'lcl': lcl, 'ucl': ucl}

    return (pd.Series(out, index=['lcl', 'ucl']))


def validate_columns(df: pd.DataFrame, rsg_vars: list, response_var: str, time_var: str = None,
                     time_unit: str = None) -> pd.DataFrame:
    # TODO: Centralize constants
    number_column_types = ['int64', 'float64', 'int32', 'Int64', 'Float64', 'Int32']
    # string_column_types = ['string', 'object', 'category', 'String', 'Object']
    date_column_types = ['int64', 'int32', 'Int64', 'Int32', 'datetime64[ns]']

    df_cols = df.columns.tolist()

    # validate columns exist in input df
    if not all(item in df_cols for item in rsg_vars):
        raise ValueError(f'One or more of the rsg_vars: {rsg_vars}: are not in the data set!')

        # Validate Time Variable - if provided
    # if time_var is not None:

    #     if not time_var in df_cols:
    #         raise ValueError(f'The time variable: {time_var}: is not in the data set!')

    #     time_type = df[time_var].dtype

    #     if not time_type in date_column_types:
    #         raise ValueError(f'The time variable: {time_var}: is not of type numeric, object or datetime!')

    # Validate response variable
    if not response_var in df_cols: raise ValueError(
        f'The response variable: {response_var}: is not in the data set!')

    response_type = df[response_var].dtype

    if not is_numeric_dtype(response_type): raise ValueError(
        f'The response variable: {response_var}: is not type numeric!')

    return df


def add_rsg_column(df: pd.DataFrame, rsg_vars: list) -> pd.DataFrame:
    df = add_composite_column(df=df, cols_to_combine=rsg_vars, col_name=RSG_VARIABLE_NAME)

    return df


def c4(n: int) -> float: #xbar chart
    # Calculate bias constant for xbar and S charts
    out = np.sqrt(2 / (n - 1)) * (np.exp(scipy.special.loggamma(n / 2) - scipy.special.loggamma((n - 1) / 2)))
    return (out)


def b3(n) -> float: #S chart
    out = 1 - (SIGMA_MULTIPLIER / c4(n) * math.sqrt(1 - math.pow(c4(n), 2)))

    return (0 if out < 0 else out)


def b4(n) -> float: #S chart
    out = 1 + (SIGMA_MULTIPLIER / c4(n) * math.sqrt(1 - math.pow(c4(n), 2)))

    return (out)


def detect_beyond_limits(x, lcl, ucl) -> int:
    result = lambda x, lcl, ucl: -1 if x < lcl else (1 if x > ucl else 0)
    return (result(x, lcl, ucl))
