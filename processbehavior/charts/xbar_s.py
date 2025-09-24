from __future__ import annotations

import pandas as pd

from ..spec import AnalysisSpecification
from .s import calculate_statistics_S
from .xbar import calculate_statistics_Xbar


def calculate_statistics_XbarS(
    df: pd.DataFrame, analysis_specification: AnalysisSpecification
) -> dict:
    """Combined Xbar and S chart analysis using pure functions.

    Args:
        df: Prepared dataframe with grouped data
        analysis_specification: Analysis specification

    Returns:
        Dict containing both Xbar and S analysis results
    """
    print('\nIn calculate statistics XbarS (using composition)...')

    # Use pure functions to calculate each chart type
    xbar_result = calculate_statistics_Xbar(df, analysis_specification)
    s_result = calculate_statistics_S(df, analysis_specification)

    # Combine results in expected format
    return {'Xbar': xbar_result, 'Sbar': s_result}


def xbar_s(df: pd.DataFrame, response: str, subgroup_col: str | None = None) -> dict:
    raise NotImplementedError  # implement real Xbar–S here


def xbar(df: pd.DataFrame, response: str, by: list[str], time_col: str | None = None) -> dict:
    raise NotImplementedError  # implement Xbar chart with rational subgroup ordering
