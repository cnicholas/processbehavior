from __future__ import annotations

import pandas as pd

from ..limits import calculate_limits, detect_beyond_limits
from ..spec import AnalysisSpecification


def calculate_statistics_Xbar(
    df: pd.DataFrame, analysis_specification: AnalysisSpecification
) -> dict:
    """Pure Xbar chart calculation.

    Args:
        df: Prepared dataframe with grouped data
        analysis_specification: Analysis specification

    Returns:
        Dict with 'data' and 'statistics' for Xbar chart
    """
    spec = analysis_specification
    out = df.copy()

    print('\nIn calculate statistics Xbar...')
    print(f'\nDataframe has columns: {out.columns.to_list()}')
    print(f'\n{out.head(10)}')
    print(f'\nn.max={out["n"].max()}')

    # Group and calculate statistics
    out = out.groupby(spec.rsg_var_name, as_index=False).agg(
        s=pd.NamedAgg(column=spec.response_var, aggfunc='std'),
        mean=pd.NamedAgg(column=spec.response_var, aggfunc='mean'),
        n=pd.NamedAgg(column='n', aggfunc='max'),
    )

    # Handle case where no subgroups have >1 observation
    if out.shape[0] == 0:
        raise ValueError('All subgroups have 1 or less observations!')

    _Xbar = out['mean'].mean()
    out['Xbar'] = _Xbar
    _S = out['s'].mean()
    out['S'] = _S  # Needed for Xbar limits calculation
    _N = out['n'].max()
    out['N'] = _N

    # Determine whether to use N or n for calculations
    n_max = _N
    n_to_use = 'N' if out['n'].eq(n_max).all() else 'n'
    print(f'Analysis is using: {n_to_use} for calculations!\nScenario: {1 if n_to_use=="N" else 2}')

    # CALCULATE XBAR LIMITS
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

    # Calculate beyond limits
    xbar['beyond_limits'] = xbar.apply(
        lambda row: detect_beyond_limits(x=row['mean'], ucl=row['ucl'], lcl=row['lcl']), axis=1
    )

    # Round final output to specified decimal places
    xbar = xbar.round(spec.round_to)

    # Gather statistics
    statistics = {}
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

    # Prepare output columns
    cols_to_keep = ['rsg', 'mean', 'Xbar', 'lcl', 'ucl', 'beyond_limits']
    xbar = xbar[cols_to_keep]

    return {'data': xbar, 'statistics': statistics}


def xbar(df: pd.DataFrame, response: str, by: list[str], time_col: str | None = None) -> dict:
    raise NotImplementedError  # implement Xbar chart with rational subgroup ordering
