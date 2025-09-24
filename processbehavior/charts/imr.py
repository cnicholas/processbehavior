from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis_dataset import (
    _calculate_moving_ranges,
    _package_grouped_results,
    _prepare_output_columns,
)
from ..limits import calculate_limits
from ..spec import AnalysisSpecification


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


def imr(
    df: pd.DataFrame, response: str, by: list[str] | None = None, time_col: str | None = None
) -> dict:
    raise NotImplementedError  # implement I & MR with rational subgroup ordering
