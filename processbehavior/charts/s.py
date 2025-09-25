from __future__ import annotations

import pandas as pd

from ..limits import calculate_limits, detect_beyond_limits
from ..spec import AnalysisSpecification


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


def s_chart(df: pd.DataFrame, response: str, by: list[str], time_col: str | None = None) -> dict:
    raise NotImplementedError  # implement S chart with rational subgroup ordering
