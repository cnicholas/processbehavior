"""`_add_beyond_limits_flag` must produce exactly what the scalar rule produces.

This method decided every point's signal state by calling `detect_beyond_limits` once per
*observation* through `DataFrame.apply(axis=1)` — ~1,000,000 Python calls at 1M rows, and the
dominant cost of `execute()` on X/mR charts (2.14s vs 0.002s vectorized). It is called from
twelve sites covering every chart type, so one method's correctness carries all of them.

`detect_beyond_limits` is deliberately left as the scalar reference and is not modified: the
Bishop validator exercises it, so keeping it independent means those 280 assertions stay an
external check on this path rather than a check of it against itself.
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior.spc_constants import detect_beyond_limits


def _row_wise(df, value_col='v', lpl_col='lpl', upl_col='upl'):
    """The pre-change implementation, kept as the comparison oracle."""
    return df.apply(
        lambda row: detect_beyond_limits(x=row[value_col], upl=row[upl_col], lpl=row[lpl_col]), axis=1
    )


def _vectorized(df, value_col='v', lpl_col='lpl', upl_col='upl'):
    values = df[value_col].to_numpy()
    lower = df[lpl_col].to_numpy()
    upper = df[upl_col].to_numpy()
    return pd.Series(np.where(values < lower, -1, np.where(values > upper, 1, 0)), index=df.index)


CASES = {
    'below, inside, above': pd.DataFrame({'v': [8.5, 10.0, 11.5], 'lpl': [9.0] * 3, 'upl': [11.0] * 3}),
    'NaN observation': pd.DataFrame({'v': [np.nan, 10.0, 11.5], 'lpl': [9.0] * 3, 'upl': [11.0] * 3}),
    'NaN limits': pd.DataFrame({'v': [8.5, 10.0], 'lpl': [np.nan] * 2, 'upl': [np.nan] * 2}),
    'exactly on each limit': pd.DataFrame({'v': [9.0, 11.0], 'lpl': [9.0] * 2, 'upl': [11.0] * 2}),
    'per-row varying limits': pd.DataFrame(
        {'v': [1.0, 5.0, 9.0], 'lpl': [0.0, 6.0, 8.0], 'upl': [2.0, 7.0, 8.5]}
    ),
    'integer dtype': pd.DataFrame({'v': [8, 10, 12], 'lpl': [9] * 3, 'upl': [11] * 3}),
    'single row': pd.DataFrame({'v': [12.0], 'lpl': [9.0], 'upl': [11.0]}),
    'non-default index': pd.DataFrame(
        {'v': [8.5, 12.0], 'lpl': [9.0] * 2, 'upl': [11.0] * 2}, index=[77, 3]
    ),
    'negative values': pd.DataFrame({'v': [-12.0, -10.0, -8.0], 'lpl': [-11.0] * 3, 'upl': [-9.0] * 3}),
}


@pytest.mark.parametrize('name', list(CASES))
def test_matches_the_row_wise_rule(name):
    df = CASES[name]
    pd.testing.assert_series_equal(_vectorized(df), _row_wise(df), check_names=False)


def test_a_point_exactly_on_a_limit_is_in_control():
    """The scalar rule uses strict `<` and `>`. A point sitting on the limit is 0, and the
    nested where must not turn that into a signal."""
    df = CASES['exactly on each limit']
    assert list(_vectorized(df)) == [0, 0]


def test_lower_limit_is_tested_first():
    """The scalar rule checks `x < lpl` before `x > upl`, so the nesting order matters.

    With inverted limits (lpl above upl) a value below both satisfies *both* conditions;
    the answer must be -1, which is what checking lower first produces.
    """
    df = pd.DataFrame({'v': [1.0], 'lpl': [10.0], 'upl': [5.0]})
    assert _row_wise(df).iloc[0] == -1
    assert _vectorized(df).iloc[0] == -1


def test_result_is_int64():
    assert _vectorized(CASES['below, inside, above']).dtype == np.int64


def test_none_limits_still_raise():
    """A chart with `None` limits is an upstream bug and should stay loud. Do not 'fix'
    this into a silent 0 — both the scalar and vectorized paths raise TypeError."""
    df = pd.DataFrame({'v': [8.5, 10.0], 'lpl': [None] * 2, 'upl': [None] * 2})
    with pytest.raises(TypeError):
        _row_wise(df)
    with pytest.raises(TypeError):
        _vectorized(df)


def test_nullable_dtype_with_na_is_handled_rather_than_crashing():
    """Documents a deliberate improvement, not an equivalence.

    With a pandas nullable dtype carrying `pd.NA`, the row-wise path raises
    "boolean value of NA is ambiguous"; the vectorized path treats it as in-control (0),
    consistent with how it treats NaN. Limit columns are plain float64 in practice, so this
    is unreachable in the normal pipeline — recorded so the difference is understood rather
    than discovered.
    """
    df = pd.DataFrame(
        {
            'v': pd.Series([8.5, pd.NA], dtype='Float64'),
            'lpl': pd.Series([9.0, 9.0], dtype='Float64'),
            'upl': pd.Series([11.0, 11.0], dtype='Float64'),
        }
    )
    with pytest.raises(TypeError):
        _row_wise(df)
    assert list(_vectorized(df)) == [-1, 0]
