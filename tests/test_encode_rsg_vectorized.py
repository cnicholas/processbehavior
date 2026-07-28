"""`encode_rsg_series` — vectorized RSG encoding, and the upcast bug it fixed.

`df[cols].apply(lambda row: encode_rsg(tuple(row)), axis=1)` materialises each row as a
Series, which **upcasts mixed dtypes**. With an int factor beside a float factor the int
encoded as `'1.0'`. Because plan expansion encodes from Python values (`study.py`), observed
rsg keys could then never match expected ones, and `DesignReport.missing_combos` /
`extra_combos` reported every combination as both missing and extra even when the plan
matched the data exactly.

Converting per column keeps each column on its own dtype, which is both ~31x faster and the
encoding plan expansion already uses. These tests pin the equivalence for the cases where the
two paths agree, and the corrected behaviour for the case where they never did.
"""

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.data_preparation import encode_rsg, encode_rsg_series
from processbehavior.exceptions import ValidationError


def _row_wise(df, cols, delimiter='_'):
    """The pre-change implementation, kept here as the comparison oracle."""
    return df[cols].apply(lambda row: encode_rsg(tuple(row), delimiter=delimiter), axis=1)


# Cases where row-wise and column-wise agree — the equivalence the speedup rests on.
AGREEING_CASES = {
    'int + int': pd.DataFrame({'a': [1, 2, 3], 'b': [3, 4, 5]}),
    'int + object': pd.DataFrame({'a': [1, 2, 3], 'b': ['X', 'Y', 'Z']}),
    'float + object': pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': ['X', 'Y', 'Z']}),
    'object + object': pd.DataFrame({'a': ['P', 'Q', 'R'], 'b': ['X', 'Y', 'Z']}),
    'float + float': pd.DataFrame({'a': [1.5, 2.5, 3.5], 'b': [4.5, 5.5, 6.5]}),
    'bool + int': pd.DataFrame({'a': [True, False, True], 'b': [1, 2, 3]}),
    'category + int': pd.DataFrame({'a': pd.Series(['P', 'Q', 'R']).astype('category'), 'b': [1, 2, 3]}),
    'nullable Int64': pd.DataFrame({'a': pd.Series([1, 2, 3], dtype='Int64'), 'b': [4, 5, 6]}),
    'three factors': pd.DataFrame({'a': [1, 2], 'b': ['X', 'Y'], 'c': ['P', 'Q']}),
    'single factor': pd.DataFrame({'a': ['X', 'Y', 'Z']}),
    'datetime + int': pd.DataFrame({'a': pd.to_datetime(['2024-01-01', '2024-01-02']), 'b': [1, 2]}),
    'datetime with time': pd.DataFrame(
        {'a': pd.to_datetime(['2024-01-01 13:30', '2024-01-02 09:00']), 'b': [1, 2]}
    ),
}


@pytest.mark.parametrize('name', list(AGREEING_CASES))
def test_matches_the_row_wise_encoding(name):
    df = AGREEING_CASES[name]
    cols = list(df.columns)
    pd.testing.assert_series_equal(encode_rsg_series(df, cols), _row_wise(df, cols), check_names=False)


@pytest.mark.parametrize('delimiter', ['_', '-', '||'])
def test_delimiter_is_honoured(delimiter):
    df = pd.DataFrame({'a': [1, 2], 'b': ['X', 'Y']})
    pd.testing.assert_series_equal(
        encode_rsg_series(df, ['a', 'b'], delimiter=delimiter),
        _row_wise(df, ['a', 'b'], delimiter=delimiter),
        check_names=False,
    )


def test_index_is_preserved():
    df = pd.DataFrame({'a': [1, 2], 'b': ['X', 'Y']}, index=[100, 200])
    assert list(encode_rsg_series(df, ['a', 'b']).index) == [100, 200]


def test_no_columns_raises():
    with pytest.raises(ValidationError, match='at least one column'):
        encode_rsg_series(pd.DataFrame({'a': [1]}), [])


# ---------------------------------------------------------------------------
# The bug: row-wise upcasting made observed keys unable to match expected ones
# ---------------------------------------------------------------------------


def test_mixed_numeric_dtypes_no_longer_upcast():
    """int beside float encodes as '1', not '1.0'.

    This is the one case where the new path deliberately DIFFERS from the old one — the
    old answer was wrong, because `encode_rsg` on the plan side produces '1'.
    """
    df = pd.DataFrame({'f1': [1, 2], 'f2': [1.5, 2.5]})

    assert list(_row_wise(df, ['f1', 'f2'])) == ['1.0_1.5', '2.0_2.5'], 'precondition: old path upcast'
    assert list(encode_rsg_series(df, ['f1', 'f2'])) == ['1_1.5', '2_2.5']
    # …and the new answer is exactly what encoding the Python values gives.
    assert encode_rsg((1, 1.5)) == '1_1.5'


def test_observed_keys_match_plan_expansion_for_mixed_dtype_factors():
    """The user-visible consequence: a plan that exactly matches the data must report
    nothing missing and nothing extra."""
    n = 4
    df = pd.DataFrame(
        {
            'f1': np.repeat([1, 2], 2 * n),
            'f2': np.tile(np.repeat([1.5, 2.5], n), 2),
            't': np.tile([1, 2, 3, 4], 4)[: 4 * n],
            'y': np.random.default_rng(0).normal(100, 1, 4 * n),
        }
    )
    study = pb.ProcessBehavior(df).formulate(
        response='y', plan={'factors': {'f1': [1, 2], 'f2': [1.5, 2.5]}, 'T': 4, 'N': 2}, time='t',
    )

    assert sorted(study.dataset['rsg'].unique()) == ['1_1.5', '1_2.5', '2_1.5', '2_2.5']
    report = study.design()
    assert report.missing_combos == [], 'every planned combination is present in the data'
    assert report.extra_combos == [], 'no combination in the data is outside the plan'


def test_single_factor_studies_are_unaffected():
    """Regression guard: the common case must not shift.

    Single-factor and same-dtype studies encoded correctly before, and their rsg keys must
    be byte-identical after — otherwise this change would move strata for everyone.
    """
    df = pd.DataFrame(
        {
            'f': np.repeat(['A', 'B'], 8),
            't': np.tile([1, 2, 3, 4], 4),
            'y': np.random.default_rng(1).normal(10, 1, 16),
        }
    )
    study = pb.ProcessBehavior(df).formulate(response='y', factors=['f'], time='t')
    assert sorted(study.dataset['rsg'].unique()) == ['A', 'B']
