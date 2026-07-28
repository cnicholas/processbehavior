"""`ProcessBehavior.__init__` cleaning — fast paths must be output-identical to slow ones.

Two optimisations live in `__init__`, and both are pure performance: they must not change a
single dtype or value, because client code reads post-init dtypes to decide what a column can
be used for (the Streamlit app builds its response/factor pickers from exactly that).

1. Phase 1 skips the garbage-token scan on columns that cannot hold a string.
2. Phase 2 cleans the *distinct* values and maps back, instead of cleaning every row.

The direct implementation (`_try_clean_numeric_strings`) is unchanged and serves as the
reference oracle for (2): the shortcut is correct iff it agrees with the direct path on every
column shape. The traps that make this non-obvious are called out in their own tests below.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.formulation_spec import FormulationSpec
from processbehavior.process_behavior import (
    _clean_numeric_strings_via_uniques,
    _is_text_like,
    _try_clean_numeric_strings,
)
from processbehavior.sds_detector import SDSRegistry

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
N = 10_000  # large enough that the low-cardinality shortcut actually engages


def _series(values, dtype=object):
    return pd.Series(values, dtype=dtype)


# ---------------------------------------------------------------------------
# The shortcut agrees with the direct path — the whole justification
# ---------------------------------------------------------------------------

EQUIVALENCE_CASES = {
    'labels only': _series([f'F_{i % 5}' for i in range(N)]),
    'clean numeric strings': _series([f'{i % 50}.5' for i in range(N)]),
    'currency': _series(['$1,234.50', '$2,000.00'] * (N // 2)),
    'accounting negatives': _series(['(1,234.56)', '2000'] * (N // 2)),
    'percentages': _series(['12%', '45%'] * (N // 2)),
    'unicode currency': _series(['€1,5', '(2,0)'] * (N // 2)),
    'whitespace': _series([' 1 ', ' 2 '] * (N // 2)),
    'with NAs': _series(['1.5', None, '2.5', np.nan] * (N // 4)),
    'NA-heavy': _series(['1.5'] + [None] * (N - 1)),
    'mixed types': _series([1, '2', 3.0, 'x'] * (N // 4)),
    'all NA': _series([None] * 100),
    'high cardinality labels': _series([f'v_{i}' for i in range(N)]),
    'high cardinality numeric': _series([f'{i}.5' for i in range(N)]),
    'exactly at threshold': _series(['1'] * 8000 + [f'x{i}' for i in range(2000)]),
    'just under threshold': _series(['1'] * 7999 + [f'x{i}' for i in range(2001)]),
    'category of strings': pd.Series(['1', '2'] * (N // 2)).astype('category'),
    'category of garbage': pd.Series(['*', '1'] * (N // 2)).astype('category'),
    'string dtype': pd.Series(['1', '2'] * (N // 2), dtype='string'),
    'float': pd.Series([1.0, 2.0] * (N // 2)),
    'int': pd.Series([1, 2] * (N // 2)),
    'bool': pd.Series([True, False] * (N // 2)),
    'datetime': pd.to_datetime(pd.Series(['2024-01-01', '2024-01-02'] * (N // 2))),
}


@pytest.mark.parametrize('name', list(EQUIVALENCE_CASES))
def test_unique_shortcut_matches_the_direct_path(name):
    """Same values AND same dtype. Dtype matters as much as values — a client picking
    "numeric columns" sees dtype, not content."""
    series = EQUIVALENCE_CASES[name]
    direct = _try_clean_numeric_strings(series)
    shortcut = _clean_numeric_strings_via_uniques(series)

    if direct is None:
        assert shortcut is None, f'{name}: direct declined but the shortcut converted'
        return
    assert shortcut is not None, f'{name}: direct converted but the shortcut declined'
    pd.testing.assert_series_equal(direct, shortcut)


def test_frequency_weighted_threshold_is_preserved():
    """The trap that makes the naive shortcut wrong.

    The 80% acceptance test is over *non-NA values*, so it is frequency-weighted. Scoring
    it over distinct values instead flips the decision: 9,900 copies of '1' plus 100
    distinct labels is 99% convertible by row and 1% by distinct value. The shortcut must
    transform the uniques but score the full column.
    """
    series = _series(['1'] * 9900 + [f'lab_{i}' for i in range(100)])

    by_row = pd.to_numeric(series, errors='coerce').notna().sum() / series.notna().sum()
    by_distinct = pd.to_numeric(pd.Series(series.unique()), errors='coerce').notna().sum() / series.nunique()
    assert by_row > 0.8 > by_distinct, 'precondition: the two scorings must disagree'

    assert _try_clean_numeric_strings(series) is not None
    assert _clean_numeric_strings_via_uniques(series) is not None, 'shortcut scored over uniques'


# ---------------------------------------------------------------------------
# The dtype guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'series,expected',
    [
        (pd.Series(['*', '1']), True),
        (pd.Series(['*', 'a']).astype('category'), True),
        (pd.Series(['*', 'a'], dtype='string'), True),
        (pd.Series([1.0, np.nan, np.inf]), False),
        (pd.Series([1, 2], dtype='int64'), False),
        (pd.Series([1, 2], dtype='Int64'), False),
        (pd.to_datetime(pd.Series(['2024-01-01'])), False),
        (pd.Series([True, False]), False),
    ],
)
def test_is_text_like(series, expected):
    assert _is_text_like(series) is expected


def test_categorical_of_strings_is_still_cleaned():
    """The trap in the dtype guard.

    A `category` column holds strings and can contain a garbage token, but it is not
    `object` — a guard written as "not numeric" would skip it and silently stop cleaning
    it. This is why the guard enumerates the dtypes to *include*.
    """
    df = pd.DataFrame({'f': pd.Series(['1', '*', '2'] * 100).astype('category')})
    cleaned = ProcessBehavior(df).data
    assert cleaned['f'].isna().sum() == 100, 'the * values should have become NA'


def test_numeric_columns_are_untouched_by_the_scan():
    values = [1.0, 2.5, np.nan, np.inf, -np.inf]
    df = pd.DataFrame({'x': values * 100})
    cleaned = ProcessBehavior(df).data
    pd.testing.assert_series_equal(cleaned['x'], df['x'])


# ---------------------------------------------------------------------------
# Design-state detection must be unmoved — the reason this change is safe at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'response,expected_ods,expected_ads',
    [('PM SDS 1', 1, 1), ('PM SDS 2', 2, 2), ('PM SDS 3', 3, 3),
     ('PM SDS 4', 4, 1), ('PM SDS 5', 5, 2), ('PM SDS 6', 6, 3)],
)
def test_design_states_unmoved(response, expected_ods, expected_ads):
    """Read WITHOUT na_values so the '*' markers arrive as raw strings — the garbage the
    cleaning exists to handle. Both lineage states must match what they were before."""
    raw = pd.read_csv(REFERENCE_CSV)
    study = ProcessBehavior(raw).formulate(
        response=response, factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME',
    )
    assert study.observed_design_state.sds == expected_ods
    assert study.analytical_design_state.sds == expected_ads


def test_detection_does_not_depend_on_init_cleaning():
    """Pins the property the whole change rests on.

    `_build_structure_view` normalises missing tokens and canonicalises the kt columns
    itself, on a minimal projection — so N_kt is the same whether or not __init__ cleaned
    first. If a future change makes detection rely on pre-cleaned input, this fails and
    the "cleaning is safe to narrow" argument no longer holds.
    """
    raw = pd.read_csv(REFERENCE_CSV)
    cleaned = ProcessBehavior(raw.copy()).data
    registry = SDSRegistry()

    for response in ('PM SDS 4', 'PM SDS 5', 'PM SDS 6'):
        spec = FormulationSpec(
            response_var=response, rsg_vars=('FACTOR 1', 'FACTOR 2'), time_var='PRODUCTION TIME',
        )
        view_raw, kt = registry._build_structure_view(raw, spec, response)
        view_clean, _ = registry._build_structure_view(cleaned, spec, response)
        n_raw = view_raw.groupby(kt, observed=True)[response].count()
        n_clean = view_clean.groupby(kt, observed=True)[response].count()
        pd.testing.assert_series_equal(n_raw, n_clean, obj=f'N_kt for {response}')


def test_post_init_dtypes_drive_client_column_pickers():
    """A `['235.5', '*', '237.2']` column must be numeric after init.

    The app's response picker is `[c for c in df.columns if is_numeric_dtype(df[c])]`, so a
    column that stays object here is one an analyst cannot select — the exact column the
    cleaning exists to rescue.
    """
    df = pd.DataFrame({'measurement': ['235.5', '*', '237.2'] * 100, 'label': ['a', 'b', 'c'] * 100})
    cleaned = ProcessBehavior(df).data
    assert pd.api.types.is_numeric_dtype(cleaned['measurement'])
    assert not pd.api.types.is_numeric_dtype(cleaned['label'])
