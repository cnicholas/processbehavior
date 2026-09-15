"""Rows with a missing factor value leave the study, for one factor or many, with a warning.

Found on a survey file: 85,101 rows, 7,980 with a blank SURVEY QUESTION. It formulated with
either factor alone and raised with both, because the two-factor path built the composite
subgroup label before the missing-value drop. Nothing in the app surfaced that for long
enough to read.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior import ProcessBehaviorWarning


@pytest.fixture
def survey():
    """Two hospitals x 10 questions x 12 months, 6 replicates, with 8% of question labels blank.

    Six replicates so that no cell falls below two after the blanks leave: the study stays
    at full replication and the test checks the drop, not a change of design state.
    """
    rng = np.random.default_rng(11)
    rows = []
    for month in range(1, 13):
        for hospital in ('A', 'B'):
            for q in ('LR', 'NC', 'NI', 'NP', 'PC', 'PI', 'PT', 'QF', 'SA', 'SD'):
                for _ in range(6):
                    rows.append(
                        {'MONTH': month, 'HOSPITAL': hospital, 'QUESTION': q, 'SCORE': rng.choice([0, 25, 50, 75, 100])}
                    )
    df = pd.DataFrame(rows)
    blank = rng.random(len(df)) < 0.08
    df.loc[blank, 'QUESTION'] = np.nan
    return df, int(blank.sum())


def _formulate(df, factors):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        st = pb.formulate(df, response='SCORE', factors=factors, time='MONTH')
    return st, [w for w in caught if issubclass(w.category, ProcessBehaviorWarning)]


def test_two_factors_formulate_with_missing_factor_values(survey):
    df, n_blank = survey
    st, pbw = _formulate(df, ['QUESTION', 'HOSPITAL'])
    assert st.analytical_design_state.sds == 1
    assert len(st._ads.analysis_dataset) == len(df) - n_blank
    assert st.design().K == 20  # 10 questions x 2 hospitals, no phantom 'nan' level


def test_one_factor_and_two_factors_drop_the_same_rows(survey):
    df, n_blank = survey
    one, _ = _formulate(df, ['QUESTION'])
    two, _ = _formulate(df, ['QUESTION', 'HOSPITAL'])
    assert len(one._ads.analysis_dataset) == len(two._ads.analysis_dataset) == len(df) - n_blank
    assert 'nan' not in set(one._ads.analysis_dataset['rsg'].astype(str))


def test_the_drop_is_reported_with_counts(survey):
    df, n_blank = survey
    _, pbw = _formulate(df, ['QUESTION', 'HOSPITAL'])
    texts = [str(w.message) for w in pbw]
    hit = [t for t in texts if 'no value in a factor column' in t]
    assert hit, texts
    assert f'Dropped {n_blank:,} of {len(df):,} rows' in hit[0]
    assert f'QUESTION ({n_blank:,} missing)' in hit[0]
    assert 'HOSPITAL' not in hit[0]  # only the columns that actually had gaps are named


def test_no_warning_when_factor_columns_are_complete(survey):
    df, _ = survey
    complete = df.dropna(subset=['QUESTION'])
    _, pbw = _formulate(complete, ['QUESTION', 'HOSPITAL'])
    assert not [w for w in pbw if 'factor column' in str(w.message)]


def test_missing_values_in_both_factors_are_named_per_column():
    df = pd.DataFrame(
        {
            'lane': ['A', 'A', None, 'B', 'B', 'B'],
            'head': [1, 1, 1, None, 2, 2],
            't': [1, 1, 1, 1, 1, 1],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        st = pb.formulate(df, response='y', factors=['lane', 'head'], time='t')
    msg = next(str(w.message) for w in caught if 'factor column' in str(w.message))
    assert 'Dropped 2 of 6 rows' in msg and 'lane (1 missing)' in msg and 'head (1 missing)' in msg
    assert len(st._ads.analysis_dataset) == 4
