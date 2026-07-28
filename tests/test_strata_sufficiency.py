"""`_split_strata_by_sufficiency` — counted in one pass, not one table scan per stratum.

This helper decides which strata are publishable (the focus()/strata contract requires every
published stratum to be focusable). It did so with a full-column comparison *inside a loop
over strata* — O(strata x rows) — plus an O(strata^2) list-membership test. At 200K rows and
632 strata that was 2.4s, 89% of a stratified execute; at 1M rows the whole call took 26.5s.

The rule is unchanged; only how the count is obtained. These tests compare against a row-wise
reference implementation of the old form, so a future "simplification" back to the loop is
caught by behaviour rather than by a timing assertion.
"""

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.analysis import _split_strata_by_sufficiency


def _reference(out, stratify_col, strata, min_obs=2):
    """The pre-change implementation, kept as the comparison oracle."""
    insufficient = [s for s in strata if len(out[out[stratify_col] == s]) < min_obs]
    published = [s for s in strata if s not in insufficient]
    return insufficient, published


CASES = {
    'object strata': (pd.DataFrame({'k': ['a', 'a', 'b', 'c', 'c', 'c']}), ['a', 'b', 'c']),
    'stratum absent from frame': (pd.DataFrame({'k': ['a', 'a']}), ['a', 'ghost']),
    'integer strata': (pd.DataFrame({'k': [1, 1, 2, 3, 3, 3]}), [1, 2, 3]),
    'numpy integer strata': (
        pd.DataFrame({'k': np.array([1, 1, 2], dtype=np.int64)}),
        [np.int64(1), np.int64(2)],
    ),
    'NaN in the stratify column': (pd.DataFrame({'k': ['a', 'a', np.nan, 'b', 'b']}), ['a', 'b', np.nan]),
    'duplicate entries in strata': (pd.DataFrame({'k': ['a', 'a', 'b']}), ['a', 'a', 'b']),
    'every stratum insufficient': (pd.DataFrame({'k': ['a', 'b', 'c']}), ['a', 'b', 'c']),
    'empty frame': (pd.DataFrame({'k': pd.Series([], dtype=object)}), ['a']),
    'mixed int and str keys': (pd.DataFrame({'k': [1, 1, '1', 'x', 'x']}), [1, '1', 'x']),
    'empty strata list': (pd.DataFrame({'k': ['a', 'a']}), []),
}


@pytest.mark.parametrize('name', list(CASES))
def test_matches_the_row_wise_reference(name):
    out, strata = CASES[name]
    assert _split_strata_by_sufficiency(out, 'k', strata) == _reference(out, 'k', strata)


def test_min_obs_boundary_is_inclusive():
    """A stratum with exactly `min_obs` rows is published; one row fewer is not."""
    out = pd.DataFrame({'k': ['keep', 'keep', 'drop']})
    insufficient, published = _split_strata_by_sufficiency(out, 'k', ['keep', 'drop'], min_obs=2)
    assert insufficient == ['drop']
    assert published == ['keep']


@pytest.mark.parametrize('min_obs', [1, 2, 3, 5])
def test_respects_min_obs(min_obs):
    out = pd.DataFrame({'k': ['a'] * 4 + ['b'] * 2})
    assert _split_strata_by_sufficiency(out, 'k', ['a', 'b'], min_obs=min_obs) == _reference(
        out, 'k', ['a', 'b'], min_obs=min_obs
    )


def test_order_follows_the_input_strata():
    """Callers rely on strata ordering for chart layout, so neither list may be re-sorted."""
    out = pd.DataFrame({'k': ['c', 'c', 'a', 'a', 'b']})
    insufficient, published = _split_strata_by_sufficiency(out, 'k', ['c', 'b', 'a'])
    assert published == ['c', 'a'], 'published must follow input order, not frequency or sort order'
    assert insufficient == ['b']


def test_the_two_lists_partition_the_input():
    out = pd.DataFrame({'k': ['a', 'a', 'b']})
    strata = ['a', 'b', 'ghost']
    insufficient, published = _split_strata_by_sufficiency(out, 'k', strata)
    assert sorted(insufficient + published) == sorted(strata)
    assert not set(insufficient) & set(published)


def test_stratified_execute_publishes_focusable_strata():
    """The contract this helper guards, exercised end to end.

    Unit equivalence does not prove the published strata are focusable — that is the actual
    requirement, and it is what breaks if the count is ever computed against the wrong frame.
    """
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            'f1': np.repeat(['A', 'B', 'C'], n // 3),
            'f2': np.tile(['X', 'Y'], n // 2),
            't': np.tile(np.arange(1, 11), n // 10),
            'y': rng.normal(100, 1, n),
        }
    )
    study = pb.ProcessBehavior(df).formulate(response='y', factors=['f1', 'f2'], time='t')
    result = study.execute(chart='X', by=['f1', 'f2'])

    assert result.is_stratified
    assert result.strata, 'expected published strata'
    for stratum in result.strata:
        focused = result.focus(stratum)
        assert len(focused.get_chart('X')) >= 2, f'{stratum} was published but is not focusable'
