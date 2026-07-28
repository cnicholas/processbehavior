"""`Analysis._calculate_lane_boundaries` — X-chart dividers, computed without a per-row loop.

Lane boundaries are visual only: the vertical dashed lines marking where a *collapsed* factor
(one not in `by`) changes value. They cost 2.9s of a 4s `execute()` at 1M rows — a row-wise
`.agg('_'.join, axis=1)` plus a `df.index.get_loc()` call per boundary, which is not O(1) on
these frames.

The replacement compares the label array against itself shifted by one, so positions come from
`np.flatnonzero` directly. These tests pin the boundary contract the plotting layer reads
(`plotting/lane_boundaries.py`) — position, label, variables — not the implementation.
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior.analysis import Analysis


def _boundaries(df, collapsed_vars):
    """Call the method without constructing a full Analysis — it touches no instance state."""
    return Analysis._calculate_lane_boundaries(None, df, collapsed_vars)


def _reference(df, collapsed_vars):
    """Row-wise oracle: walk the frame and record every position whose key differs from the
    previous row's."""
    if not collapsed_vars or df.empty:
        return []
    keys = ['_'.join(str(v) for v in row) for row in df[collapsed_vars].to_numpy()]
    return [
        {'position': i, 'label': keys[i], 'variables': collapsed_vars}
        for i in range(1, len(keys))
        if keys[i] != keys[i - 1]
    ]


def _frame(**cols):
    return pd.DataFrame(cols)


CASES = {
    'single collapsed var': (_frame(f=['A', 'A', 'B', 'B', 'C']), ['f']),
    'two collapsed vars': (_frame(a=['A', 'A', 'A', 'B'], b=[1, 1, 2, 2]), ['a', 'b']),
    'three collapsed vars': (_frame(a=['A', 'A'], b=[1, 2], c=['X', 'X']), ['a', 'b', 'c']),
    'no changes at all': (_frame(f=['A'] * 5), ['f']),
    'every row changes': (_frame(f=list('ABCDE')), ['f']),
    'single row': (_frame(f=['A']), ['f']),
    'numeric collapsed var': (_frame(f=[1, 1, 2, 3, 3]), ['f']),
    'repeating pattern': (_frame(f=['A', 'B', 'A', 'B']), ['f']),
}


@pytest.mark.parametrize('name', list(CASES))
def test_matches_the_row_wise_reference(name):
    df, collapsed = CASES[name]
    assert _boundaries(df, collapsed) == _reference(df, collapsed)


def test_no_collapsed_vars_returns_empty():
    assert _boundaries(_frame(f=['A', 'B']), []) == []


def test_empty_frame_returns_empty():
    """The positional form indexes element 0 to seed the change mask, so an empty frame
    must short-circuit rather than raise."""
    assert _boundaries(pd.DataFrame({'f': pd.Series([], dtype=object)}), ['f']) == []


def test_first_row_is_never_a_boundary():
    """A chart starts in its first lane; the divider belongs at each *change*."""
    result = _boundaries(_frame(f=['A', 'A', 'B']), ['f'])
    assert all(b['position'] > 0 for b in result)
    assert [b['position'] for b in result] == [2]


def test_boundary_contract_shape():
    """The plotting layer reads these keys — position must be a plain int, not np.int64."""
    boundary = _boundaries(_frame(f=['A', 'B']), ['f'])[0]
    assert set(boundary) == {'position', 'label', 'variables'}
    assert isinstance(boundary['position'], int)
    assert boundary['label'] == 'B', 'the label names the lane being entered'
    assert boundary['variables'] == ['f']


def test_positions_are_positional_not_index_labels():
    """Regression guard for a latent bug in the old `df.index.get_loc(pos)` lookup.

    When a boundary landed on a *duplicated* index label, `get_loc` returned a boolean mask
    rather than an integer, so the boundary dict came back malformed — `position` a numpy
    array and `label` a Series instead of a string, which the plotting layer cannot use:

        {'position': array([True, False, True]), 'label': 10  A\\n10  C\\nName: f, ...}

    The positional form cannot express that. Note the index here is both non-monotonic and
    non-unique *at a boundary row*, which is what it takes to reach the old defect.
    """
    df = pd.DataFrame({'f': ['A', 'B', 'C']}, index=[10, 5, 10])
    assert _boundaries(df, ['f']) == [
        {'position': 1, 'label': 'B', 'variables': ['f']},
        {'position': 2, 'label': 'C', 'variables': ['f']},
    ]


def test_datetime_collapsed_var_uses_astype_rendering():
    """Deliberate divergence from `encode_rsg_series`, pinned so the two are not "unified".

    Lane labels are display strings. `DataFrame.astype(str)` renders a midnight timestamp as
    '2024-01-01'; `encode_rsg_series` would render it as '2024-01-01 00:00:00' because
    stratum identity must match `str(Timestamp)` on the plan side. Sharing the encoder would
    silently change what a chart divider reads.
    """
    df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'])})
    result = _boundaries(df, ['d'])
    assert result == [{'position': 2, 'label': '2024-01-02', 'variables': ['d']}]


def test_matches_reference_on_a_larger_frame():
    """Randomised cross-check at a size where an off-by-one would surface."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame({'a': rng.integers(0, 4, 2000), 'b': rng.choice(list('XY'), 2000)})
    assert _boundaries(df, ['a', 'b']) == _reference(df, ['a', 'b'])
