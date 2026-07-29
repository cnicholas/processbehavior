"""Binning puts every observation in a bin, including the extremes.

``pd.cut`` leaves the outer edge open on the bounded side, and ``include_lowest`` only
closes the *lowest* one — and only bites for ``right=True``. Under ``right=False``, the
default, every observation equal to the column maximum binned to ``NaN``.

It was silent in the worst way: ``_range_labels`` renders that final interval with a closed
bracket (``[239.03, 243.225]``) and its comment says the rendering exists "so the extreme
value is included". The label promised what the binning did not deliver, and a row whose
category is NaN leaves the analysis entirely once the binned column is used as a factor.

Scale: **every** observation tied at the maximum, not one. Discretised gauges make ties at
the extremes routine, so this was not a boundary curiosity.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processbehavior import Derivation, evaluate

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
SOURCE = 'PM SDS 1'
# Methods whose edges are fitted from the data, so an edge coincides with an observation.
BOUNDED_METHODS = ('equal_freq', 'equal_width')
# sd and breaks bound with +/-inf, so no observation can sit on the outer edge.
UNBOUNDED_METHODS = ('sd', 'breaks')


@pytest.fixture(scope='module')
def column():
    return pd.read_csv(REFERENCE_CSV, na_values=['*'])[SOURCE]


def _lost(spec, col):
    """Present values that ended up with no category."""
    values = evaluate(spec, col).values
    return int((values.isna() & col.notna()).sum())


def _spec(method, **kwargs):
    if method == 'breaks':
        kwargs.setdefault('breaks', [230.0, 238.0, 243.0])
        kwargs.pop('n', None)
    return Derivation.bin(SOURCE, method=method, **kwargs)


# ---------------------------------------------------------------------------
# Nothing is lost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('right', [False, True])
@pytest.mark.parametrize('method', [*BOUNDED_METHODS, *UNBOUNDED_METHODS])
def test_every_observation_lands_in_a_bin(column, method, right):
    assert _lost(_spec(method, n=4, right=right), column) == 0


@pytest.mark.parametrize('right', [False, True])
def test_the_extreme_values_specifically(column, right):
    """The maximum under right=False, the minimum under right=True."""
    spec = _spec('equal_width', n=4, right=right)
    values = evaluate(spec, column).values

    assert values[column.idxmax()] is not pd.NA and not pd.isna(values[column.idxmax()])
    assert not pd.isna(values[column.idxmin()])


@pytest.mark.parametrize('n_tied', [1, 5, 50])
def test_all_observations_tied_at_the_maximum_survive(n_tied):
    """The regression's real size — it dropped every tied row, not a single boundary case."""
    col = pd.Series([float(v) for v in range(1000)] + [999.0] * (n_tied - 1))
    spec = Derivation.bin('x', method='equal_width', n=4)
    assert int((evaluate(spec, col).values.isna() & col.notna()).sum()) == 0


@pytest.mark.parametrize('right', [False, True])
@pytest.mark.parametrize('n_tied', [1, 5, 50])
def test_observations_tied_at_the_minimum_are_binned(n_tied, right):
    """The minimum was never the bug — asserted so the fix stays one-sided.

    ``include_lowest`` already closed the bottom edge for ``right=True``, and a half-open
    ``[a, b)`` interval includes it for ``right=False``. Only the top needed widening; if
    someone later mirrors the fix onto the bottom edge, they are adding dead code.
    """
    col = pd.Series([0.0] * n_tied + [float(v) for v in range(1, 1000)])
    spec = Derivation.bin('x', method='equal_width', n=4, right=right)
    values = evaluate(spec, col).values
    assert not values[col == col.min()].isna().any()


def test_counts_account_for_every_present_value(column):
    """What the app's preview shows: the bar chart must total the input."""
    result = evaluate(_spec('equal_freq', n=4), column)
    assert result.values.value_counts(sort=False).sum() == int(column.notna().sum())


# ---------------------------------------------------------------------------
# The fit is still reported honestly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('method', list(BOUNDED_METHODS))
def test_reported_edges_are_the_true_cut_points(column, method):
    """The widening applies to the cut alone.

    Nudging `fitted['edges']` would be a different bug: the preview, the range labels and a
    saved study all read those numbers, and they must stay the real quantile / linspace
    boundaries rather than a float tick above them.
    """
    edges = evaluate(_spec(method, n=4), column).fitted['edges']
    assert edges[0] == column.min()
    assert edges[-1] == column.max()


def test_range_labels_still_describe_the_true_interval(column):
    edges = evaluate(_spec('equal_width', n=4), column).fitted['edges']
    labels = evaluate(_spec('equal_width', n=4), column).fitted['labels']
    assert labels[-1].endswith(']'), 'the closed bracket is the promise being kept'
    assert f'{edges[-1]:g}' in labels[-1]


def test_unbounded_methods_are_untouched(column):
    """sd and breaks bound with +/-inf; the widening must not apply to them."""
    for method in UNBOUNDED_METHODS:
        edges = evaluate(_spec(method, n=4), column).fitted['edges']
        assert np.isinf(edges[0]) and np.isinf(edges[-1])


# ---------------------------------------------------------------------------
# Bin membership is otherwise unchanged
# ---------------------------------------------------------------------------


def test_only_the_extreme_changed(column):
    """A one-float widening must not move any other observation between bins.

    Reconstructed against the pre-fix behaviour: everything except values sitting exactly on
    the outer edge binned identically, so the counts differ only in the final bin.
    """
    result = evaluate(_spec('equal_freq', n=4), column)
    edges = result.fitted['edges']
    counts = result.values.value_counts(sort=False)

    manual = pd.cut(column, bins=edges, right=False, labels=result.fitted['labels'], ordered=True)
    on_edge = int((column == edges[-1]).sum())

    assert counts.iloc[-1] == manual.value_counts(sort=False).iloc[-1] + on_edge
    assert list(counts.iloc[:-1]) == list(manual.value_counts(sort=False).iloc[:-1])


def test_equal_freq_bins_stay_balanced(column):
    """Quantile bins still hold ~equal counts.

    On the reference data: 1000 / 1000 / 999 / 1001. The 999 is a quantile artifact that
    predates this fix; the 1001 is the recovered maximum. Asserted as a tolerance rather
    than exact counts, so the test states "balanced" rather than pinning arithmetic that
    would change with the data.
    """
    counts = evaluate(_spec('equal_freq', n=4), column).values.value_counts(sort=False)
    expected = counts.sum() / len(counts)
    assert (counts - expected).abs().max() <= 0.01 * expected
