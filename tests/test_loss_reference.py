"""
LossResult.reference: what the PDC, time and interaction parts show from noise alone.

The parts are averages of squared deviations of estimated means, so with no effects each is still
positive. The reference is their expected value under independent readings with a common variance,
for the study's own layout of occupied cells and cell sizes.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior import ProcessBehavior


def _loss(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return pb.formulate(df, response='y', factors=['k'], time='t').loss_function(**kwargs)


def test_balanced_grid_matches_closed_forms():
    K, T, n = 4, 10, 3
    df = pd.DataFrame([(k, t) for k in range(K) for t in range(T) for _ in range(n)], columns=['k', 't'])
    df['y'] = np.random.default_rng(0).normal(size=len(df))
    loss = _loss(df)
    s2 = loss.unexplained
    assert loss.reference['pdc'] == pytest.approx(s2 * (K - 1) / (K * T * n))
    assert loss.reference['time'] == pytest.approx(s2 * (T - 1) / (K * T * n))
    assert loss.reference['interaction'] == pytest.approx(s2 * (K - 1) * (T - 1) / (K * T * n))


def test_unbalanced_incomplete_layout_matches_simulation():
    """With no effects, the average of each part over replicates matches the average reference."""
    rng = np.random.default_rng(11)
    cells = [(k, t, int(rng.integers(2, 5))) for k in range(3) for t in range(8) if rng.random() > 0.2]
    layout = pd.DataFrame([(k, t) for k, t, n in cells for _ in range(n)], columns=['k', 't'])
    parts, refs = [], []
    for _ in range(200):
        df = layout.assign(y=rng.normal(size=len(layout)))
        loss = _loss(df)
        parts.append([loss.pdc, loss.time, loss.interaction])
        refs.append([loss.reference['pdc'], loss.reference['time'], loss.reference['interaction']])
    parts, refs = np.array(parts), np.array(refs)
    diff = parts - refs
    se = diff.std(axis=0, ddof=1) / np.sqrt(len(diff))
    assert np.all(np.abs(diff.mean(axis=0)) < 3 * se), (diff.mean(axis=0), se)


@pytest.fixture(scope='module')
def t100():
    return ProcessBehavior.read_csv('validation/PBTESTDATABASE_T100.csv')


def test_no_interaction_reference_without_replication(t100):
    loss = t100.formulate(response='PM SDS 2', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME').loss_function()
    assert loss.reference['interaction'] is None
    assert loss.reference['pdc'] > 0 and loss.reference['time'] > 0
    assert 'PDCxPT INT' not in repr(loss).split('No-effect reference')[1]


def test_shown_in_repr_and_as_dict(t100):
    loss = t100.formulate(response='PM SDS 1', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME').loss_function(
        target=237.0
    )
    assert set(loss.as_dict()['reference']) == {'pdc', 'time', 'interaction'}
    assert 'No-effect reference (share from noise alone): PDC ' in repr(loss)


def test_parts_unchanged(t100):
    """The reference is reported beside the parts; it changes none of them."""
    loss = t100.formulate(response='PM SDS 1', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME').loss_function(
        target=237.0
    )
    total = loss.centering + loss.unexplained + loss.pdc + loss.time + loss.interaction
    assert loss.total == pytest.approx(total)
