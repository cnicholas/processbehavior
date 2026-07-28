"""`calculate_limits_vectorized` must equal `calculate_limits`, element for element.

The chart builders called the scalar form once per subgroup through
``DataFrame.apply(axis=1)`` — ~50,000 Python calls at 1M rows, each constructing a
``pd.Series`` to carry two numbers, ~2.5s per chart. The array form does the same
arithmetic on whole columns.

`calculate_limits` is deliberately left unchanged as the scalar reference: it is what the
Bishop validator exercises, so keeping it independent means those 280 assertions stay a
genuine external check on the fast path rather than a check of it against itself. These
tests are the internal half of that argument.
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior.spc_constants import calculate_limits, calculate_limits_vectorized


def _scalar_frame(limits_type, rows, sigma_multiplier=3):
    return pd.DataFrame(
        [calculate_limits(limits_type=limits_type, sigma_multiplier=sigma_multiplier, **kw) for kw in rows]
    )


@pytest.fixture(scope='module')
def sample():
    rng = np.random.default_rng(7)
    n = 500
    return {
        'mean': rng.normal(50, 5, n),
        'sd': rng.uniform(0.1, 3, n),
        'N': rng.integers(2, 30, n),
        'mR': rng.uniform(0.1, 5, n),
    }


@pytest.mark.parametrize('sigma_multiplier', [3.0, 2.0, 1.5])
@pytest.mark.parametrize('limits_type', ['Xbar', 'S', 'XmR', 'R'])
def test_matches_the_scalar_reference_exactly(sample, limits_type, sigma_multiplier):
    """Exact equality, not approximate — limits are compared and displayed as numbers a
    user reads off a chart, so a last-digit drift is a real difference."""
    mean, sd, N, mR = sample['mean'], sample['sd'], sample['N'], sample['mR']

    if limits_type == 'Xbar':
        vec = calculate_limits_vectorized('Xbar', mean=mean, sd=sd, N=N, sigma_multiplier=sigma_multiplier)
        rows = [{'mean': m, 'sd': s, 'N': int(k)} for m, s, k in zip(mean, sd, N)]
    elif limits_type == 'S':
        vec = calculate_limits_vectorized('S', sd=sd, N=N, sigma_multiplier=sigma_multiplier)
        rows = [{'mean': 0, 'sd': s, 'N': int(k)} for s, k in zip(sd, N)]
    elif limits_type == 'XmR':
        vec = calculate_limits_vectorized('XmR', mean=mean, mR=mR, sigma_multiplier=sigma_multiplier)
        rows = [{'mean': m, 'mR': r} for m, r in zip(mean, mR)]
    else:
        vec = calculate_limits_vectorized('R', mR=mR, sigma_multiplier=sigma_multiplier)
        rows = [{'mean': 0, 'sd': 0, 'N': 0, 'mR': r} for r in mR]

    ref = _scalar_frame(limits_type, rows, sigma_multiplier)
    np.testing.assert_array_equal(vec['lpl'].to_numpy(), ref['lpl'].to_numpy())
    np.testing.assert_array_equal(vec['upl'].to_numpy(), ref['upl'].to_numpy())


@pytest.mark.parametrize('N', [2, 3, 5, 25, 26, 100])
def test_subgroup_size_constants_broadcast_correctly(N):
    """Equal N must give equal limits.

    c4/b3/b4 are evaluated once per *distinct* N and mapped back; this guards the mapping
    against silently pairing a constant with the wrong row.
    """
    means = np.full(50, 42.0)
    sizes = np.full(50, N)
    vec = calculate_limits_vectorized('Xbar', mean=means, sd=1.5, N=sizes)
    ref = calculate_limits(limits_type='Xbar', mean=42.0, sd=1.5, N=N)
    assert vec['lpl'].nunique() == 1 and vec['upl'].nunique() == 1
    assert vec['lpl'].iloc[0] == ref['lpl']
    assert vec['upl'].iloc[0] == ref['upl']


def test_mixed_subgroup_sizes_pair_with_their_own_constant():
    sizes = np.array([2, 30, 2, 30, 5])
    vec = calculate_limits_vectorized('Xbar', mean=np.zeros(5), sd=1.0, N=sizes)
    for i, n in enumerate(sizes):
        ref = calculate_limits(limits_type='Xbar', mean=0.0, sd=1.0, N=int(n))
        assert vec['upl'].iloc[i] == ref['upl'], f'row {i} (N={n}) got another size’s constant'


def test_subgroup_size_of_one_raises_like_the_scalar_path():
    """c4(1) is undefined and the scalar path raises. The array path must not quietly
    emit inf or NaN instead."""
    with pytest.raises(ValueError, match='>= 2'):
        calculate_limits(limits_type='Xbar', mean=1.0, sd=1.0, N=1)
    with pytest.raises(ValueError, match='>= 2'):
        calculate_limits_vectorized('Xbar', mean=np.array([1.0]), sd=1.0, N=np.array([1]))


def test_index_is_preserved_for_assignment():
    """`df[['lpl','upl']] = calculate_limits_vectorized(...)` aligns on index. Returning a
    fresh RangeIndex would silently produce NaN on any non-default index — which is what
    the chart frames have after filtering."""
    df = pd.DataFrame({'center': [1.0, 2.0, 3.0], 'n': [5, 5, 5]}, index=[10, 20, 30])
    out = calculate_limits_vectorized('Xbar', mean=df['center'], sd=1.0, N=df['n'])
    assert list(out.index) == [10, 20, 30]
    df[['lpl', 'upl']] = out
    assert df['lpl'].notna().all()


def test_unsupported_type_raises():
    with pytest.raises(ValueError, match='not supported'):
        calculate_limits_vectorized('Nonsense', mean=np.array([1.0]), sd=1.0, N=np.array([5]))


@pytest.mark.parametrize(
    'limits_type,kwargs',
    [('Xbar', {'mean': None, 'sd': 1.0, 'N': np.array([5])}), ('S', {'sd': None, 'N': np.array([5])}),
     ('XmR', {'mean': np.array([1.0]), 'mR': None}), ('R', {'mR': None})],
)
def test_missing_required_argument_raises(limits_type, kwargs):
    with pytest.raises(ValueError, match='requires'):
        calculate_limits_vectorized(limits_type, **kwargs)
