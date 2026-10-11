"""Xbar/S charts with subgroups of unequal size follow the 10-1 manual (Eqs 11-5, 11-7..11-9, 11-16, 11-17).

sigma_hat is the average of each subgroup's own unbiased S_r / c4(N_r); each subgroup's Xbar limits are
centre ± 3 sigma_hat / sqrt(N_r), and its S limits are B3(N_r) S-bar and B4(N_r) S-bar. N_r counts the
values the chart is drawn from, so a residual missing on one row (R2's first value) leaves that subgroup
one smaller. Tom's VAS run of the Medicare data (10/3/2026, slides 52-53) draws exactly this: ACO-001
has 3 R6 values, the other 23 organisations 4, and its limits are wider by sqrt(4/3) (Xbar) and use
B4(3) (S).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.spc_constants import b4, c4, subgroup_sigma_hat


def test_subgroup_sigma_hat_is_average_of_unbiased_estimates():
    s, n = np.array([1.0, 2.0, 1.5]), np.array([3, 4, 5])
    assert subgroup_sigma_hat(s, n) == pytest.approx(np.mean(s / np.array([c4(3), c4(4), c4(5)])))
    # equal sizes: S-bar / c4(N) (Eq 11-6)
    assert subgroup_sigma_hat(s, np.array([4, 4, 4])) == pytest.approx(s.mean() / c4(4))


@pytest.fixture(scope='module')
def medicare():
    return pb.formulate(
        pd.read_csv('validation/aco_per_capita_expenditure.csv'),
        response='PER CAPITA EXPENDITURE', factors=['ACO'], time='YEAR', precision=12,
    )


def test_subgroup_size_counts_the_charted_values(medicare):
    table = medicare.execute(chart='S', by=['ACO'], value='R6').chart_table('S')
    sizes = medicare.dataset.groupby('ACO')['R2'].count()
    assert sizes['ACO-001'] == 3 and (sizes.drop('ACO-001') == 4).all()
    # ACO-001's S limit uses B4(3), the others B4(4) (Eqs 11-8, 11-9)
    s_bar = table['center'].iloc[0]
    assert table['upl'].iloc[0] == pytest.approx(b4(3) * s_bar, rel=1e-9)
    assert table['upl'].iloc[1] == pytest.approx(b4(4) * s_bar, rel=1e-9)


def test_xbar_limits_step_with_subgroup_size(medicare):
    result = medicare.execute(chart='Xbar', by=['ACO'], value='R6', recentered=True)
    table = result.chart_table('Xbar')
    half = (table['upl'] - table['lpl']) / 2
    assert half.iloc[0] / half.iloc[1] == pytest.approx(np.sqrt(4 / 3), rel=1e-9)
    # VAS slide 52 prints the size-4 limits: half-width (11321.4 - 10340.9) / 2 = 490.25
    assert half.iloc[-1] == pytest.approx(490.25, abs=0.06)
    stats = result.get_statistics('Xbar')
    assert stats['limits_vary'] is True and stats['lpl'] is None


def test_equal_sizes_unchanged(medicare):
    """The response itself has no missing values, so every organisation has 4: the limits are constant,
    S-bar / c4(N) as before."""
    result = medicare.execute(chart='Xbar', by=['ACO'], recentered=False)
    stats = result.get_statistics('Xbar')
    assert stats.get('limits_vary') is not True and stats['lpl'] is not None


@pytest.fixture(scope='module')
def with_one_reading_cells():
    """Two conditions x four periods, three readings per cell except two cells with one (ADS 3)."""
    rng = np.random.default_rng(7)
    rows = []
    for f in ('A', 'B'):
        for t in range(1, 5):
            n = 1 if (f, t) in {('A', 2), ('B', 4)} else 3
            rows += [{'F': f, 'T': t, 'Y': 10 + rng.normal()} for _ in range(n)]
    return pb.formulate(pd.DataFrame(rows), response='Y', factors=['F'], time='T', precision=12)


def test_one_reading_subgroups_are_charted_at_three_sigma(with_one_reading_cells):
    """A one-reading subgroup has no S_r, so it takes no part in sigma_hat (Eq 11-5), but it is charted with
    limits centre ± 3 sigma_hat (Eqs 11-16/17 with N_r = 1), as VAS draws it (PM SDS 3 and 6, 10/3/2026)."""
    study = with_one_reading_cells
    result = study.execute(chart='Xbar', by=[], companion=True)
    xbar, s = result.chart_table('Xbar'), result.chart_table('S')
    cells = study.dataset.groupby(['F', 'T'])['Y'].agg(['std', 'count'])
    replicated = cells[cells['count'] >= 2]
    sigma = np.mean(replicated['std'] / c4(3))
    assert len(xbar) == 8 and len(s) == 6
    one = xbar[xbar['n'] == 1]
    assert len(one) == 2
    assert (one['upl'] - one['center']).to_numpy() == pytest.approx([3 * sigma] * 2, rel=1e-9)
    three = xbar[xbar['n'] == 3]
    assert (three['upl'] - three['center']).to_numpy() == pytest.approx([3 * sigma / np.sqrt(3)] * 6, rel=1e-9)
    assert result.get_statistics('Xbar')['limits_vary'] is True
    # The S chart has only the subgroups with an S, all of size 3: constant B3/B4(3) limits
    s_stats = result.get_statistics('S')
    assert s_stats['upl'] == pytest.approx(b4(3) * replicated['std'].mean(), rel=1e-9)


def test_one_reading_subgroups_are_charted_per_stratum(with_one_reading_cells):
    result = with_one_reading_cells.execute(chart='Xbar', by=['T'], companion=True)
    a = result.focus('A')
    assert len(a.chart_table('Xbar')) == 4 and len(a.chart_table('S')) == 3
