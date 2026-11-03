"""
Design-report lines that explain the analysis, and the reason given when potential capability is unavailable.

- Single-reading cells: in ADS 3 one cell with one reading switches R2 for the whole study to the
  differencing method, so the report says how many there are.
- Periods: when conditions were observed in different periods, condition effects average over
  different times, so the report says how far apart the conditions' average periods are.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior import ProcessBehavior, load_coffee_shop


def _report(df, **kwargs) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return repr(pb.formulate(df, response='y', **kwargs).design())


@pytest.fixture(scope='module')
def crossed():
    """4 conditions x 100 periods x 2 readings: every cell filled."""
    g = pd.DataFrame([(k, t, r) for k in range(4) for t in range(1, 101) for r in range(2)], columns=['k', 't', 'r'])
    g['y'] = np.random.default_rng(3).normal(size=len(g))
    return g


class TestSingleReadingCells:
    def test_one_singleton_is_reported(self, crossed):
        one = crossed.drop(index=crossed[(crossed.k == 2) & (crossed.t == 50) & (crossed.r == 1)].index)
        text = _report(one, factors=['k'], time='t')
        assert 'ADS (Analytical): 3' in text
        assert 'Single-reading cells: 1 of 400. One is enough to switch R2' in text

    def test_full_replication_has_no_line(self, crossed):
        assert 'Single-reading cells' not in _report(crossed, factors=['k'], time='t')

    def test_t100_partial_replication(self):
        d = ProcessBehavior.read_csv('validation/PBTESTDATABASE_T100.csv')
        text = repr(d.formulate(response='PM SDS 3', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME').design())
        assert 'Single-reading cells: 17 of 800.' in text


class TestPeriods:
    def test_staggered_conditions_are_reported(self):
        # condition k observed in periods 1+2k .. 18+2k of 24: average periods 9.5, 11.5, 13.5, 15.5
        rows = [(k, t, r) for k in range(4) for t in range(1 + 2 * k, 19 + 2 * k) for r in range(3)]
        s = pd.DataFrame(rows, columns=['k', 't', 'r'])
        s['y'] = np.random.default_rng(4).normal(size=len(s))
        text = _report(s, factors=['k'], time='t')
        assert 'Periods: conditions were observed in different periods' in text
        assert 'differs by up to 6.0 of 24' in text

    def test_crossed_data_has_no_line(self, crossed):
        assert 'Periods:' not in _report(crossed, factors=['k'], time='t')

    def test_nested_factor_has_no_line(self):
        coffee = load_coffee_shop().bin(
            'week', method='breaks', breaks=[8, 12, 14], bin_labels=['a', 'b', 'c', 'd'], label='stage'
        )
        text = repr(coffee.formulate(response='wait_sec', factors=['stage'], time='date').design())
        assert 'nested in date' in text
        assert 'Periods:' not in text


class TestCapabilityReason:
    def test_no_factors(self, crossed):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cap = pb.formulate(crossed, response='y', time='t').capability(lsl=-4, usl=4)
        assert cap.cp is None
        assert cap.potential_unavailable_reason == (
            'Cp/Cpk use the VAS R2 residuals, which need at least one factor; this study has none.'
        )

    def test_no_time(self, crossed):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cap = pb.formulate(crossed, response='y', factors=['k']).capability(lsl=-4, usl=4)
        assert 'which need a time variable' in cap.potential_unavailable_reason
        assert 'Cp=' not in repr(cap)
