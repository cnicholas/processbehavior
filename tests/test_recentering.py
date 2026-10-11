"""Re-centred residuals follow Bishop's 10-1 manual Eq 14-26: RCR = R + Ybar.

Re-centring moves a residual onto the measurement scale without changing what it shows, so an
effect chart of a re-centred residual plots that effect plus R2, centred on the grand mean.
"""

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb

T100 = 'validation/PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']


def _study(response, mask=None):
    df = pd.read_csv(T100, na_values=['*'])
    if mask is not None:
        df[response] = df[response].where(df[mask].notna())
    return pb.formulate(df, response=response, factors=FACTORS, time='PRODUCTION TIME', precision=12)


@pytest.mark.parametrize('response', ['PM SDS 1', 'PM SDS 2', 'PM SDS 3'])
def test_rcr_is_residual_plus_grand_mean(response):
    ds = _study(response).dataset
    for k in (1, 3, 4, 5):
        diff = (ds[f'RCR{k}'] - (ds[f'R{k}'] + ds['Ybar'])).abs().max()
        assert diff <= 1e-9, f'RCR{k} != R{k} + Ybar'


def test_pm5_inert_interaction_chart_matches_vas():
    """PM INERT under the PM SDS 5 pattern (ADS 2): Tom's 10/3/2026 deck, slides 28-29.

    The re-centred R3 individuals chart reads 0.07 (-5.88 / 6.03) with moving range 2.24 / 7.31.
    """
    study = _study('PM INERT', mask='PM SDS 5')
    r = study.execute(chart='X', by=[], value='R3', recentered=True, companion=True)
    x, mr = r.statistics('X'), r.statistics('mR')
    assert x['center'] == pytest.approx(0.07, abs=0.005)
    assert x['lpl'] == pytest.approx(-5.88, abs=0.005)
    assert x['upl'] == pytest.approx(6.03, abs=0.005)
    assert mr['center'] == pytest.approx(2.24, abs=0.005)
    assert mr['upl'] == pytest.approx(7.31, abs=0.005)


def test_period_effects_live_in_r4_not_r3():
    """By period, the re-centred R3 averages out flat (the interaction sums to zero over conditions);
    the period effects are R4's. Charting period effects with R3 would show nothing."""
    study = _study('PM SDS 1')
    by_t = ['PRODUCTION TIME']
    r3 = study.execute(chart='Xbar', by=by_t, value='R3', recentered=True).charts['Xbar']['data']['xbar']
    r4 = study.execute(chart='Xbar', by=by_t, value='R4', recentered=True).charts['Xbar']['data']['xbar']
    assert np.std(r3) < 1e-6
    assert np.std(r4) > 0.1
