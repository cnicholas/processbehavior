"""ADS 3 (partial replication) charts follow the 10-1 manual: Xbar/S is the default chart.

Tom Bishop's VAS run of 10/3/2026 draws PM SDS 3 as Xbar/S throughout (original data and the
PDC×PT interaction); the per-condition centre lines on that deck are the unweighted mean of every
subgroup mean, one-observation subgroups included.
"""

import pandas as pd
import pytest

import processbehavior as pb

T100 = 'validation/PBTESTDATABASE_T100.csv'


@pytest.fixture(scope='module')
def pm_sds_3():
    df = pd.read_csv(T100, na_values=['*'])
    return pb.formulate(df, response='PM SDS 3', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME', precision=12)


def test_ads3_recommends_xbar(pm_sds_3):
    assert pm_sds_3.analytical_design_state.sds == 3
    assert pm_sds_3.recommended_chart == 'Xbar'


def test_stratified_xbar_centre_counts_one_observation_subgroups():
    """Each stratum's centre is the mean of all its subgroup means, n = 1 subgroups included."""
    df = pd.DataFrame(
        {
            'lane': ['A'] * 5 + ['B'] * 6,
            'time': [1, 1, 2, 2, 3, 1, 1, 2, 2, 3, 3],
            'y': [10.0, 12.0, 14.0, 16.0, 30.0, 5.0, 7.0, 9.0, 11.0, 20.0, 22.0],
        }
    )
    study = pb.formulate(df, response='y', factors=['lane'], time='time', precision=12)
    result = study.execute(chart='Xbar', by=['time'])

    # Lane A: subgroup means 11, 15 and the single observation 30 -> centre 56/3
    assert result.statistics('Xbar', stratum='A')['center'] == pytest.approx(56 / 3)
    # Lane B: all subgroups replicated -> 6, 10, 21 -> centre 37/3
    assert result.statistics('Xbar', stratum='B')['center'] == pytest.approx(37 / 3)


@pytest.mark.parametrize(
    ('stratum', 'deck_centre'),
    [('2_2', 239.0), ('3_1', 236.64), ('4_1', 237.52), ('4_2', 238.18), ('1_1', 238.22)],
)
def test_pm_sds3_per_condition_centres_match_vas(pm_sds_3, stratum, deck_centre):
    """PM SDS 3 deck (10/3/2026), pages 3-18: per-condition Xbar centre lines."""
    result = pm_sds_3.execute(chart='Xbar', by=['PRODUCTION TIME'])
    assert result.statistics('Xbar', stratum=stratum)['center'] == pytest.approx(deck_centre, abs=0.005)


def test_pm_sds3_interaction_xbar_s_centres_match_vas(pm_sds_3):
    """PM SDS 3 deck (10/3/2026), slides 28-29: interaction Xbar 237.83, S 1.04."""
    r = pm_sds_3.execute(
        chart='Xbar', by=['FACTOR 1', 'FACTOR 2', 'PRODUCTION TIME'], value='R3', recentered=True, companion=True
    )
    assert r.statistics('Xbar')['center'] == pytest.approx(237.83, abs=0.005)
    assert r.statistics('S')['center'] == pytest.approx(1.04, abs=0.005)
