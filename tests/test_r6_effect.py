"""R6's factor-level effect is the condition effect averaged over the level, without R2's level mean.

R6 = α_i + R2 (10-1 manual Eq 14-25). The manual takes α_i as the level's mean of R5 (Eq 14-24, "≈ α_i");
R5 = ρ̂_k + R2 (Eq 14-19), so that mean also carries the level's average R2. VAS leaves it out: Tom's
Medicare run (10/3/2026, slide 52) centres the organisation-effects Xbar chart at 10831.1, which is the
effect without it (the mean of R5 gives 10829.8). On ADS 1 the two agree, because R2 averages to zero in
every cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb


def test_medicare_r6_effect_is_the_condition_effect():
    study = pb.formulate(
        pd.read_csv('validation/aco_per_capita_expenditure.csv'),
        response='PER CAPITA EXPENDITURE', factors=['ACO'], time='YEAR', precision=12,
    )
    result = study.execute(chart='Xbar', by=['ACO'], value='R6', recentered=True)
    frame = result._dataset
    alpha = frame['R6'] - frame['R2']
    rho = frame['Ybar_k'] - frame['Ybar']
    ok = frame['R2'].notna()
    assert np.allclose(alpha[ok], rho[ok], atol=1e-9)
    # the level's mean R2 is not in the effect (it is in the mean of R5)
    r5_mean = frame.groupby('ACO')['R5'].transform('mean')
    assert not np.allclose(alpha[ok], r5_mean[ok], atol=1e-3)
    assert round(result.get_statistics('Xbar')['center'], 1) == 10831.1  # VAS slide 52


def test_ads1_r6_effect_equals_mean_of_r5():
    df = pd.read_csv('validation/PBTESTDATABASE_T100.csv', na_values=['*'])
    study = pb.formulate(
        df, response='PM SDS 1', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME', precision=12
    )
    frame = study.execute(chart='Xbar', by=['FACTOR 1'], value='R6')._dataset
    alpha = frame['R6'] - frame['R2']
    r5_mean = frame.groupby('FACTOR 1')['R5'].transform('mean')
    assert alpha.to_numpy() == pytest.approx(r5_mean.to_numpy(), abs=1e-9)
