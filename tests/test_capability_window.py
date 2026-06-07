"""Time-windowed capability views (before/after along the declared time axis).

The window is a view over the immutable analytic dataset: current stats and the
potential *centering* (y_bar) read the windowed observed values, but the potential
*noise floor* (sigma_hat_r2) stays the full-study pooled R2 — never re-estimated on
the subset. Default (window=None) must be byte-for-byte identical to before.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.exceptions import ValidationError

DATA = Path(__file__).parent / 'fixtures' / 'data' / 'coffee_shop_demo_long.csv'
CUT_WEEK = 8
CUT_DATE = '2026-02-23'  # first date of week 8 (the process change)


@pytest.fixture
def coffee_df():
    return pd.read_csv(DATA)  # date stays a string -> formulated as a datetime axis


def _study(coffee_df, time):
    return ProcessBehavior(coffee_df).formulate(
        response='wait_sec', factors=['day_of_week'], time=time
    )


# --- The before/after story (incapable -> capable) -------------------------

@pytest.mark.parametrize(
    'time, before_win, after_win',
    [
        ('week', (None, CUT_WEEK), (CUT_WEEK, None)),
        ('date', (None, CUT_DATE), (CUT_DATE, None)),
    ],
)
def test_before_after_contrast(coffee_df, time, before_win, after_win):
    study = _study(coffee_df, time)
    before = study.capability(usl=240, target=180, window=before_win)
    after = study.capability(usl=240, target=180, window=after_win)

    # Before: baseline era ~240s, right at the USL -> roughly half over spec.
    assert before.y_bar == pytest.approx(239.9, abs=0.5)
    assert before.n == 168
    assert before.pct_above_usl > 40
    # After: post-change era well below the USL -> far fewer over spec.
    assert after.y_bar == pytest.approx(202.6, abs=0.5)
    assert after.n == 216
    assert after.pct_above_usl < 15
    assert after.y_bar < before.y_bar


def test_int_and_date_axes_give_identical_partition(coffee_df):
    by_week = _study(coffee_df, 'week').capability(usl=240, target=180, window=(None, CUT_WEEK))
    by_date = _study(coffee_df, 'date').capability(usl=240, target=180, window=(None, CUT_DATE))
    assert by_week.n == by_date.n
    assert by_week.y_bar == pytest.approx(by_date.y_bar)


# --- Principle: pooled potential, windowed centering -----------------------

def test_potential_noise_floor_is_pooled_not_reestimated(coffee_df):
    study = _study(coffee_df, 'week')
    full = study.capability(usl=240, target=180)
    before = study.capability(usl=240, target=180, window=(None, CUT_WEEK))
    after = study.capability(usl=240, target=180, window=(CUT_WEEK, None))
    # sigma_hat_r2 is the full-study residual floor in every window.
    assert before.sigma_hat_r2 == pytest.approx(full.sigma_hat_r2)
    assert after.sigma_hat_r2 == pytest.approx(full.sigma_hat_r2)


def test_centering_is_windowed(coffee_df):
    study = _study(coffee_df, 'week')
    full = study.capability(usl=240, target=180)
    before = study.capability(usl=240, target=180, window=(None, CUT_WEEK))
    assert before.y_bar != full.y_bar
    assert before.y_bar == pytest.approx(
        coffee_df[coffee_df.week < CUT_WEEK]['wait_sec'].mean()
    )


# --- Un-windowed path unchanged -------------------------------------------

def test_unwindowed_result_carries_no_window_metadata(coffee_df):
    full = _study(coffee_df, 'week').capability(usl=240, target=180)
    d = full.as_dict()
    for k in ('window', 'time_var', 'n_total', 'window_warning'):
        assert k not in d
    assert full.window is None and full.window_warning is None


def test_window_none_matches_frozen_pre_change_baseline(coffee_df):
    """The real 'pure addition' guarantor: window=None as_dict() must equal a
    baseline captured from the PRE-CHANGE code (committed JSON), not merely agree
    with a second fresh computation. Regenerate the baseline only if the default
    capability math intentionally changes.
    """
    baseline = json.loads(
        (Path(__file__).parent / 'fixtures' / 'coffee_capability_full_baseline.json').read_text()
    )
    got = _study(coffee_df, 'week').capability(usl=240, target=180).as_dict()
    assert set(got) == set(baseline)  # no new keys leak onto the default path
    for k, expected in baseline.items():
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert got[k] == pytest.approx(expected), f'{k}: {got[k]} != baseline {expected}'
        else:
            assert got[k] == expected, f'{k}: {got[k]} != baseline {expected}'


def test_windowed_result_is_self_describing(coffee_df):
    cap = _study(coffee_df, 'week').capability(usl=240, target=180, window=(CUT_WEEK, None))
    d = cap.as_dict()
    assert d['window'] == (CUT_WEEK, None)
    assert d['time_var'] == 'week'
    assert d['n_total'] == 384


# --- n-ladder: refuse < 8, warn 8..29, clean >= 30 ------------------------

def test_thin_window_refuses_below_8(coffee_df):
    study = _study(coffee_df, 'date')
    with pytest.raises(ValidationError, match=r'need >= 8'):
        study.capability(usl=240, target=180, window=('2026-01-05', '2026-01-06'))


def test_thin_window_warns_between_8_and_29(coffee_df):
    # Two weeks of one weekday ~ 8 readings -> warn but compute.
    study = _study(coffee_df, 'week')
    # window weeks [1,3) with all days = 48 obs (clean); narrow via a small int span
    # that yields 8<=n<30: a single week = 24 obs (warn).
    cap = study.capability(usl=240, target=180, window=(1, 2))
    assert cap.n == 24
    assert cap.window_warning is not None
    assert 'unstable' in cap.window_warning


def test_wide_window_has_no_warning(coffee_df):
    cap = _study(coffee_df, 'week').capability(usl=240, target=180, window=(CUT_WEEK, None))
    assert cap.n >= 30
    assert cap.window_warning is None


# --- Guards ---------------------------------------------------------------

def test_window_requires_time_variable(coffee_df):
    study = ProcessBehavior(coffee_df).formulate(response='wait_sec', factors=['day_of_week'])
    with pytest.raises(ValidationError, match='time variable'):
        study.capability(usl=240, target=180, window=(None, 8))


def test_bound_type_mismatch_raises_clean_error(coffee_df):
    study = _study(coffee_df, 'week')  # integer axis
    with pytest.raises(ValidationError, match='not comparable'):
        study.capability(usl=240, target=180, window=(None, '2026-02-23'))
