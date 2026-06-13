"""Tests for bundled dataset loaders."""

import pandas as pd

from processbehavior import ProcessBehavior, load_coffee_shop


def test_load_coffee_shop_returns_ready_processbehavior():
    pb = load_coffee_shop()
    assert isinstance(pb, ProcessBehavior)
    assert pb.data.shape == (384, 6)
    assert {'date', 'day_of_week', 'week', 'subgroup_id', 'reading_no', 'wait_sec'} <= set(pb.data.columns)


def test_load_coffee_shop_is_formulatable_with_both_axes():
    pb = load_coffee_shop()
    # integer time axis
    s_week = pb.formulate(response='wait_sec', factors=['day_of_week'], time='week')
    assert s_week.analytical_design_state.sds >= 1
    # date time axis (string dates -> datetime64 at formulation)
    s_date = pb.formulate(response='wait_sec', factors=['day_of_week'], time='date')
    assert pd.api.types.is_datetime64_any_dtype(s_date.dataset['date'])


def test_load_coffee_shop_before_after_capability_story():
    study = load_coffee_shop().formulate(
        response='wait_sec', factors=['day_of_week'], time='week'
    )
    before = study.capability(usl=240, target=180, window=(None, 8))
    after = study.capability(usl=240, target=180, window=(8, None))
    full = study.capability(usl=240, target=180)
    # incapable -> capable
    assert before.y_bar > after.y_bar
    assert before.pct_above_usl > 40 and after.pct_above_usl < 15
    # pooled potential noise floor (not re-estimated on the window)
    assert before.sigma_hat_r2 == full.sigma_hat_r2 == after.sigma_hat_r2
