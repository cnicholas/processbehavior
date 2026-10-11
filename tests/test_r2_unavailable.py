"""R2 is unavailable, with a stated reason, when the layout leaves the R2 scale factor undefined.

With a one-observation subgroup, R2 is the condition-and-period-adjusted series differenced and divided
by the R2 scale factor c(K, M) (10-1 manual Eqs 14-4..14-13). c needs at least two process design
conditions; with one condition the adjusted series is identically zero and R2 carries no information.
"""

import numpy as np
import pytest

import processbehavior as pb
from processbehavior import ChartNotAvailableError, ProcessBehaviorWarning, ValidationError


@pytest.fixture(scope='module')
def one_condition():
    df = pb.make_design(2, K1=1, K2=1, T=15, seed=1)
    with pytest.warns(ProcessBehaviorWarning, match='R2 is unavailable'):
        study = pb.formulate(df, response='y', factors=['factor 1'], time='time')
    return study


def test_formulate_succeeds_and_r2_is_nan_not_inf(one_condition):
    ds = one_condition.dataset
    assert one_condition.analytical_design_state.sds == 2
    assert ds['R2'].isna().all()
    assert not np.isinf(ds[['R2', 'R3', 'R4', 'R5']].to_numpy(dtype=float)).any()
    assert ds['R1'].notna().all()


def test_r2_family_not_offered(one_condition):
    offered = {v for _, v in one_condition.residual_charts}
    assert offered.isdisjoint({'R2', 'R3', 'R4', 'R5', 'R6'})
    assert list(one_condition.residuals) == ['R1']


@pytest.mark.parametrize('value', ['R2', 'R3', 'RCR5'])
def test_execute_r2_family_raises_with_reason(one_condition, value):
    with pytest.raises(ChartNotAvailableError, match='R2 scale factor'):
        one_condition.execute(chart='X', by=[], value=value)


def test_why_not_gives_the_reason(one_condition):
    assert 'R2 scale factor' in one_condition.why_not('X', value='R2')


def test_methods_needing_r2(one_condition):
    methods = one_condition.available_analysis_methods.set_index('method')['available']
    assert not methods['Loss Function'] and not methods['Maximum Information']
    with pytest.raises(ValidationError, match='R2 scale factor'):
        one_condition.loss_function()
    with pytest.raises(ValidationError, match='R2 scale factor'):
        one_condition.maximum_information()
    cap = one_condition.capability(usl=100.0, lsl=0.0)
    assert cap.cp is None
    assert 'R2 scale factor' in cap.potential_unavailable_reason
