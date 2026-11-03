"""
A factor nested in time: each time period holds one of its levels (a stage, a before/after split).

Its effect cannot be separated from time. The design report says why the grid has empty cells,
and the loss function warns that the decomposition counts the factor more than once.
"""

import warnings

import pandas as pd
import pytest

from processbehavior import ProcessBehavior, ProcessBehaviorWarning, load_coffee_shop
from processbehavior.sds_detector import factors_nested_in_time


@pytest.fixture(scope='module')
def coffee():
    """Coffee-shop data with a stage factor: one stage per date."""
    return load_coffee_shop().bin(
        'week',
        method='breaks',
        breaks=[8, 12, 14],
        bin_labels=['baseline', 'new machine', 'new POS', 'new baristas'],
        label='stage',
    )


@pytest.fixture(scope='module')
def t100():
    return ProcessBehavior.read_csv('validation/PBTESTDATABASE_T100.csv')


def _loss_warnings(study) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        study.loss_function()
    return [str(w.message) for w in caught if issubclass(w.category, ProcessBehaviorWarning)]


class TestHelper:
    def test_flags_a_stage(self, coffee):
        frame = coffee.formulate(response='wait_sec', factors=['stage', 'daypart'], time='date')._ads.analysis_dataset
        assert factors_nested_in_time(frame, ['stage', 'daypart'], 'date') == ['stage']

    def test_crossed_factors_are_not_flagged(self, t100):
        assert factors_nested_in_time(t100.data, ['FACTOR 1', 'FACTOR 2'], 'PRODUCTION TIME') == []

    def test_one_shared_period_is_enough_to_cross(self):
        df = pd.DataFrame({'t': [1, 1, 2, 3], 'f': ['A', 'B', 'A', 'B']})
        assert factors_nested_in_time(df, ['f'], 't') == []

    def test_a_single_level_is_not_flagged(self):
        df = pd.DataFrame({'t': [1, 2, 3], 'f': ['A', 'A', 'A']})
        assert factors_nested_in_time(df, ['f'], 't') == []

    def test_no_time_or_no_factors(self, coffee):
        assert factors_nested_in_time(coffee.data, ['daypart'], None) == []
        assert factors_nested_in_time(coffee.data, [], 'date') == []


class TestDesignReport:
    def test_empty_cells_say_why(self, coffee):
        text = repr(coffee.formulate(response='wait_sec', factors=['stage'], time='date').design())
        assert '288 empty cells (stage is nested in date: each date has one stage)' in text

    def test_crossed_study_has_no_nesting_note(self, coffee):
        text = repr(coffee.formulate(response='wait_sec', factors=['daypart'], time='date').design())
        assert 'nested' not in text

    def test_structure_is_not_checked_without_a_plan(self, coffee):
        design = coffee.formulate(response='wait_sec', factors=['stage'], time='date').design()
        assert design.structure_summary == 'Not checked (no sampling plan supplied)'
        assert 'Structure: Not checked (no sampling plan supplied)' in repr(design)


class TestLossWarning:
    @pytest.mark.parametrize('factors', [['stage'], ['stage', 'daypart']])
    def test_warns_when_a_factor_is_nested(self, coffee, factors):
        messages = _loss_warnings(coffee.formulate(response='wait_sec', factors=factors, time='date'))
        assert len(messages) == 1
        assert "'stage' is nested in 'date'" in messages[0]
        assert 'phased=True' in messages[0]

    def test_crossed_studies_do_not_warn(self, coffee, t100):
        assert _loss_warnings(coffee.formulate(response='wait_sec', factors=['daypart'], time='date')) == []
        study = t100.formulate(response='PM SDS 1', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME')
        assert _loss_warnings(study) == []

    def test_loss_values_are_unchanged(self, coffee):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loss = coffee.formulate(response='wait_sec', factors=['stage'], time='date').loss_function()
        assert loss.total == pytest.approx(loss.unexplained + loss.pdc + loss.time + loss.interaction)
