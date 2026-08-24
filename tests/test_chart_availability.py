"""`why_not()` / `support` must answer the same question `execute()` enforces.

The bug these pin: on ADS 2 (no replication), `why_not('Xbar')` reported
"'Xbar' IS available" while `execute(chart='Xbar')` raised. Two different
questions were being answered — `valid_charts` asks whether the chart family
has any legitimate use, `execute` asks whether *this* call will compute — and
nothing said so, which made the flagship diagnostic feature contradict the
thing it exists to explain.

The fix keeps `valid_charts` meaning what it always meant (membership must not
change: Xbar-of-a-residual and Xbar-of-a-pooled-subgroup are legal on ADS 2)
and makes the advisory surfaces data-aware.
"""

import pytest

from processbehavior import ProcessBehavior, ValidationError
from processbehavior.datasets.synthetic import make_design

FACTORS = ['factor 1', 'factor 2']


def study_for(state):
    df = make_design(state=state, seed=42)
    return ProcessBehavior(df).formulate(response='y', factors=FACTORS, time='time')


@pytest.fixture(scope='module')
def ads2():
    """No replication — every factor x time cell holds one observation."""
    study = study_for(2)
    assert study.analytical_design_state.sds == 2, 'fixture no longer produces ADS 2'
    return study


@pytest.fixture(scope='module')
def ads1():
    study = study_for(1)
    assert study.analytical_design_state.sds == 1
    return study


def _primary_row(study, chart):
    support = study.support
    row = support[(support['chart'] == chart) & (support['category'] == 'primary')]
    assert len(row) == 1, f'expected one primary row for {chart}, got {len(row)}'
    return row.iloc[0]


class TestAdvisoryMatchesEnforcement:
    """The contradiction itself: whatever `why_not` says, `execute` must agree."""

    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    def test_why_not_reports_unavailable_when_execute_refuses(self, ads2, chart):
        with pytest.raises(ValidationError):
            ads2.execute(chart=chart)

        explanation = ads2.why_not(chart)
        assert 'IS available' not in explanation
        assert 'unavailable' in explanation.lower()

    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    def test_support_marks_unavailable_with_a_reason(self, ads2, chart):
        row = _primary_row(ads2, chart)
        assert row['available'] is False or row['available'] == False  # noqa: E712
        assert row['reason'], 'unavailable chart must carry a reason'

    @pytest.mark.parametrize('chart', ['Xbar', 'S', 'X', 'mR', 'Histogram'])
    def test_available_charts_actually_execute(self, ads2, chart):
        """Nothing marked available may raise — the contradiction in reverse."""
        row = _primary_row(ads2, chart)
        if not row['available']:
            pytest.skip(f'{chart} reported unavailable')
        kwargs = {'by': []} if chart in ('X', 'mR') else {}
        assert ads2.execute(chart=chart, **kwargs).charts


class TestEscapeHatchesInTheMessageWork:
    """Every alternative the message offers must be real.

    A refusal that suggests something that also fails is worse than one that
    suggests nothing — the first draft of this message recommended
    `by=[time]`, which raises on exactly this data.
    """

    def test_individuals_chart_works(self, ads2):
        assert ads2.execute(chart='X', by=[]).charts

    def test_pooled_subgroup_works(self, ads2):
        assert ads2.execute(chart='Xbar', by=['factor 1']).charts

    def test_residual_works(self, ads2):
        assert ads2.execute(chart='Xbar', value='R6', by=['factor 1']).charts

    def test_message_names_those_routes(self, ads2):
        message = ads2.why_not('Xbar')
        assert "chart='X'" in message
        assert 'by=' in message
        assert 'R6' in message


class TestValidChartsMembershipUnchanged:
    """`valid_charts` is chart-family validity and must not narrow.

    Narrowing it would break the legal ADS-2 paths, since
    `_validate_execute_request` gates on membership.
    """

    def test_xbar_still_listed_on_ads2(self, ads2):
        assert 'Xbar' in ads2.valid_charts
        assert 'S' in ads2.valid_charts

    def test_full_family_listed_on_ads2(self, ads2):
        assert set(ads2.valid_charts) == {'Histogram', 'Xbar', 'S', 'X', 'mR'}

    def test_residual_availability_unaffected(self, ads2):
        assert ('Xbar', 'R6') in ads2.residual_charts
        assert 'IS available' in ads2.why_not('Xbar', value='R6')


class TestReplicatedDataUnaffected:
    """ADS 1 has replication, so nothing here should change."""

    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    def test_available_and_executes(self, ads1, chart):
        assert 'IS available' in ads1.why_not(chart)
        assert _primary_row(ads1, chart)['available']
        assert ads1.execute(chart=chart).charts


class TestEnforcementMessagesUnchanged:
    """The execute-side text is pinned; extracting it must not have edited it."""

    def test_xbar_message(self, ads2):
        with pytest.raises(ValidationError) as exc:
            ads2.execute(chart='Xbar')
        assert str(exc.value) == (
            'No subgroups with n > 1 found — Xbar chart requires replicated observations.\n'
            'This data has Analytical Design State 2.\n'
            "Use chart='X' for individual values, or chart='Xbar' with value='R6' for effects analysis."
        )

    def test_s_message(self, ads2):
        with pytest.raises(ValidationError) as exc:
            ads2.execute(chart='S')
        assert str(exc.value) == (
            'No subgroups with n > 1 found — S chart requires replicated observations.\n'
            'This data has Analytical Design State 2.\n'
            "Use chart='X' for individual values."
        )


class TestHistogramQuestion:
    """`why_not('Histogram')` used to end in a trailing empty question."""

    def test_histogram_answer_carries_a_question(self, ads1):
        answer = ads1.why_not('Histogram')
        assert 'IS available' in answer
        assert not answer.rstrip().endswith('available.')
