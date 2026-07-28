"""`why_not()` must answer from the same rule `execute()` enforces.

The defect this file exists to prevent: `execute(chart='Xbar', value='R2')` raised a correct
error ending "Use study.why_not('Xbar', value='R2') for details", and that call replied
"'Xbar' (R2) is not a recognized chart type" — false, since Xbar is recognized — then referred
on to `study.support`, which had no row for the pair at all. A two-hop referral that dead-ended
in a wrong statement, at the exact moment the library's teaching surface was invoked.

Cause: `why_not` answered only from `support`, whose residual rows come from a hand-maintained
list of *potentially valid* pairs, so "no row" was conflated with "unknown chart".

Both callers now route through `Study._residual_pair_problem`. These tests assert the two can
never disagree again — not by sampling, but over the full chart x residual grid.
"""

from pathlib import Path

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.exceptions import ChartNotAvailableError
from processbehavior.spc_constants import VALID_BASE_CHARTS
from processbehavior.study import RESIDUAL_CODES, _base_residual_code

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'


@pytest.fixture(scope='module')
def t100():
    return pd.read_csv(REFERENCE_CSV, na_values=['*'])


@pytest.fixture(scope='module')
def study(t100):
    """ADS 1 — every residual available."""
    return ProcessBehavior(t100).formulate(response='PM SDS 1', factors=FACTORS, time=TIME)


def _execute_kwargs(chart, code):
    """Arguments that isolate the chart x residual rule from unrelated requirements.

    X/mR with factors need an explicit `by`; mR residuals need `companion`; R5/R6 are
    factor-effect residuals and need a factor named. Without these, a pair could raise for
    a reason that has nothing to do with the rule under test.
    """
    kwargs = {'chart': chart, 'value': code}
    if code in ('R5', 'R6'):
        kwargs['by'] = ['FACTOR 1']
    elif chart in ('X', 'mR'):
        kwargs['by'] = []
    if chart == 'mR':
        kwargs['companion'] = True
    return kwargs


# ---------------------------------------------------------------------------
# The regression this file is named for
# ---------------------------------------------------------------------------


def test_the_round_trip_closes(study):
    """Follow the error's own instruction and get a consistent answer.

    This is the test that would have caught the defect: it does exactly what a user does —
    reads the error, calls the method it names, and reads the reply.
    """
    with pytest.raises(ChartNotAvailableError) as exc:
        study.execute(chart='Xbar', value='R2')
    error_text = str(exc.value)
    assert "study.why_not('Xbar', value='R2')" in error_text, 'error should refer the user onward'

    answer = study.why_not('Xbar', value='R2')

    assert 'not a recognized chart type' not in answer, 'Xbar IS a recognized chart type'
    assert 'Valid charts for R2' in answer
    assert 'S' in answer and 'X' in answer
    # The referral must not contradict what sent the user there.
    assert answer.splitlines()[0] == error_text.splitlines()[0]


# ---------------------------------------------------------------------------
# The invariant: one rule, two callers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('response,expected_ads', [('PM SDS 1', 1), ('PM SDS 2', 2), ('PM SDS 3', 3)])
def test_execute_and_why_not_agree_across_the_whole_grid(t100, response, expected_ads):
    """For every chart x residual pair, `execute()` raising must coincide exactly with
    `_residual_pair_problem()` reporting a problem.

    Asserted over the full grid rather than sampled, and across all three design states,
    because `residual_charts` varies by ADS — a rule that agrees at ADS 1 and diverges at
    ADS 3 is exactly the failure this guards.
    """
    study = ProcessBehavior(t100).formulate(response=response, factors=FACTORS, time=TIME)
    assert study.analytical_design_state.sds == expected_ads

    available = set(study.residuals)
    checked = 0
    for chart in sorted(VALID_BASE_CHARTS):
        for code in RESIDUAL_CODES:
            if code not in available:
                continue  # residual absent for this design state — a different rule
            predicate_says_problem = study._residual_pair_problem(chart, code) is not None
            try:
                study.execute(**_execute_kwargs(chart, code))
                execute_raised = False
            except ChartNotAvailableError:
                execute_raised = True
            except Exception:
                # Rejected by a rule other than the chart x residual pairing — e.g. an S
                # chart at ADS 2 has no replicated subgroups. The predicate governs pair
                # validity only, so those are out of scope here and must not count as
                # a pair rejection.
                execute_raised = False
            assert predicate_says_problem == execute_raised, (
                f'disagreement for ({chart}, {code}) at ADS {expected_ads}: '
                f'predicate={predicate_says_problem} execute_raised={execute_raised}'
            )
            checked += 1
    assert checked >= 20, f'expected a meaningful grid, only checked {checked} pairs'


def test_why_not_never_contradicts_execute(study):
    """Whatever `execute()` rejects, `why_not()` must also describe as unavailable."""
    for chart in sorted(VALID_BASE_CHARTS):
        for code in RESIDUAL_CODES:
            answer = study.why_not(chart, value=code)
            rejected = study._residual_pair_problem(chart, code) is not None
            if rejected:
                assert 'IS available' not in answer, f'({chart},{code}) rejected but why_not says available'
            else:
                assert 'IS available' in answer, f'({chart},{code}) allowed but why_not says otherwise'


# ---------------------------------------------------------------------------
# Distinguishing the cases that used to be conflated
# ---------------------------------------------------------------------------


def test_unrecognized_chart_says_so_and_names_the_valid_ones(study):
    answer = study.why_not('Bogus')
    assert 'not a recognized chart type' in answer
    for chart in VALID_BASE_CHARTS:
        assert chart in answer, 'the message should name what IS valid'


def test_unrecognized_residual_is_distinct_from_unrecognized_chart(study):
    """'R9' is a bad residual, not a bad chart — these used to give the same wrong message."""
    answer = study.why_not('Xbar', value='R9')
    assert 'not a recognized residual' in answer
    assert 'not a recognized chart type' not in answer
    assert 'R6' in answer, 'should name the valid residual codes'


def test_recognized_chart_with_invalid_pairing_is_not_called_unrecognized(study):
    """The exact conflation that caused the defect."""
    answer = study.why_not('Xbar', value='R2')
    assert 'not a recognized' not in answer
    assert 'is not valid for' in answer


# ---------------------------------------------------------------------------
# The three regimes
# ---------------------------------------------------------------------------


def test_histogram_accepts_any_residual(study):
    for code in RESIDUAL_CODES:
        assert 'IS available' in study.why_not('Histogram', value=code)


def test_mr_reports_available_and_names_the_companion_requirement(study):
    answer = study.why_not('mR', value='R2')
    assert 'IS available' in answer
    assert 'companion=True' in answer, 'mR residuals need companion; the answer should say so'


def test_s_chart_of_r1_is_rejected_by_both(study):
    """R1 is a location shift, so its S chart duplicates the response's exactly.

    Previously `execute` exempted R1 from the pair check while `residual_charts` excluded it —
    the one place the two authorities genuinely disagreed. Now both reject it.
    """
    with pytest.raises(ChartNotAvailableError) as exc:
        study.execute(chart='S', value='R1')
    assert 'Valid charts for R1' in str(exc.value)
    answer = study.why_not('S', value='R1')
    assert 'is not valid for' in answer
    assert 'Xbar' in answer and 'X' in answer


def test_location_charts_of_r1_still_work(study):
    """The narrowing must not catch the pairings R1 is actually for."""
    for chart in ('Xbar', 'X'):
        kwargs = {'chart': chart, 'value': 'R1'}
        if chart == 'X':
            kwargs['by'] = []
        study.execute(**kwargs)
        assert 'IS available' in study.why_not(chart, value='R1')


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_recentered_and_lowercase_forms_normalise():
    assert _base_residual_code('RCR5') == 'R5'
    assert _base_residual_code('r2') == 'R2'
    assert _base_residual_code('R6') == 'R6'


def test_why_not_accepts_recentered_and_lowercase(study):
    """`execute` upper-cases before comparing; `why_not` now normalises identically."""
    assert 'IS available' in study.why_not('Xbar', value='RCR5')
    assert study.why_not('Xbar', value='r2') == study.why_not('Xbar', value='R2').replace('R2', 'r2', 1)
