"""Stored vs request residuals — the two kinds, and one lookup that knows the difference.

The defect: a single result object gave three answers about R6. ``result.dataset['R6']`` held
4,000 values, ``result.residuals`` omitted it, and ``result.get_residual('R6')`` logged
"Residual 'R6' not found" and returned an **empty Series** — so a caller who did not check
``len()`` computed on nothing.

``result.residuals`` omitting R6 is correct and stays that way: R6 is a function of the
request's ``by=``, so there is no canonical study-level R6 to put in a study-level frame. What
was wrong was looking in only one place, and lying about the result.

Vocabulary under test (see spc_constants):
  stored  — R1-R5, RCR1-RCR5: computed at formulate(), independent of by=
  request — R6, RCR6: computed at execute() from by=, belonging to that one result
"""

from pathlib import Path

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.exceptions import ValidationError
from processbehavior.spc_constants import ALL_RESIDUALS, REQUEST_RESIDUALS, STORED_RESIDUALS

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'


@pytest.fixture(scope='module')
def study():
    df = pd.read_csv(REFERENCE_CSV, na_values=['*'])
    return ProcessBehavior(df).formulate(response='PM SDS 1', factors=FACTORS, time=TIME)


@pytest.fixture(scope='module')
def r6_result(study):
    """A result that actually asked for R6."""
    return study.execute(chart='Xbar', value='R6', by=['FACTOR 1'])


# ---------------------------------------------------------------------------
# The regression
# ---------------------------------------------------------------------------


def test_the_three_views_of_r6_agree(r6_result):
    """dataset, get_residual and plot_residuals must all see the same R6.

    Previously: 4000 values / omitted / empty-with-a-false-warning.
    """
    from_dataset = r6_result.dataset['R6']
    from_accessor = r6_result.get_residual('R6')

    assert len(from_accessor) == len(from_dataset) == 4000
    pd.testing.assert_series_equal(from_accessor, from_dataset, check_names=False)
    r6_result.plot_residuals('R6')  # must not raise


def test_result_residuals_still_excludes_r6(r6_result):
    """Deliberate, and a guard against a future 'helpful' fix.

    Putting R6 in this frame would assert a canonical study-level R6, which does not exist —
    the values depend on by=. See test_r6_depends_on_by.
    """
    assert list(r6_result.residuals.columns) == list(STORED_RESIDUALS)
    assert 'R6' not in r6_result.residuals.columns


# ---------------------------------------------------------------------------
# The property the whole vocabulary rests on
# ---------------------------------------------------------------------------


def test_stored_residuals_do_not_depend_on_by(study):
    """R5 is the same series whatever by= you pass — that is what makes it *stored*."""
    a = study.execute(chart='Xbar', value='R5', by=['FACTOR 1']).get_residual('R5')
    b = study.execute(chart='Xbar', value='R5', by=['FACTOR 2']).get_residual('R5')
    pd.testing.assert_series_equal(a, b)


def test_r6_depends_on_by(study):
    """R6 differs by factor — that is what makes it *request*-scoped.

    If this ever passes as equal, the two concepts have collapsed and the naming (plus
    result.residuals' exclusion of R6) is no longer justified.
    """
    a = study.execute(chart='Xbar', value='R6', by=['FACTOR 1']).get_residual('R6')
    b = study.execute(chart='Xbar', value='R6', by=['FACTOR 2']).get_residual('R6')
    assert not a.equals(b)


# ---------------------------------------------------------------------------
# The formula — pins R6/RCR6 numerically, bit for bit
# ---------------------------------------------------------------------------


def test_r6_matches_hand_computed_alpha_plus_r2(study):
    """R6 = α_i + R2 where α_i = mean(R5 | factor level). Exact, no tolerance.

    Written against the study-level frame so it holds regardless of *where* the
    library computes R6 — it pins the numbers themselves.
    """
    result = study.execute(chart='Xbar', value='R6', by=['FACTOR 1'])
    ds = study.dataset
    alpha = ds.groupby('FACTOR 1')['R5'].transform('mean')
    expected = alpha + ds['R2']
    pd.testing.assert_series_equal(result.get_residual('R6'), expected, check_names=False)


def test_rcr6_matches_hand_computed_reconstruction(study):
    """RCR6 = Ybar + α_i + R2, in that exact expression order.

    Float addition is non-associative: (Ybar + α) + R2 differs from Ybar + (α + R2)
    in the last ulp. The library evaluates left-to-right from Ybar; so does this.
    """
    result = study.execute(chart='Xbar', value='R6', by=['FACTOR 1'], recentered=True)
    ds = study.dataset
    alpha = ds.groupby('FACTOR 1')['R5'].transform('mean')
    expected = ds['Ybar'] + alpha + ds['R2']
    pd.testing.assert_series_equal(result.get_residual('RCR6'), expected, check_names=False)


def test_r6_multi_factor_groupby_matches_hand_computed(study):
    """by= with two factors groups on the list, not on either factor alone."""
    result = study.execute(chart='Xbar', value='R6', by=['FACTOR 1', 'FACTOR 2'])
    ds = study.dataset
    alpha = ds.groupby(['FACTOR 1', 'FACTOR 2'])['R5'].transform('mean')
    expected = alpha + ds['R2']
    pd.testing.assert_series_equal(result.get_residual('R6'), expected, check_names=False)


# ---------------------------------------------------------------------------
# The staleness trap
# ---------------------------------------------------------------------------


def test_a_result_never_returns_another_requests_r6(study):
    """Computing R6 writes the column onto the study-level dataset, so a result created
    afterwards inherits an R6 column it never asked for.

    Returning it would be worse than the original bug: silently wrong numbers from a
    different by= rather than an empty Series. The lookup gates on this result's own chart
    metadata, so it must raise here even though the column is right there.
    """
    study.execute(chart='Xbar', value='R6', by=['FACTOR 1'])
    plain = study.execute(chart='Xbar')

    assert 'R6' in plain.dataset.columns, 'precondition: the stale column is present'
    with pytest.raises(ValidationError, match='computed per request'):
        plain.get_residual('R6')


# ---------------------------------------------------------------------------
# Error cases are distinct and actionable
# ---------------------------------------------------------------------------


def test_unknown_code_names_the_valid_ones(r6_result):
    with pytest.raises(ValidationError, match='not a recognized residual'):
        r6_result.get_residual('R99')


def test_request_residual_error_says_how_to_get_it(study):
    plain = study.execute(chart='Xbar')
    with pytest.raises(ValidationError) as exc:
        plain.get_residual('R6')
    message = str(exc.value)
    assert 'by=' in message
    assert "value='R6'" in message, 'the message should show the call that produces it'


def test_no_lookup_returns_an_empty_series(study, r6_result):
    """The silent-wrongness path is gone: every outcome is data or an exception."""
    for result, code in ((r6_result, 'R2'), (r6_result, 'R6'), (r6_result, 'RCR2')):
        assert len(result.get_residual(code)) > 0


# ---------------------------------------------------------------------------
# Stored recentered forms, and drill-down
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('code', ['R1', 'R2', 'R3', 'R4', 'R5', 'RCR1', 'RCR2', 'RCR5'])
def test_stored_and_recentered_forms_are_reachable(study, code):
    """RCR1-RCR5 live on the dataset rather than the snapshot; both are 'stored'."""
    assert len(study.execute(chart='Xbar').get_residual(code)) > 0


def test_rcr6_reachable_on_a_recentered_request(study):
    result = study.execute(chart='Xbar', value='R6', by=['FACTOR 1'], recentered=True)
    assert len(result.get_residual('RCR6')) > 0


def test_lookup_works_on_a_focused_result(study):
    """Stratified drill-down returns a FocusedAnalysisResult with a filtered dataset."""
    stratified = study.execute(chart='Xbar', by=[TIME])
    if not stratified.is_stratified:
        pytest.skip('expected a stratified result for this grouping')
    focused = stratified.focus(stratified.strata[0])
    r2 = focused.get_residual('R2')
    assert len(r2) > 0
    assert len(r2) < len(study.dataset), 'a focused result should carry only its stratum'


# ---------------------------------------------------------------------------
# The named sets
# ---------------------------------------------------------------------------


def test_the_two_sets_are_disjoint_and_complete():
    """Catches a future R7 being added to only one of them."""
    stored, request = set(STORED_RESIDUALS), set(REQUEST_RESIDUALS)
    assert not stored & request, 'a residual cannot be both stored and request-scoped'
    assert stored | request == set(ALL_RESIDUALS)
