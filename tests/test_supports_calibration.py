"""``Study.supports_calibration`` must agree with ``execute()`` on every request.

The app hand-wrote this predicate because the rule was private, and its copy was wrong
on a row nobody would guess: Xbar with ``by=[time]`` is refused for the **response** and
accepted for a **residual**, because Xbar/S only stratify on time when charting the
response (``analysis.py`` ``_resolve_by_grouping``). That asymmetry is why the answer
belongs in the library.

The load-bearing test here is :func:`test_prediction_matches_execution`, a biconditional
over the chart x by x value grid: for every executable request, ``supports_calibration()``
is True exactly when ``execute(calibration=...)`` does not raise. It fails if either side
changes without the other, which is the drift this method exists to prevent.
"""

from pathlib import Path

import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.calibration import Calibration
from processbehavior.exceptions import (
    CalibrationNotSupportedError,
    ProcessBehaviorError,
    ValidationError,
)

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'
CAL = Calibration(label='ref', mean=10.0, sigma=1.0)


@pytest.fixture(scope='module')
def df():
    return pd.read_csv(REFERENCE_CSV, na_values=['*'])


@pytest.fixture(scope='module')
def study(df):
    return pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)


def _requests():
    """The chart x by x value grid, as kwargs for both execute() and the predicate."""
    for chart in ('Xbar', 'S', 'X', 'mR'):
        for by in (None, [], ['FACTOR 1'], FACTORS, [TIME], FACTORS + [TIME]):
            for value in (None, 'R5', 'R2'):
                for recentered in ((False, True) if value else (False,)):
                    for phased in ((False, True) if chart in ('X', 'mR') else (False,)):
                        kwargs = dict(
                            chart=chart, by=by, recentered=recentered, phased=phased,
                        )
                        if value is not None:
                            kwargs['value'] = value
                        yield kwargs


# ---------------------------------------------------------------------------
# The biconditional
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('response', ['PM SDS 1', 'PM SDS 2', 'PM SDS 3'])
def test_prediction_matches_execution(df, response):
    """Prediction == outcome, for every executable request in the grid.

    Runs across all three analytical design states, because which requests are even
    executable — and which stratify — changes with ADS.
    """
    study = pb.formulate(df, response=response, factors=FACTORS, time=TIME)
    checked = 0
    mismatches = []

    for kwargs in _requests():
        try:
            study.execute(**kwargs)
        except Exception:
            # Not executable at all, so there is no calibration question to answer.
            # Broad on purpose: one request in this grid still escapes as a raw pandas
            # ValueError — see test_stratified_S_with_no_sufficient_strata below.
            continue

        predicted = study.supports_calibration(**kwargs)
        try:
            study.execute(calibration=CAL, **kwargs)
            actual = True
        except CalibrationNotSupportedError:
            actual = False

        checked += 1
        if predicted != actual:
            mismatches.append(f'{kwargs} predicted={predicted} actual={actual}')

    assert not mismatches, 'predicate disagrees with execute():\n' + '\n'.join(mismatches)
    assert checked > 20, f'grid collapsed to {checked} executable requests — it is not testing much'


# ---------------------------------------------------------------------------
# The specific rows a hand-written copy gets wrong
# ---------------------------------------------------------------------------


def test_response_and_residual_differ_on_by_time(study):
    """The row the app's copy got wrong, pinned on its own.

    Xbar/S stratify by factor combination when grouping on time — but only for the
    response. Charting a residual over time is a single chart, so it calibrates.
    """
    assert study.supports_calibration(chart='Xbar', by=[TIME]) is False
    assert study.supports_calibration(chart='Xbar', by=[TIME], value='R5') is True
    assert study.supports_calibration(chart='S', by=[TIME]) is False
    assert study.supports_calibration(chart='S', by=[TIME], value='R5') is True


def test_xbar_grouped_paths_calibrate(study):
    """Grouping is not stratifying: Xbar/S on factors stay one chart."""
    assert study.supports_calibration(chart='Xbar', by=[]) is True
    assert study.supports_calibration(chart='Xbar', by=['FACTOR 1']) is True
    assert study.supports_calibration(chart='Xbar', by=FACTORS) is True
    assert study.supports_calibration(chart='Xbar', by=FACTORS + [TIME]) is True


def test_xmr_stratifies_on_any_non_empty_by(study):
    """X/mR read `by` the other way — anything non-empty splits into separate charts."""
    assert study.supports_calibration(chart='X', by=[]) is True
    assert study.supports_calibration(chart='X', by=['FACTOR 1']) is False
    assert study.supports_calibration(chart='X', by=FACTORS) is False
    assert study.supports_calibration(chart='mR', by=[]) is True
    assert study.supports_calibration(chart='mR', by=['FACTOR 1']) is False


def test_phased_xmr_is_refused_only_when_collapsing(study):
    """`by=[]` collapses factors into the sequence, which is what phasing acts on."""
    assert study.supports_calibration(chart='X', by=[], phased=False) is True
    assert study.supports_calibration(chart='X', by=[], phased=True) is False


def test_companion_does_not_change_the_answer(study):
    """Xbar+S together must agree with Xbar alone — they share the grouping."""
    for by in ([], [TIME], FACTORS):
        assert study.supports_calibration(chart='Xbar', by=by, companion=True) == \
            study.supports_calibration(chart='Xbar', by=by)


# ---------------------------------------------------------------------------
# The typed exception
# ---------------------------------------------------------------------------


def test_raises_the_typed_exception(study):
    with pytest.raises(CalibrationNotSupportedError) as exc:
        study.execute(chart='X', by=['FACTOR 1'], calibration=CAL)
    assert exc.value.context == 'stratified X/mR'


def test_context_names_the_path(study):
    with pytest.raises(CalibrationNotSupportedError) as exc:
        study.execute(chart='Xbar', by=[TIME], calibration=CAL)
    assert exc.value.context == 'stratified Xbar'

    with pytest.raises(CalibrationNotSupportedError) as exc:
        study.execute(chart='X', by=[], phased=True, calibration=CAL)
    assert exc.value.context == 'phased X/mR'


def test_still_a_ValidationError(study):
    """Existing handlers catch ValidationError; the new type must not escape them."""
    with pytest.raises(ValidationError):
        study.execute(chart='X', by=['FACTOR 1'], calibration=CAL)


def test_message_still_matches_the_old_text(study):
    """A client matching the message (as the app does today) keeps working."""
    with pytest.raises(CalibrationNotSupportedError) as exc:
        study.execute(chart='X', by=['FACTOR 1'], calibration=CAL)
    assert 'calibration is not supported' in str(exc.value)
    assert 'supports_calibration' in str(exc.value), 'the error should name the predicate'


def test_exported_from_the_package():
    assert pb.CalibrationNotSupportedError is CalibrationNotSupportedError


# ---------------------------------------------------------------------------
# Honest failure on unanswerable questions
# ---------------------------------------------------------------------------


def test_unexecutable_request_raises_rather_than_returning_False(study):
    """X/mR with factors and no `by` is not a chart, so it is not a calibration question."""
    with pytest.raises(ProcessBehaviorError):
        study.supports_calibration(chart='X', by=None)


def test_unknown_chart_raises(study):
    with pytest.raises(ProcessBehaviorError):
        study.supports_calibration(chart='NotAChart', by=[])


def test_stratify_by_alias_agrees_with_by(study):
    assert study.supports_calibration(chart='X', stratify_by=['FACTOR 1']) is False
    assert study.supports_calibration(chart='X', stratify_by=['FACTOR 1']) == \
        study.supports_calibration(chart='X', by=['FACTOR 1'])
    with pytest.raises(ValidationError, match='not both'):
        study.supports_calibration(chart='X', by=[], stratify_by=['FACTOR 1'])


# ---------------------------------------------------------------------------
# Surfaced by the grid; not this change's to fix
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason='Pre-existing: stratified S with zero sufficient strata concatenates an empty '
           'list and pandas raises. Verified present at a99a534, before any of this work.',
    strict=True,
)
def test_stratified_S_with_no_sufficient_strata(df):
    """A raw ``ValueError: No objects to concatenate`` reaches the analyst.

    ADS 2 has no replication, so every stratum fails the subgroup-size check and the
    stratified S path is left with nothing to concatenate. Whatever the right answer is
    — an empty result or a self-diagnostic error — a pandas internal message is not it.
    Flipping to a pass means someone fixed it; update this test then.
    """
    study = pb.formulate(df, response='PM SDS 2', factors=FACTORS, time=TIME)
    assert study.analytical_design_state.sds == 2
    with pytest.raises(ProcessBehaviorError):
        study.execute(chart='S', by=[TIME])


def test_predicate_does_not_execute_the_analysis(study, monkeypatch):
    """It must be cheap enough to call while rendering a control."""
    from processbehavior.analysis import Analysis

    def boom(self):
        raise AssertionError('supports_calibration ran the analysis')

    monkeypatch.setattr(Analysis, 'calculate', boom)
    assert study.supports_calibration(chart='X', by=[]) is True
