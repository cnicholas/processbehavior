"""A moving-range chart of residuals works standalone, not only as a companion.

``execute(chart='mR', value='R2')`` used to raise *"Chart type 'mR' (moving range) is not
supported for residual charts"* while ``execute(chart='mR', value='R2', companion=True)``
returned exactly the chart it claimed was unsupported. Three things said it was an
oversight rather than a methodology rule:

- ``_residual_pair_problem`` — the single source of truth for the chart x residual rule —
  already returned ``None`` for mR.
- ``_RESIDUAL_SOLO_STRATEGY_MAP`` omitted ``'mR'`` while ``_SOLO_STRATEGY_MAP`` carried
  ``'mR': '_calculate_r'``, under a comment claiming both tables held the same entries.
- ``why_not('mR', 'R2')`` answered "IS available", disagreeing with ``execute``.

A moving range is the absolute difference between consecutive values of whatever series is
plotted, so it is as well defined for a residual as for the response. The load-bearing test
here is :func:`test_solo_matches_the_companion_chart` — the numbers must be the ones the
companion path already produced, because this adds a route to an existing computation and
must not alter it.
"""

from pathlib import Path

import pandas as pd
import pandas.testing as pt
import pytest

import processbehavior as pb
from processbehavior.analysis import (
    _RESIDUAL_COMPANION_STRATEGY_MAP,
    _RESIDUAL_SOLO_STRATEGY_MAP,
    _SOLO_STRATEGY_MAP,
)

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'
RESIDUALS = ['R1', 'R2', 'R3', 'R4', 'R5']


@pytest.fixture(scope='module')
def df():
    return pd.read_csv(REFERENCE_CSV, na_values=['*'])


@pytest.fixture(scope='module')
def study(df):
    return pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)


# ---------------------------------------------------------------------------
# The numbers must not move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('value', RESIDUALS)
def test_solo_matches_the_companion_chart(study, value):
    """Same computation, reached a different way — so the output must be identical."""
    solo = study.execute(chart='mR', value=value, by=[])
    companion = study.execute(chart='mR', value=value, by=[], companion=True)

    pt.assert_frame_equal(
        solo.get_chart('mR').reset_index(drop=True),
        companion.get_chart('mR').reset_index(drop=True),
    )
    assert solo.get_statistics('mR') == companion.get_statistics('mR')


@pytest.mark.parametrize('value', RESIDUALS)
def test_solo_returns_only_the_mr_chart(study, value):
    """Solo means solo — the X chart appears only when asked for."""
    assert list(study.execute(chart='mR', value=value, by=[]).charts) == ['mR']
    assert list(study.execute(chart='mR', value=value, by=[], companion=True).charts) == ['X', 'mR']


def test_recentered_residuals_work_too(study):
    solo = study.execute(chart='mR', value='R5', by=[], recentered=True)
    companion = study.execute(chart='mR', value='R5', by=[], recentered=True, companion=True)
    pt.assert_frame_equal(
        solo.get_chart('mR').reset_index(drop=True),
        companion.get_chart('mR').reset_index(drop=True),
    )


@pytest.mark.parametrize('response', ['PM SDS 1', 'PM SDS 2', 'PM SDS 3'])
def test_available_in_every_design_state(df, response):
    """mR takes any residual regardless of ADS — unlike Xbar/S/X, which are ADS-gated."""
    study = pb.formulate(df, response=response, factors=FACTORS, time=TIME)
    for value in RESIDUALS:
        assert list(study.execute(chart='mR', value=value, by=[]).charts) == ['mR']


def test_the_response_mr_chart_is_unchanged(study):
    """Guards the dispatch edit: adding a residual route must not alter the response route."""
    result = study.execute(chart='mR', by=[])
    assert list(result.charts) == ['mR']
    assert len(result.get_chart('mR')) > 0


# ---------------------------------------------------------------------------
# why_not agreed with itself but not with execute
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('value', RESIDUALS)
def test_why_not_agrees_with_execute(study, value):
    """The disagreement that exposed the bug: why_not said available, execute raised."""
    explanation = study.why_not('mR', value)
    assert 'IS available' in explanation

    study.execute(chart='mR', value=value, by=[])  # must not raise


def test_why_not_no_longer_demands_companion(study):
    """It used to instruct the caller to pass companion=True to get anything at all."""
    explanation = study.why_not('mR', 'R2')
    assert 'IS available as a companion chart' not in explanation
    assert 'companion=True' in explanation, 'mentioning the companion option is still useful'


# ---------------------------------------------------------------------------
# The dispatch tables
# ---------------------------------------------------------------------------


def test_residual_solo_map_mirrors_the_response_solo_map():
    """The omission was invisible because nothing compared the two tables.

    Their comment claims residual context uses the same entry points; assert it.
    """
    assert set(_RESIDUAL_SOLO_STRATEGY_MAP) == set(_SOLO_STRATEGY_MAP)


def test_solo_and_companion_residual_maps_cover_the_same_charts():
    assert set(_RESIDUAL_SOLO_STRATEGY_MAP) == set(_RESIDUAL_COMPANION_STRATEGY_MAP)


def test_mr_dispatches_to_the_moving_range_calculator():
    assert _RESIDUAL_SOLO_STRATEGY_MAP['mR'] == '_calculate_r'
    assert _RESIDUAL_SOLO_STRATEGY_MAP['mR'] == _SOLO_STRATEGY_MAP['mR']


# ---------------------------------------------------------------------------
# residual_charts is not the whole rule
# ---------------------------------------------------------------------------


def test_mr_and_histogram_are_absent_from_residual_charts_yet_valid(study):
    """Pinned because a consumer gating on `residual_charts` alone hides both.

    This is exactly the trap the app's chart gating has to avoid, and the reason
    ``_residual_pair_problem`` treats these two charts as their own regime.
    """
    assert not any(chart == 'mR' for chart, _ in study.residual_charts)
    assert not any(chart == 'Histogram' for chart, _ in study.residual_charts)

    study.execute(chart='mR', value='R2', by=[])
    study.execute(chart='Histogram', value='R2', by=[])
