"""Residual chart titles carry a readable label for every residual code.

`_generate_title` resolves a residual code through `_RESIDUAL_LABELS` with
``.get(code, code)``, so a missing key silently degrades to the code itself and renders
as ``"R6 (R6)"``. That is exactly how R6 shipped unlabeled, and how R1 shipped carrying
R2's concept ("Within-Subgroup") — nothing in the suite asserted on this map, because
every other title test passes an explicit ``title=`` and takes the override path.

These tests close that gap: they drive the real allowlist, so a residual can neither be
added without a label nor labeled with the wrong concept.
"""

from pathlib import Path

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.plotting.plotter import _RESIDUAL_LABELS

pytestmark = pytest.mark.plotting

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
ALL_CODES = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6'}


@pytest.fixture(scope='module')
def study():
    """ADS 1 study from Bishop's reference data — all six residuals available."""
    df = pd.read_csv(REFERENCE_CSV, na_values=['*'])
    return ProcessBehavior(df).formulate(
        response='PM SDS 1', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME',
    )


def _title(study, chart, value):
    """Auto-generated title for a residual chart (no `title=` — that path bypasses the map)."""
    kwargs = {'chart': chart, 'value': value}
    if value in ('R5', 'R6'):
        kwargs['by'] = ['FACTOR 1']  # factor-effect residuals need an explicit factor
    elif chart in ('X', 'mR'):
        kwargs['by'] = []  # X/mR with factors require an explicit by (single overall chart)
    result = study.execute(**kwargs)
    if result.is_stratified:
        result = result.focus(result.strata[0])
    return result.plot(chart=chart).figure.layout.title.text


# ---------------------------------------------------------------------------
# The map itself
# ---------------------------------------------------------------------------


def test_every_residual_code_has_a_label():
    """A residual added without a label renders as 'Rn (Rn)'. Fail here instead."""
    assert set(_RESIDUAL_LABELS) == ALL_CODES


def test_labels_are_distinct_and_non_empty():
    labels = list(_RESIDUAL_LABELS.values())
    assert all(labels), 'every residual needs a non-empty label'
    assert len(set(labels)) == len(labels), f'duplicate residual labels: {labels}'


def test_no_label_is_just_its_own_code():
    """Guards the exact `.get(code, code)` fallback that produced 'R6 (R6)'."""
    for code, label in _RESIDUAL_LABELS.items():
        assert label != code, f'{code} falls back to its own code'


# ---------------------------------------------------------------------------
# Rendered titles
# ---------------------------------------------------------------------------


def test_no_rendered_title_repeats_its_code(study):
    """The single assertion that would have caught the R6 bug.

    Drives `study.residual_charts` rather than a hardcoded list, so a newly-chartable
    residual is covered automatically.
    """
    checked = set()
    for chart, value in study.residual_charts:
        title = _title(study, chart, value)
        assert f'{value} ({value})' not in title, f'{chart}/{value} renders as {title!r}'
        checked.add(value)
    assert checked == ALL_CODES, f'expected all six residuals to be chartable, got {checked}'


def test_r1_is_not_labeled_with_r2s_concept(study):
    """Regression guard: R1 read 'Within-Subgroup', which is R2's quantity.

    Bishop 13.1 defines R1 = Y_ktn - Ybar.. — the response centred at 0, a location
    shift. 'Within-*' anything is the wrong concept.
    """
    title = _title(study, 'Xbar', 'R1')
    assert 'Within' not in title, f'R1 labeled with a within-subgroup concept: {title!r}'
    assert _RESIDUAL_LABELS['R1'] in title


def test_explicit_title_still_overrides_the_map(study):
    """The override path short-circuits before the map — must stay that way."""
    result = study.execute(chart='Xbar', value='R1')
    fig = result.plot(chart='Xbar', title='Custom')
    assert fig.figure.layout.title.text == 'Custom'


# ---------------------------------------------------------------------------
# R1's deliberate location-only exposure
# ---------------------------------------------------------------------------


def test_r1_is_chartable_on_location_charts_only(study):
    """R1 is the response shifted by a constant, so S/mR of R1 would duplicate the
    response's dispersion charts exactly. Xbar/X only — locked so the asymmetry isn't
    'helpfully' completed later."""
    r1_charts = {chart for chart, value in study.residual_charts if value == 'R1'}
    assert r1_charts == {'Xbar', 'X'}


def test_r1_accessor_and_allowlist_agree(study):
    """`study.residuals` advertised R1 while `residual_charts` omitted it — the
    inconsistency that surfaced the mislabel. They must not diverge again."""
    assert 'R1' in list(study.residuals)
    assert any(value == 'R1' for _chart, value in study.residual_charts)


def test_r1_is_the_response_centered_at_zero(study):
    """The label is a claim about the arithmetic; assert the arithmetic."""
    data = study.dataset
    expected = data['PM SDS 1'] - data['PM SDS 1'].mean()
    pd.testing.assert_series_equal(data['R1'], expected, check_names=False)
