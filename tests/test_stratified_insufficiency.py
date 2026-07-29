"""Stratified Xbar/S handle strata that have nothing to subgroup.

An Xbar or S chart needs n >= 2 within a subgroup. A stratum whose every subgroup is a
single observation contributes nothing, and the stratified paths dropped it with a bare
``continue`` while still publishing it in ``strata``. Three failures came out of that one
omission:

1. **Every stratum insufficient** — ``all_s_frames`` was empty and ``pd.concat([])``
   surfaced as ``ValueError: No objects to concatenate``, a pandas internal reaching the
   analyst. The ungrouped S path and the stratified *Xbar* path both already raised a
   self-diagnostic error here; stratified S was the gap.
2. **Some strata insufficient** — the dropped stratum stayed in ``result.strata`` with no
   row in ``data`` and no entry in ``statistics``, so ``result.focus(stratum)`` raised
   "No data found for stratum". That is exactly the contract
   ``_split_strata_by_sufficiency`` was written to keep on the X/mR path; Xbar/S never
   adopted it.
3. **Companion Xbar+S** — Xbar passed its raw stratum list through ``_intermediates``
   while ``per_stratum`` held only the computed ones, so S raised ``KeyError``.

The reference data cannot express case 2 — every stratum is sufficient in ADS 1 and
ADS 3 — so ``mixed`` below thins one factor combination down to a single observation per
cell and leaves the rest replicated.
"""

from pathlib import Path

import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.exceptions import ProcessBehaviorError, ValidationError

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'
THINNED = '1_1'  # the factor combination `mixed` strips of replication


@pytest.fixture(scope='module')
def df():
    return pd.read_csv(REFERENCE_CSV, na_values=['*'])


@pytest.fixture(scope='module')
def mixed(df):
    """One factor combination with no replication; the other seven keep theirs."""
    data = df[[*FACTORS, TIME, 'PM SDS 1']].copy()
    victim = (data['FACTOR 1'] == 1) & (data['FACTOR 2'] == 1)
    thinned = data[victim].groupby([TIME], as_index=False).head(1)
    return pd.concat([data[~victim], thinned], ignore_index=True)


@pytest.fixture(scope='module')
def mixed_study(mixed):
    return pb.formulate(mixed, response='PM SDS 1', factors=FACTORS, time=TIME)


@pytest.fixture(scope='module')
def no_replication_study(df):
    """ADS 2 — every cell is a single observation, so every stratum is insufficient."""
    study = pb.formulate(df, response='PM SDS 2', factors=FACTORS, time=TIME)
    assert study.analytical_design_state.sds == 2
    return study


def _meta(result, chart):
    return result.charts[chart]['metadata']


# ---------------------------------------------------------------------------
# 1. Every stratum insufficient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('chart', ['S', 'Xbar'])
def test_all_strata_insufficient_raises_a_library_error(no_replication_study, chart):
    """The regression: a raw pandas ValueError used to escape from the S path."""
    with pytest.raises(ProcessBehaviorError) as exc:
        no_replication_study.execute(chart=chart, by=[TIME])
    message = str(exc.value)
    assert 'requires replicated observations' in message
    assert chart in message


@pytest.mark.parametrize('chart', ['S', 'Xbar'])
def test_the_error_is_self_diagnostic(no_replication_study, chart):
    """It must say what the data is and what to do instead, not just that it failed."""
    with pytest.raises(ValidationError) as exc:
        no_replication_study.execute(chart=chart, by=[TIME])
    message = str(exc.value)
    assert 'Analytical Design State 2' in message
    assert 'no replication' in message
    assert "chart='X'" in message, 'should name a chart that does work on this data'


def test_error_names_the_unusable_strata(no_replication_study):
    with pytest.raises(ValidationError) as exc:
        no_replication_study.execute(chart='S', by=[TIME])
    assert 'All 8 strata were unusable' in str(exc.value)


def test_stratified_S_matches_the_ungrouped_S_message(no_replication_study):
    """Both refuse for the same reason, so they should read the same way."""
    with pytest.raises(ValidationError) as stratified:
        no_replication_study.execute(chart='S', by=[TIME])
    with pytest.raises(ValidationError) as ungrouped:
        no_replication_study.execute(chart='S', by=[])
    opening = 'No subgroups with n > 1 found — S chart requires replicated observations.'
    assert str(stratified.value).startswith(opening)
    assert str(ungrouped.value).startswith(opening)


# ---------------------------------------------------------------------------
# 2. Some strata insufficient — the focus() contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('chart', 'companion'), [('Xbar', False), ('S', False), ('S', True)],
)
def test_every_published_stratum_is_focusable(mixed_study, chart, companion):
    """The contract: if it is in `strata`, focus() must return data for it."""
    request = 'Xbar' if companion else chart
    result = mixed_study.execute(chart=request, by=[TIME], companion=companion)

    assert result.strata, 'nothing published at all'
    for stratum in result.strata:
        assert len(result.focus(stratum).get_chart(chart)) > 0, stratum


@pytest.mark.parametrize('chart', ['Xbar', 'S'])
def test_strata_statistics_and_data_agree(mixed_study, chart):
    """Three views of the same set; a stratum in one and not another is the defect."""
    result = mixed_study.execute(chart=chart, by=[TIME])
    stats = result.charts[chart]['statistics']
    rows = result.get_chart(chart)[_meta(result, chart)['stratify_col']]

    assert set(result.strata) == set(stats)
    assert set(result.strata) == set(rows.unique())


@pytest.mark.parametrize('chart', ['Xbar', 'S'])
def test_the_unusable_stratum_is_reported_not_hidden(mixed_study, chart):
    """Dropping it silently would understate the design; metadata records it."""
    result = mixed_study.execute(chart=chart, by=[TIME])
    assert _meta(result, chart)['insufficient_strata'] == [THINNED]
    assert THINNED not in result.strata


@pytest.mark.parametrize('chart', ['Xbar', 'S'])
def test_the_other_strata_are_all_published(mixed_study, chart):
    """One bad stratum must not cost the other seven."""
    result = mixed_study.execute(chart=chart, by=[TIME])
    assert len(result.strata) == 7


def test_companion_no_longer_raises_KeyError(mixed_study):
    """Xbar passed its raw stratum list to S, whose per_stratum lacked the skipped one."""
    result = mixed_study.execute(chart='Xbar', by=[TIME], companion=True)
    assert set(result.charts['Xbar']['strata']) == set(result.charts['S']['strata'])
    assert THINNED not in result.charts['S']['strata']


def test_companion_and_solo_agree(mixed_study):
    """Computing S alongside Xbar must not change which strata it publishes."""
    solo = mixed_study.execute(chart='S', by=[TIME])
    paired = mixed_study.execute(chart='Xbar', by=[TIME], companion=True)
    assert solo.charts['S']['strata'] == paired.charts['S']['strata']


# ---------------------------------------------------------------------------
# Fully-replicated data is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('response', ['PM SDS 1', 'PM SDS 3'])
@pytest.mark.parametrize('chart', ['Xbar', 'S'])
def test_sufficient_data_publishes_every_stratum(df, response, chart):
    """The common path: nothing is dropped and nothing is reported."""
    study = pb.formulate(df, response=response, factors=FACTORS, time=TIME)
    result = study.execute(chart=chart, by=[TIME])

    assert len(result.strata) == 8
    assert _meta(result, chart)['insufficient_strata'] is None
    for stratum in result.strata:
        assert len(result.focus(stratum).get_chart(chart)) > 0
