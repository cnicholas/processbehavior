"""Three additive ergonomics affordances for the notebook path.

All three came out of reading the public surface as an analyst meeting it cold:

1. ``pb.formulate(df, ...)`` — the advertised idiom, ``pb.formulate(...).execute().plot()``,
   was not actually available at module level. The docs bind ``pb = ProcessBehavior(df)``, so
   ``pb.formulate`` was an *instance* method shadowing the conventional module alias.
2. ``result.statistics()`` — ``get_statistics()`` returns ``{'N','center','lpl','upl'}`` for an
   unstratified result and ``{stratum: {...}}`` for a stratified one, so ``stats['center']``
   works on one dataset and raises ``KeyError`` on the next.
3. ``execute(stratify_by=...)`` — ``by=`` composes subgroups for Xbar/S but splits into
   separate charts for X/mR. The alias puts the intent in the call site.

Nothing existing changes behaviour; these tests pin that as much as the new surface.
"""

from pathlib import Path

import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.exceptions import ValidationError

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'


@pytest.fixture(scope='module')
def df():
    return pd.read_csv(REFERENCE_CSV, na_values=['*'])


@pytest.fixture(scope='module')
def study(df):
    return pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)


# ---------------------------------------------------------------------------
# 1. The one-call entry point
# ---------------------------------------------------------------------------


def test_the_advertised_chain_runs_end_to_end(df):
    """`pb.formulate(df, ...).execute().plot()` — the whole idiom in one expression."""
    figure = pb.formulate(df, response='PM SDS 1', factors=['FACTOR 1'], time=TIME).execute().plot()
    assert figure.figure is not None


def test_formulate_is_exported(df):
    assert 'formulate' in pb.__all__


def test_equivalent_to_the_two_step_form(df, study):
    """The free function must be a pure convenience — same Study, not a different one."""
    two_step = pb.ProcessBehavior(df).formulate(response='PM SDS 1', factors=FACTORS, time=TIME)
    assert study.response == two_step.response
    assert study.factors == two_step.factors
    assert study.analytical_design_state.sds == two_step.analytical_design_state.sds
    pd.testing.assert_frame_equal(study.dataset, two_step.dataset)


def test_positional_data_argument(df):
    """Data comes first and positionally — the shape every dataframe API uses."""
    assert pb.formulate(df, 'PM SDS 1', ['FACTOR 1'], TIME).response == 'PM SDS 1'


def test_still_reachable_through_ProcessBehavior_for_derivations(df):
    """The class stays the path for the fluent verbs, which must attach before formulating."""
    study = pb.ProcessBehavior(df).bin('PM SDS 1', method='equal_freq', n=3, label='band').formulate(
        response='PM SDS 2', factors=['band'], time=TIME,
    )
    assert 'band' in study.dataset.columns


# ---------------------------------------------------------------------------
# 2. Type-stable statistics
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {'N', 'center', 'lpl', 'upl'}


def test_same_keys_whether_or_not_stratified(study):
    """The whole point: one shape, regardless of what the data did."""
    flat = study.execute(chart='Xbar', by=[])
    strat = study.execute(chart='Xbar', by=[TIME])
    assert not flat.is_stratified and strat.is_stratified

    assert set(flat.statistics('Xbar')) >= EXPECTED_KEYS
    assert set(strat.statistics('Xbar', stratum=strat.strata[0])) >= EXPECTED_KEYS
    assert isinstance(flat.statistics('Xbar')['center'], float)


def test_stratified_without_a_stratum_raises_and_names_them(study):
    """Rather than returning a differently-shaped dict, which is the defect being fixed."""
    strat = study.execute(chart='Xbar', by=[TIME])
    with pytest.raises(ValidationError) as exc:
        strat.statistics('Xbar')
    message = str(exc.value)
    assert 'stratum is required' in message
    assert strat.strata[0] in message, 'the error should name what to pass'
    assert 'get_statistics' in message, 'and point at the all-strata alternative'


def test_stratum_on_an_unstratified_result_raises(study):
    """A call naming a stratum must never silently return whole-result numbers."""
    flat = study.execute(chart='Xbar', by=[])
    with pytest.raises(ValidationError, match='not stratified'):
        flat.statistics('Xbar', stratum='anything')


def test_unknown_stratum_raises(study):
    strat = study.execute(chart='Xbar', by=[TIME])
    with pytest.raises(ValidationError, match='not found'):
        strat.statistics('Xbar', stratum='no-such-stratum')


def test_values_agree_with_get_statistics(study):
    """Same numbers, different packaging — this is an accessor, not a recomputation."""
    flat = study.execute(chart='Xbar', by=[])
    assert flat.statistics('Xbar') == flat.get_statistics('Xbar')

    strat = study.execute(chart='Xbar', by=[TIME])
    stratum = strat.strata[0]
    assert strat.statistics('Xbar', stratum=stratum) == dict(strat.get_statistics('Xbar')[stratum])


def test_get_statistics_behaviour_is_unchanged(study):
    """The old accessor keeps its value-dependent shape; this change is purely additive."""
    assert set(study.execute(chart='Xbar', by=[]).get_statistics('Xbar')) >= EXPECTED_KEYS
    strat = study.execute(chart='Xbar', by=[TIME])
    assert set(strat.get_statistics('Xbar')) == set(strat.strata)


# ---------------------------------------------------------------------------
# 3. stratify_by
# ---------------------------------------------------------------------------


def test_stratify_by_matches_by(study):
    by_result = study.execute(chart='X', by=['FACTOR 1'])
    alias_result = study.execute(chart='X', stratify_by=['FACTOR 1'])
    assert alias_result.strata == by_result.strata
    assert alias_result.is_stratified


def test_passing_both_raises(study):
    """They set the same parameter, so accepting both would hide a contradiction."""
    with pytest.raises(ValidationError, match="not both"):
        study.execute(chart='X', by=['FACTOR 1'], stratify_by=['FACTOR 2'])


def test_by_still_works_unchanged(study):
    """Existing calls must be untouched — the alias is additive."""
    assert study.execute(chart='Xbar', by=['FACTOR 1']).is_stratified is False
    assert study.execute(chart='X', by=['FACTOR 1']).is_stratified is True


def test_the_dual_semantics_still_hold(study):
    """Documenting the behaviour the alias exists to make legible, not to change.

    `by` composes the subgroup for Xbar/S (one chart) and splits for X/mR (many). That
    asymmetry is deliberate — subgrouping and stratification are different operations.
    """
    assert study.execute(chart='Xbar', by=['FACTOR 1']).is_stratified is False
    assert study.execute(chart='X', stratify_by=['FACTOR 1']).is_stratified is True
