"""Any permutation of the rational subgroup names the same analysis.

``by=`` carries two meanings at once, and that is the point:

- **which analysis** — decided by the *set* of columns. Every spelling of the rational
  subgroup (``by=[]``, ``by=[f1, f2, time]``, ``by=[time, f2, f1]``) is the same grouping.
- **how it reads** — decided by the *order*, where the resolver preserves it. ``by=[f1,f2]``
  groups f1-major, ``by=[f2,f1]`` groups f2-major. Same statistics, presented the way the
  analyst asked.

This is not free, and that is what these tests protect. Permutations cross **different code
paths**: ``by=[f1,f2]`` in ``rsg_vars`` order resolves to the precomputed ``Ybar_k`` column,
while ``by=[f2,f1]`` falls through to runtime aggregation. ``tests/test_resolve_by_grouping``
asserts which path each spelling takes; nothing asserted the paths *agree numerically*. A
change to either one could silently make the same analysis return different answers
depending on how the analyst happened to type it.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.analysis import Analysis
from processbehavior.formulation_spec import ChartRequest

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
F1, F2 = 'FACTOR 1', 'FACTOR 2'
TIME = 'PRODUCTION TIME'
FACTORS = [F1, F2]


@pytest.fixture(scope='module')
def study():
    df = pd.read_csv(REFERENCE_CSV, na_values=['*'])
    return pb.formulate(df, response='PM SDS 1', factors=FACTORS, time=TIME)


def _values(study, by, chart='Xbar', **kwargs):
    """Sorted chart values — comparable across spellings that order rows differently."""
    data = study.execute(chart=chart, by=by, **kwargs).get_chart(chart)
    col = {'Xbar': 'xbar', 'S': 's'}[chart]
    return np.sort(data[col].dropna().round(9).to_numpy())


def _resolve(study, by):
    """(groupby_cols, ybar_col, stratify_by) for this spelling of `by`."""
    request = ChartRequest(
        chart='Xbar', by=tuple(by), value_col=None, residual=None,
        residual_chart_type=None, recentered=False, companion=False,
        bins=None, phased=False, n_sigma=3.0, n_mode='actual',
    )
    analysis = Analysis(study._spec, request, analysis_dataset=study._ads)
    return analysis._resolve_by_grouping(study._spec.response_var)


# ---------------------------------------------------------------------------
# Every spelling of the rational subgroup is the rational subgroup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('permutation', list(itertools.permutations([F1, F2, TIME])))
@pytest.mark.parametrize('chart', ['Xbar', 'S'])
def test_every_permutation_of_the_cell_matches_by_empty(study, permutation, chart):
    """All six orderings of factors + time equal ``by=[]``.

    ``by=[]`` is the rational subgroup — one design condition at one point in time. Naming
    its columns explicitly, in any order, must select the same thing.
    """
    assert np.allclose(_values(study, list(permutation), chart), _values(study, [], chart))


@pytest.mark.parametrize('permutation', list(itertools.permutations(FACTORS)))
@pytest.mark.parametrize('chart', ['Xbar', 'S'])
def test_every_permutation_of_the_factors_agrees(study, permutation, chart):
    """Factor-level grouping is order-independent in its statistics."""
    assert np.allclose(_values(study, list(permutation), chart), _values(study, FACTORS, chart))


@pytest.mark.parametrize('value', ['R3', 'R5'])
def test_permutation_invariance_holds_for_residuals(study, value):
    """Residual charts take the same `by`, so they inherit the same guarantee."""
    for permutation in itertools.permutations(FACTORS):
        assert np.allclose(
            _values(study, list(permutation), 'Xbar', value=value),
            _values(study, FACTORS, 'Xbar', value=value),
        )


# ---------------------------------------------------------------------------
# The reason the above is not trivial: permutations cross code paths
# ---------------------------------------------------------------------------


def test_permutations_really_do_take_different_paths(study):
    """Guards the guard.

    If the resolver ever normalised every spelling to one path, the tests above would
    still pass while testing nothing. They are meaningful precisely because
    ``by=[f1,f2]`` is served from a precomputed mean and ``by=[f2,f1]`` is aggregated at
    runtime.
    """
    _, canonical_ybar, _ = _resolve(study, FACTORS)
    _, reversed_ybar, _ = _resolve(study, [F2, F1])

    assert canonical_ybar == 'Ybar_k', 'canonical order should use the precomputed mean'
    assert reversed_ybar is None, 'reversed order should aggregate at runtime'


def test_the_precomputed_and_runtime_paths_agree(study):
    """The invariant the two paths must jointly satisfy, stated directly.

    ``Ybar_k`` is computed once during ``formulate()``; the runtime path re-aggregates from
    the analysis dataset. They are different code, and this is the assertion that keeps
    them honest.
    """
    for chart in ('Xbar', 'S'):
        precomputed = _values(study, FACTORS, chart)        # Ybar_k
        runtime = _values(study, [F2, F1], chart)           # groupby at execute time
        assert np.allclose(precomputed, runtime), chart


def test_cell_level_permutations_use_the_precomputed_cell_mean(study):
    """All six orderings resolve to ``Ybar_kt`` on the canonical grain."""
    for permutation in itertools.permutations([F1, F2, TIME]):
        groupby, ybar, _ = _resolve(study, list(permutation))
        assert ybar == 'Ybar_kt', permutation
        assert groupby == [F1, F2, TIME], (
            f'{permutation} should normalise to the canonical cell grain, got {groupby}'
        )


# ---------------------------------------------------------------------------
# Order semantics — set decides the analysis, order decides the presentation
# ---------------------------------------------------------------------------


def test_factor_order_changes_presentation_not_statistics(study):
    """`by=[f1,f2]` reads f1-major; `by=[f2,f1]` reads f2-major.

    Same eight subgroups either way — the analyst chooses which factor varies slowest.
    """
    f1_major = study.execute(chart='Xbar', by=[F1, F2]).get_chart('Xbar')
    f2_major = study.execute(chart='Xbar', by=[F2, F1]).get_chart('Xbar')

    assert len(f1_major) == len(f2_major) == 8
    assert np.allclose(_values(study, [F1, F2]), _values(study, [F2, F1]))

    groups_f1 = list(f1_major.iloc[:, 0])
    groups_f2 = list(f2_major.iloc[:, 0])
    assert groups_f1 != groups_f2, 'the ordering should be visible in the row order'


def test_cell_level_order_is_normalised_not_preserved(study):
    """The one asymmetry, pinned so it is a decision rather than a surprise.

    A *partial or permuted factor* grouping keeps the analyst's order, because there is no
    canonical one. The full factor × time cell does have a canonical grain, so every
    spelling normalises to it — which is also why all six permutations can share the
    precomputed ``Ybar_kt``.
    """
    for permutation in itertools.permutations([F1, F2, TIME]):
        groupby, _, _ = _resolve(study, list(permutation))
        assert groupby == [F1, F2, TIME]

    groupby, _, _ = _resolve(study, [F2, F1])
    assert groupby == [F2, F1], 'permuted factor order is preserved'


def test_single_factor_is_a_partial_subset(study):
    """`by=[f1]` collapses f2 — a different analysis, not a permutation."""
    assert not np.allclose(
        len(_values(study, [F1])), len(_values(study, FACTORS)),
    )
    groupby, ybar, _ = _resolve(study, [F1])
    assert groupby == [F1] and ybar is None
