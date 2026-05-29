"""Regression tests for the design-state lineage (PDS / ODS / ADS).

Two contracts pinned here:

1. The synthetic generators ``make_design(state=N)`` produce data whose **Observed
   Design State** classification equals N. This catches the drift that lived
   in ``make_sds4 / make_sds5 / make_sds6`` until the rewrite that produced
   these tests — the prior implementations were scenario archetypes (single
   stream, nested, regime change) that didn't actually classify as Bishop's
   structural ODS 4/5/6.

2. The **lineage collapse** through tidying is correct:

   - ODS ∈ {1, 2, 3}  → ADS ∈ {1, 2, 3}  (no collapse; complete grid)
   - ODS 4            → ADS 1            (empty cells drop; replicated cells survive)
   - ODS 5            → ADS 2            (empty cells drop; singleton cells survive)
   - ODS 6            → ADS 3            (empty cells drop; mixed singleton/replicated)
"""

from __future__ import annotations

import pytest

import processbehavior as pb
from processbehavior.exceptions import ValidationError


@pytest.mark.parametrize('target_sds', [1, 2, 3, 4, 5, 6])
def test_make_sds_produces_target_ods(target_sds: int) -> None:
    """make_design(state=N) generates data that classifies as ODS N."""
    df = pb.make_design(state=target_sds, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    assert study.observed_design_state.sds == target_sds, (
        f'make_design(state={target_sds}) produced data classifying as '
        f'ODS {study.observed_design_state.sds}, not {target_sds}'
    )


@pytest.mark.parametrize(
    'target_sds,expected_ads',
    [
        (1, 1),  # full replication — no collapse
        (2, 2),  # no replication — no collapse
        (3, 3),  # partial replication — no collapse
        (4, 1),  # incomplete, all occupied replicated → ADS 1 after tidy
        (5, 2),  # incomplete, all occupied singletons → ADS 2 after tidy
        (6, 3),  # incomplete, mixed → ADS 3 after tidy
    ],
)
def test_lineage_collapses_correctly(target_sds: int, expected_ads: int) -> None:
    """ODS → ADS lineage matches Bishop's collapse rule."""
    df = pb.make_design(state=target_sds, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    assert study.analytical_design_state.sds == expected_ads, (
        f'ODS {study.observed_design_state.sds} expected to collapse to '
        f'ADS {expected_ads}, got ADS {study.analytical_design_state.sds}'
    )


@pytest.mark.parametrize('target_sds', [4, 5, 6])
def test_incomplete_designs_have_empty_cells(target_sds: int) -> None:
    """ODS 4/5/6 generators produce at least one empty cell."""
    df = pb.make_design(state=target_sds, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    assert study.observed_design_state.n_empty_cells > 0, (
        f'make_design(state={target_sds}) produced no empty cells, so the result cannot classify as an incomplete ODS'
    )


@pytest.mark.parametrize('target_sds', [1, 2, 3])
def test_complete_designs_have_no_empty_cells(target_sds: int) -> None:
    """ODS 1/2/3 generators produce no empty cells."""
    df = pb.make_design(state=target_sds, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    assert study.observed_design_state.n_empty_cells == 0, (
        f'make_design(state={target_sds}) produced '
        f'{study.observed_design_state.n_empty_cells} empty cells, '
        'but the result should be a complete grid'
    )


def test_ods5_has_only_singletons() -> None:
    """ODS 5: every occupied cell has exactly one observation."""
    df = pb.make_design(state=5, seed=42)
    occupied = df.dropna(subset=['y'])
    cell_sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
    assert (cell_sizes == 1).all(), (
        f'ODS 5 occupied cells should all be singletons; saw cell sizes: {cell_sizes.unique()}'
    )


def test_ods4_has_no_singletons() -> None:
    """ODS 4: every occupied cell has at least 2 observations (no singletons)."""
    df = pb.make_design(state=4, seed=42)
    occupied = df.dropna(subset=['y'])
    cell_sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
    assert (cell_sizes >= 2).all(), f'ODS 4 occupied cells should all have N>=2; saw cell sizes: {cell_sizes.unique()}'


def test_ods6_has_both_singletons_and_replicates() -> None:
    """ODS 6: occupied cells include both singletons and replicated."""
    df = pb.make_design(state=6, seed=42)
    occupied = df.dropna(subset=['y'])
    cell_sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
    assert (cell_sizes == 1).any(), 'ODS 6 must include at least one singleton cell'
    assert (cell_sizes >= 2).any(), 'ODS 6 must include at least one replicated cell'


# =============================================================================
# Contract robustness — pin make_design(state=N) → ODS N beyond the default
# seed / grid. The prior drift only surfaced because make_sds4/5/6 silently
# produced ODS 2 regardless of input; these matrices guard that the contract
# holds across the parameter space a user might actually pass.
# =============================================================================

_FORMULATE_KW = dict(response='y', factors=['factor 1', 'factor 2'], time='time')


def _observed_state(df) -> int:
    return pb.ProcessBehavior(df).formulate(**_FORMULATE_KW).observed_design_state.sds


class TestContractAcrossParameters:
    """``make_design(state=N) → ODS N`` must hold across seeds and grid sizes."""

    @pytest.mark.parametrize('state', [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize('seed', [0, 1, 7, 99, 123, 2024])
    def test_contract_holds_across_seeds(self, state: int, seed: int) -> None:
        df = pb.make_design(state=state, seed=seed)
        assert _observed_state(df) == state, (
            f'make_design(state={state}, seed={seed}) classified as '
            f'ODS {_observed_state(df)} — contract is seed-fragile'
        )

    @pytest.mark.parametrize('state', [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize('K1,K2,T', [(2, 2, 4), (3, 2, 6), (4, 3, 8), (2, 3, 10)])
    def test_contract_holds_across_grid_sizes(
        self, state: int, K1: int, K2: int, T: int
    ) -> None:
        df = pb.make_design(state=state, K1=K1, K2=K2, T=T, seed=42)
        assert _observed_state(df) == state, (
            f'make_design(state={state}, K1={K1}, K2={K2}, T={T}) classified '
            f'as ODS {_observed_state(df)} — contract is grid-size-fragile'
        )


class TestDegenerateParamsRaise:
    """Policy: parameters that contradict the requested design state raise a
    clear ValidationError rather than silently producing the wrong ODS.

    The mixed states (3, 6) require both singleton and replicated cells; mix
    params at the extremes can't satisfy that, so the generator's
    post-generation structural check rejects them."""

    def test_state3_all_replicated_raises(self) -> None:
        with pytest.raises(ValidationError, match='both single and multiple'):
            pb.make_design(state=3, p_replicated=0.99, seed=42)

    def test_state3_none_replicated_raises(self) -> None:
        with pytest.raises(ValidationError, match='both single and multiple'):
            pb.make_design(state=3, p_replicated=0.01, seed=42)

    def test_state3_balanced_mix_ok(self) -> None:
        """A sane mix still produces ODS 3 — the guard only rejects extremes."""
        df = pb.make_design(state=3, p_replicated=0.5, seed=42)
        assert _observed_state(df) == 3
