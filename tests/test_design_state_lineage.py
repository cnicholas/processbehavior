"""Regression tests for the design-state lineage (PDS / ODS / ADS).

Two contracts pinned here:

1. The synthetic generators ``make_sds(sds=N)`` produce data whose **Observed
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


@pytest.mark.parametrize('target_sds', [1, 2, 3, 4, 5, 6])
def test_make_sds_produces_target_ods(target_sds: int) -> None:
    """make_sds(sds=N) generates data that classifies as ODS N."""
    df = pb.make_sds(sds=target_sds, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    assert study.observed_design_state.sds == target_sds, (
        f'make_sds(sds={target_sds}) produced data classifying as '
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
    df = pb.make_sds(sds=target_sds, seed=42)
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
    df = pb.make_sds(sds=target_sds, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    assert study.observed_design_state.n_empty_cells > 0, (
        f'make_sds(sds={target_sds}) produced no empty cells, so the result cannot classify as an incomplete ODS'
    )


@pytest.mark.parametrize('target_sds', [1, 2, 3])
def test_complete_designs_have_no_empty_cells(target_sds: int) -> None:
    """ODS 1/2/3 generators produce no empty cells."""
    df = pb.make_sds(sds=target_sds, seed=42)
    study = pb.ProcessBehavior(df).formulate(
        response='y',
        factors=['factor 1', 'factor 2'],
        time='time',
    )
    assert study.observed_design_state.n_empty_cells == 0, (
        f'make_sds(sds={target_sds}) produced '
        f'{study.observed_design_state.n_empty_cells} empty cells, '
        'but the result should be a complete grid'
    )


def test_ods5_has_only_singletons() -> None:
    """ODS 5: every occupied cell has exactly one observation."""
    df = pb.make_sds(sds=5, seed=42)
    occupied = df.dropna(subset=['y'])
    cell_sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
    assert (cell_sizes == 1).all(), (
        f'ODS 5 occupied cells should all be singletons; saw cell sizes: {cell_sizes.unique()}'
    )


def test_ods4_has_no_singletons() -> None:
    """ODS 4: every occupied cell has at least 2 observations (no singletons)."""
    df = pb.make_sds(sds=4, seed=42)
    occupied = df.dropna(subset=['y'])
    cell_sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
    assert (cell_sizes >= 2).all(), f'ODS 4 occupied cells should all have N>=2; saw cell sizes: {cell_sizes.unique()}'


def test_ods6_has_both_singletons_and_replicates() -> None:
    """ODS 6: occupied cells include both singletons and replicated."""
    df = pb.make_sds(sds=6, seed=42)
    occupied = df.dropna(subset=['y'])
    cell_sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
    assert (cell_sizes == 1).any(), 'ODS 6 must include at least one singleton cell'
    assert (cell_sizes >= 2).any(), 'ODS 6 must include at least one replicated cell'
