"""Tests for strata/focus/chart_table consistency in companion-pair results.

Two regressions this file pins:

1. ``AnalysisResult.strata`` previously returned the first chart's strata
   list, even when companion charts had different focusable strata. In
   particular, the mR chart drops first-row-per-stratum (NaN moving range),
   so any single-observation cell becomes empty in mR's data — but the
   inherited X-chart strata list still listed it. ``focus(stratum)`` then
   raised on those strata, breaking any caller driving ``focus`` from
   ``strata``.

2. ``chart_table`` raised ``ValueError`` when joining the ``n`` column from
   the analysis dataset, if the chart's by-column was stringified during
   construction (e.g. PRODUCTION TIME → object) while the ads kept the
   column numeric. The merge now coerces both sides to ``str`` when dtypes
   differ.
"""

from __future__ import annotations

import pandas as pd
import pytest

from processbehavior import ProcessBehavior

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def df_with_single_obs_cell():
    """Synthetic DataFrame where one factor level has exactly one observation.

    For chart='X', by=['factor 1'], the mR chart will drop the lone row of
    that stratum (NaN moving range), leaving the stratum empty in mR's data
    while the X chart still carries the single point.
    """
    return pd.DataFrame(
        {
            'factor 1': [1, 1, 1, 2, 2, 2, 3],  # factor=3 → single observation
            'time': [1, 2, 3, 1, 2, 3, 1],
            'y': [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0],
        }
    )


@pytest.fixture
def t100_sds4_study():
    """T100 reference study with response=PM SDS 4 — triggers chart_table
    dtype mismatch on by=[time] residual+recentered configurations."""
    df = pd.read_csv('validation/PBTESTDATABASE_T100.csv', na_values=['*'])
    return ProcessBehavior(df).formulate(
        response='PM SDS 4',
        factors=['FACTOR 1', 'FACTOR 2'],
        time='PRODUCTION TIME',
    )


# =============================================================================
# Symptom 1 — strata/focus consistency for X+mR companion
# =============================================================================


class TestStrataExcludesUnfocusable:
    def test_mr_strata_excludes_single_obs_cell(self, df_with_single_obs_cell):
        """mR's published strata list excludes the single-obs stratum."""
        pb = ProcessBehavior(df_with_single_obs_cell)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='X', by=['factor 1'], companion=True)

        mr_strata = list(result.charts['mR']['strata'])
        assert '3' not in mr_strata, f'mR strata still contains the single-obs stratum: {mr_strata}'

    def test_result_strata_is_intersection(self, df_with_single_obs_cell):
        """result.strata returns the intersection across charts."""
        pb = ProcessBehavior(df_with_single_obs_cell)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='X', by=['factor 1'], companion=True)

        x_strata = set(result.charts['X']['strata'])
        mr_strata = set(result.charts['mR']['strata'])
        expected_intersection = x_strata & mr_strata
        assert set(result.strata) == expected_intersection

    def test_focus_succeeds_for_every_stratum_in_strata(
        self,
        df_with_single_obs_cell,
    ):
        """The strata/focus contract: focus(s) succeeds for every s in result.strata."""
        pb = ProcessBehavior(df_with_single_obs_cell)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='X', by=['factor 1'], companion=True)

        for stratum in result.strata:
            # If focus raises, the strata/focus contract is broken.
            focused = result.focus(stratum)
            assert focused is not None

    def test_strata_order_preserved(self, df_with_single_obs_cell):
        """Intersection preserves the ordering of the first chart's strata list."""
        pb = ProcessBehavior(df_with_single_obs_cell)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='X', by=['factor 1'], companion=True)

        # The first chart's strata that survive the intersection should
        # appear in the same relative order.
        first_chart_strata = next(iter(result.charts.values()))['strata']
        intersected = set(result.strata)
        expected_order = [s for s in first_chart_strata if s in intersected]
        assert list(result.strata) == expected_order


# =============================================================================
# Symptom 2 — chart_table dtype mismatch
# =============================================================================


class TestChartTableDtypeMismatch:
    def test_time_main_effects_xbar_table_non_empty(self, t100_sds4_study):
        """Xbar chart-data-table populates for residual+recentered Time Effects."""
        result = t100_sds4_study.execute(
            chart='Xbar',
            by=['PRODUCTION TIME'],
            value='R4',
            companion=True,
            recentered=True,
        )
        table = result.chart_table(chart='Xbar')
        assert len(table) > 0
        assert 'value' in table.columns
        assert 'center' in table.columns

    def test_time_main_effects_s_table_non_empty(self, t100_sds4_study):
        """S table populates for the same configuration."""
        result = t100_sds4_study.execute(
            chart='Xbar',
            by=['PRODUCTION TIME'],
            value='R4',
            companion=True,
            recentered=True,
        )
        table = result.chart_table(chart='S')
        assert len(table) > 0

    def test_xbar_table_n_column_populated(self, t100_sds4_study):
        """The n-join survives dtype coercion — every row has a valid n."""
        result = t100_sds4_study.execute(
            chart='Xbar',
            by=['PRODUCTION TIME'],
            value='R4',
            companion=True,
            recentered=True,
        )
        table = result.chart_table(chart='Xbar')
        assert 'n' in table.columns
        assert table['n'].notna().all(), f'Found NaN in n column: {table[table["n"].isna()]}'

    def test_chart_table_synthetic_dtype_mismatch(self):
        """Unit-level: directly mismatched dtypes between chart_data and ads."""
        df = pd.DataFrame(
            {
                'factor 1': [1, 1, 1, 2, 2, 2],
                'time': [1, 2, 3, 1, 2, 3],
                'y': [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            }
        )
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        # by=['time'] residual on Xbar — triggers stringification of time
        result = study.execute(
            chart='Xbar',
            by=['time'],
            value='R4',
            companion=True,
            recentered=True,
        )
        # Just confirm it doesn't raise; the table should be non-empty.
        table = result.chart_table(chart='Xbar')
        assert len(table) > 0
