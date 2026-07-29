"""Regression tests for P1 (R6 mutation) and P2 (R + residual validation).

Uses validation dataset per CLAUDE.md.
"""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior


@pytest.fixture
def sds1_study():
    """SDS1 study from validation dataset."""
    pb = ProcessBehavior.read_csv('validation/PBTESTDATABASE_T100.csv')
    return pb.formulate(
        response='PM SDS 1',
        time='PRODUCTION TIME',
        factors=['FACTOR 1', 'FACTOR 2'],
    )


class TestR6MutationSafety:
    """P1: Repeated R6 execute calls must not corrupt earlier results."""

    def test_r6_repeated_calls_do_not_corrupt(self, sds1_study):
        """Calling execute(value='R6') with different by= must not mutate prior results."""
        result1 = sds1_study.execute(chart='Xbar', value='R6', by=['FACTOR 1'])
        xbar_values_1 = result1.charts['Xbar']['data']['xbar'].copy()

        # Second call with different factor grouping recalculates R6
        result2 = sds1_study.execute(chart='Xbar', value='R6', by=['FACTOR 2'])

        # result1's chart data must be unchanged
        pd.testing.assert_series_equal(
            result1.charts['Xbar']['data']['xbar'],
            xbar_values_1,
            check_names=False,
        )

        # Sanity: the two results should differ (different grouping factors)
        assert len(result1.charts['Xbar']['data']) != len(result2.charts['Xbar']['data'])


class TestRChartResidualValidation:
    """chart='mR' with a residual value works, standalone or as a companion.

    These three cases asserted the opposite until the restriction was lifted. The P2 fix in
    ``bb9f68e`` was aimed at a *late unhelpful error* — the real defect was a missing
    ``'mR'`` entry in ``_RESIDUAL_SOLO_STRATEGY_MAP``, which failed deep inside
    ``Analysis.calculate()``. The guard made that failure fail *nicely* rather than making
    it work. Adding the dispatch entry produces output identical to the companion path
    (see tests/test_solo_mr_residuals.py), so the guard is gone and these now assert the
    working behaviour. The early-error goal of the original fix is unaffected: genuinely
    invalid pairs still raise in validation, via ``_residual_pair_problem``.
    """

    def test_r_chart_with_residual_works(self, sds1_study):
        """execute(chart='mR', value='R3') returns a standalone moving-range chart."""
        result = sds1_study.execute(chart='mR', value='R3', by=[])
        assert list(result.charts) == ['mR']

    def test_r_chart_with_r5_works(self, sds1_study):
        result = sds1_study.execute(chart='mR', value='R5', by=[])
        assert list(result.charts) == ['mR']

    def test_r_chart_with_residual_no_companion_returns_only_mr(self, sds1_study):
        """companion=False means the X chart is not included — not that it fails."""
        result = sds1_study.execute(chart='mR', value='R3', by=[], companion=False)
        assert list(result.charts) == ['mR']

    def test_r_chart_with_residual_companion_works(self, sds1_study):
        """execute(chart='mR', value='R3', companion=True) returns both X and mR."""
        result = sds1_study.execute(chart='mR', value='R3', by=[], companion=True)
        assert 'X' in result.charts
        assert 'mR' in result.charts

    def test_r_chart_with_residual_companion_recentered_works(self, sds1_study):
        """execute(chart='mR', value='R3', companion=True, recentered=True) works."""
        result = sds1_study.execute(chart='mR', value='R3', by=[], companion=True, recentered=True)
        assert 'X' in result.charts
        assert 'mR' in result.charts

    def test_xbar_with_residual_still_works(self, sds1_study):
        """Xbar + residual must still work (no false positive)."""
        result = sds1_study.execute(chart='Xbar', value='R3')
        assert 'Xbar' in result.charts
