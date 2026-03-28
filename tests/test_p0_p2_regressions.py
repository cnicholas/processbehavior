"""Regression tests for P1 (R6 mutation) and P2 (R + residual validation).

Uses validation dataset per CLAUDE.md.
"""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior, ValidationError


@pytest.fixture
def sds1_study():
    """SDS1 study from validation dataset."""
    pb = ProcessBehavior.read_csv('validation/TABVASTESTDATABASE.csv')
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
    """P2: chart='R' with a residual value must raise ValidationError early."""

    def test_r_chart_with_residual_raises_validation_error(self, sds1_study):
        """execute(chart='R', value='R3') must raise ValidationError."""
        with pytest.raises(ValidationError, match="not supported for residual"):
            sds1_study.execute(chart='R', value='R3', by=[])

    def test_r_chart_with_r5_raises_validation_error(self, sds1_study):
        """execute(chart='R', value='R5') must raise ValidationError."""
        with pytest.raises(ValidationError, match="not supported for residual"):
            sds1_study.execute(chart='R', value='R5', by=[])

    def test_r_chart_with_residual_no_companion_still_raises(self, sds1_study):
        """execute(chart='R', value='R3', companion=False) must still raise."""
        with pytest.raises(ValidationError, match="not supported for residual"):
            sds1_study.execute(chart='R', value='R3', by=[], companion=False)

    def test_r_chart_with_residual_companion_works(self, sds1_study):
        """execute(chart='R', value='R3', companion=True) returns both XmR and R."""
        result = sds1_study.execute(chart='R', value='R3', by=[], companion=True)
        assert 'XmR' in result.charts
        assert 'R' in result.charts

    def test_r_chart_with_residual_companion_recentered_works(self, sds1_study):
        """execute(chart='R', value='R3', companion=True, recentered=True) works."""
        result = sds1_study.execute(chart='R', value='R3', by=[], companion=True, recentered=True)
        assert 'XmR' in result.charts
        assert 'R' in result.charts

    def test_xmr_with_residual_still_works(self, sds1_study):
        """XmR + residual must still work (no false positive)."""
        result = sds1_study.execute(chart='XmR', value='R3', by=[])
        assert 'XmR' in result.charts
