"""Tests for SRP refactor of Xbar/S and X/mR chart calculation.

These tests verify that the `companion` parameter works correctly and that
the refactored chart methods follow Single Responsibility Principle.

Issue #69 Phase 1: Xbar/S SRP Refactor
Issue #69 Phase 2: X/mR SRP Refactor
"""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sds1_study():
    """SDS1 study with two factors and time - has both Xbar and S available."""
    df = synthetic.make_design(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


@pytest.fixture
def sds3_study():
    """SDS3 study (single factor, time, partial replication) - has Xbar and S."""
    df = synthetic.make_design(3, K1=4, T=8, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1'])


# =============================================================================
# SRP COMPLIANCE TESTS
# =============================================================================


class TestSRPCompliance:
    """Test that chart methods follow Single Responsibility Principle.

    With companion=False (default), each chart method returns only its own chart.
    """

    def test_calculate_xbar_returns_only_xbar(self, sds1_study):
        """Xbar chart should return only Xbar data when companion=False."""
        result = sds1_study.execute(chart='Xbar')

        # Should have Xbar
        assert 'Xbar' in result.charts

        # Should NOT have S (SRP compliance)
        assert 'S' not in result.charts

    def test_calculate_s_returns_only_s(self, sds1_study):
        """S chart should return only S data when companion=False."""
        result = sds1_study.execute(chart='S')

        # Should have S
        assert 'S' in result.charts

        # Should NOT have Xbar (SRP compliance)
        assert 'Xbar' not in result.charts

    def test_calculate_xmr_returns_only_xmr(self, sds1_study):
        """X chart should return only X data when companion=False."""
        result = sds1_study.execute(chart='X', by=[])

        # Should have X
        assert 'X' in result.charts

        # Should NOT have mR (SRP compliance)
        assert 'mR' not in result.charts

    def test_calculate_r_returns_only_r(self, sds1_study):
        """mR chart should return only mR data when companion=False."""
        result = sds1_study.execute(chart='mR', by=[])

        # Should have mR
        assert 'mR' in result.charts

        # Should NOT have X (SRP compliance)
        assert 'X' not in result.charts


# =============================================================================
# COMPANION BEHAVIOR TESTS
# =============================================================================


class TestCompanionBehavior:
    """Test that companion=True bundles chart pairs together."""

    def test_companion_xbar_returns_both_charts(self, sds1_study):
        """With companion=True, requesting Xbar should return both Xbar and S."""
        result = sds1_study.execute(chart='Xbar', companion=True)

        # Should have both charts
        assert 'Xbar' in result.charts
        assert 'S' in result.charts

    def test_companion_s_returns_both_charts(self, sds1_study):
        """With companion=True, requesting S should return both Xbar and S."""
        result = sds1_study.execute(chart='S', companion=True)

        # Should have both charts
        assert 'Xbar' in result.charts
        assert 'S' in result.charts

    def test_companion_xmr_returns_both_charts(self, sds1_study):
        """With companion=True, requesting X should return both X and mR."""
        result = sds1_study.execute(chart='X', by=[], companion=True)

        # Should have both charts
        assert 'X' in result.charts
        assert 'mR' in result.charts

    def test_companion_r_returns_both_charts(self, sds1_study):
        """With companion=True, requesting mR should return both X and mR."""
        result = sds1_study.execute(chart='mR', by=[], companion=True)

        # Should have both charts
        assert 'X' in result.charts
        assert 'mR' in result.charts


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================


class TestConsistency:
    """Test that chart data is identical whether calculated companion or uncompanioned."""

    def test_xbar_data_identical_companion_vs_uncompanioned(self, sds1_study):
        """Xbar data should be identical whether calculated alone or with S."""
        result_solo = sds1_study.execute(chart='Xbar')
        result_companion = sds1_study.execute(chart='Xbar', companion=True)

        xbar_solo = result_solo.charts['Xbar']['data']
        xbar_companion = result_companion.charts['Xbar']['data']

        pd.testing.assert_frame_equal(
            xbar_solo.reset_index(drop=True), xbar_companion.reset_index(drop=True), check_names=False
        )

    def test_xbar_statistics_identical_companion_vs_uncompanioned(self, sds1_study):
        """Xbar statistics should be identical whether calculated alone or with S."""
        result_solo = sds1_study.execute(chart='Xbar')
        result_companion = sds1_study.execute(chart='Xbar', companion=True)

        stats_solo = result_solo.charts['Xbar']['statistics']
        stats_companion = result_companion.charts['Xbar']['statistics']

        assert stats_solo == stats_companion

    def test_s_data_identical_companion_vs_independent(self, sds1_study):
        """S data should be identical whether calculated via companion or independently."""
        result_solo = sds1_study.execute(chart='S')
        result_companion = sds1_study.execute(chart='S', companion=True)

        s_solo = result_solo.charts['S']['data']
        s_companion = result_companion.charts['S']['data']

        pd.testing.assert_frame_equal(
            s_solo.reset_index(drop=True), s_companion.reset_index(drop=True), check_names=False
        )

    def test_s_statistics_identical_companion_vs_independent(self, sds1_study):
        """S statistics should be identical whether calculated via companion or independently."""
        result_solo = sds1_study.execute(chart='S')
        result_companion = sds1_study.execute(chart='S', companion=True)

        stats_solo = result_solo.charts['S']['statistics']
        stats_companion = result_companion.charts['S']['statistics']

        assert stats_solo == stats_companion

    def test_xmr_data_identical_companion_vs_uncompanioned(self, sds1_study):
        """X data should be identical whether calculated alone or with mR."""
        result_solo = sds1_study.execute(chart='X', by=[])
        result_companion = sds1_study.execute(chart='X', by=[], companion=True)

        xmr_solo = result_solo.charts['X']['data']
        xmr_companion = result_companion.charts['X']['data']

        pd.testing.assert_frame_equal(
            xmr_solo.reset_index(drop=True), xmr_companion.reset_index(drop=True), check_names=False
        )

    def test_xmr_statistics_identical_companion_vs_uncompanioned(self, sds1_study):
        """X statistics should be identical whether calculated alone or with mR."""
        result_solo = sds1_study.execute(chart='X', by=[])
        result_companion = sds1_study.execute(chart='X', by=[], companion=True)

        stats_solo = result_solo.charts['X']['statistics']
        stats_companion = result_companion.charts['X']['statistics']

        assert stats_solo == stats_companion

    def test_r_data_identical_companion_vs_independent(self, sds1_study):
        """mR data should be identical whether calculated via companion or independently."""
        result_solo = sds1_study.execute(chart='mR', by=[])
        result_companion = sds1_study.execute(chart='mR', by=[], companion=True)

        r_solo = result_solo.charts['mR']['data']
        r_companion = result_companion.charts['mR']['data']

        pd.testing.assert_frame_equal(
            r_solo.reset_index(drop=True), r_companion.reset_index(drop=True), check_names=False
        )

    def test_r_statistics_identical_companion_vs_independent(self, sds1_study):
        """mR statistics should be identical whether calculated via companion or independently."""
        result_solo = sds1_study.execute(chart='mR', by=[])
        result_companion = sds1_study.execute(chart='mR', by=[], companion=True)

        stats_solo = result_solo.charts['mR']['statistics']
        stats_companion = result_companion.charts['mR']['statistics']

        assert stats_solo == stats_companion


# =============================================================================
# DEFAULT BEHAVIOR TESTS
# =============================================================================


class TestDefaultBehavior:
    """Test that companion=False is the default (SRP-compliant by default)."""

    def test_companion_false_is_default_xbar(self, sds1_study):
        """Default behavior should be companion=False (SRP-compliant) for Xbar."""
        # Execute without specifying companion
        result = sds1_study.execute(chart='Xbar')

        # Should only have Xbar (default is companion=False)
        assert 'Xbar' in result.charts
        assert 'S' not in result.charts

    def test_companion_false_is_default_xmr(self, sds1_study):
        """Default behavior should be companion=False (SRP-compliant) for X."""
        # Execute without specifying companion
        result = sds1_study.execute(chart='X', by=[])

        # Should only have X (default is companion=False)
        assert 'X' in result.charts
        assert 'mR' not in result.charts

    def test_histogram_rejects_companion(self, sds1_study):
        """Histogram should reject companion parameter (no companion histogram)."""
        from processbehavior.exceptions import ValidationError

        with pytest.raises(ValidationError, match='companion.*Histogram'):
            sds1_study.execute(chart='Histogram', companion=True)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases for the companion parameter."""

    def test_companion_with_by_parameter(self, sds1_study):
        """Companion should work correctly with by parameter."""
        result = sds1_study.execute(chart='Xbar', by=['factor 1'], companion=True)

        # Should have both charts with aggregation by factor 1
        assert 'Xbar' in result.charts
        assert 'S' in result.charts

        # Both should have same grouping column
        xbar_cols = result.charts['Xbar']['data'].columns.tolist()
        s_cols = result.charts['S']['data'].columns.tolist()

        # Single-factor by preserves the original column name
        assert 'factor 1' in xbar_cols
        assert 'factor 1' in s_cols

    def test_companion_with_sds3(self, sds3_study):
        """Companion should work with different SDS types."""
        result = sds3_study.execute(chart='Xbar', companion=True)

        assert 'Xbar' in result.charts
        assert 'S' in result.charts

    def test_companion_xbar_and_s_have_same_groups(self, sds1_study):
        """When companion, Xbar and S should have the same groups."""
        result = sds1_study.execute(chart='Xbar', companion=True)

        xbar_data = result.charts['Xbar']['data']
        s_data = result.charts['S']['data']

        # Should have same number of rows
        assert len(xbar_data) == len(s_data)

    def test_xmr_srp_stratified(self, sds1_study):
        """X SRP should work correctly with stratification (by parameter)."""
        # Stratify by all factors
        result = sds1_study.execute(chart='X', by=['factor 1', 'factor 2'])

        # Should have X only
        assert 'X' in result.charts
        assert 'mR' not in result.charts

        # Should have strata metadata
        assert 'strata' in result.charts['X']

    def test_xmr_srp_ungrouped(self, sds1_study):
        """X SRP should work correctly without stratification (by=[])."""
        result = sds1_study.execute(chart='X', by=[])

        # Should have X only
        assert 'X' in result.charts
        assert 'mR' not in result.charts

        # Should NOT have strata (ungrouped)
        assert 'strata' not in result.charts['X']

    def test_r_srp_stratified(self, sds1_study):
        """mR SRP should work correctly with stratification."""
        # Stratify by all factors
        result = sds1_study.execute(chart='mR', by=['factor 1', 'factor 2'])

        # Should have mR only
        assert 'mR' in result.charts
        assert 'X' not in result.charts

        # Should have strata metadata
        assert 'strata' in result.charts['mR']

    def test_r_srp_ungrouped(self, sds1_study):
        """mR SRP should work correctly without stratification."""
        result = sds1_study.execute(chart='mR', by=[])

        # Should have mR only
        assert 'mR' in result.charts
        assert 'X' not in result.charts

    def test_companion_xmr_with_stratification(self, sds1_study):
        """Companion X/mR should work with stratification."""
        result = sds1_study.execute(chart='X', by=['factor 1', 'factor 2'], companion=True)

        # Should have both charts
        assert 'X' in result.charts
        assert 'mR' in result.charts

        # Both should have strata
        assert 'strata' in result.charts['X']
        assert 'strata' in result.charts['mR']

    def test_companion_xmr_and_r_have_consistent_strata(self, sds1_study):
        """When companion, X and mR should have the same strata."""
        result = sds1_study.execute(chart='X', by=['factor 1', 'factor 2'], companion=True)

        xmr_strata = result.charts['X']['strata']
        r_strata = result.charts['mR']['strata']

        # Should have same strata
        assert xmr_strata == r_strata
