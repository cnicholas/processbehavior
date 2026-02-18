"""Tests for SRP refactor of Xbar/S and XmR/R chart calculation.

These tests verify that the `paired` parameter works correctly and that
the refactored chart methods follow Single Responsibility Principle.

Issue #69 Phase 1: Xbar/S SRP Refactor
Issue #69 Phase 2: XmR/R SRP Refactor
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
    df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1', 'factor 2']
    )


@pytest.fixture
def sds3_study():
    """SDS3 study (single factor, time, partial replication) - has Xbar and S."""
    df = synthetic.make_sds(3, K1=4, T=8, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1']
    )


# =============================================================================
# SRP COMPLIANCE TESTS
# =============================================================================

class TestSRPCompliance:
    """Test that chart methods follow Single Responsibility Principle.

    With paired=False (default), each chart method returns only its own chart.
    """

    def test_calculate_xbar_returns_only_xbar(self, sds1_study):
        """Xbar chart should return only Xbar data when paired=False."""
        result = sds1_study.execute(chart='Xbar')

        # Should have Xbar
        assert 'Xbar' in result.charts

        # Should NOT have S (SRP compliance)
        assert 'S' not in result.charts

    def test_calculate_s_returns_only_s(self, sds1_study):
        """S chart should return only S data when paired=False."""
        result = sds1_study.execute(chart='S')

        # Should have S
        assert 'S' in result.charts

        # Should NOT have Xbar (SRP compliance)
        assert 'Xbar' not in result.charts

    def test_calculate_xmr_returns_only_xmr(self, sds1_study):
        """XmR chart should return only XmR data when paired=False."""
        result = sds1_study.execute(chart='XmR', by=[])

        # Should have XmR
        assert 'XmR' in result.charts

        # Should NOT have R (SRP compliance)
        assert 'R' not in result.charts

    def test_calculate_r_returns_only_r(self, sds1_study):
        """R chart should return only R data when paired=False."""
        result = sds1_study.execute(chart='R', by=[])

        # Should have R
        assert 'R' in result.charts

        # Should NOT have XmR (SRP compliance)
        assert 'XmR' not in result.charts


# =============================================================================
# PAIRED BEHAVIOR TESTS
# =============================================================================

class TestPairedBehavior:
    """Test that paired=True bundles chart pairs together."""

    def test_paired_xbar_returns_both_charts(self, sds1_study):
        """With paired=True, requesting Xbar should return both Xbar and S."""
        result = sds1_study.execute(chart='Xbar', paired=True)

        # Should have both charts
        assert 'Xbar' in result.charts
        assert 'S' in result.charts

    def test_paired_s_returns_both_charts(self, sds1_study):
        """With paired=True, requesting S should return both Xbar and S."""
        result = sds1_study.execute(chart='S', paired=True)

        # Should have both charts
        assert 'Xbar' in result.charts
        assert 'S' in result.charts

    def test_paired_xmr_returns_both_charts(self, sds1_study):
        """With paired=True, requesting XmR should return both XmR and R."""
        result = sds1_study.execute(chart='XmR', by=[], paired=True)

        # Should have both charts
        assert 'XmR' in result.charts
        assert 'R' in result.charts

    def test_paired_r_returns_both_charts(self, sds1_study):
        """With paired=True, requesting R should return both XmR and R."""
        result = sds1_study.execute(chart='R', by=[], paired=True)

        # Should have both charts
        assert 'XmR' in result.charts
        assert 'R' in result.charts


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

class TestConsistency:
    """Test that chart data is identical whether calculated paired or unpaired."""

    def test_xbar_data_identical_paired_vs_unpaired(self, sds1_study):
        """Xbar data should be identical whether calculated alone or with S."""
        result_solo = sds1_study.execute(chart='Xbar')
        result_paired = sds1_study.execute(chart='Xbar', paired=True)

        xbar_solo = result_solo.charts['Xbar']['data']
        xbar_paired = result_paired.charts['Xbar']['data']

        pd.testing.assert_frame_equal(
            xbar_solo.reset_index(drop=True),
            xbar_paired.reset_index(drop=True),
            check_names=False
        )

    def test_xbar_statistics_identical_paired_vs_unpaired(self, sds1_study):
        """Xbar statistics should be identical whether calculated alone or with S."""
        result_solo = sds1_study.execute(chart='Xbar')
        result_paired = sds1_study.execute(chart='Xbar', paired=True)

        stats_solo = result_solo.charts['Xbar']['statistics']
        stats_paired = result_paired.charts['Xbar']['statistics']

        assert stats_solo == stats_paired

    def test_s_data_identical_paired_vs_independent(self, sds1_study):
        """S data should be identical whether calculated via paired or independently."""
        result_solo = sds1_study.execute(chart='S')
        result_paired = sds1_study.execute(chart='S', paired=True)

        s_solo = result_solo.charts['S']['data']
        s_paired = result_paired.charts['S']['data']

        pd.testing.assert_frame_equal(
            s_solo.reset_index(drop=True),
            s_paired.reset_index(drop=True),
            check_names=False
        )

    def test_s_statistics_identical_paired_vs_independent(self, sds1_study):
        """S statistics should be identical whether calculated via paired or independently."""
        result_solo = sds1_study.execute(chart='S')
        result_paired = sds1_study.execute(chart='S', paired=True)

        stats_solo = result_solo.charts['S']['statistics']
        stats_paired = result_paired.charts['S']['statistics']

        assert stats_solo == stats_paired

    def test_xmr_data_identical_paired_vs_unpaired(self, sds1_study):
        """XmR data should be identical whether calculated alone or with R."""
        result_solo = sds1_study.execute(chart='XmR', by=[])
        result_paired = sds1_study.execute(chart='XmR', by=[], paired=True)

        xmr_solo = result_solo.charts['XmR']['data']
        xmr_paired = result_paired.charts['XmR']['data']

        pd.testing.assert_frame_equal(
            xmr_solo.reset_index(drop=True),
            xmr_paired.reset_index(drop=True),
            check_names=False
        )

    def test_xmr_statistics_identical_paired_vs_unpaired(self, sds1_study):
        """XmR statistics should be identical whether calculated alone or with R."""
        result_solo = sds1_study.execute(chart='XmR', by=[])
        result_paired = sds1_study.execute(chart='XmR', by=[], paired=True)

        stats_solo = result_solo.charts['XmR']['statistics']
        stats_paired = result_paired.charts['XmR']['statistics']

        assert stats_solo == stats_paired

    def test_r_data_identical_paired_vs_independent(self, sds1_study):
        """R data should be identical whether calculated via paired or independently."""
        result_solo = sds1_study.execute(chart='R', by=[])
        result_paired = sds1_study.execute(chart='R', by=[], paired=True)

        r_solo = result_solo.charts['R']['data']
        r_paired = result_paired.charts['R']['data']

        pd.testing.assert_frame_equal(
            r_solo.reset_index(drop=True),
            r_paired.reset_index(drop=True),
            check_names=False
        )

    def test_r_statistics_identical_paired_vs_independent(self, sds1_study):
        """R statistics should be identical whether calculated via paired or independently."""
        result_solo = sds1_study.execute(chart='R', by=[])
        result_paired = sds1_study.execute(chart='R', by=[], paired=True)

        stats_solo = result_solo.charts['R']['statistics']
        stats_paired = result_paired.charts['R']['statistics']

        assert stats_solo == stats_paired


# =============================================================================
# DEFAULT BEHAVIOR TESTS
# =============================================================================

class TestDefaultBehavior:
    """Test that paired=False is the default (SRP-compliant by default)."""

    def test_paired_false_is_default_xbar(self, sds1_study):
        """Default behavior should be paired=False (SRP-compliant) for Xbar."""
        # Execute without specifying paired
        result = sds1_study.execute(chart='Xbar')

        # Should only have Xbar (default is paired=False)
        assert 'Xbar' in result.charts
        assert 'S' not in result.charts

    def test_paired_false_is_default_xmr(self, sds1_study):
        """Default behavior should be paired=False (SRP-compliant) for XmR."""
        # Execute without specifying paired
        result = sds1_study.execute(chart='XmR', by=[])

        # Should only have XmR (default is paired=False)
        assert 'XmR' in result.charts
        assert 'R' not in result.charts

    def test_histogram_ignores_paired(self, sds1_study):
        """Histogram should ignore paired parameter (no paired histogram)."""
        result = sds1_study.execute(chart='Histogram', paired=True)

        # Should only have Histogram
        assert 'Histogram' in result.charts
        assert len(result.charts) == 1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases for the paired parameter."""

    def test_paired_with_by_parameter(self, sds1_study):
        """Paired should work correctly with by parameter."""
        result = sds1_study.execute(chart='Xbar', by=['factor 1'], paired=True)

        # Should have both charts with aggregation by factor 1
        assert 'Xbar' in result.charts
        assert 'S' in result.charts

        # Both should have same grouping column
        xbar_cols = result.charts['Xbar']['data'].columns.tolist()
        s_cols = result.charts['S']['data'].columns.tolist()

        # Both should have subgroup column (from by parameter)
        assert 'subgroup' in xbar_cols
        assert 'subgroup' in s_cols

    def test_paired_with_sds3(self, sds3_study):
        """Paired should work with different SDS types."""
        result = sds3_study.execute(chart='Xbar', paired=True)

        assert 'Xbar' in result.charts
        assert 'S' in result.charts

    def test_paired_xbar_and_s_have_same_groups(self, sds1_study):
        """When paired, Xbar and S should have the same groups."""
        result = sds1_study.execute(chart='Xbar', paired=True)

        xbar_data = result.charts['Xbar']['data']
        s_data = result.charts['S']['data']

        # Should have same number of rows
        assert len(xbar_data) == len(s_data)

    def test_xmr_srp_stratified(self, sds1_study):
        """XmR SRP should work correctly with stratification (by parameter)."""
        # Stratify by all factors
        result = sds1_study.execute(chart='XmR', by=['factor 1', 'factor 2'])

        # Should have XmR only
        assert 'XmR' in result.charts
        assert 'R' not in result.charts

        # Should have strata metadata
        assert 'strata' in result.charts['XmR']

    def test_xmr_srp_ungrouped(self, sds1_study):
        """XmR SRP should work correctly without stratification (by=[])."""
        result = sds1_study.execute(chart='XmR', by=[])

        # Should have XmR only
        assert 'XmR' in result.charts
        assert 'R' not in result.charts

        # Should NOT have strata (ungrouped)
        assert 'strata' not in result.charts['XmR']

    def test_r_srp_stratified(self, sds1_study):
        """R SRP should work correctly with stratification."""
        # Stratify by all factors
        result = sds1_study.execute(chart='R', by=['factor 1', 'factor 2'])

        # Should have R only
        assert 'R' in result.charts
        assert 'XmR' not in result.charts

        # Should have strata metadata
        assert 'strata' in result.charts['R']

    def test_r_srp_ungrouped(self, sds1_study):
        """R SRP should work correctly without stratification."""
        result = sds1_study.execute(chart='R', by=[])

        # Should have R only
        assert 'R' in result.charts
        assert 'XmR' not in result.charts

    def test_paired_xmr_with_stratification(self, sds1_study):
        """Paired XmR/R should work with stratification."""
        result = sds1_study.execute(chart='XmR', by=['factor 1', 'factor 2'], paired=True)

        # Should have both charts
        assert 'XmR' in result.charts
        assert 'R' in result.charts

        # Both should have strata
        assert 'strata' in result.charts['XmR']
        assert 'strata' in result.charts['R']

    def test_paired_xmr_and_r_have_consistent_strata(self, sds1_study):
        """When paired, XmR and R should have the same strata."""
        result = sds1_study.execute(chart='XmR', by=['factor 1', 'factor 2'], paired=True)

        xmr_strata = result.charts['XmR']['strata']
        r_strata = result.charts['R']['strata']

        # Should have same strata
        assert xmr_strata == r_strata
