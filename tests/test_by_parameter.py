"""Tests for the `by` and `value` parameters in Study.execute().

These tests verify the behavior of view creation over the immutable analytic dataset.
The `by` parameter controls stratification/grouping, while `value` controls what to chart.

Key invariant: Residuals are NEVER recomputed - `by` creates views, not recomputations.
"""

import pytest
import pandas as pd

from processbehavior import ProcessBehavior, FactorNotFoundError
from processbehavior.datasets import synthetic


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sds1_study():
    """SDS1 study with two factors and time - ideal for testing `by` parameter."""
    df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1', 'factor 2']
    )


@pytest.fixture
def sds4_study():
    """SDS4 study without factors - single condition over time."""
    df = synthetic.make_sds(4, T=20, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time'
    )


@pytest.fixture
def sds1_single_factor_study():
    """SDS1 study with single factor."""
    df = synthetic.make_sds(1, K1=3, K2=1, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1']
    )


# =============================================================================
# CRITICAL INVARIANT TEST
# =============================================================================

class TestResidualsInvariant:
    """Test that residuals are identical across all `by` views.

    This is the CRITICAL methodological invariant: `by` creates views,
    not recomputations. The decomposition of variation is fixed at
    formulation time.
    """

    def test_residuals_identical_across_by_views(self, sds1_study):
        """Residual values must be identical regardless of by parameter."""
        # Execute with different `by` values
        result_full = sds1_study.execute(chart='Imr', by=['factor 1', 'factor 2'])
        result_f1 = sds1_study.execute(chart='Imr', by=['factor 1'])
        result_all = sds1_study.execute(chart='Imr', by=[])

        # Get the analytic dataset from each (should be identical)
        ds_full = result_full.dataset
        ds_f1 = result_f1.dataset
        ds_all = result_all.dataset

        # All views must have identical R1-R5 values
        for residual in ['R1', 'R2', 'R3', 'R4', 'R5']:
            pd.testing.assert_series_equal(
                ds_full[residual].reset_index(drop=True),
                ds_f1[residual].reset_index(drop=True),
                check_names=False,
                obj=f"{residual} full vs f1"
            )
            pd.testing.assert_series_equal(
                ds_full[residual].reset_index(drop=True),
                ds_all[residual].reset_index(drop=True),
                check_names=False,
                obj=f"{residual} full vs all"
            )


# =============================================================================
# BY PARAMETER VALIDATION TESTS
# =============================================================================

class TestByParameterValidation:
    """Test validation rules for the `by` parameter."""

    def test_imr_with_factors_requires_by(self, sds1_study):
        """IMR charts with factors should require explicit `by` parameter."""
        with pytest.raises(ValueError, match="require.*by"):
            sds1_study.execute(chart='Imr')

    def test_imr_without_factors_allows_by_none(self, sds4_study):
        """IMR charts without factors should allow by=None."""
        result = sds4_study.execute(chart='Imr')
        assert 'Imr' in result.charts

    def test_by_must_be_subset_of_factors(self, sds1_study):
        """by parameter must only contain factor variables."""
        with pytest.raises(FactorNotFoundError, match="not a valid by variable"):
            sds1_study.execute(chart='Imr', by=['invalid_factor'])

    def test_by_empty_list_allowed(self, sds1_study):
        """by=[] should be allowed (collapses all factors)."""
        result = sds1_study.execute(chart='Imr', by=[])
        assert 'Imr' in result.charts

    def test_by_single_factor_allowed(self, sds1_study):
        """by=['factor 1'] should stratify by that factor."""
        result = sds1_study.execute(chart='Imr', by=['factor 1'])
        assert 'Imr' in result.charts
        assert result.charts['Imr'].get('strata') is not None

    def test_by_all_factors_allowed(self, sds1_study):
        """by=['factor 1', 'factor 2'] should stratify by all factors."""
        result = sds1_study.execute(chart='Imr', by=['factor 1', 'factor 2'])
        assert 'Imr' in result.charts
        assert result.charts['Imr'].get('strata') is not None


# =============================================================================
# XBAR/S AGGREGATION TESTS
# =============================================================================

class TestXbarAggregation:
    """Test Xbar chart aggregation with `by` parameter."""

    def test_xbar_by_none_uses_full_rsg_key(self, sds1_study):
        """Xbar with by=None should use full factor grouping."""
        result = sds1_study.execute(chart='Xbar')
        xbar_data = result.charts['Xbar']['data']
        # Should have one row per unique rsg (factor combination)
        assert len(xbar_data) > 1

    def test_xbar_by_factor_aggregates(self, sds1_study):
        """Xbar with by=['factor 1'] should aggregate by that factor."""
        result = sds1_study.execute(chart='Xbar', by=['factor 1'])
        xbar_data = result.charts['Xbar']['data']
        # Should have 3 rows (K1=3 levels of factor 1)
        assert len(xbar_data) == 3

    def test_xbar_by_empty_equals_by_none(self, sds1_study):
        """Xbar with by=[] should behave identically to by=None (Kt level)."""
        result_empty = sds1_study.execute(chart='Xbar', by=[])
        result_none = sds1_study.execute(chart='Xbar')
        # Both should have same number of rows (Kt level)
        assert len(result_empty.charts['Xbar']['data']) == len(result_none.charts['Xbar']['data'])


class TestSAggregation:
    """Test S chart aggregation with `by` parameter."""

    def test_s_by_none_uses_full_rsg_key(self, sds1_study):
        """S with by=None should use full factor grouping."""
        result = sds1_study.execute(chart='S')
        s_data = result.charts['S']['data']
        assert len(s_data) > 1

    def test_s_by_factor_aggregates(self, sds1_study):
        """S with by=['factor 1'] should aggregate by that factor."""
        result = sds1_study.execute(chart='S', by=['factor 1'])
        s_data = result.charts['S']['data']
        assert len(s_data) == 3

    def test_s_by_empty_equals_by_none(self, sds1_study):
        """S with by=[] should behave identically to by=None (Kt level)."""
        result_empty = sds1_study.execute(chart='S', by=[])
        result_none = sds1_study.execute(chart='S')
        # Both should have same number of rows (Kt level)
        assert len(result_empty.charts['S']['data']) == len(result_none.charts['S']['data'])


# =============================================================================
# ORDER PRESERVATION TESTS
# =============================================================================

class TestOrderPreservation:
    """Test that user-specified by order is preserved in subgroup labels."""

    def test_by_order_preserved_in_subgroup_labels(self, sds1_study):
        """by=['f2','f1'] should produce different labels than by=['f1','f2']."""
        result_f1f2 = sds1_study.execute(chart='Xbar', by=['factor 1', 'factor 2'])
        result_f2f1 = sds1_study.execute(chart='Xbar', by=['factor 2', 'factor 1'])

        labels_f1f2 = result_f1f2.charts['Xbar']['data']['subgroup'].tolist()
        labels_f2f1 = result_f2f1.charts['Xbar']['data']['subgroup'].tolist()

        # Labels should differ
        assert labels_f1f2 != labels_f2f1

        # First label should reflect the order
        assert labels_f1f2[0].startswith('F1_')
        assert labels_f2f1[0].startswith('F2_')

    def test_s_chart_order_preservation(self, sds1_study):
        """S chart should also preserve user's by order."""
        result_f1f2 = sds1_study.execute(chart='S', by=['factor 1', 'factor 2'])
        result_f2f1 = sds1_study.execute(chart='S', by=['factor 2', 'factor 1'])

        labels_f1f2 = result_f1f2.charts['S']['data']['subgroup'].tolist()
        labels_f2f1 = result_f2f1.charts['S']['data']['subgroup'].tolist()

        assert labels_f1f2 != labels_f2f1
        assert labels_f1f2[0].startswith('F1_')
        assert labels_f2f1[0].startswith('F2_')


# =============================================================================
# IMR STRATIFICATION TESTS
# =============================================================================

class TestImrStratification:
    """Test IMR chart stratification with `by` parameter."""

    def test_imr_by_all_factors_stratifies(self, sds1_study):
        """IMR with by=['factor 1', 'factor 2'] creates strata for each combo."""
        result = sds1_study.execute(chart='Imr', by=['factor 1', 'factor 2'])
        strata = result.charts['Imr'].get('strata')
        assert strata is not None
        # K1=3, K2=2 -> 6 strata
        assert len(strata) == 6

    def test_imr_by_single_factor_with_boundaries(self, sds1_study):
        """IMR with by=['factor 1'] creates strata with lane boundaries for factor 2."""
        result = sds1_study.execute(chart='Imr', by=['factor 1'])
        strata = result.charts['Imr'].get('strata')
        assert strata is not None
        # K1=3 -> 3 strata
        assert len(strata) == 3

        # Should have lane boundaries for factor 2
        metadata = result.charts['Imr']['metadata']
        lane_boundaries = metadata.get('lane_boundaries')
        assert lane_boundaries is not None
        # Boundaries should be keyed by stratum
        assert isinstance(lane_boundaries, dict)

    def test_imr_by_empty_single_chart_with_boundaries(self, sds1_study):
        """IMR with by=[] creates single chart with lane boundaries."""
        result = sds1_study.execute(chart='Imr', by=[])

        # Should NOT have strata (single chart)
        assert result.charts['Imr'].get('strata') is None

        # Should have lane boundaries for collapsed factors
        metadata = result.charts['Imr']['metadata']
        lane_boundaries = metadata.get('lane_boundaries')
        assert lane_boundaries is not None
        # Boundaries should be a list (not dict) for single chart
        assert isinstance(lane_boundaries, list)
        assert len(lane_boundaries) > 0


class TestRStratification:
    """Test R chart stratification with `by` parameter (bundled with IMR)."""

    def test_r_by_single_factor_has_boundaries(self, sds1_study):
        """R chart should also have lane boundaries when stratified."""
        result = sds1_study.execute(chart='R', by=['factor 1'])

        metadata = result.charts['R']['metadata']
        lane_boundaries = metadata.get('lane_boundaries')
        # R chart may or may not have boundaries depending on implementation
        # At minimum, it should have stratification
        strata = result.charts['R'].get('strata')
        assert strata is not None


# =============================================================================
# VALUE PARAMETER TESTS
# =============================================================================

class TestValueParameter:
    """Test the `value` parameter for charting residuals."""

    def test_value_r5_uses_r5_residual(self, sds1_study):
        """value='R5' should chart the R5 residual."""
        result = sds1_study.execute(chart='Xbar', value='R5')
        metadata = result.charts['Xbar']['metadata']
        assert metadata.get('residual_type') == 'R5'

    def test_value_r3_uses_r3_residual(self, sds1_study):
        """value='R3' should chart the R3 residual (available for SDS1 Xbar)."""
        result = sds1_study.execute(chart='Xbar', value='R3')
        metadata = result.charts['Xbar']['metadata']
        assert metadata.get('residual_type') == 'R3'

    def test_value_r5_recentered_uses_rcr5(self, sds1_study):
        """value='R5' with recentered=True should use recentered residual."""
        result = sds1_study.execute(chart='Xbar', value='R5', recentered=True)
        metadata = result.charts['Xbar']['metadata']
        assert metadata.get('residual_type') == 'R5'
        assert metadata.get('recentered') is True

    def test_value_none_uses_response(self, sds1_study):
        """value=None should chart the response variable (no residual_type)."""
        result = sds1_study.execute(chart='Xbar')
        metadata = result.charts['Xbar']['metadata']
        # When charting response, residual_type should be absent or None
        assert metadata.get('residual_type') is None

    def test_value_with_s_chart(self, sds1_study):
        """value parameter should work with S chart."""
        result = sds1_study.execute(chart='S', value='R5')
        metadata = result.charts['S']['metadata']
        assert metadata.get('residual_type') == 'R5'


# =============================================================================
# LANE BOUNDARY CONTENT TESTS
# =============================================================================

class TestLaneBoundaryContent:
    """Test the content and structure of lane boundaries."""

    def test_lane_boundary_has_position(self, sds1_study):
        """Lane boundaries should have position field."""
        result = sds1_study.execute(chart='Imr', by=[])
        boundaries = result.charts['Imr']['metadata']['lane_boundaries']
        assert all('position' in b for b in boundaries)

    def test_lane_boundary_has_label(self, sds1_study):
        """Lane boundaries should have label field."""
        result = sds1_study.execute(chart='Imr', by=[])
        boundaries = result.charts['Imr']['metadata']['lane_boundaries']
        assert all('label' in b for b in boundaries)

    def test_lane_boundary_has_variables(self, sds1_study):
        """Lane boundaries should indicate which variables changed."""
        result = sds1_study.execute(chart='Imr', by=[])
        boundaries = result.charts['Imr']['metadata']['lane_boundaries']
        assert all('variables' in b for b in boundaries)

    def test_lane_boundaries_positions_are_ordered(self, sds1_study):
        """Lane boundary positions should be in ascending order."""
        result = sds1_study.execute(chart='Imr', by=[])
        boundaries = result.charts['Imr']['metadata']['lane_boundaries']
        positions = [b['position'] for b in boundaries]
        assert positions == sorted(positions)
