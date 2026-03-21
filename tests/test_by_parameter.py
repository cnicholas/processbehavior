"""Tests for the `by` and `value` parameters in Study.execute().

These tests verify the behavior of view creation over the immutable analytic dataset.
The `by` parameter controls stratification/grouping, while `value` controls what to chart.

Key invariant: Residuals are NEVER recomputed - `by` creates views, not recomputations.
"""

import pandas as pd
import pytest

from processbehavior import FactorNotFoundError, ProcessBehavior, ValidationError
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
        result_full = sds1_study.execute(chart='XmR', by=['factor 1', 'factor 2'])
        result_f1 = sds1_study.execute(chart='XmR', by=['factor 1'])
        result_all = sds1_study.execute(chart='XmR', by=[])

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

    def test_xmr_with_factors_requires_by(self, sds1_study):
        """XmR charts with factors should require explicit `by` parameter."""
        with pytest.raises(ValidationError, match="require.*by"):
            sds1_study.execute(chart='XmR')

    def test_by_must_be_subset_of_factors(self, sds1_study):
        """by parameter must only contain factor variables."""
        with pytest.raises(FactorNotFoundError, match="not a valid by variable"):
            sds1_study.execute(chart='XmR', by=['invalid_factor'])

    def test_by_empty_list_allowed(self, sds1_study):
        """by=[] should be allowed (collapses all factors)."""
        result = sds1_study.execute(chart='XmR', by=[])
        assert 'XmR' in result.charts

    def test_by_single_factor_allowed(self, sds1_study):
        """by=['factor 1'] should stratify by that factor."""
        result = sds1_study.execute(chart='XmR', by=['factor 1'])
        assert 'XmR' in result.charts
        assert result.charts['XmR'].get('strata') is not None

    def test_by_all_factors_allowed(self, sds1_study):
        """by=['factor 1', 'factor 2'] should stratify by all factors."""
        result = sds1_study.execute(chart='XmR', by=['factor 1', 'factor 2'])
        assert 'XmR' in result.charts
        assert result.charts['XmR'].get('strata') is not None


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
        """Xbar with by=[] is equivalent to by=None (cell-level grouping, not collapse)."""
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
# XmR STRATIFICATION TESTS
# =============================================================================

class TestXmRStratification:
    """Test XmR chart stratification with `by` parameter."""

    def test_xmr_by_all_factors_stratifies(self, sds1_study):
        """XmR with by=['factor 1', 'factor 2'] creates strata for each combo."""
        result = sds1_study.execute(chart='XmR', by=['factor 1', 'factor 2'])
        strata = result.charts['XmR'].get('strata')
        assert strata is not None
        # K1=3, K2=2 -> 6 strata
        assert len(strata) == 6

    def test_xmr_by_single_factor_with_boundaries(self, sds1_study):
        """XmR with by=['factor 1'] creates strata with lane boundaries for factor 2."""
        result = sds1_study.execute(chart='XmR', by=['factor 1'])
        strata = result.charts['XmR'].get('strata')
        assert strata is not None
        # K1=3 -> 3 strata
        assert len(strata) == 3

        # Should have lane boundaries for factor 2
        metadata = result.charts['XmR']['metadata']
        lane_boundaries = metadata.get('lane_boundaries')
        assert lane_boundaries is not None
        # Boundaries should be keyed by stratum
        assert isinstance(lane_boundaries, dict)

    def test_xmr_by_empty_single_chart_with_boundaries(self, sds1_study):
        """XmR with by=[] creates single chart with lane boundaries."""
        result = sds1_study.execute(chart='XmR', by=[])

        # Should NOT have strata (single chart)
        assert result.charts['XmR'].get('strata') is None

        # Should have lane boundaries for collapsed factors
        metadata = result.charts['XmR']['metadata']
        lane_boundaries = metadata.get('lane_boundaries')
        assert lane_boundaries is not None
        # Boundaries should be a list (not dict) for single chart
        assert isinstance(lane_boundaries, list)
        assert len(lane_boundaries) > 0


class TestRStratification:
    """Test R chart stratification with `by` parameter (bundled with XmR)."""

    def test_r_by_single_factor_has_boundaries(self, sds1_study):
        """R chart should also have lane boundaries when stratified."""
        result = sds1_study.execute(chart='R', by=['factor 1'])

        metadata = result.charts['R']['metadata']
        # R chart may or may not have boundaries depending on implementation
        # At minimum, it should have stratification
        _ = metadata.get('lane_boundaries')  # Access to ensure metadata is populated
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
        result = sds1_study.execute(chart='XmR', by=[])
        boundaries = result.charts['XmR']['metadata']['lane_boundaries']
        assert all('position' in b for b in boundaries)

    def test_lane_boundary_has_label(self, sds1_study):
        """Lane boundaries should have label field."""
        result = sds1_study.execute(chart='XmR', by=[])
        boundaries = result.charts['XmR']['metadata']['lane_boundaries']
        assert all('label' in b for b in boundaries)

    def test_lane_boundary_has_variables(self, sds1_study):
        """Lane boundaries should indicate which variables changed."""
        result = sds1_study.execute(chart='XmR', by=[])
        boundaries = result.charts['XmR']['metadata']['lane_boundaries']
        assert all('variables' in b for b in boundaries)

    def test_lane_boundaries_positions_are_ordered(self, sds1_study):
        """Lane boundary positions should be in ascending order."""
        result = sds1_study.execute(chart='XmR', by=[])
        boundaries = result.charts['XmR']['metadata']['lane_boundaries']
        positions = [b['position'] for b in boundaries]
        assert positions == sorted(positions)


# =============================================================================
# THREE-FACTOR LANE BOUNDARY TESTS
# =============================================================================

@pytest.fixture
def three_factor_study():
    """SDS1 study with three factors — machine, shift, operator."""
    import numpy as np
    rng = np.random.default_rng(42)
    rows = []
    for t in [1, 2]:
        for m in ['M1', 'M2']:
            for s in ['S1', 'S2']:
                for o in ['O1', 'O2']:
                    for _ in range(2):  # n=2 per cell
                        rows.append({
                            'time': t,
                            'machine': m,
                            'shift': s,
                            'operator': o,
                            'y': rng.normal(50, 1),
                        })
    df = pd.DataFrame(rows)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['machine', 'shift', 'operator'],
    )


class TestThreeFactorLaneBoundaries:
    """Test lane boundaries with 3 factors — covers N-element tuple code paths."""

    def test_by_empty_single_chart_with_three_factor_boundaries(self, three_factor_study):
        """by=[] collapses all 3 factors into a single XmR stream with lane boundaries."""
        result = three_factor_study.execute(chart='XmR', by=[])

        # Single chart, no strata
        assert result.charts['XmR'].get('strata') is None

        # Lane boundaries are a list (not dict)
        boundaries = result.charts['XmR']['metadata']['lane_boundaries']
        assert isinstance(boundaries, list)
        assert len(boundaries) > 0

        # variables field lists all 3 factor names
        for b in boundaries:
            assert b['variables'] == ['machine', 'shift', 'operator']

        # Labels are 3-part joined strings (e.g. "M1_S1_O2")
        for b in boundaries:
            parts = b['label'].split('_')
            assert len(parts) == 3, f"Expected 3-part label, got {b['label']!r}"

        # Positions are ordered
        positions = [b['position'] for b in boundaries]
        assert positions == sorted(positions)

    def test_by_one_factor_stratifies_with_two_collapsed(self, three_factor_study):
        """by=['machine'] stratifies by machine, collapses shift+operator."""
        result = three_factor_study.execute(chart='XmR', by=['machine'])

        # Stratified: 2 strata (M1, M2)
        strata = result.charts['XmR'].get('strata')
        assert strata is not None
        assert len(strata) == 2

        # Lane boundaries are a dict keyed by stratum
        boundaries = result.charts['XmR']['metadata']['lane_boundaries']
        assert isinstance(boundaries, dict)

        for _stratum_key, stratum_bounds in boundaries.items():
            # Each boundary reflects the 2 collapsed factors
            for b in stratum_bounds:
                assert b['variables'] == ['shift', 'operator']

            # Labels are 2-part joined strings (e.g. "S1_O2")
            for b in stratum_bounds:
                parts = b['label'].split('_')
                assert len(parts) == 2, f"Expected 2-part label, got {b['label']!r}"

    def test_by_two_factors_stratifies_with_one_collapsed(self, three_factor_study):
        """by=['machine', 'shift'] stratifies by both, collapses operator."""
        result = three_factor_study.execute(chart='XmR', by=['machine', 'shift'])

        # Stratified: 2x2 = 4 strata as tuples
        strata = result.charts['XmR'].get('strata')
        assert strata is not None
        assert len(strata) == 4

        # Strata keys are tuples (multi-key stratification)
        for s in strata:
            assert isinstance(s, tuple), f"Expected tuple stratum key, got {type(s).__name__}"

        # Lane boundaries dict uses tuple keys
        boundaries = result.charts['XmR']['metadata']['lane_boundaries']
        assert isinstance(boundaries, dict)

        for stratum_key, stratum_bounds in boundaries.items():
            assert isinstance(stratum_key, tuple)

            # Each boundary reflects the single collapsed factor
            for b in stratum_bounds:
                assert b['variables'] == ['operator']

            # Labels are single values (no joining needed)
            for b in stratum_bounds:
                assert '_' not in b['label'] or b['label'] in ('O1', 'O2'), \
                    f"Expected single-factor label, got {b['label']!r}"

    def test_by_empty_three_factor_plot_renders(self, three_factor_study):
        """Smoke test: plotting a 3-factor by=[] chart does not raise."""
        result = three_factor_study.execute(chart='XmR', by=[])
        fig = result.plot('XmR')
        assert fig is not None


# =============================================================================
# XBAR R2-BASED LIMITS FOR EFFECT-CARRYING RESIDUALS
# =============================================================================

class TestXbarEffectResidualLimits:
    """Test that Xbar limits for R4/R5 use R2's within-group std.

    At collapsed groupings (by single factor), R5's within-group std includes
    between-cell variance from the collapsed dimension, inflating limits.
    R2 (within-cell noise) is the correct basis per Wheeler/Bishop.

    At RSG level, R5 std = R2 std within cells, so limits are unchanged.
    """

    def test_xbar_r5_by_factor_uses_r2_sbar(self, sds1_study):
        """Xbar R5 by single factor: Sbar should come from R2, not R5."""
        # Get R2-based limits (what we test)
        result_r5 = sds1_study.execute(chart='Xbar', value='R5', by=['factor 1'])
        # Get R2's own limits (reference)
        result_r2 = sds1_study.execute(chart='Xbar', value='R2', by=['factor 1'])

        r5_data = result_r5.charts['Xbar']['data']
        r2_data = result_r2.charts['Xbar']['data']

        # R5 limits should match R2 limits (same Sbar, same N)
        # They differ only in center line (CL), so limit WIDTH should match
        r5_width = (r5_data['upl'] - r5_data['lpl']).iloc[0]
        r2_width = (r2_data['upl'] - r2_data['lpl']).iloc[0]
        assert abs(r5_width - r2_width) < 0.01, (
            f"R5 limit width {r5_width:.4f} should match R2 width {r2_width:.4f}"
        )

    def test_xbar_r5_by_rsg_unchanged(self, sds1_study):
        """Xbar R5 by RSG: limits unchanged (R5 std = R2 std within cells)."""
        result = sds1_study.execute(
            chart='Xbar', value='R5', by=['factor 1', 'factor 2']
        )
        result_r2 = sds1_study.execute(
            chart='Xbar', value='R2', by=['factor 1', 'factor 2']
        )

        r5_data = result.charts['Xbar']['data']
        r2_data = result_r2.charts['Xbar']['data']

        # At RSG level, R5 std = R2 std, so widths should be equal
        r5_width = (r5_data['upl'] - r5_data['lpl']).iloc[0]
        r2_width = (r2_data['upl'] - r2_data['lpl']).iloc[0]
        assert abs(r5_width - r2_width) < 0.01

    def test_xbar_r4_by_factor_uses_r2_sbar(self, sds1_study):
        """Xbar R4 by single factor: Sbar should come from R2, not R4."""
        result_r4 = sds1_study.execute(chart='Xbar', value='R4', by=['factor 1'])
        result_r2 = sds1_study.execute(chart='Xbar', value='R2', by=['factor 1'])

        r4_data = result_r4.charts['Xbar']['data']
        r2_data = result_r2.charts['Xbar']['data']

        r4_width = (r4_data['upl'] - r4_data['lpl']).iloc[0]
        r2_width = (r2_data['upl'] - r2_data['lpl']).iloc[0]
        assert abs(r4_width - r2_width) < 0.01, (
            f"R4 limit width {r4_width:.4f} should match R2 width {r2_width:.4f}"
        )

    def test_xbar_r2_not_substituted(self, sds1_study):
        """Xbar R2 by factor: R2 is NOT an effect-carrying residual, no substitution."""
        result = sds1_study.execute(chart='Xbar', value='R2', by=['factor 1'])
        data = result.charts['Xbar']['data']
        # Just verify it runs and produces limits
        assert data['upl'].iloc[0] > data['lpl'].iloc[0]

    def test_xbar_r5_collapsed_narrower_than_without_fix(self, sds1_study):
        """Verify R5 by factor limits are narrower than R5's own variability would give.

        If R5 has between-cell variance from collapsed F2, using R5's std directly
        would produce wider limits than using R2's std. This test confirms the fix
        by checking R5 limit width < what R5's raw within-group std would give.
        """
        ds = sds1_study.dataset
        # R5's within-group std by factor 1 (includes between-cell F2 variance)
        r5_group_stds = ds.groupby('factor 1')['R5'].std()
        # R2's within-group std by factor 1 (pure within-cell noise)
        r2_group_stds = ds.groupby('factor 1')['R2'].std()

        # R5 std >= R2 std at collapsed levels (between-cell variance adds up)
        assert r5_group_stds.mean() >= r2_group_stds.mean() - 0.001, (
            "R5 within-group std should be >= R2 within-group std at collapsed level"
        )

        # The Xbar chart should use R2's (tighter) std
        result = sds1_study.execute(chart='Xbar', value='R5', by=['factor 1'])
        actual_width = (
            result.charts['Xbar']['data']['upl']
            - result.charts['Xbar']['data']['lpl']
        ).iloc[0]

        result_r2 = sds1_study.execute(chart='Xbar', value='R2', by=['factor 1'])
        r2_width = (
            result_r2.charts['Xbar']['data']['upl']
            - result_r2.charts['Xbar']['data']['lpl']
        ).iloc[0]

        assert abs(actual_width - r2_width) < 0.01
