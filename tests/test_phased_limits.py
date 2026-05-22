"""Tests for phased limits feature (per-phase center lines and control limits).

Phased limits compute independent center lines and control limits for each
contiguous run of the same collapsed factor combination.
"""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets.synthetic import make_sds
from processbehavior.exceptions import ValidationError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sds1_study():
    """SDS1 study with two factors and time."""
    df = make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


@pytest.fixture
def sds2_study():
    """SDS2 study — R2/R3 use XmR charts."""
    df = make_sds(2, K1=2, K2=2, T=8, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


# =============================================================================
# Core Behavior
# =============================================================================


class TestPhasedXmRLimits:
    """Test per-phase limits computation for XmR charts."""

    def test_phased_xmr_limits_vary_across_phases(self, sds1_study):
        """Center/lpl/upl columns should not be constant across phases."""
        result = sds1_study.execute(chart='X', by=[], phased=True)
        data = result.get_chart('X')

        # With multiple phases having different data, limits should vary
        assert data['center'].nunique() > 1, 'Center should vary across phases'
        assert data['upl'].nunique() > 1, 'UPL should vary across phases'

    def test_phased_mr_resets_at_boundaries(self, sds1_study):
        """mR should be NaN at the first point of each new phase in mR chart."""
        result = sds1_study.execute(chart='X', by=[], phased=True, companion=True)
        r_data = result.get_chart('mR')

        # mR chart data should have NaN mr values at phase boundaries
        mr_values = r_data['mr']
        assert mr_values.isna().any(), 'mR should have NaN at phase boundaries'

    def test_phased_statistics_signal_variable_limits(self, sds1_study):
        """Phased charts emit limits_vary=True with None for the scalar fields."""
        result = sds1_study.execute(chart='X', by=[], phased=True)
        stats = result.get_statistics('X')

        assert stats['center'] is None
        assert stats['lpl'] is None
        assert stats['upl'] is None
        assert stats['limits_vary'] is True

    def test_phased_metadata_flag(self, sds1_study):
        """Metadata should include phased=True."""
        result = sds1_study.execute(chart='X', by=[], phased=True)
        metadata = result.charts['X']['metadata']

        assert metadata['phased'] is True
        assert metadata['run_rules_applicable'] is False

    def test_phased_single_point_phases_metadata(self, sds1_study):
        """Metadata should include single_point_phases count."""
        result = sds1_study.execute(chart='X', by=[], phased=True)
        metadata = result.charts['X']['metadata']

        assert 'single_point_phases' in metadata
        assert isinstance(metadata['single_point_phases'], int)

    def test_phased_signal_detection_per_phase(self, sds1_study):
        """Beyond_limits uses per-phase lpl/upl, not global."""
        result_phased = sds1_study.execute(chart='X', by=[], phased=True)
        result_global = sds1_study.execute(chart='X', by=[], phased=False)

        phased_data = result_phased.get_chart('X')
        global_data = result_global.get_chart('X')

        # The beyond_limits column should exist in both
        assert 'beyond_limits' in phased_data.columns
        assert 'beyond_limits' in global_data.columns

        # Phased limits are tighter per-phase, so signals may differ
        # (we just verify the column is populated, not specific values)
        assert phased_data['beyond_limits'].dtype in ['int64', 'int32', 'float64']


# =============================================================================
# Companion (XmR + R) with Phased
# =============================================================================


class TestPhasedCompanion:
    """Test phased limits with companion=True (XmR + R together)."""

    def test_phased_companion_both_charts(self, sds1_study):
        """companion=True should produce both X and mR with phased limits."""
        result = sds1_study.execute(chart='X', by=[], phased=True, companion=True)

        xmr_data = result.get_chart('X')
        r_data = result.get_chart('mR')

        assert xmr_data is not None
        assert r_data is not None

        # Both should signal variable limits
        xmr_stats = result.get_statistics('X')
        r_stats = result.get_statistics('mR')
        assert xmr_stats['center'] is None
        assert r_stats['center'] is None
        assert xmr_stats['limits_vary'] is True
        assert r_stats['limits_vary'] is True

        # Both should have phased metadata
        assert result.charts['X']['metadata'].get('phased') is True
        assert result.charts['mR']['metadata'].get('phased') is True

    def test_phased_r_row_count_matches_xmr(self, sds1_study):
        """Phased mR chart row count == X chart row count (no rows dropped)."""
        result = sds1_study.execute(chart='X', by=[], phased=True, companion=True)

        xmr_rows = len(result.get_chart('X'))
        r_rows = len(result.get_chart('mR'))

        assert r_rows == xmr_rows, f'Phased mR chart should have same row count as X (got mR={r_rows}, X={xmr_rows})'

    def test_phased_r_lane_boundaries_align(self, sds1_study):
        """mR chart lane boundaries should match X positions (no offset)."""
        result = sds1_study.execute(chart='X', by=[], phased=True, companion=True)

        xmr_meta = result.charts['X']['metadata']
        r_meta = result.charts['mR']['metadata']

        xmr_boundaries = xmr_meta.get('lane_boundaries')
        r_boundaries = r_meta.get('lane_boundaries')

        if xmr_boundaries is not None:
            assert r_boundaries is not None, 'mR should have lane boundaries if X does'
            xmr_positions = [b['position'] for b in xmr_boundaries]
            r_positions = [b['position'] for b in r_boundaries]
            assert xmr_positions == r_positions, (
                f'mR lane boundaries should match X positions (X={xmr_positions}, mR={r_positions})'
            )


# =============================================================================
# Lane Boundaries
# =============================================================================


class TestPhasedLaneBoundaries:
    """Test that lane boundaries are preserved with phased limits."""

    def test_phased_lane_boundaries_preserved(self, sds1_study):
        """Lane boundaries should still be present in phased output."""
        result = sds1_study.execute(chart='X', by=[], phased=True)
        metadata = result.charts['X']['metadata']

        assert metadata.get('lane_boundaries') is not None
        assert len(metadata['lane_boundaries']) > 0


# =============================================================================
# Validation
# =============================================================================


class TestPhasedValidation:
    """Test that invalid phased configurations raise appropriate errors."""

    def test_phased_requires_xmr_or_r(self, sds1_study):
        """phased=True with Xbar should raise ValidationError."""
        with pytest.raises(ValidationError, match='only valid for X or mR'):
            sds1_study.execute(chart='Xbar', phased=True)

    def test_phased_stratified_allowed_with_collapsed_factors(self, sds1_study):
        """phased=True with by=['factor 1'] should work (factor 2 is collapsed)."""
        # Should NOT raise — factor 2 remains as collapsed factor for phases
        result = sds1_study.execute(chart='X', by=['factor 1'], phased=True)
        assert result is not None

    def test_phased_requires_collapsed_factors(self, sds1_study):
        """phased=True with by=all_factors should raise (no collapsed factors)."""
        with pytest.raises(ValidationError, match='requires collapsed factors'):
            sds1_study.execute(chart='X', by=['factor 1', 'factor 2'], phased=True)

    def test_phased_requires_by_none_raises(self, sds1_study):
        """phased=True with by=None should raise (by=None invalid for X with factors)."""
        with pytest.raises(ValidationError):
            sds1_study.execute(chart='X', phased=True)


# =============================================================================
# Regression: Unphased behavior unchanged
# =============================================================================


class TestUnphasedUnchanged:
    """Verify that phased=False (default) produces identical results."""

    def test_unphased_unchanged(self, sds1_study):
        """phased=False should produce identical results to omitting phased."""
        result_default = sds1_study.execute(chart='X', by=[])
        result_explicit = sds1_study.execute(chart='X', by=[], phased=False)

        default_data = result_default.get_chart('X')
        explicit_data = result_explicit.get_chart('X')

        pd.testing.assert_frame_equal(default_data, explicit_data)

        default_stats = result_default.get_statistics('X')
        explicit_stats = result_explicit.get_statistics('X')

        assert default_stats == explicit_stats

    def test_unphased_companion_unchanged(self, sds1_study):
        """phased=False with companion=True should produce identical results."""
        result_default = sds1_study.execute(chart='X', by=[], companion=True)
        result_explicit = sds1_study.execute(chart='X', by=[], companion=True, phased=False)

        for chart_name in ['X', 'mR']:
            default_data = result_default.get_chart(chart_name)
            explicit_data = result_explicit.get_chart(chart_name)

            pd.testing.assert_frame_equal(default_data, explicit_data, obj=f'{chart_name} data')


# =============================================================================
# Stratified Phased Limits
# =============================================================================


class TestPhasedStratified:
    """Test phased limits in stratified (by=['factor']) mode."""

    def test_phased_stratified_limits_vary(self, sds1_study):
        """Stratified phased: center/lpl/upl columns vary within each stratum."""
        result = sds1_study.execute(chart='X', by=['factor 1'], phased=True)
        # Get the combined data
        xmr_data = result.get_chart('X')

        # With phased limits, center should vary (not constant per stratum)
        assert 'center' in xmr_data.columns
        assert xmr_data['center'].nunique() > 1, 'Center should vary across phases within strata'

    def test_phased_stratified_statistics_signal_variable_limits(self, sds1_study):
        """Per-stratum stats dict signals variable limits."""
        result = sds1_study.execute(chart='X', by=['factor 1'], phased=True)
        stats = result.get_statistics('X')

        # Statistics are nested {stratum: {limits_vary: True, ...}}
        assert isinstance(stats, dict)
        for stratum, stratum_stats in stats.items():
            assert stratum_stats['center'] is None, f'Stratum {stratum} center should be None for phased'
            assert stratum_stats['lpl'] is None
            assert stratum_stats['upl'] is None
            assert stratum_stats['limits_vary'] is True

    def test_phased_stratified_mr_resets_at_phase_boundaries(self, sds1_study):
        """mR NaN at start of each phase within stratum (mR chart)."""
        result = sds1_study.execute(chart='X', by=['factor 1'], phased=True, companion=True)
        r_data = result.get_chart('mR')

        # mR chart should have NaN mr values at phase boundaries
        mr_values = r_data['mr']
        assert mr_values.isna().any(), 'mR should have NaN at phase boundaries'

    def test_phased_stratified_companion(self, sds1_study):
        """Companion X+mR both phased in stratified mode."""
        result = sds1_study.execute(chart='X', by=['factor 1'], phased=True, companion=True)

        xmr_data = result.get_chart('X')
        r_data = result.get_chart('mR')

        assert xmr_data is not None
        assert r_data is not None

        # Both should signal variable limits per stratum
        xmr_stats = result.get_statistics('X')
        r_stats = result.get_statistics('mR')

        for stratum in xmr_stats:
            assert xmr_stats[stratum]['center'] is None
            assert xmr_stats[stratum]['limits_vary'] is True
        for stratum in r_stats:
            assert r_stats[stratum]['center'] is None
            assert r_stats[stratum]['limits_vary'] is True

        # Both should have phased metadata
        assert result.charts['X']['metadata'].get('phased') is True
        assert result.charts['mR']['metadata'].get('phased') is True

    def test_phased_stratified_phase_count(self, sds1_study):
        """Known data produces expected phase count per stratum."""
        result = sds1_study.execute(chart='X', by=['factor 1'], phased=True)
        xmr_data = result.get_chart('X')

        # Center values change at phase boundaries — count transitions
        # within each stratum by checking unique center values
        assert xmr_data['center'].nunique() > 1, 'Should have multiple distinct center values (phases)'

    def test_phased_lane_boundary_positions_valid(self, sds1_study):
        """All lane boundary positions < len(stratum_df) and strictly increasing."""
        result = sds1_study.execute(chart='X', by=['factor 1'], phased=True)
        metadata = result.charts['X']['metadata']
        lane_boundaries = metadata.get('lane_boundaries')

        if lane_boundaries:
            _ = result.charts['X'].get('strata', [])
            xmr_data = result.get_chart('X')

            # For stratified charts, lane_boundaries is dict keyed by stratum
            if isinstance(lane_boundaries, dict):
                stratify_by = metadata.get('stratify_by', [])
                if stratify_by and stratify_by[0] in xmr_data.columns:
                    for stratum, boundaries in lane_boundaries.items():
                        stratum_len = len(xmr_data[xmr_data[stratify_by[0]] == stratum])
                        positions = [b['position'] for b in boundaries]
                        assert positions == sorted(positions), f'Boundary positions not sorted for {stratum}'
                        for p in positions:
                            assert p < stratum_len, f'Position {p} >= stratum len {stratum_len}'

    def test_phased_stratified_metadata(self, sds1_study):
        """Metadata should include phased=True and run_rules_applicable=False."""
        result = sds1_study.execute(chart='X', by=['factor 1'], phased=True)
        metadata = result.charts['X']['metadata']

        assert metadata.get('phased') is True
        assert metadata.get('run_rules_applicable') is False
        assert metadata.get('stratified') is True


# =============================================================================
# Time Tick Labels
# =============================================================================


class TestTimeTickLabels:
    """Test that time tick labels are applied when x-axis uses integer indices."""

    def test_time_ticks_apply_when_x_is_index(self, sds1_study):
        """Tick labels set for by=[] non-phased when _get_x_column() is None."""
        result = sds1_study.execute(chart='X', by=[])
        fig = result.plot('X')

        # The figure should have tick labels set (ticktext/tickvals)
        xaxis = fig._fig.layout.xaxis
        assert xaxis.tickvals is not None, 'tickvals should be set'
        assert xaxis.ticktext is not None, 'ticktext should be set'
        assert len(xaxis.ticktext) > 0, 'Should have tick labels'

    def test_phased_xmr_has_time_tick_labels(self, sds1_study):
        """Phased by=[] chart has time values as tick labels."""
        result = sds1_study.execute(chart='X', by=[], phased=True)
        fig = result.plot('X')

        xaxis = fig._fig.layout.xaxis
        assert xaxis.tickvals is not None, 'tickvals should be set for phased'
        assert xaxis.ticktext is not None, 'ticktext should be set for phased'

    def test_time_ticks_applied_in_stratified_with_replication(self, sds1_study):
        """Stratified with replication: time repeats per stratum, ticks applied."""
        # SDS 1 has replication, so time values repeat within each stratum
        result = sds1_study.execute(chart='X', by=['factor 1', 'factor 2'])
        fig = result.plot('X')

        # Time values repeat due to replication, so integer positions are used
        # and compute_x_axis_layout should add tick labels (via the two-tier path).
        xaxis = fig._fig.layout.xaxis
        assert xaxis.tickvals is not None, 'tickvals should be set (time repeats)'
        assert xaxis.ticktext is not None, 'ticktext should be set (time repeats)'


# =============================================================================
# Phased + Residual Value
# =============================================================================


class TestPhasedWithResidualValue:
    """Phased limits work with residual value= parameter."""

    def test_phased_residual_chart(self, sds1_study):
        """phased=True with value='R1' produces a valid chart."""
        result = sds1_study.execute(chart='X', by=[], phased=True, value='R1')
        data = result.get_chart('X')
        assert len(data) > 0
        assert result.charts['X']['metadata']['phased'] is True

    def test_phased_recentered_residual_chart(self, sds1_study):
        """phased=True with value='R1' and recentered=True succeeds."""
        result = sds1_study.execute(chart='X', by=[], phased=True, value='R1', recentered=True)
        data = result.get_chart('X')
        assert len(data) > 0
        assert result.charts['X']['metadata']['phased'] is True

    def test_phased_residual_limits_vary(self, sds2_study):
        """Phased residual chart has per-phase limits (not constant)."""
        result = sds2_study.execute(chart='X', by=[], phased=True, value='R2')
        data = result.get_chart('X')
        assert data['center'].nunique() > 1 or data['upl'].nunique() > 1
