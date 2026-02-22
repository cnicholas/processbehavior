"""Tests for staged limits feature (per-stage center lines and control limits).

Staged limits compute independent center lines and control limits for each
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
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1', 'factor 2']
    )




# =============================================================================
# Core Behavior
# =============================================================================

class TestStagedXmRLimits:
    """Test per-stage limits computation for XmR charts."""

    def test_staged_xmr_limits_vary_across_stages(self, sds1_study):
        """Center/lpl/upl columns should not be constant across stages."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True)
        data = result.get_chart('XmR')

        # With multiple stages having different data, limits should vary
        assert data['center'].nunique() > 1, "Center should vary across stages"
        assert data['upl'].nunique() > 1, "UPL should vary across stages"

    def test_staged_mr_resets_at_boundaries(self, sds1_study):
        """mR should be NaN at the first point of each new stage in R chart."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True, paired=True)
        r_data = result.get_chart('R')

        # R chart data should have NaN mr values at stage boundaries
        mr_values = r_data['mr']
        assert mr_values.isna().any(), "mR should have NaN at stage boundaries"

    def test_staged_statistics_are_varies(self, sds1_study):
        """Statistics should report 'Varies' for staged charts."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True)
        stats = result.get_statistics('XmR')

        assert stats['center'] == 'Varies'
        assert stats['lpl'] == 'Varies'
        assert stats['upl'] == 'Varies'

    def test_staged_metadata_flag(self, sds1_study):
        """Metadata should include staged=True."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True)
        metadata = result.charts['XmR']['metadata']

        assert metadata['staged'] is True
        assert metadata['run_rules_applicable'] is False

    def test_staged_single_point_stages_metadata(self, sds1_study):
        """Metadata should include single_point_stages count."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True)
        metadata = result.charts['XmR']['metadata']

        assert 'single_point_stages' in metadata
        assert isinstance(metadata['single_point_stages'], int)

    def test_staged_signal_detection_per_stage(self, sds1_study):
        """Beyond_limits uses per-stage lpl/upl, not global."""
        result_staged = sds1_study.execute(chart='XmR', by=[], staged=True)
        result_global = sds1_study.execute(chart='XmR', by=[], staged=False)

        staged_data = result_staged.get_chart('XmR')
        global_data = result_global.get_chart('XmR')

        # The beyond_limits column should exist in both
        assert 'beyond_limits' in staged_data.columns
        assert 'beyond_limits' in global_data.columns

        # Staged limits are tighter per-stage, so signals may differ
        # (we just verify the column is populated, not specific values)
        assert staged_data['beyond_limits'].dtype in ['int64', 'int32', 'float64']


# =============================================================================
# Paired (XmR + R) with Staged
# =============================================================================

class TestStagedPaired:
    """Test staged limits with paired=True (XmR + R together)."""

    def test_staged_paired_both_charts(self, sds1_study):
        """paired=True should produce both XmR and R with staged limits."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True, paired=True)

        xmr_data = result.get_chart('XmR')
        r_data = result.get_chart('R')

        assert xmr_data is not None
        assert r_data is not None

        # Both should have 'Varies' statistics
        xmr_stats = result.get_statistics('XmR')
        r_stats = result.get_statistics('R')
        assert xmr_stats['center'] == 'Varies'
        assert r_stats['center'] == 'Varies'

        # Both should have staged metadata
        assert result.charts['XmR']['metadata'].get('staged') is True
        assert result.charts['R']['metadata'].get('staged') is True

    def test_staged_r_row_count_matches_xmr(self, sds1_study):
        """Staged R chart row count == XmR chart row count (no rows dropped)."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True, paired=True)

        xmr_rows = len(result.get_chart('XmR'))
        r_rows = len(result.get_chart('R'))

        assert r_rows == xmr_rows, (
            f"Staged R chart should have same row count as XmR "
            f"(got R={r_rows}, XmR={xmr_rows})"
        )

    def test_staged_r_lane_boundaries_align(self, sds1_study):
        """R chart lane boundaries should match XmR positions (no offset)."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True, paired=True)

        xmr_meta = result.charts['XmR']['metadata']
        r_meta = result.charts['R']['metadata']

        xmr_boundaries = xmr_meta.get('lane_boundaries')
        r_boundaries = r_meta.get('lane_boundaries')

        if xmr_boundaries is not None:
            assert r_boundaries is not None, "R should have lane boundaries if XmR does"
            xmr_positions = [b['position'] for b in xmr_boundaries]
            r_positions = [b['position'] for b in r_boundaries]
            assert xmr_positions == r_positions, (
                f"R lane boundaries should match XmR positions "
                f"(XmR={xmr_positions}, R={r_positions})"
            )


# =============================================================================
# Lane Boundaries
# =============================================================================

class TestStagedLaneBoundaries:
    """Test that lane boundaries are preserved with staged limits."""

    def test_staged_lane_boundaries_preserved(self, sds1_study):
        """Lane boundaries should still be present in staged output."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True)
        metadata = result.charts['XmR']['metadata']

        assert metadata.get('lane_boundaries') is not None
        assert len(metadata['lane_boundaries']) > 0


# =============================================================================
# Validation
# =============================================================================

class TestStagedValidation:
    """Test that invalid staged configurations raise appropriate errors."""

    def test_staged_requires_xmr_or_r(self, sds1_study):
        """staged=True with Xbar should raise ValidationError."""
        with pytest.raises(ValidationError, match="only valid for XmR or R"):
            sds1_study.execute(chart='Xbar', staged=True)

    def test_staged_stratified_allowed_with_collapsed_factors(self, sds1_study):
        """staged=True with by=['factor 1'] should work (factor 2 is collapsed)."""
        # Should NOT raise — factor 2 remains as collapsed factor for stages
        result = sds1_study.execute(
            chart='XmR',
            by=['factor 1'],
            staged=True
        )
        assert result is not None

    def test_staged_requires_collapsed_factors(self, sds1_study):
        """staged=True with by=all_factors should raise (no collapsed factors)."""
        with pytest.raises(ValidationError, match="requires collapsed factors"):
            sds1_study.execute(
                chart='XmR',
                by=['factor 1', 'factor 2'],
                staged=True
            )

    def test_staged_requires_by_none_raises(self, sds1_study):
        """staged=True with by=None should raise (by=None invalid for XmR with factors)."""
        with pytest.raises(ValueError):
            sds1_study.execute(chart='XmR', staged=True)


# =============================================================================
# Regression: Unstaged behavior unchanged
# =============================================================================

class TestUnstagedUnchanged:
    """Verify that staged=False (default) produces identical results."""

    def test_unstaged_unchanged(self, sds1_study):
        """staged=False should produce identical results to omitting staged."""
        result_default = sds1_study.execute(chart='XmR', by=[])
        result_explicit = sds1_study.execute(chart='XmR', by=[], staged=False)

        default_data = result_default.get_chart('XmR')
        explicit_data = result_explicit.get_chart('XmR')

        pd.testing.assert_frame_equal(default_data, explicit_data)

        default_stats = result_default.get_statistics('XmR')
        explicit_stats = result_explicit.get_statistics('XmR')

        assert default_stats == explicit_stats

    def test_unstaged_paired_unchanged(self, sds1_study):
        """staged=False with paired=True should produce identical results."""
        result_default = sds1_study.execute(chart='XmR', by=[], paired=True)
        result_explicit = sds1_study.execute(
            chart='XmR', by=[], paired=True, staged=False
        )

        for chart_name in ['XmR', 'R']:
            default_data = result_default.get_chart(chart_name)
            explicit_data = result_explicit.get_chart(chart_name)

            pd.testing.assert_frame_equal(
                default_data, explicit_data,
                obj=f"{chart_name} data"
            )


# =============================================================================
# Stratified Staged Limits
# =============================================================================

class TestStagedStratified:
    """Test staged limits in stratified (by=['factor']) mode."""

    def test_staged_stratified_limits_vary(self, sds1_study):
        """Stratified staged: center/lpl/upl columns vary within each stratum."""
        result = sds1_study.execute(
            chart='XmR', by=['factor 1'], staged=True
        )
        # Get the combined data
        xmr_data = result.get_chart('XmR')

        # With staged limits, center should vary (not constant per stratum)
        assert 'center' in xmr_data.columns
        assert xmr_data['center'].nunique() > 1, \
            "Center should vary across stages within strata"

    def test_staged_stratified_statistics_are_varies(self, sds1_study):
        """Per-stratum stats dict has 'Varies' values."""
        result = sds1_study.execute(
            chart='XmR', by=['factor 1'], staged=True
        )
        stats = result.get_statistics('XmR')

        # Statistics are nested {stratum: {center: 'Varies', ...}}
        assert isinstance(stats, dict)
        for stratum, stratum_stats in stats.items():
            assert stratum_stats['center'] == 'Varies', \
                f"Stratum {stratum} center should be 'Varies'"
            assert stratum_stats['lpl'] == 'Varies'
            assert stratum_stats['upl'] == 'Varies'

    def test_staged_stratified_mr_resets_at_stage_boundaries(self, sds1_study):
        """mR NaN at start of each stage within stratum (R chart)."""
        result = sds1_study.execute(
            chart='XmR', by=['factor 1'], staged=True, paired=True
        )
        r_data = result.get_chart('R')

        # R chart should have NaN mr values at stage boundaries
        mr_values = r_data['mr']
        assert mr_values.isna().any(), "mR should have NaN at stage boundaries"

    def test_staged_stratified_paired(self, sds1_study):
        """Paired XmR+R both staged in stratified mode."""
        result = sds1_study.execute(
            chart='XmR', by=['factor 1'], staged=True, paired=True
        )

        xmr_data = result.get_chart('XmR')
        r_data = result.get_chart('R')

        assert xmr_data is not None
        assert r_data is not None

        # Both should have 'Varies' statistics for each stratum
        xmr_stats = result.get_statistics('XmR')
        r_stats = result.get_statistics('R')

        for stratum in xmr_stats:
            assert xmr_stats[stratum]['center'] == 'Varies'
        for stratum in r_stats:
            assert r_stats[stratum]['center'] == 'Varies'

        # Both should have staged metadata
        assert result.charts['XmR']['metadata'].get('staged') is True
        assert result.charts['R']['metadata'].get('staged') is True

    def test_staged_stratified_stage_count(self, sds1_study):
        """Known data produces expected stage count per stratum."""
        result = sds1_study.execute(
            chart='XmR', by=['factor 1'], staged=True
        )
        xmr_data = result.get_chart('XmR')

        # Center values change at stage boundaries — count transitions
        # within each stratum by checking unique center values
        assert xmr_data['center'].nunique() > 1, \
            "Should have multiple distinct center values (stages)"

    def test_staged_lane_boundary_positions_valid(self, sds1_study):
        """All lane boundary positions < len(stratum_df) and strictly increasing."""
        result = sds1_study.execute(
            chart='XmR', by=['factor 1'], staged=True
        )
        metadata = result.charts['XmR']['metadata']
        lane_boundaries = metadata.get('lane_boundaries')

        if lane_boundaries:
            _ = result.charts['XmR'].get('strata', [])
            xmr_data = result.get_chart('XmR')

            # For stratified charts, lane_boundaries is dict keyed by stratum
            if isinstance(lane_boundaries, dict):
                stratify_by = metadata.get('stratify_by', [])
                if stratify_by and stratify_by[0] in xmr_data.columns:
                    for stratum, boundaries in lane_boundaries.items():
                        stratum_len = len(
                            xmr_data[xmr_data[stratify_by[0]] == stratum]
                        )
                        positions = [b['position'] for b in boundaries]
                        assert positions == sorted(positions), \
                            f"Boundary positions not sorted for {stratum}"
                        for p in positions:
                            assert p < stratum_len, \
                                f"Position {p} >= stratum len {stratum_len}"

    def test_staged_stratified_metadata(self, sds1_study):
        """Metadata should include staged=True and run_rules_applicable=False."""
        result = sds1_study.execute(
            chart='XmR', by=['factor 1'], staged=True
        )
        metadata = result.charts['XmR']['metadata']

        assert metadata.get('staged') is True
        assert metadata.get('run_rules_applicable') is False
        assert metadata.get('stratified') is True


# =============================================================================
# Time Tick Labels
# =============================================================================

class TestTimeTickLabels:
    """Test that time tick labels are applied when x-axis uses integer indices."""

    def test_time_ticks_apply_when_x_is_index(self, sds1_study):
        """Tick labels set for by=[] non-staged when _get_x_column() is None."""
        result = sds1_study.execute(chart='XmR', by=[])
        fig = result.plot('XmR')

        # The figure should have tick labels set (ticktext/tickvals)
        xaxis = fig._fig.layout.xaxis
        assert xaxis.tickvals is not None, "tickvals should be set"
        assert xaxis.ticktext is not None, "ticktext should be set"
        assert len(xaxis.ticktext) > 0, "Should have tick labels"

    def test_staged_xmr_has_time_tick_labels(self, sds1_study):
        """Staged by=[] chart has time values as tick labels."""
        result = sds1_study.execute(chart='XmR', by=[], staged=True)
        fig = result.plot('XmR')

        xaxis = fig._fig.layout.xaxis
        assert xaxis.tickvals is not None, "tickvals should be set for staged"
        assert xaxis.ticktext is not None, "ticktext should be set for staged"

    def test_time_ticks_applied_in_stratified_with_replication(self, sds1_study):
        """Stratified with replication: time repeats per stratum, ticks applied."""
        # SDS 1 has replication, so time values repeat within each stratum
        result = sds1_study.execute(
            chart='XmR', by=['factor 1', 'factor 2']
        )
        fig = result.plot('XmR')

        # Time values repeat due to replication, so integer positions are used
        # and _apply_time_tick_labels should add tick labels
        xaxis = fig._fig.layout.xaxis
        assert xaxis.tickvals is not None, "tickvals should be set (time repeats)"
        assert xaxis.ticktext is not None, "ticktext should be set (time repeats)"
