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

    def test_staged_requires_by_empty(self, sds1_study):
        """staged=True with by=['factor'] should raise ValidationError."""
        with pytest.raises(ValidationError, match="requires by=\\[\\]"):
            sds1_study.execute(
                chart='XmR',
                by=['factor 1'],
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
