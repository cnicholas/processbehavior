"""Tests for n_sigma and n_mode parameters on Xbar/S charts.

These parameters control limit sensitivity and subgroup-size handling:
- n_sigma: replaces hard-coded 3-sigma multiplier (wider/narrower limits)
- n_mode: "actual" uses per-subgroup N_k, "average" uses mean N-bar
"""

import numbers

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic
from processbehavior.exceptions import ValidationError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sds1_study():
    """SDS1 study with constant subgroup sizes (n=3 per cell)."""
    df = synthetic.make_sds(1, K1=3, K2=2, T=4, n_min=3, n_max=3, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


@pytest.fixture
def sds1_varying_n_study():
    """SDS1 study with VARYING subgroup sizes (n_min=2, n_max=5)."""
    df = synthetic.make_sds(1, K1=3, K2=2, T=4, n_min=2, n_max=5, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


@pytest.fixture
def xmr_study():
    """XmR-only study (time series, no factors)."""
    df = synthetic.make_sds(4, T=20, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


# =============================================================================
# Default Behavior Unchanged
# =============================================================================


class TestDefaultBehavior:
    """Verify n_sigma=3.0, n_mode="actual" produces identical results."""

    def test_explicit_defaults_match_omitted(self, sds1_study):
        """Passing n_sigma=3.0, n_mode='actual' explicitly is identical to omitting."""
        result_default = sds1_study.execute(chart='Xbar', companion=True)
        result_explicit = sds1_study.execute(chart='Xbar', companion=True, n_sigma=3.0, n_mode='actual')

        # Xbar data identical
        pd.testing.assert_frame_equal(
            result_default.charts['Xbar']['data'],
            result_explicit.charts['Xbar']['data'],
        )
        # S data identical
        pd.testing.assert_frame_equal(
            result_default.charts['S']['data'],
            result_explicit.charts['S']['data'],
        )

    def test_xbar_statistics_unchanged(self, sds1_study):
        """Statistics match when using default n_sigma."""
        result = sds1_study.execute(chart='Xbar')
        stats = result.charts['Xbar']['statistics']
        assert 'center' in stats
        assert stats['center'] is not None

    def test_metadata_includes_n_sigma_n_mode(self, sds1_study):
        """Metadata always includes n_sigma and n_mode."""
        result = sds1_study.execute(chart='Xbar', companion=True)
        xbar_meta = result.charts['Xbar']['metadata']
        s_meta = result.charts['S']['metadata']

        assert xbar_meta['n_sigma'] == 3.0
        assert xbar_meta['n_mode'] == 'actual'
        assert s_meta['n_sigma'] == 3.0
        assert s_meta['n_mode'] == 'actual'

    def test_metadata_no_n_avg_when_actual(self, sds1_study):
        """n_avg should NOT appear in metadata when n_mode='actual'."""
        result = sds1_study.execute(chart='Xbar', companion=True)
        assert 'n_avg' not in result.charts['Xbar']['metadata']
        assert 'n_avg' not in result.charts['S']['metadata']


# =============================================================================
# n_sigma Affects Limit Width
# =============================================================================


class TestNSigmaWidth:
    """n_sigma controls how wide/narrow the limits are."""

    def test_wider_sigma_gives_wider_limits(self, sds1_study):
        """n_sigma=4 should produce wider limits than n_sigma=3."""
        r3 = sds1_study.execute(chart='Xbar', n_sigma=3.0)
        r4 = sds1_study.execute(chart='Xbar', n_sigma=4.0)

        d3 = r3.charts['Xbar']['data']
        d4 = r4.charts['Xbar']['data']

        # UPL should be farther from center with larger sigma
        assert (d4['upl'] >= d3['upl']).all()
        # LPL should be farther from center (lower) with larger sigma
        assert (d4['lpl'] <= d3['lpl']).all()

    def test_narrower_sigma_gives_narrower_limits(self, sds1_study):
        """n_sigma=2 should produce narrower limits than n_sigma=3."""
        r2 = sds1_study.execute(chart='Xbar', n_sigma=2.0)
        r3 = sds1_study.execute(chart='Xbar', n_sigma=3.0)

        d2 = r2.charts['Xbar']['data']
        d3 = r3.charts['Xbar']['data']

        assert (d2['upl'] <= d3['upl']).all()
        assert (d2['lpl'] >= d3['lpl']).all()

    def test_monotonic_ordering(self, sds1_study):
        """upl_2 < upl_3 < upl_4 and lpl_2 > lpl_3 > lpl_4."""
        r2 = sds1_study.execute(chart='Xbar', n_sigma=2.0)
        r3 = sds1_study.execute(chart='Xbar', n_sigma=3.0)
        r4 = sds1_study.execute(chart='Xbar', n_sigma=4.0)

        d2 = r2.charts['Xbar']['data']
        d3 = r3.charts['Xbar']['data']
        d4 = r4.charts['Xbar']['data']

        assert (d2['upl'] < d3['upl']).all()
        assert (d3['upl'] < d4['upl']).all()
        assert (d2['lpl'] > d3['lpl']).all()
        assert (d3['lpl'] > d4['lpl']).all()

    def test_s_chart_sigma_affects_limits(self, sds1_study):
        """n_sigma also controls S chart limits via b3/b4."""
        r2 = sds1_study.execute(chart='Xbar', companion=True, n_sigma=2.0)
        r3 = sds1_study.execute(chart='Xbar', companion=True, n_sigma=3.0)

        s2 = r2.charts['S']['data']
        s3 = r3.charts['S']['data']

        # S chart UPL should be narrower with smaller sigma
        assert (s2['upl'] <= s3['upl']).all()

    def test_center_unchanged_by_n_sigma(self, sds1_study):
        """Center line should be identical regardless of n_sigma."""
        r2 = sds1_study.execute(chart='Xbar', n_sigma=2.0)
        r3 = sds1_study.execute(chart='Xbar', n_sigma=3.0)

        d2 = r2.charts['Xbar']['data']
        d3 = r3.charts['Xbar']['data']

        pd.testing.assert_series_equal(d2['center'], d3['center'])


# =============================================================================
# n_mode="average" Makes Limits Constant
# =============================================================================


class TestNModeAverage:
    """n_mode='average' uses mean N-bar for constant limits."""

    def test_average_mode_constant_limits(self, sds1_varying_n_study):
        """With n_mode='average', all rows have identical lpl/upl."""
        result = sds1_varying_n_study.execute(chart='Xbar', n_mode='average')
        data = result.charts['Xbar']['data']

        # All lpl values should be identical
        assert data['lpl'].nunique() == 1, f'Expected 1 unique lpl, got {data["lpl"].nunique()}'
        # All upl values should be identical
        assert data['upl'].nunique() == 1, f'Expected 1 unique upl, got {data["upl"].nunique()}'

    def test_average_mode_scalar_statistics(self, sds1_varying_n_study):
        """With n_mode='average', statistics show scalar values (not None)."""
        result = sds1_varying_n_study.execute(chart='Xbar', n_mode='average')
        stats = result.charts['Xbar']['statistics']

        assert isinstance(stats['lpl'], numbers.Number), f'Expected numeric lpl, got {stats["lpl"]!r}'
        assert isinstance(stats['upl'], numbers.Number), f'Expected numeric upl, got {stats["upl"]!r}'
        assert isinstance(stats['N'], numbers.Number), f'Expected numeric N, got {stats["N"]!r}'

    def test_average_mode_s_chart_constant(self, sds1_varying_n_study):
        """S chart also has constant limits with n_mode='average'."""
        result = sds1_varying_n_study.execute(chart='Xbar', companion=True, n_mode='average')
        s_data = result.charts['S']['data']

        assert s_data['lpl'].nunique() == 1
        assert s_data['upl'].nunique() == 1

    def test_average_mode_metadata_includes_n_avg(self, sds1_varying_n_study):
        """Metadata includes n_avg when n_mode='average'."""
        result = sds1_varying_n_study.execute(chart='Xbar', companion=True, n_mode='average')
        xbar_meta = result.charts['Xbar']['metadata']
        s_meta = result.charts['S']['metadata']

        assert 'n_avg' in xbar_meta
        assert isinstance(xbar_meta['n_avg'], (int, float))
        assert xbar_meta['n_avg'] > 0

        assert 'n_avg' in s_meta
        assert s_meta['n_avg'] == xbar_meta['n_avg']


# =============================================================================
# n_mode="average" Uses Correct N-bar
# =============================================================================


class TestNAvgCorrectness:
    """Verify that the computed N-bar matches the actual mean subgroup size."""

    def test_n_avg_value(self, sds1_varying_n_study):
        """n_avg in metadata should equal the mean of subgroup sizes."""
        result = sds1_varying_n_study.execute(chart='Xbar', n_mode='average')
        meta = result.charts['Xbar']['metadata']

        # Compute expected n_avg from the study's dataset
        ds = sds1_varying_n_study.dataset
        spec = sds1_varying_n_study._spec
        # Get actual subgroup sizes from the analysis
        groupby_cols = spec.rsg_vars_list + [spec.time_var]
        n_per_group = ds.groupby(groupby_cols, observed=True)[spec.response_var].count()
        # Filter out n=1 groups (same as Xbar calculation)
        n_per_group = n_per_group[n_per_group > 1]
        expected_n_avg = round(n_per_group.mean(), spec.round_to)

        assert meta['n_avg'] == expected_n_avg


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Verify invalid inputs are rejected with clear errors."""

    def test_n_sigma_zero_raises(self, sds1_study):
        with pytest.raises(ValidationError, match='n_sigma must be a finite number > 0'):
            sds1_study.execute(chart='Xbar', n_sigma=0)

    def test_n_sigma_negative_raises(self, sds1_study):
        with pytest.raises(ValidationError, match='n_sigma must be a finite number > 0'):
            sds1_study.execute(chart='Xbar', n_sigma=-1)

    def test_n_sigma_inf_raises(self, sds1_study):
        with pytest.raises(ValidationError, match='n_sigma must be a finite number > 0'):
            sds1_study.execute(chart='Xbar', n_sigma=float('inf'))

    def test_n_mode_invalid_raises(self, sds1_study):
        with pytest.raises(ValidationError, match="n_mode must be 'actual' or 'average'"):
            sds1_study.execute(chart='Xbar', n_mode='foo')

    def test_xmr_n_sigma_raises(self, xmr_study):
        with pytest.raises(ValidationError, match='only supported for Xbar/S'):
            xmr_study.execute(chart='X', by=[], n_sigma=2.0)

    def test_xmr_n_mode_raises(self, xmr_study):
        with pytest.raises(ValidationError, match='only supported for Xbar/S'):
            xmr_study.execute(chart='X', by=[], n_mode='average')

    def test_histogram_n_sigma_raises(self, sds1_study):
        with pytest.raises(ValidationError, match='only supported for Xbar/S'):
            sds1_study.execute(chart='Histogram', n_sigma=2.0)

    def test_default_values_no_validation_error_on_non_xbar(self, xmr_study):
        """Default n_sigma=3.0, n_mode='actual' should NOT raise on XmR."""
        result = xmr_study.execute(chart='X', by=[])
        assert result is not None


# =============================================================================
# Companion Consistency
# =============================================================================


class TestCompanionConsistency:
    """Companion Xbar+S should both use the same n_sigma and n_mode."""

    def test_companion_n_sigma_applied_to_both(self, sds1_study):
        """Both Xbar and S use the custom n_sigma."""
        result = sds1_study.execute(chart='Xbar', companion=True, n_sigma=2.5)
        xbar_meta = result.charts['Xbar']['metadata']
        s_meta = result.charts['S']['metadata']

        assert xbar_meta['n_sigma'] == 2.5
        assert s_meta['n_sigma'] == 2.5

    def test_companion_n_mode_applied_to_both(self, sds1_varying_n_study):
        """Both Xbar and S use n_mode='average'."""
        result = sds1_varying_n_study.execute(chart='Xbar', companion=True, n_mode='average')
        xbar_meta = result.charts['Xbar']['metadata']
        s_meta = result.charts['S']['metadata']

        assert xbar_meta['n_mode'] == 'average'
        assert s_meta['n_mode'] == 'average'

    def test_companion_combined_options(self, sds1_varying_n_study):
        """Both n_sigma and n_mode applied together in companion mode."""
        result = sds1_varying_n_study.execute(chart='Xbar', companion=True, n_sigma=2.5, n_mode='average')

        xbar_data = result.charts['Xbar']['data']
        s_data = result.charts['S']['data']

        # Constant limits from n_mode="average"
        assert xbar_data['lpl'].nunique() == 1
        assert s_data['lpl'].nunique() == 1

        # Metadata correct
        assert result.charts['Xbar']['metadata']['n_sigma'] == 2.5
        assert result.charts['S']['metadata']['n_sigma'] == 2.5
