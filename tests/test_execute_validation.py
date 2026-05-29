"""Tests for execute() parameter validation — invalid combinations raise ValidationError.

Covers validation checks:
- companion + Histogram
- phased + value
- bins + non-Histogram
- recentered value resolution
- recentered requires VAS decomposition
"""

import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic
from processbehavior.exceptions import ValidationError

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope='module')
def sds1_study():
    """SDS1 study with two factors — supports all chart types."""
    df = synthetic.make_design(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


@pytest.fixture(scope='module')
def sds1_single_factor_study():
    """SDS1 study with single factor — for XmR tests without ambiguity."""
    df = synthetic.make_design(1, K1=3, K2=1, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1'])


# =============================================================================
# Invalid combinations
# =============================================================================


class TestInvalidCombinations:
    """Validation for invalid parameter combinations."""

    def test_companion_histogram_raises(self, sds1_study):
        """companion=True with Histogram raises ValidationError."""
        with pytest.raises(ValidationError, match='companion.*Histogram'):
            sds1_study.execute(chart='Histogram', companion=True)

    def test_phased_with_value_allowed(self, sds2_study):
        """phased=True with value= (residual) is allowed."""
        result = sds2_study.execute(chart='X', by=[], phased=True, value='R2')
        assert len(result.get_chart('X')) > 0

    def test_bins_non_histogram_raises(self, sds1_single_factor_study):
        """bins with non-Histogram chart raises ValidationError."""
        with pytest.raises(ValidationError, match='bins.*Histogram'):
            sds1_single_factor_study.execute(chart='X', by=[], bins=10)

    def test_bins_non_histogram_raises_xbar(self, sds1_study):
        """bins with Xbar chart raises ValidationError."""
        with pytest.raises(ValidationError, match='bins.*Histogram'):
            sds1_study.execute(chart='Xbar', bins=5)


# =============================================================================
# Valid combinations (sanity checks)
# =============================================================================


class TestValidCombinations:
    """Verify that valid parameter combinations still succeed."""

    def test_histogram_default_bins_succeeds(self, sds1_study):
        """chart='Histogram' without explicit bins uses default."""
        result = sds1_study.execute(chart='Histogram')
        assert result is not None

    def test_xmr_without_bins_succeeds(self, sds1_single_factor_study):
        """chart='X' without bins (None) succeeds."""
        result = sds1_single_factor_study.execute(chart='X', by=[])
        assert result is not None

    def test_histogram_with_bins_succeeds(self, sds1_study):
        """chart='Histogram' with explicit bins succeeds."""
        result = sds1_study.execute(chart='Histogram', bins=15)
        assert result is not None

    def test_companion_xmr_succeeds(self, sds1_single_factor_study):
        """companion=True with X succeeds."""
        result = sds1_single_factor_study.execute(chart='X', by=[], companion=True)
        assert result is not None


# =============================================================================
# Companion charts with residual values
# =============================================================================


class TestCompanionWithResiduals:
    """companion=True should return both charts when charting residuals."""

    def test_xmr_companion_with_residual_returns_both(self, sds2_study):
        """X + companion + value='R2' → both X and mR."""
        result = sds2_study.execute(chart='X', by=[], companion=True, value='R2')
        assert 'X' in result.charts
        assert 'mR' in result.charts

    def test_xbar_companion_with_residual_returns_both(self, sds1_study):
        """Xbar + companion + value='R5' → both Xbar and S."""
        result = sds1_study.execute(chart='Xbar', companion=True, value='R5')
        assert 'Xbar' in result.charts
        assert 'S' in result.charts

    def test_xmr_no_companion_with_residual_returns_single(self, sds2_study):
        """X + companion=False + value='R2' → only X."""
        result = sds2_study.execute(chart='X', by=[], companion=False, value='R2')
        assert 'X' in result.charts
        assert 'mR' not in result.charts

    def test_companion_residual_preserves_metadata(self, sds1_study):
        """Companion residual charts should have residual metadata on both."""
        result = sds1_study.execute(chart='Xbar', companion=True, value='R5')
        for chart_name in ('Xbar', 'S'):
            metadata = result.charts[chart_name]['metadata']
            assert metadata.get('residual_type') == 'R5'


# =============================================================================
# Recentered value resolution
# =============================================================================


@pytest.fixture(scope='module')
def sds2_study():
    """SDS2 study — supports R2_XmR residual charts."""
    df = synthetic.make_design(2, K1=2, K2=2, T=8, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


class TestRecenteredValueResolution:
    """Verify recentered parameter selects the correct data column."""

    def test_recentered_false_uses_r2(self, sds2_study):
        """value='R2', recentered=False → charts R2 column."""
        result = sds2_study.execute(chart='X', by=[], value='R2', recentered=False)
        assert result.charts['X']['metadata']['value_col'] == 'R2'

    def test_recentered_true_uses_rcr2(self, sds2_study):
        """value='R2', recentered=True → charts RCR2 column."""
        result = sds2_study.execute(chart='X', by=[], value='R2', recentered=True)
        assert result.charts['X']['metadata']['value_col'] == 'RCR2'

    def test_rcr2_passthrough(self, sds2_study):
        """value='RCR2' directly selects RCR2 without recentered flag."""
        result = sds2_study.execute(chart='X', by=[], value='RCR2')
        assert result.charts['X']['metadata']['value_col'] == 'RCR2'

    def test_recentered_true_without_value_raises(self, sds1_study):
        """recentered=True with value=None raises ValidationError."""
        with pytest.raises(ValidationError, match='recentered.*requires.*residual'):
            sds1_study.execute(chart='Xbar', recentered=True)


# =============================================================================
# Recentered requires VAS decomposition
# =============================================================================


@pytest.fixture(scope='module')
def no_time_study():
    """Study with factors but no time — no VAS residuals."""
    df = synthetic.make_design(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(response='y', factors=['factor 1', 'factor 2'])


class TestRecenteredRequiresVAS:
    def test_recentered_without_time_raises(self, no_time_study):
        """recentered=True without time raises clear ValidationError."""
        with pytest.raises(ValidationError, match='factors.*time'):
            no_time_study.execute(chart='X', by=[], value='R2', recentered=True)


# =============================================================================
# Residual accessor
# =============================================================================


class TestResidualAccessor:
    """study.residuals accessor provides IDE-friendly residual access."""

    def test_accessor_has_residual_attributes(self, sds1_study):
        """Full study exposes R1-R5 as attributes."""
        r = sds1_study.residuals
        assert r.R1 == 'R1'
        assert r.R2 == 'R2'
        assert r.R3 == 'R3'
        assert r.R4 == 'R4'
        assert r.R5 == 'R5'

    def test_accessor_returns_string(self, sds1_study):
        """Accessor values are strings passable to execute(value=...)."""
        assert isinstance(sds1_study.residuals.R2, str)

    def test_accessor_empty_without_time(self, no_time_study):
        """Study without time has empty residuals accessor."""
        assert len(no_time_study.residuals) == 0

    def test_accessor_no_attribute_without_time(self, no_time_study):
        """Accessing R2 on factors-only study raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = no_time_study.residuals.R2

    def test_residual_charts_empty_without_time(self, no_time_study):
        """residual_charts also empty when residuals not computed."""
        assert no_time_study.residual_charts == []

    def test_execute_with_accessor(self, sds1_study):
        """End-to-end: study.execute(chart=..., value=study.residuals.R5)."""
        result = sds1_study.execute(chart=sds1_study.charts.Xbar, value=sds1_study.residuals.R5)
        assert result is not None
