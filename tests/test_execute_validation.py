"""Tests for execute() parameter validation — invalid combinations raise ValidationError.

Covers the three new validation checks:
- companion + Histogram
- phased + value
- bins + non-Histogram
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
    df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1', 'factor 2']
    )


@pytest.fixture(scope='module')
def sds1_single_factor_study():
    """SDS1 study with single factor — for XmR tests without ambiguity."""
    df = synthetic.make_sds(1, K1=3, K2=1, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1']
    )


# =============================================================================
# Invalid combinations
# =============================================================================


class TestInvalidCombinations:
    """Validation for invalid parameter combinations."""

    def test_companion_histogram_raises(self, sds1_study):
        """companion=True with Histogram raises ValidationError."""
        with pytest.raises(ValidationError, match="companion.*Histogram"):
            sds1_study.execute(chart='Histogram', companion=True)

    def test_phased_with_value_raises(self, sds1_study):
        """phased=True with value= raises ValidationError."""
        with pytest.raises(ValidationError, match="phased.*value"):
            sds1_study.execute(
                chart='XmR', by=[], phased=True, value='R2'
            )

    def test_bins_non_histogram_raises(self, sds1_single_factor_study):
        """bins with non-Histogram chart raises ValidationError."""
        with pytest.raises(ValidationError, match="bins.*Histogram"):
            sds1_single_factor_study.execute(chart='XmR', by=[], bins=10)

    def test_bins_non_histogram_raises_xbar(self, sds1_study):
        """bins with Xbar chart raises ValidationError."""
        with pytest.raises(ValidationError, match="bins.*Histogram"):
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
        """chart='XmR' without bins (None) succeeds."""
        result = sds1_single_factor_study.execute(chart='XmR', by=[])
        assert result is not None

    def test_histogram_with_bins_succeeds(self, sds1_study):
        """chart='Histogram' with explicit bins succeeds."""
        result = sds1_study.execute(chart='Histogram', bins=15)
        assert result is not None

    def test_companion_xmr_succeeds(self, sds1_single_factor_study):
        """companion=True with XmR succeeds."""
        result = sds1_single_factor_study.execute(
            chart='XmR', by=[], companion=True
        )
        assert result is not None
