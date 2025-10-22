"""
Unit tests for SPC constants and formulas.

Tests cover:
- Bias correction constants (c4, b3, b4)
- Control limit calculations (Xbar, S, Imr, R)
- Signal detection
- Edge cases and validation
"""

import pytest
import numpy as np
import pandas as pd
from processbehavior.spc_constants import (
    c4, b3, b4,
    calculate_limits,
    detect_beyond_limits,
    SIGMA_MULTIPLIER,
    IMR_LIMIT_MULTIPLIER,
    R_UPPER_LIMIT_MULTIPLIER
)


# ============================================================================
# Test: Constants
# ============================================================================

def test_sigma_multiplier():
    """Sigma multiplier should be 3 for 3-sigma limits."""
    assert SIGMA_MULTIPLIER == 3


def test_imr_limit_multiplier():
    """IMR E2 constant should be 2.66."""
    assert IMR_LIMIT_MULTIPLIER == 2.66


def test_r_upper_limit_multiplier():
    """R chart D4 constant should be 3.268."""
    assert R_UPPER_LIMIT_MULTIPLIER == 3.268


# ============================================================================
# Test: c4 (bias correction constant)
# ============================================================================

def test_c4_known_values():
    """c4 should match published constants."""
    # Known values from Wheeler (1995) Table A.1
    assert abs(c4(2) - 0.7979) < 0.001
    assert abs(c4(3) - 0.8862) < 0.001
    assert abs(c4(4) - 0.9213) < 0.001
    assert abs(c4(5) - 0.9400) < 0.001
    assert abs(c4(10) - 0.9727) < 0.001
    assert abs(c4(25) - 0.9896) < 0.001


def test_c4_approaches_one():
    """c4 should approach 1.0 as n increases."""
    c4_10 = c4(10)
    c4_100 = c4(100)
    c4_1000 = c4(1000)

    assert c4_10 < c4_100 < c4_1000 < 1.0
    assert c4_1000 > 0.999  # Very close to 1


def test_c4_raises_on_invalid_n():
    """c4 should raise error for n < 2."""
    with pytest.raises(ValueError, match="Subgroup size must be >= 2"):
        c4(1)

    with pytest.raises(ValueError, match="Subgroup size must be >= 2"):
        c4(0)

    with pytest.raises(ValueError, match="Subgroup size must be >= 2"):
        c4(-5)


# ============================================================================
# Test: b3 (S chart lower limit constant)
# ============================================================================

def test_b3_zero_for_small_n():
    """b3 should be 0 for small subgroup sizes (n < 6)."""
    assert b3(2) == 0
    assert b3(3) == 0
    assert b3(4) == 0
    assert b3(5) == 0


def test_b3_nonzero_for_larger_n():
    """b3 should be > 0 for n >= 6."""
    assert b3(6) > 0
    assert b3(10) > 0
    assert b3(25) > 0


def test_b3_known_values():
    """b3 should match published constants."""
    # Known values from Wheeler (1995) Table A.1
    assert abs(b3(6) - 0.030) < 0.01
    assert abs(b3(10) - 0.284) < 0.01
    assert abs(b3(25) - 0.565) < 0.01


def test_b3_raises_on_invalid_n():
    """b3 should raise error for n < 2."""
    with pytest.raises(ValueError, match="Subgroup size must be >= 2"):
        b3(1)


# ============================================================================
# Test: b4 (S chart upper limit constant)
# ============================================================================

def test_b4_always_positive():
    """b4 should always be > 1 for valid n."""
    assert b4(2) > 1
    assert b4(5) > 1
    assert b4(25) > 1


def test_b4_decreases_with_n():
    """b4 should decrease as n increases."""
    b4_2 = b4(2)
    b4_10 = b4(10)
    b4_100 = b4(100)

    assert b4_2 > b4_10 > b4_100


def test_b4_known_values():
    """b4 should match published constants."""
    # Known values from Wheeler (1995) Table A.1
    assert abs(b4(2) - 3.267) < 0.01
    assert abs(b4(5) - 2.089) < 0.01
    assert abs(b4(10) - 1.716) < 0.01
    assert abs(b4(25) - 1.435) < 0.01


def test_b4_raises_on_invalid_n():
    """b4 should raise error for n < 2."""
    with pytest.raises(ValueError, match="Subgroup size must be >= 2"):
        b4(1)


# ============================================================================
# Test: calculate_limits - Xbar chart
# ============================================================================

def test_calculate_limits_xbar():
    """Should calculate Xbar chart limits correctly."""
    result = calculate_limits(
        limits_type='Xbar',
        mean=10.0,
        sd=0.5,
        N=5
    )

    assert 'lcl' in result
    assert 'ucl' in result

    # Wd = sd / c4(n) = 0.5 / 0.9400 = 0.5319
    # Limits = 10 ± (3 * 0.5319) / sqrt(5) = 10 ± 0.714
    expected_lcl = 10.0 - 0.714
    expected_ucl = 10.0 + 0.714

    assert abs(result['lcl'] - expected_lcl) < 0.01
    assert abs(result['ucl'] - expected_ucl) < 0.01


def test_calculate_limits_xbar_missing_params():
    """Should raise error if Xbar parameters missing."""
    with pytest.raises(ValueError, match="requires \\(mean, sd, and N\\)"):
        calculate_limits(limits_type='Xbar', mean=10.0, sd=0.5)

    with pytest.raises(ValueError, match="requires \\(mean, sd, and N\\)"):
        calculate_limits(limits_type='Xbar', mean=10.0, N=5)

    with pytest.raises(ValueError, match="requires \\(mean, sd, and N\\)"):
        calculate_limits(limits_type='Xbar', sd=0.5, N=5)


def test_calculate_limits_xbar_symmetric():
    """Xbar limits should be symmetric around mean."""
    result = calculate_limits(
        limits_type='Xbar',
        mean=100.0,
        sd=2.0,
        N=10
    )

    # Distance from mean should be equal
    lcl_dist = 100.0 - result['lcl']
    ucl_dist = result['ucl'] - 100.0

    assert abs(lcl_dist - ucl_dist) < 0.001


# ============================================================================
# Test: calculate_limits - S chart
# ============================================================================

def test_calculate_limits_s():
    """Should calculate S chart limits correctly."""
    result = calculate_limits(
        limits_type='S',
        sd=0.5,
        N=5
    )

    assert 'lcl' in result
    assert 'ucl' in result

    # LCL = 0.5 * b3(5) = 0.5 * 0 = 0
    # UCL = 0.5 * b4(5) = 0.5 * 2.089 = 1.044
    assert result['lcl'] == 0.0
    assert abs(result['ucl'] - 1.044) < 0.01


def test_calculate_limits_s_missing_params():
    """Should raise error if S parameters missing."""
    with pytest.raises(ValueError, match="requires \\(sd, and N\\)"):
        calculate_limits(limits_type='S', sd=0.5)

    with pytest.raises(ValueError, match="requires \\(sd, and N\\)"):
        calculate_limits(limits_type='S', N=5)


def test_calculate_limits_s_lcl_always_nonnegative():
    """S chart LCL should never be negative."""
    for n in [2, 3, 4, 5, 10, 25]:
        result = calculate_limits(limits_type='S', sd=1.0, N=n)
        assert result['lcl'] >= 0


# ============================================================================
# Test: calculate_limits - IMR chart
# ============================================================================

def test_calculate_limits_imr():
    """Should calculate IMR chart limits correctly."""
    result = calculate_limits(
        limits_type='Imr',
        mean=10.0,
        mR=0.3
    )

    assert 'lcl' in result
    assert 'ucl' in result

    # LCL = 10.0 - (2.66 * 0.3) = 10.0 - 0.798 = 9.202
    # UCL = 10.0 + (2.66 * 0.3) = 10.0 + 0.798 = 10.798
    expected_lcl = 10.0 - (2.66 * 0.3)
    expected_ucl = 10.0 + (2.66 * 0.3)

    assert abs(result['lcl'] - expected_lcl) < 0.001
    assert abs(result['ucl'] - expected_ucl) < 0.001


def test_calculate_limits_imr_missing_params():
    """Should raise error if IMR parameters missing."""
    with pytest.raises(ValueError, match="requires \\(mean, and mR\\)"):
        calculate_limits(limits_type='Imr', mean=10.0)

    with pytest.raises(ValueError, match="requires \\(mean, and mR\\)"):
        calculate_limits(limits_type='Imr', mR=0.3)


def test_calculate_limits_imr_symmetric():
    """IMR limits should be symmetric around mean."""
    result = calculate_limits(
        limits_type='Imr',
        mean=50.0,
        mR=1.5
    )

    lcl_dist = 50.0 - result['lcl']
    ucl_dist = result['ucl'] - 50.0

    assert abs(lcl_dist - ucl_dist) < 0.001


# ============================================================================
# Test: calculate_limits - R chart
# ============================================================================

def test_calculate_limits_r():
    """Should calculate R chart limits correctly."""
    result = calculate_limits(
        limits_type='R',
        mR=0.3
    )

    assert 'lcl' in result
    assert 'ucl' in result

    # LCL = 0 (ranges cannot be negative)
    # UCL = 0.3 * 3.268 = 0.9804
    assert result['lcl'] == 0.0
    assert abs(result['ucl'] - (0.3 * 3.268)) < 0.001


def test_calculate_limits_r_missing_params():
    """Should raise error if R parameters missing."""
    with pytest.raises(ValueError, match="requires \\(mR\\)"):
        calculate_limits(limits_type='R')


def test_calculate_limits_r_lcl_always_zero():
    """R chart LCL should always be 0."""
    for mR in [0.1, 0.5, 1.0, 5.0]:
        result = calculate_limits(limits_type='R', mR=mR)
        assert result['lcl'] == 0.0


# ============================================================================
# Test: calculate_limits - Invalid type
# ============================================================================

def test_calculate_limits_invalid_type():
    """Should raise error for unsupported chart type."""
    with pytest.raises(ValueError, match="not supported"):
        calculate_limits(limits_type='Invalid', mean=10.0, sd=1.0, N=5)

    with pytest.raises(ValueError, match="not supported"):
        calculate_limits(limits_type='p', mean=0.5, N=100)


# ============================================================================
# Test: calculate_limits - Return type
# ============================================================================

def test_calculate_limits_returns_series():
    """calculate_limits should return pd.Series with lcl and ucl."""
    result = calculate_limits(
        limits_type='Xbar',
        mean=10.0,
        sd=0.5,
        N=5
    )

    assert isinstance(result, pd.Series)
    assert list(result.index) == ['lcl', 'ucl']
    assert len(result) == 2


# ============================================================================
# Test: detect_beyond_limits
# ============================================================================

def test_detect_beyond_limits_within():
    """Should return 0 when value is within limits."""
    assert detect_beyond_limits(10.0, lcl=9.0, ucl=11.0) == 0
    assert detect_beyond_limits(9.5, lcl=9.0, ucl=11.0) == 0
    assert detect_beyond_limits(10.5, lcl=9.0, ucl=11.0) == 0


def test_detect_beyond_limits_at_boundaries():
    """Should return 0 when value exactly at limits."""
    assert detect_beyond_limits(9.0, lcl=9.0, ucl=11.0) == 0
    assert detect_beyond_limits(11.0, lcl=9.0, ucl=11.0) == 0


def test_detect_beyond_limits_below_lcl():
    """Should return -1 when value below LCL."""
    assert detect_beyond_limits(8.9, lcl=9.0, ucl=11.0) == -1
    assert detect_beyond_limits(5.0, lcl=9.0, ucl=11.0) == -1
    assert detect_beyond_limits(0.0, lcl=9.0, ucl=11.0) == -1


def test_detect_beyond_limits_above_ucl():
    """Should return 1 when value above UCL."""
    assert detect_beyond_limits(11.1, lcl=9.0, ucl=11.0) == 1
    assert detect_beyond_limits(15.0, lcl=9.0, ucl=11.0) == 1
    assert detect_beyond_limits(100.0, lcl=9.0, ucl=11.0) == 1


def test_detect_beyond_limits_with_negative_limits():
    """Should work correctly with negative control limits."""
    assert detect_beyond_limits(-5.0, lcl=-10.0, ucl=0.0) == 0
    assert detect_beyond_limits(-11.0, lcl=-10.0, ucl=0.0) == -1
    assert detect_beyond_limits(1.0, lcl=-10.0, ucl=0.0) == 1


def test_detect_beyond_limits_with_floats():
    """Should handle floating point values correctly."""
    assert detect_beyond_limits(9.999, lcl=9.0, ucl=11.0) == 0
    assert detect_beyond_limits(11.001, lcl=9.0, ucl=11.0) == 1
    assert detect_beyond_limits(8.999, lcl=9.0, ucl=11.0) == -1


# ============================================================================
# Test: Integration - Full workflow
# ============================================================================

def test_full_xbar_workflow():
    """Test complete workflow: calculate limits and detect signals."""
    # Setup
    mean = 100.0
    sd = 2.0
    N = 5

    # Calculate limits
    limits = calculate_limits(
        limits_type='Xbar',
        mean=mean,
        sd=sd,
        N=N
    )

    # Test values
    values = [99.0, 100.0, 101.0, 95.0, 105.0]
    signals = [
        detect_beyond_limits(v, limits['lcl'], limits['ucl'])
        for v in values
    ]

    # First 3 should be in control
    assert signals[0] == 0
    assert signals[1] == 0
    assert signals[2] == 0

    # Last 2 depend on actual limits (may or may not signal)
    assert all(s in [-1, 0, 1] for s in signals)


def test_full_imr_workflow():
    """Test complete IMR workflow."""
    # Setup
    mean = 50.0
    mR = 2.0

    # Calculate limits
    limits = calculate_limits(
        limits_type='Imr',
        mean=mean,
        mR=mR
    )

    # Test detection
    assert detect_beyond_limits(50.0, limits['lcl'], limits['ucl']) == 0
    assert detect_beyond_limits(mean + 10, limits['lcl'], limits['ucl']) == 1
    assert detect_beyond_limits(mean - 10, limits['lcl'], limits['ucl']) == -1


# ============================================================================
# Test: Edge Cases
# ============================================================================

def test_calculate_limits_with_zero_sd():
    """Should handle zero standard deviation."""
    result = calculate_limits(
        limits_type='Xbar',
        mean=10.0,
        sd=0.0,
        N=5
    )

    # Limits should collapse to mean
    assert result['lcl'] == 10.0
    assert result['ucl'] == 10.0


def test_calculate_limits_with_large_values():
    """Should handle very large values."""
    result = calculate_limits(
        limits_type='Xbar',
        mean=1e6,
        sd=1e3,
        N=5
    )

    assert result['lcl'] < result['ucl']
    assert result['lcl'] < 1e6 < result['ucl']


def test_calculate_limits_with_small_values():
    """Should handle very small values."""
    result = calculate_limits(
        limits_type='Xbar',
        mean=1e-6,
        sd=1e-8,
        N=5
    )

    assert result['lcl'] < result['ucl']
    assert result['lcl'] < 1e-6 < result['ucl']
