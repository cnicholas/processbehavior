"""
Unit tests for SPC constants and formulas.

Tests cover:
- Bias correction constants (c4, b3, b4)
- Control limit calculations (Xbar, S, XmR, R)
- Signal detection
- Edge cases and validation
"""

import pandas as pd
import pytest

from processbehavior.spc_constants import (
    b3,
    b4,
    c4,
    calculate_limits,
    detect_beyond_limits,
)

# ============================================================================
# Test: c4 (bias correction constant)
# ============================================================================


@pytest.mark.parametrize(
    'n, expected, tol',
    [
        (2, 0.7979, 0.001),
        (3, 0.8862, 0.001),
        (4, 0.9213, 0.001),
        (5, 0.9400, 0.001),
        (10, 0.9727, 0.001),
        (25, 0.9896, 0.001),
        (100, None, None),  # monotonic check only
        (1000, None, None),  # monotonic + close-to-1 check
    ],
    ids=[
        'c4(2)=0.7979',
        'c4(3)=0.8862',
        'c4(4)=0.9213',
        'c4(5)=0.9400',
        'c4(10)=0.9727',
        'c4(25)=0.9896',
        'c4(100)-monotonic',
        'c4(1000)-approaches-1',
    ],
)
def test_c4_values(n, expected, tol):
    """c4 should match published constants and approach 1.0 as n increases."""
    val = c4(n)
    if expected is not None:
        assert abs(val - expected) < tol
    # All values must be < 1.0 and monotonically increasing
    assert val < 1.0
    if n >= 10:
        assert val > c4(n - 1)
    if n == 1000:
        assert val > 0.999


@pytest.mark.parametrize('n', [1, 0, -5], ids=['n=1', 'n=0', 'n=-5'])
def test_c4_raises_on_invalid_n(n):
    """c4 should raise error for n < 2."""
    with pytest.raises(ValueError, match='Subgroup size must be >= 2'):
        c4(n)


# ============================================================================
# Test: b3 and b4 (S chart limit constants)
# ============================================================================


@pytest.mark.parametrize(
    'func, n, expected, tol',
    [
        # b3: clamped to 0 for small n
        (b3, 2, 0, 0.001),
        (b3, 3, 0, 0.001),
        (b3, 4, 0, 0.001),
        (b3, 5, 0, 0.001),
        # b3: positive for larger n, known values
        (b3, 6, 0.030, 0.01),
        (b3, 10, 0.284, 0.01),
        (b3, 25, 0.565, 0.01),
        # b4: always > 1, known values
        (b4, 2, 3.267, 0.01),
        (b4, 5, 2.089, 0.01),
        (b4, 10, 1.716, 0.01),
        (b4, 25, 1.435, 0.01),
    ],
    ids=[
        'b3(2)=0-clamped',
        'b3(3)=0-clamped',
        'b3(4)=0-clamped',
        'b3(5)=0-clamped',
        'b3(6)=0.030',
        'b3(10)=0.284',
        'b3(25)=0.565',
        'b4(2)=3.267',
        'b4(5)=2.089',
        'b4(10)=1.716',
        'b4(25)=1.435',
    ],
)
def test_b3_b4_known_values(func, n, expected, tol):
    """b3/b4 should match published constants from Wheeler (1995) Table A.1."""
    assert abs(func(n) - expected) < tol


def test_b4_decreases_with_n():
    """b4 should decrease as n increases."""
    assert b4(2) > b4(10) > b4(100)


@pytest.mark.parametrize('func', [b3, b4], ids=['b3', 'b4'])
def test_b3_b4_raises_on_invalid_n(func):
    """b3/b4 should raise error for n < 2."""
    with pytest.raises(ValueError, match='Subgroup size must be >= 2'):
        func(1)


# ============================================================================
# Test: calculate_limits
# ============================================================================


@pytest.mark.parametrize(
    'limits_type, kwargs, expected_lpl, expected_upl, lpl_tol, upl_tol',
    [
        # Xbar: mean=10, sd=0.5, N=5 → Wd=0.5/c4(5)≈0.5319, limits=10±(3*0.5319)/√5≈10±0.714
        ('Xbar', dict(mean=10.0, sd=0.5, N=5), 9.286, 10.714, 0.01, 0.01),
        # S: sd=0.5, N=5 → LPL=0.5*b3(5)=0, UPL=0.5*b4(5)≈1.044
        ('S', dict(sd=0.5, N=5), 0.0, 1.044, 0.001, 0.01),
        # XmR: mean=10, mR=0.3 → 10±2.66*0.3=10±0.798
        ('XmR', dict(mean=10.0, mR=0.3), 10.0 - 2.66 * 0.3, 10.0 + 2.66 * 0.3, 0.001, 0.001),
        # R: mR=0.3 → LPL=0, UPL=0.3*3.268
        ('R', dict(mR=0.3), 0.0, 0.3 * 3.268, 0.001, 0.001),
    ],
    ids=['Xbar', 'S', 'XmR', 'R'],
)
def test_calculate_limits(limits_type, kwargs, expected_lpl, expected_upl, lpl_tol, upl_tol):
    """Should calculate chart limits correctly for each chart type."""
    result = calculate_limits(limits_type=limits_type, **kwargs)

    assert isinstance(result, pd.Series)
    assert list(result.index) == ['lpl', 'upl']
    assert abs(result['lpl'] - expected_lpl) < lpl_tol
    assert abs(result['upl'] - expected_upl) < upl_tol


@pytest.mark.parametrize(
    'limits_type, kwargs, match_pattern',
    [
        ('Xbar', dict(mean=10.0, sd=0.5), r'requires \(mean, sd, and N\)'),
        ('Xbar', dict(mean=10.0, N=5), r'requires \(mean, sd, and N\)'),
        ('Xbar', dict(sd=0.5, N=5), r'requires \(mean, sd, and N\)'),
        ('S', dict(sd=0.5), r'requires \(sd, and N\)'),
        ('S', dict(N=5), r'requires \(sd, and N\)'),
        ('XmR', dict(mean=10.0), r'requires \(mean, and mR\)'),
        ('XmR', dict(mR=0.3), r'requires \(mean, and mR\)'),
        ('R', dict(), r'requires \(mR\)'),
        ('Invalid', dict(mean=10.0, sd=1.0, N=5), 'not supported'),
        ('p', dict(mean=0.5, N=100), 'not supported'),
    ],
    ids=[
        'Xbar-missing-N',
        'Xbar-missing-sd',
        'Xbar-missing-mean',
        'S-missing-N',
        'S-missing-sd',
        'XmR-missing-mR',
        'XmR-missing-mean',
        'R-missing-mR',
        'invalid-type',
        'unsupported-p-chart',
    ],
)
def test_calculate_limits_missing_or_invalid(limits_type, kwargs, match_pattern):
    """Should raise error for missing params or unsupported chart types."""
    with pytest.raises(ValueError, match=match_pattern):
        calculate_limits(limits_type=limits_type, **kwargs)


@pytest.mark.parametrize(
    'limits_type, kwargs',
    [
        ('Xbar', dict(mean=100.0, sd=2.0, N=10)),
        ('XmR', dict(mean=50.0, mR=1.5)),
    ],
    ids=['Xbar-symmetric', 'XmR-symmetric'],
)
def test_calculate_limits_symmetric(limits_type, kwargs):
    """Xbar and XmR limits should be symmetric around mean."""
    result = calculate_limits(limits_type=limits_type, **kwargs)
    mean = kwargs['mean']
    lcl_dist = mean - result['lpl']
    ucl_dist = result['upl'] - mean
    assert abs(lcl_dist - ucl_dist) < 0.001


def test_calculate_limits_s_lcl_clamped_for_small_n():
    """S chart LPL is clamped to 0 for small subgroup sizes."""
    for n in [2, 3]:
        result = calculate_limits(limits_type='S', sd=1.0, N=n)
        assert result['lpl'] == 0

    # Large n: LPL should be positive
    result = calculate_limits(limits_type='S', sd=1.0, N=10)
    assert result['lpl'] > 0


def test_calculate_limits_r_lcl_always_zero():
    """R chart LPL should always be 0."""
    for mR in [0.1, 0.5, 1.0, 5.0]:
        result = calculate_limits(limits_type='R', mR=mR)
        assert result['lpl'] == 0.0


# ============================================================================
# Test: detect_beyond_limits
# ============================================================================


@pytest.mark.parametrize(
    'value, lpl, upl, expected',
    [
        # Within limits
        (10.0, 9.0, 11.0, 0),
        (9.5, 9.0, 11.0, 0),
        (10.5, 9.0, 11.0, 0),
        # At boundaries (not beyond)
        (9.0, 9.0, 11.0, 0),
        (11.0, 9.0, 11.0, 0),
        # Below LPL
        (8.9, 9.0, 11.0, -1),
        (5.0, 9.0, 11.0, -1),
        (0.0, 9.0, 11.0, -1),
        # Above UPL
        (11.1, 9.0, 11.0, 1),
        (15.0, 9.0, 11.0, 1),
        (100.0, 9.0, 11.0, 1),
        # Negative limits
        (-5.0, -10.0, 0.0, 0),
        (-11.0, -10.0, 0.0, -1),
        (1.0, -10.0, 0.0, 1),
        # Floating point precision
        (9.999, 9.0, 11.0, 0),
        (11.001, 9.0, 11.0, 1),
        (8.999, 9.0, 11.0, -1),
    ],
    ids=[
        'within-center',
        'within-low',
        'within-high',
        'at-lpl',
        'at-upl',
        'below-lpl-near',
        'below-lpl-far',
        'below-lpl-zero',
        'above-upl-near',
        'above-upl-far',
        'above-upl-extreme',
        'negative-limits-within',
        'negative-limits-below',
        'negative-limits-above',
        'float-within',
        'float-above',
        'float-below',
    ],
)
def test_detect_beyond_limits(value, lpl, upl, expected):
    """Should correctly classify values relative to control limits."""
    assert detect_beyond_limits(value, lpl=lpl, upl=upl) == expected


# ============================================================================
# Test: Integration - Full workflow
# ============================================================================


def test_full_xbar_workflow():
    """Test complete workflow: calculate limits and detect signals."""
    limits = calculate_limits(limits_type='Xbar', mean=100.0, sd=2.0, N=5)
    values = [99.0, 100.0, 101.0, 95.0, 105.0]
    signals = [detect_beyond_limits(v, limits['lpl'], limits['upl']) for v in values]

    # First 3 should be in control
    assert signals[0] == 0
    assert signals[1] == 0
    assert signals[2] == 0
    assert all(s in [-1, 0, 1] for s in signals)


def test_full_xmr_workflow():
    """Test complete XmR workflow."""
    mean = 50.0
    limits = calculate_limits(limits_type='XmR', mean=mean, mR=2.0)

    assert detect_beyond_limits(50.0, limits['lpl'], limits['upl']) == 0
    assert detect_beyond_limits(mean + 10, limits['lpl'], limits['upl']) == 1
    assert detect_beyond_limits(mean - 10, limits['lpl'], limits['upl']) == -1


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.parametrize(
    'mean, sd, N',
    [
        (10.0, 0.0, 5),
        (1e6, 1e3, 5),
        (1e-6, 1e-8, 5),
    ],
    ids=['zero-sd', 'large-values', 'small-values'],
)
def test_calculate_limits_edge_cases(mean, sd, N):
    """Should handle edge-case numeric values."""
    result = calculate_limits(limits_type='Xbar', mean=mean, sd=sd, N=N)
    assert result['lpl'] <= mean <= result['upl']
    if sd == 0.0:
        assert result['lpl'] == mean
        assert result['upl'] == mean
    else:
        assert result['lpl'] < result['upl']
