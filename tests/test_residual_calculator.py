"""
Unit tests for ResidualCalculator - Pure Function Testing!

This showcases the power of pure functions:
- No setup required
- Test inputs → outputs directly
- Fast, focused tests
- Easy to understand

Tests cover:
- All mean calculations (Ybar, Ybar_k, Ybar_kt, Ybar_t)
- All residuals (R1-R5)
- SDS-specific R2 calculations
- Orchestration class
"""

import pandas as pd
import pytest

from processbehavior.analysis_dataset import AnalysisSpecification
from processbehavior.residual_calculator import (
    # Orchestration
    ResidualCalculator,
    calculate_cell_means,
    calculate_factor_means,
    # Pure functions for means
    calculate_grand_mean,
    # Pure functions for residuals
    calculate_r1_residual,
    calculate_r2_residual_sds1,
    calculate_r2_residual_sds2,
    calculate_r2_residual_sds3,
    calculate_r3_residual,
    calculate_r4_residual,
    calculate_r5_residual,
    calculate_time_means,
)

# ============================================================================
# Test: Pure Functions for Means
# ============================================================================

def test_calculate_grand_mean():
    """Grand mean is simple average of all values."""
    df = pd.DataFrame({'weight': [10.0, 10.5, 9.5]})

    result = calculate_grand_mean(df, 'weight')

    assert result == 10.0


def test_calculate_grand_mean_with_floats():
    """Should handle floating point precision."""
    df = pd.DataFrame({'weight': [10.1, 10.3, 9.9]})

    result = calculate_grand_mean(df, 'weight')

    assert pytest.approx(result, 0.01) == 10.1


def test_calculate_factor_means_broadcasts_to_rows():
    """Factor means should broadcast to all rows in each group."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'weight': [10.0, 10.5, 9.0, 9.5]
    })

    result = calculate_factor_means(df, 'weight', 'lane')

    expected = pd.Series([10.25, 10.25, 9.25, 9.25])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_time_means_broadcasts_to_rows():
    """Time means should broadcast to all rows at each time point."""
    df = pd.DataFrame({
        'pull': [1, 1, 2, 2],
        'weight': [10.0, 10.5, 9.0, 9.5]
    })

    result = calculate_time_means(df, 'weight', 'pull')

    expected = pd.Series([10.25, 10.25, 9.25, 9.25])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_cell_means_broadcasts_to_rows():
    """Cell means should broadcast to all rows in each (factor × time) cell."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'pull': [1, 1, 2, 2],
        'weight': [10.0, 10.5, 9.0, 9.5]
    })

    result = calculate_cell_means(df, 'weight', 'lane', 'pull')

    expected = pd.Series([10.25, 10.25, 9.25, 9.25])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_cell_means_with_unequal_cell_sizes():
    """Should work with varying observations per cell."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'B', 'B'],  # A has 3, B has 2
        'pull': [1, 1, 1, 1, 1],
        'weight': [10.0, 10.5, 10.2, 9.0, 9.5]
    })

    result = calculate_cell_means(df, 'weight', 'lane', 'pull')

    # A cells should average (10.0 + 10.5 + 10.2) / 3 = 10.233...
    # B cells should average (9.0 + 9.5) / 2 = 9.25
    assert pytest.approx(result.iloc[0], 0.01) == 10.23
    assert pytest.approx(result.iloc[3], 0.01) == 9.25


# ============================================================================
# Test: Pure Functions for Residuals
# ============================================================================

def test_calculate_r1_residual():
    """R1 = Y - Ybar (total deviation from grand mean)."""
    df = pd.DataFrame({'weight': [10.1, 10.3, 9.9]})
    grand_mean = 10.1

    result = calculate_r1_residual(df, 'weight', grand_mean)

    expected = pd.Series([0.0, 0.2, -0.2])
    pd.testing.assert_series_equal(result, expected, check_names=False, atol=0.01)


def test_calculate_r1_residual_sum_is_zero():
    """Sum of R1 residuals should always be zero (or very close)."""
    df = pd.DataFrame({'weight': [10.1, 10.3, 9.9, 10.5, 9.7]})
    grand_mean = df['weight'].mean()

    r1 = calculate_r1_residual(df, 'weight', grand_mean)

    assert pytest.approx(r1.sum(), abs=1e-10) == 0.0


def test_calculate_r2_residual_sds1():
    """R2 for SDS 1 = Y - Ybar_kt (within-cell deviation)."""
    df = pd.DataFrame({'weight': [10.0, 10.5, 9.0, 9.5]})
    cell_means = pd.Series([10.25, 10.25, 9.25, 9.25])

    result = calculate_r2_residual_sds1(df, 'weight', cell_means)

    expected = pd.Series([-0.25, 0.25, -0.25, 0.25])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_r2_residual_sds2_moving_average():
    """R2 for SDS 2 uses backward 2-point moving average per Tom Bishop."""
    # Sorted data: [10, 11, 12] within group
    df = pd.DataFrame({
        'rsg': ['A', 'A', 'A'],
        'weight': [10.0, 11.0, 12.0]
    }).sort_values(['rsg'])  # Must be sorted!

    result = calculate_r2_residual_sds2(df, 'weight', 'rsg')

    # Per Tom Bishop Eq. 65: Y_ma = (Y_j + Y_{j-1}) / 2
    # R2_j = Y_j - Y_ma = (Y_j - Y_{j-1}) / 2
    #
    # Point 0 (j=1): No lag, R2 = NaN
    # Point 1 (j=2): MA = (11 + 10)/2 = 10.5, R2 = 11 - 10.5 = 0.5
    # Point 2 (j=3): MA = (12 + 11)/2 = 11.5, R2 = 12 - 11.5 = 0.5
    assert pd.isna(result.iloc[0])  # First point has no lag
    assert pytest.approx(result.iloc[1], 0.01) == 0.5
    assert pytest.approx(result.iloc[2], 0.01) == 0.5


def test_calculate_r2_residual_sds3_hybrid():
    """R2 for SDS 3 uses within-cell for n>1, zero for n=1."""
    df = pd.DataFrame({
        'rsg': ['A', 'A', 'B'],  # A has n=2, B has n=1
        'time': [1, 1, 1],
        'weight': [10.0, 10.5, 9.0]
    })
    cell_means = pd.Series([10.25, 10.25, 9.0])

    result = calculate_r2_residual_sds3(df, 'weight', cell_means, 'rsg', 'time')

    # A: n=2, use within-cell: [10.0-10.25, 10.5-10.25] = [-0.25, 0.25]
    # B: n=1, use zero: [0.0]
    expected = pd.Series([-0.25, 0.25, 0.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_r3_residual():
    """R3 = Y - Ybar_k - Ybar_t + Ybar (interaction)."""
    df = pd.DataFrame({'weight': [10.0, 10.5, 9.0, 9.5]})
    factor_means = pd.Series([10.25, 10.25, 9.25, 9.25])
    time_means = pd.Series([9.75, 9.75, 10.0, 10.0])
    grand_mean = 9.875

    result = calculate_r3_residual(df, 'weight', factor_means, time_means, grand_mean)

    # R3 = Y - Ybar_k - Ybar_t + Ybar
    # Row 0: 10.0 - 10.25 - 9.75 + 9.875 = -0.125
    assert pytest.approx(result.iloc[0], 0.01) == -0.125


def test_calculate_r4_residual():
    """R4 = Ybar_t - Ybar + R2 (time effects + unexplained)."""
    time_means = pd.Series([10.25, 9.75])
    grand_mean = 10.0
    r2 = pd.Series([0.1, -0.1])

    result = calculate_r4_residual(time_means, grand_mean, r2)

    # R4 = Ybar_t - Ybar + R2
    # Row 0: 10.25 - 10.0 + 0.1 = 0.35
    # Row 1: 9.75 - 10.0 + (-0.1) = -0.35
    expected = pd.Series([0.35, -0.35])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_r5_residual():
    """R5 = Ybar_k - Ybar + R2 (factor effects + unexplained)."""
    factor_means = pd.Series([10.25, 9.75])
    grand_mean = 10.0
    r2 = pd.Series([0.1, -0.1])

    result = calculate_r5_residual(factor_means, grand_mean, r2)

    # R5 = Ybar_k - Ybar + R2
    # Row 0: 10.25 - 10.0 + 0.1 = 0.35
    # Row 1: 9.75 - 10.0 + (-0.1) = -0.35
    expected = pd.Series([0.35, -0.35])
    pd.testing.assert_series_equal(result, expected, check_names=False)


# ============================================================================
# Test: Orchestration Class (ResidualCalculator)
# ============================================================================

@pytest.fixture
def calc():
    """Create ResidualCalculator instance."""
    return ResidualCalculator()


@pytest.fixture
def sds1_df():
    """SDS 1 data - full replication (all cells have n≥2)."""
    return pd.DataFrame({
        'rsg': ['A', 'A', 'B', 'B'] * 2,  # Each cell has n=2
        'time': [1, 1, 1, 1, 2, 2, 2, 2],
        'weight': [10.0, 10.5, 9.0, 9.5, 10.2, 10.4, 9.1, 9.3]
    })


@pytest.fixture
def spec_sds1():
    """Specification for SDS 1 data."""
    return AnalysisSpecification('Xbar', {
        'rsg_vars': ['rsg'],
        'rsg_var_name': 'rsg',
        'time_var': 'time',
        'response_var': 'weight'
    })


def test_calculate_residuals_adds_all_mean_columns(calc, sds1_df, spec_sds1):
    """Should add Ybar, Ybar_k, Ybar_t, Ybar_kt columns."""
    result = calc.calculate_residuals(sds1_df, spec_sds1, sds=1)

    mean_cols = ['Ybar', 'Ybar_k', 'Ybar_t', 'Ybar_kt']
    for col in mean_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_calculate_residuals_adds_all_residual_columns(calc, sds1_df, spec_sds1):
    """Should add R1, R2, R3, R4, R5 columns."""
    result = calc.calculate_residuals(sds1_df, spec_sds1, sds=1)

    residual_cols = ['R1', 'R2', 'R3', 'R4', 'R5']
    for col in residual_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_calculate_residuals_sds1_uses_within_cell_variance(calc, sds1_df, spec_sds1):
    """SDS 1 should use within-cell variance for R2."""
    result = calc.calculate_residuals(sds1_df, spec_sds1, sds=1)

    # R2 should be Y - Ybar_kt
    expected_r2 = result['weight'] - result['Ybar_kt']
    pd.testing.assert_series_equal(result['R2'], expected_r2, check_names=False)


def test_calculate_residuals_preserves_original_data(calc, sds1_df, spec_sds1):
    """Should not modify original DataFrame columns."""
    original_cols = sds1_df.columns.tolist()

    result = calc.calculate_residuals(sds1_df, spec_sds1, sds=1)

    # Original columns should still be present
    for col in original_cols:
        assert col in result.columns


def test_calculate_residuals_raises_on_unsupported_sds(calc, sds1_df, spec_sds1):
    """Should raise helpful error for unsupported SDS."""
    with pytest.raises(ValueError, match="not supported for SDS 0"):
        calc.calculate_residuals(sds1_df, spec_sds1, sds=0)

    with pytest.raises(ValueError, match="not supported for SDS 4"):
        calc.calculate_residuals(sds1_df, spec_sds1, sds=4)


def test_calculate_residuals_raises_without_grouping(calc, sds1_df):
    """Should raise if no grouping variables (VAS requires grouping)."""
    spec_no_grouping = AnalysisSpecification('Imr', {
        'response_var': 'weight'
    })

    with pytest.raises(ValueError, match="require grouping structure"):
        calc.calculate_residuals(sds1_df, spec_no_grouping, sds=1)


def test_calculate_residuals_r1_sum_is_zero(calc, sds1_df, spec_sds1):
    """R1 residuals should sum to zero (deviations from mean)."""
    result = calc.calculate_residuals(sds1_df, spec_sds1, sds=1)

    assert pytest.approx(result['R1'].sum(), abs=1e-10) == 0.0


def test_calculate_residuals_sds2_uses_moving_average(calc, spec_sds1):
    """SDS 2 should use backward moving average for R2."""
    # SDS 2: Each cell has n=1
    sds2_df = pd.DataFrame({
        'rsg': ['A', 'A', 'A'],
        'time': [1, 2, 3],
        'weight': [10.0, 11.0, 12.0]
    }).sort_values(['rsg', 'time'])

    result = calc.calculate_residuals(sds2_df, spec_sds1, sds=2)

    # R2 should use backward moving average: R2 = (Y_j - Y_{j-1}) / 2
    assert 'R2' in result.columns
    # First observation should be NaN (no lag)
    assert pd.isna(result['R2'].iloc[0])
    # Remaining should equal (Y_j - Y_{j-1}) / 2
    assert pytest.approx(result['R2'].iloc[1], 0.01) == 0.5  # (11-10)/2
    assert pytest.approx(result['R2'].iloc[2], 0.01) == 0.5  # (12-11)/2


def test_calculate_residuals_sds3_hybrid_approach(calc, spec_sds1):
    """SDS 3 should use hybrid: within-cell for n>1, zero for n=1."""
    # Mix of n=2 and n=1 cells
    sds3_df = pd.DataFrame({
        'rsg': ['A', 'A', 'B'],  # A has n=2, B has n=1
        'time': [1, 1, 1],
        'weight': [10.0, 10.5, 9.0]
    })

    result = calc.calculate_residuals(sds3_df, spec_sds1, sds=3)

    # A (n=2): Should have non-zero R2 (within-cell variation)
    # B (n=1): Should have R2 = 0
    a_rows = result[result['rsg'] == 'A']
    b_rows = result[result['rsg'] == 'B']

    assert not (a_rows['R2'] == 0).all()  # A has within-cell variation
    assert (b_rows['R2'] == 0).all()      # B has no variation (n=1)


# ============================================================================
# Test: Pure Functions Are Truly Pure
# ============================================================================

def test_pure_functions_dont_modify_inputs():
    """Pure functions should never modify input DataFrames."""
    df = pd.DataFrame({'weight': [10.0, 10.5]})
    df_copy = df.copy()

    # Call multiple pure functions
    calculate_grand_mean(df, 'weight')
    calculate_r1_residual(df, 'weight', 10.25)

    # Original should be unchanged
    pd.testing.assert_frame_equal(df, df_copy)


def test_pure_functions_same_input_same_output():
    """Pure functions should give same output for same input."""
    df = pd.DataFrame({'weight': [10.1, 10.3, 9.9]})

    # Call twice
    result1 = calculate_grand_mean(df, 'weight')
    result2 = calculate_grand_mean(df, 'weight')

    assert result1 == result2


# ============================================================================
# Test: Edge Cases
# ============================================================================

def test_calculate_residuals_with_single_observation_per_row(calc):
    """Should handle edge case of minimal data."""
    df = pd.DataFrame({
        'rsg': ['A', 'A'],
        'time': [1, 1],
        'weight': [10.0, 10.5]
    })
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['rsg'],
        'rsg_var_name': 'rsg',
        'time_var': 'time',
        'response_var': 'weight'
    })

    result = calc.calculate_residuals(df, spec, sds=1)

    # Should complete without error
    assert len(result) == 2
    assert all(col in result.columns for col in ['R1', 'R2', 'R3', 'R4', 'R5'])


def test_r2_sds2_handles_single_group(calc):
    """SDS 2 R2 calculation should handle minimal data correctly."""
    df = pd.DataFrame({
        'rsg': ['A', 'A'],  # Only 2 points
        'weight': [10.0, 11.0]
    }).sort_values('rsg')

    result = calculate_r2_residual_sds2(df, 'weight', 'rsg')

    # First point has no lag (NaN), second point has R2 = (11-10)/2 = 0.5
    assert len(result) == 2
    assert pd.isna(result.iloc[0])  # First point: no lag
    assert pytest.approx(result.iloc[1], 0.01) == 0.5  # Second: (11-10)/2


# ============================================================================
# Test: Tom Bishop Validation - R2 = MR/2 for SDS 2
# ============================================================================

def test_r2_sds2_equals_half_moving_range():
    """
    Validate Tom Bishop's formula: R2 = (Y_j - Y_{j-1}) / 2 = MR / 2.

    This test confirms the mathematical relationship between R2 residuals
    and the moving range used in IMR charts.
    """
    df = pd.DataFrame({
        'rsg': ['A'] * 5,
        'weight': [10.0, 12.0, 11.0, 13.0, 12.5]
    })

    r2 = calculate_r2_residual_sds2(df, 'weight', 'rsg')

    # Calculate moving range: MR_j = |Y_j - Y_{j-1}|
    # For our data:
    # j=1: MR = NA
    # j=2: MR = |12 - 10| = 2, R2 should = 2/2 = 1.0
    # j=3: MR = |11 - 12| = 1, R2 should = -1/2 = -0.5 (signed!)
    # j=4: MR = |13 - 11| = 2, R2 should = 2/2 = 1.0
    # j=5: MR = |12.5 - 13| = 0.5, R2 should = -0.5/2 = -0.25

    assert pd.isna(r2.iloc[0])
    assert pytest.approx(r2.iloc[1], 0.01) == 1.0   # (12-10)/2
    assert pytest.approx(r2.iloc[2], 0.01) == -0.5  # (11-12)/2
    assert pytest.approx(r2.iloc[3], 0.01) == 1.0   # (13-11)/2
    assert pytest.approx(r2.iloc[4], 0.01) == -0.25 # (12.5-13)/2


def test_r2_sds2_multiple_groups():
    """
    SDS 2 R2 calculation should work independently per group.

    Validates that the backward MA is calculated within each group
    (process design condition), not across groups.
    """
    df = pd.DataFrame({
        'rsg': ['A', 'A', 'A', 'B', 'B', 'B'],
        'time': [1, 2, 3, 1, 2, 3],
        'weight': [10.0, 11.0, 12.0, 20.0, 22.0, 21.0]
    }).sort_values(['rsg', 'time'])

    result = calculate_r2_residual_sds2(df, 'weight', 'rsg')

    # Group A (indices 0-2):
    assert pd.isna(result.iloc[0])  # First in group A
    assert pytest.approx(result.iloc[1], 0.01) == 0.5   # (11-10)/2
    assert pytest.approx(result.iloc[2], 0.01) == 0.5   # (12-11)/2

    # Group B (indices 3-5):
    assert pd.isna(result.iloc[3])  # First in group B
    assert pytest.approx(result.iloc[4], 0.01) == 1.0   # (22-20)/2
    assert pytest.approx(result.iloc[5], 0.01) == -0.5  # (21-22)/2


def test_r2_sds2_tom_bishop_example():
    """
    Test with values inspired by Tom Bishop's Figure 30 example.

    This validates that our implementation produces the "unexplained"
    variation pattern that Tom describes - removing PDC and PT effects
    to leave only the residual variation.
    """
    # Simulated data with trend (PT effect) and noise
    df = pd.DataFrame({
        'rsg': ['Lane4'] * 10,
        'time': list(range(1, 11)),
        'weight': [238.0, 239.0, 240.0, 239.5, 240.5,
                   241.0, 240.0, 241.5, 241.0, 242.0]
    })

    result = calculate_r2_residual_sds2(df, 'weight', 'rsg')

    # First point should be NaN
    assert pd.isna(result.iloc[0])

    # Remaining R2 values should be small (trend removed)
    # and should equal (Y_j - Y_{j-1}) / 2
    for i in range(1, len(result)):
        y_current = df['weight'].iloc[i]
        y_previous = df['weight'].iloc[i-1]
        expected_r2 = (y_current - y_previous) / 2.0
        assert pytest.approx(result.iloc[i], 0.01) == expected_r2

    # Verify that R2 values are generally smaller than original variation
    # (trend has been removed by the moving average)
    assert result[1:].abs().max() < df['weight'].std()
