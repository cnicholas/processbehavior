"""
Unit tests for VAS residual calculations.

Tests cover:
- All mean calculations (Ybar, Ybar_k, Ybar_kt, Ybar_t)
- All residuals (R1-R5)
- R2 method variants (exact, ma2, hybrid)
- Orchestration via calculate_vas_residuals
- Tom Bishop validation data
"""

import pandas as pd
import pytest

from processbehavior.data_preparation import DataPreparation
from processbehavior.formulation_spec import FormulationSpec
from processbehavior.residual_calculator import (
    calculate_cell_means,
    calculate_factor_means,
    calculate_grand_mean,
    calculate_r1_residual,
    calculate_r2,
    calculate_r3_residual,
    calculate_r4_residual,
    calculate_r5_residual,
    calculate_time_means,
    calculate_vas_residuals,
)

# ============================================================================
# Helpers
# ============================================================================


def _prepare_for_vas(df: pd.DataFrame, spec: FormulationSpec) -> pd.DataFrame:
    """Run DataPreparation pipeline to add required keys for calculate_vas_residuals."""
    prep = DataPreparation()
    out = prep.prepare_dataset(df, spec)
    return prep.build_keys(out, spec)


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
    df = pd.DataFrame({'lane': ['A', 'A', 'B', 'B'], 'weight': [10.0, 10.5, 9.0, 9.5]})

    result = calculate_factor_means(df, 'weight', 'lane')

    expected = pd.Series([10.25, 10.25, 9.25, 9.25])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_time_means_broadcasts_to_rows():
    """Time means should broadcast to all rows at each time point."""
    df = pd.DataFrame({'pull': [1, 1, 2, 2], 'weight': [10.0, 10.5, 9.0, 9.5]})

    result = calculate_time_means(df, 'weight', 'pull')

    expected = pd.Series([10.25, 10.25, 9.25, 9.25])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_cell_means_broadcasts_to_rows():
    """Cell means should broadcast to all rows in each (factor × time) cell."""
    df = pd.DataFrame({'lane': ['A', 'A', 'B', 'B'], 'pull': [1, 1, 2, 2], 'weight': [10.0, 10.5, 9.0, 9.5]})

    result = calculate_cell_means(df, 'weight', 'lane', 'pull')

    expected = pd.Series([10.25, 10.25, 9.25, 9.25])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_cell_means_with_unequal_cell_sizes():
    """Should work with varying observations per cell."""
    df = pd.DataFrame(
        {
            'lane': ['A', 'A', 'A', 'B', 'B'],  # A has 3, B has 2
            'pull': [1, 1, 1, 1, 1],
            'weight': [10.0, 10.5, 10.2, 9.0, 9.5],
        }
    )

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


def test_calculate_r2_exact():
    """R2 exact = Y - Ybar_kt (within-cell deviation)."""
    # Build data with required keys
    df = pd.DataFrame(
        {
            'weight': [10.0, 10.5, 9.0, 9.5],
            'Ybar_kt': [10.25, 10.25, 9.25, 9.25],
            'cell_key': [('A', 1), ('A', 1), ('B', 1), ('B', 1)],
            'rsg_key': ['A', 'A', 'B', 'B'],
            'sort_key': [(('A',), 1, 0), (('A',), 1, 1), (('B',), 1, 0), (('B',), 1, 1)],
        }
    )

    result = calculate_r2(df, 'weight', r2_method='exact')

    expected = pd.Series([-0.25, 0.25, -0.25, 0.25])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_calculate_r2_ma2():
    """R2 ma2 uses backward 2-point moving average per Tom Bishop."""
    df = pd.DataFrame(
        {
            'weight': [10.0, 11.0, 12.0],
            'Ybar_kt': [10.0, 11.0, 12.0],
            'cell_key': [('A', 1), ('A', 2), ('A', 3)],
            'rsg_key': ['A', 'A', 'A'],
            'sort_key': [(('A',), 1, 0), (('A',), 2, 0), (('A',), 3, 0)],
        }
    )

    result = calculate_r2(df, 'weight', r2_method='ma2')

    # R2_j = (Y_j - Y_{j-1}) / 2
    # Point 0: NaN (no predecessor — Bishop leaves j=1 blank)
    # Point 1: (11-10)/2 = 0.5
    # Point 2: (12-11)/2 = 0.5
    assert pd.isna(result.iloc[0])
    assert pytest.approx(result.iloc[1], 0.01) == 0.5
    assert pytest.approx(result.iloc[2], 0.01) == 0.5


def test_calculate_r2_hybrid():
    """R2 with any singletons uses MA2 across entire sorted stream.

    When any cell has n=1, ALL observations use MA2 on the full
    canonical-sorted stream — no per-cell exact/MA2 selection.
    Bishop Eq 13.7-13.9: j=2,...,J with no grouping; only j=1 gets 0.
    """
    # A×1 has n=2, B×1 has n=1, B×2 has n=1
    # sort_key order: A×1(0), A×1(1), B×1(0), B×2(0)
    df = pd.DataFrame(
        {
            'weight': [10.0, 10.5, 9.0, 11.0],
            'Ybar_kt': [10.25, 10.25, 9.0, 11.0],
            'cell_key': [('A', 1), ('A', 1), ('B', 1), ('B', 2)],
            'rsg_key': ['A', 'A', 'B', 'B'],
            'sort_key': [(('A',), 1, 0), (('A',), 1, 1), (('B',), 1, 0), (('B',), 2, 0)],
        }
    )
    n_per_cell = pd.Series([2, 2, 1, 1])

    result = calculate_r2(df, 'weight', r2_method='hybrid', n_per_cell=n_per_cell)

    # MA2 across full sorted stream (no grouping):
    # j=0 (10.0): first obs → NaN (no predecessor)
    # j=1 (10.5): (10.5 - 10.0)/2 = 0.25
    # j=2 (9.0):  (9.0 - 10.5)/2 = -0.75
    # j=3 (11.0): (11.0 - 9.0)/2 = 1.0
    assert pd.isna(result.iloc[0])
    assert pytest.approx(result.iloc[1], 0.01) == 0.25
    assert pytest.approx(result.iloc[2], 0.01) == -0.75
    assert pytest.approx(result.iloc[3], 0.01) == 1.0


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
# Test: calculate_vas_residuals orchestration
# ============================================================================


@pytest.fixture
def sds1_df():
    """SDS 1 data - full replication (all cells have n>=2)."""
    return pd.DataFrame(
        {
            'lane': ['A', 'A', 'B', 'B'] * 2,  # Each cell has n=2
            'time': [1, 1, 1, 1, 2, 2, 2, 2],
            'weight': [10.0, 10.5, 9.0, 9.5, 10.2, 10.4, 9.1, 9.3],
        }
    )


@pytest.fixture
def spec_sds1():
    """Specification for SDS 1 data."""
    return FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        rsg_var_name='rsg',
        time_var='time',
    )


def test_calculate_vas_residuals_adds_all_mean_columns(sds1_df, spec_sds1):
    """Should add Ybar, Ybar_k, Ybar_t, Ybar_kt columns."""
    df = _prepare_for_vas(sds1_df, spec_sds1)
    result = calculate_vas_residuals(df, spec_sds1, r2_method='exact')

    mean_cols = ['Ybar', 'Ybar_k', 'Ybar_t', 'Ybar_kt']
    for col in mean_cols:
        assert col in result.columns, f'Missing column: {col}'


def test_calculate_vas_residuals_adds_all_residual_columns(sds1_df, spec_sds1):
    """Should add R1, R2, R3, R4, R5 columns."""
    df = _prepare_for_vas(sds1_df, spec_sds1)
    result = calculate_vas_residuals(df, spec_sds1, r2_method='exact')

    residual_cols = ['R1', 'R2', 'R3', 'R4', 'R5']
    for col in residual_cols:
        assert col in result.columns, f'Missing column: {col}'


def test_calculate_vas_residuals_sds1_uses_within_cell_variance(sds1_df, spec_sds1):
    """SDS 1 (exact) should use within-cell variance for R2."""
    df = _prepare_for_vas(sds1_df, spec_sds1)
    result = calculate_vas_residuals(df, spec_sds1, r2_method='exact')

    # R2 should be Y - Ybar_kt
    expected_r2 = result['weight'] - result['Ybar_kt']
    pd.testing.assert_series_equal(result['R2'], expected_r2, check_names=False)


def test_calculate_vas_residuals_preserves_original_data(sds1_df, spec_sds1):
    """Should not modify original DataFrame columns."""
    df = _prepare_for_vas(sds1_df, spec_sds1)
    original_cols = df.columns.tolist()

    result = calculate_vas_residuals(df, spec_sds1, r2_method='exact')

    # Original columns should still be present
    for col in original_cols:
        assert col in result.columns


def test_calculate_vas_residuals_raises_on_missing_keys(sds1_df, spec_sds1):
    """Should raise if required key columns are missing."""
    # Raw df without build_keys — missing rsg_key, obs_id, cell_key
    with pytest.raises(ValueError, match='Missing required columns'):
        calculate_vas_residuals(sds1_df, spec_sds1, r2_method='exact')


def test_calculate_vas_residuals_raises_without_grouping():
    """Should raise if no grouping variables (VAS requires grouping)."""
    spec_no_grouping = FormulationSpec(
        response_var='weight',
    )
    # Need to provide keys to get past _validate_prerequisites
    df = pd.DataFrame({'weight': [10.0, 10.5]})
    df['rsg_key'] = 'A'
    df['obs_id'] = range(len(df))
    df['cell_key'] = 'X'

    with pytest.raises(ValueError, match='require grouping structure'):
        calculate_vas_residuals(df, spec_no_grouping, r2_method='exact')


def test_calculate_vas_residuals_r1_sum_is_zero(sds1_df, spec_sds1):
    """R1 residuals should sum to zero (deviations from mean)."""
    df = _prepare_for_vas(sds1_df, spec_sds1)
    result = calculate_vas_residuals(df, spec_sds1, r2_method='exact')

    assert pytest.approx(result['R1'].sum(), abs=1e-10) == 0.0


def test_calculate_vas_residuals_sds2_uses_moving_average(spec_sds1):
    """SDS 2 (ma2) should use backward moving average for R2."""
    sds2_df = pd.DataFrame({'lane': ['A', 'A', 'A'], 'time': [1, 2, 3], 'weight': [10.0, 11.0, 12.0]})
    df = _prepare_for_vas(sds2_df, spec_sds1)
    result = calculate_vas_residuals(df, spec_sds1, r2_method='ma2')

    assert 'R2' in result.columns
    # R2 = (Y_j - Y_{j-1}) / 2
    # First observation: NaN (no predecessor — Bishop leaves j=1 blank)
    assert pd.isna(result['R2'].iloc[0])
    # Remaining: (11-10)/2=0.5, (12-11)/2=0.5
    assert pytest.approx(result['R2'].iloc[1], 0.01) == 0.5
    assert pytest.approx(result['R2'].iloc[2], 0.01) == 0.5


def test_calculate_vas_residuals_sds3_hybrid_uses_ma2_for_all(spec_sds1):
    """SDS 3 (any singletons) uses MA2 across entire sorted stream.

    When any cell has n=1, ALL observations use MA2 on the full
    canonical-sorted stream — no per-cell exact/MA2 selection.
    Only j=1 gets R2=NaN (no predecessor).
    """
    # A has n=2 at time=1, B has n=1 at time=1 and n=1 at time=2
    sds3_df = pd.DataFrame({'lane': ['A', 'A', 'B', 'B'], 'time': [1, 1, 1, 2], 'weight': [10.0, 10.5, 9.0, 11.0]})
    df = _prepare_for_vas(sds3_df, spec_sds1)
    n_per_cell = df.groupby('cell_key', observed=True)['weight'].transform('size')
    result = calculate_vas_residuals(df, spec_sds1, r2_method='hybrid', n_per_cell=n_per_cell)

    # MA2 across full sorted stream (A×1(0), A×1(1), B×1(0), B×2(0)):
    # Only the very first observation in the entire stream gets R2=NaN
    # All others get (Y_j - Y_{j-1})/2
    a_rows = result[result['rsg'] == 'A']
    assert pd.isna(a_rows['R2'].iloc[0])  # j=1: first in stream → NaN

    b_rows = result[result['rsg'] == 'B']
    assert b_rows['R2'].iloc[0] != 0  # Not first in stream — gets MA2 value
    assert b_rows['R2'].iloc[1] != 0  # Also gets MA2 value


def test_calculate_vas_residuals_sds1_exact_replicated(spec_sds1):
    """Full replication should use exact R2 = Y - Ybar_kt."""
    sds4_df = pd.DataFrame(
        {
            'lane': ['A', 'A', 'B', 'B'] * 2,
            'time': [1, 1, 1, 1, 2, 2, 2, 2],
            'weight': [10.0, 10.5, 9.0, 9.5, 10.2, 10.4, 9.1, 9.3],
        }
    )
    df = _prepare_for_vas(sds4_df, spec_sds1)
    result = calculate_vas_residuals(df, spec_sds1, r2_method='exact')

    expected_r2 = result['weight'] - result['Ybar_kt']
    pd.testing.assert_series_equal(result['R2'], expected_r2, check_names=False)


def test_calculate_vas_residuals_sparse_uses_moving_average(spec_sds1):
    """Sparse data (all n=1) should use MA2 for R2."""
    sds6_df = pd.DataFrame({'lane': ['A', 'A', 'A'], 'time': [1, 2, 3], 'weight': [10.0, 11.0, 12.0]})
    df = _prepare_for_vas(sds6_df, spec_sds1)
    result = calculate_vas_residuals(df, spec_sds1, r2_method='ma2')

    assert pd.isna(result['R2'].iloc[0])  # j=1: no predecessor → NaN
    assert pytest.approx(result['R2'].iloc[1], 0.01) == 0.5
    assert pytest.approx(result['R2'].iloc[2], 0.01) == 0.5


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


def test_calculate_vas_residuals_with_single_observation_per_row():
    """Should handle edge case of minimal data."""
    df = pd.DataFrame({'lane': ['A', 'A'], 'time': [1, 1], 'weight': [10.0, 10.5]})
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        rsg_var_name='rsg',
        time_var='time',
    )
    prepared = _prepare_for_vas(df, spec)
    result = calculate_vas_residuals(prepared, spec, r2_method='exact')

    assert len(result) == 2
    assert all(col in result.columns for col in ['R1', 'R2', 'R3', 'R4', 'R5'])


# ============================================================================
# Test: No NaN in any residual for any R2 method
# ============================================================================


@pytest.fixture
def sds2_df():
    """SDS 2 data - all cells have n=1 (uses MA2)."""
    return pd.DataFrame(
        {
            'lane': ['A', 'A', 'A', 'B', 'B', 'B'],
            'time': [1, 2, 3, 1, 2, 3],
            'weight': [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        }
    )


@pytest.fixture
def sds3_df():
    """SDS 3 data - mixed cell sizes (uses hybrid)."""
    return pd.DataFrame(
        {
            'lane': ['A', 'A', 'A', 'A', 'B', 'B', 'B'],
            'time': [1, 1, 2, 2, 1, 2, 3],
            'weight': [10.0, 10.5, 11.0, 11.5, 20.0, 21.0, 22.0],
        }
    )


@pytest.mark.parametrize(
    'r2_method,df_key',
    [
        ('exact', 'sds1'),
        ('ma2', 'sds2'),
        ('hybrid', 'sds3'),
    ],
)
def test_no_nan_residuals_all_r2_methods(r2_method, df_key, sds1_df, sds2_df, sds3_df, spec_sds1):
    """Exact method has no NaN residuals. MA2 methods have NaN at j=1.

    Bishop leaves the j=1 MA2 residual blank (no predecessor for the
    moving average). R3-R5 inherit NaN from R2 via arithmetic propagation.
    R1 is always defined (Y - Ybar).
    """
    dfs = {'sds1': sds1_df, 'sds2': sds2_df, 'sds3': sds3_df}
    raw_df = dfs[df_key]
    df = _prepare_for_vas(raw_df, spec_sds1)

    n_per_cell = df.groupby('cell_key', observed=True)['weight'].transform('size')
    result = calculate_vas_residuals(
        df,
        spec_sds1,
        r2_method=r2_method,
        n_per_cell=n_per_cell,
    )

    # R1 is always NaN-free (Y - Ybar is always defined)
    assert result['R1'].isna().sum() == 0, 'R1 should never have NaN'

    if r2_method == 'exact':
        # Exact method: all cells n≥2, no MA2 → no NaN
        for col in ['R2', 'R3', 'R4', 'R5']:
            assert result[col].isna().sum() == 0, f'{col} has NaN with r2_method=exact'
    else:
        # MA2 methods (ma2, hybrid): j=1 has no predecessor → exactly 1 NaN
        for col in ['R2', 'R3', 'R4', 'R5']:
            assert result[col].isna().sum() == 1, (
                f'{col} should have exactly 1 NaN (j=1) with r2_method={r2_method}, got {result[col].isna().sum()}'
            )


# ============================================================================
# Test: Tom Bishop Validation - R2 = MR/2 for MA2
# ============================================================================


def test_r2_ma2_equals_half_moving_range():
    """
    Validate Tom Bishop's formula: R2 = (Y_j - Y_{j-1}) / 2 = MR / 2.

    This test confirms the mathematical relationship between R2 residuals
    and the moving range used in XmR charts.
    """
    df = pd.DataFrame(
        {
            'weight': [10.0, 12.0, 11.0, 13.0, 12.5],
            'Ybar_kt': [10.0, 12.0, 11.0, 13.0, 12.5],
            'cell_key': [('A', i) for i in range(1, 6)],
            'rsg_key': ['A'] * 5,
            'sort_key': [(('A',), i, 0) for i in range(1, 6)],
        }
    )

    r2 = calculate_r2(df, 'weight', r2_method='ma2')

    # R2_j = (Y_j - Y_{j-1}) / 2
    # j=1: NaN (no predecessor)
    # j=2: (12-10)/2 = 1.0
    # j=3: (11-12)/2 = -0.5
    # j=4: (13-11)/2 = 1.0
    # j=5: (12.5-13)/2 = -0.25
    assert pd.isna(r2.iloc[0])
    assert pytest.approx(r2.iloc[1], 0.01) == 1.0
    assert pytest.approx(r2.iloc[2], 0.01) == -0.5
    assert pytest.approx(r2.iloc[3], 0.01) == 1.0
    assert pytest.approx(r2.iloc[4], 0.01) == -0.25


def test_r2_ma2_multiple_groups():
    """
    MA2 R2 runs across the entire canonical-sorted stream, not per group.

    Bishop Eq 13.7-13.9: j=2,...,J with no grouping. Only j=1 gets R2=NaN.
    The MA2 continues across rsg_key boundaries.
    """
    df = pd.DataFrame(
        {
            'weight': [10.0, 11.0, 12.0, 20.0, 22.0, 21.0],
            'Ybar_kt': [10.0, 11.0, 12.0, 20.0, 22.0, 21.0],
            'cell_key': [('A', 1), ('A', 2), ('A', 3), ('B', 1), ('B', 2), ('B', 3)],
            'rsg_key': ['A', 'A', 'A', 'B', 'B', 'B'],
            'sort_key': [
                (('A',), 1, 0),
                (('A',), 2, 0),
                (('A',), 3, 0),
                (('B',), 1, 0),
                (('B',), 2, 0),
                (('B',), 3, 0),
            ],
        }
    )

    result = calculate_r2(df, 'weight', r2_method='ma2')

    # Full stream MA2 (no grouping):
    # j=0 (10.0): first obs → NaN (no predecessor)
    # j=1 (11.0): (11-10)/2 = 0.5
    # j=2 (12.0): (12-11)/2 = 0.5
    # j=3 (20.0): (20-12)/2 = 4.0  (crosses A→B boundary)
    # j=4 (22.0): (22-20)/2 = 1.0
    # j=5 (21.0): (21-22)/2 = -0.5
    assert pd.isna(result.iloc[0])
    assert pytest.approx(result.iloc[1], 0.01) == 0.5
    assert pytest.approx(result.iloc[2], 0.01) == 0.5
    assert pytest.approx(result.iloc[3], 0.01) == 4.0
    assert pytest.approx(result.iloc[4], 0.01) == 1.0
    assert pytest.approx(result.iloc[5], 0.01) == -0.5


def test_r2_ma2_tom_bishop_example():
    """
    Test with values inspired by Tom Bishop's Figure 30 example.

    Validates that R2 = (Y_j - Y_{j-1}) / 2 for each consecutive pair,
    removing trend (PT effect) and leaving unexplained variation.
    """
    weights = [238.0, 239.0, 240.0, 239.5, 240.5, 241.0, 240.0, 241.5, 241.0, 242.0]
    df = pd.DataFrame(
        {
            'weight': weights,
            'Ybar_kt': weights,  # n=1 per cell, so Ybar_kt = Y
            'cell_key': [('Lane4', i) for i in range(1, 11)],
            'rsg_key': ['Lane4'] * 10,
            'sort_key': [(('Lane4',), i, 0) for i in range(1, 11)],
        }
    )

    result = calculate_r2(df, 'weight', r2_method='ma2')

    assert pd.isna(result.iloc[0])  # j=1: no predecessor → NaN
    for i in range(1, len(result)):
        y_current = df['weight'].iloc[i]
        y_previous = df['weight'].iloc[i - 1]
        expected_r2 = (y_current - y_previous) / 2.0
        assert pytest.approx(result.iloc[i], 0.01) == expected_r2

    # R2 values should be smaller than original variation (trend removed)
    assert result.iloc[1:].abs().max() < df['weight'].std()


def test_r2_ma2_handles_single_group():
    """MA2 R2 calculation should handle minimal data (2 points) correctly."""
    df = pd.DataFrame(
        {
            'weight': [10.0, 11.0],
            'Ybar_kt': [10.0, 11.0],
            'cell_key': [('A', 1), ('A', 2)],
            'rsg_key': ['A', 'A'],
            'sort_key': [(('A',), 1, 0), (('A',), 2, 0)],
        }
    )

    result = calculate_r2(df, 'weight', r2_method='ma2')

    assert len(result) == 2
    assert pd.isna(result.iloc[0])  # j=1: no predecessor → NaN
    assert pytest.approx(result.iloc[1], 0.01) == 0.5  # (11-10)/2


# ============================================================================
# Test: R1/RCR1 VAS Invariants (replaces zero_center functionality)
# ============================================================================


def test_r1_rcr1_invariants():
    """
    R1 and RCR1 replace zero_center functionality.

    VAS semantics guarantee:
    - R1 = Y - Ȳ (zero-centered, mean ≈ 0)
    - RCR1 = R1 + Ȳ = Y (original scale)
    - RCR1 - R1 = Ȳ (constant difference)
    """
    import numpy as np

    from processbehavior import ProcessBehavior
    from processbehavior.datasets.synthetic import make_sds

    df = make_sds(1, seed=42)
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', time='time', factors=['factor 1', 'factor 2'])
    result = study.execute()

    ds = result.dataset
    y_mean = np.mean(ds['y'].values)
    ybar = ds['Ybar'].iloc[0]  # unweighted grand mean (mean of cell means)

    # RCR1 = Ybar + R1 = Y (always true regardless of how Ybar is computed)
    assert np.allclose(ds['RCR1'].values, ds['y'].values, atol=1e-8)

    # RCR1 - R1 = Ȳ (constant for all rows)
    diff = ds['RCR1'].values - ds['R1'].values
    assert np.allclose(diff, ybar, atol=1e-8)

    # mean(RCR1) = mean(Y) (since RCR1 = Y)
    assert np.isclose(np.mean(ds['RCR1'].values), y_mean, atol=1e-10)


# ============================================================================
# Test: Residual Chart Column Selection (Issue #66)
# ============================================================================


def test_residual_chart_uses_correct_column():
    """
    Verify R5_Xbar uses R5 column, not response variable.

    This test catches the bug where residual charts incorrectly calculated
    statistics on Y instead of the residual column. By constructing a dataset
    where Y is constant but R5 varies, we can detect if the wrong column is used.

    If the bug exists: R5_Xbar center would equal Y's constant value (100)
    After fix: R5_Xbar center should equal mean(R5) ≈ 3.5
    """
    import numpy as np

    from processbehavior import ProcessBehavior
    from processbehavior.datasets.synthetic import make_sds

    # Create SDS 1 data (has R5 column)
    df = make_sds(1, seed=42)
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', time='time', factors=['factor 1', 'factor 2'])
    result = study.execute()

    # Get actual values from the dataset
    ds = result.dataset
    # Bishop VAS Xbar centers use mean of cell means at the residual's natural
    # grain, not observation-weighted df[col].mean(). For Y, that's the
    # pre-computed Ybar (full cell grid). For R5, it's the rsg-level grain.
    y_center_expected = ds['Ybar'].iloc[0]
    r5_center_expected = ds.groupby(['factor 1', 'factor 2'], observed=True)['R5'].mean().mean()

    # Y and R5 should have different centers (otherwise test is not meaningful)
    assert not np.isclose(y_center_expected, r5_center_expected, atol=0.1), (
        'Test data should have different Y and R5 centers'
    )

    # Get Xbar center (uses Y) - should equal Bishop unweighted Y center.
    xbar_stats = result.get_statistics('Xbar')
    assert np.isclose(xbar_stats['center'], y_center_expected, atol=0.01), (
        f'Xbar center {xbar_stats["center"]} should equal {y_center_expected}'
    )

    # Get R5 Xbar center (should use R5) - Bishop unweighted at rsg grain.
    r5_result = study.execute(chart='Xbar', value='R5')
    r5_xbar_stats = r5_result.get_statistics('Xbar')
    assert np.isclose(r5_xbar_stats['center'], r5_center_expected, atol=0.01), (
        f'R5 Xbar center {r5_xbar_stats["center"]} should equal {r5_center_expected}'
    )

    # Most importantly: R5 Xbar center should NOT equal Y center
    # This catches the bug where residual charts used Y instead of R5
    assert not np.isclose(r5_xbar_stats['center'], y_center_expected, atol=0.1), (
        f'R5 Xbar center {r5_xbar_stats["center"]} should NOT equal {y_center_expected}'
    )


# ============================================================================
# Test: Unweighted Means (mean of cell means) for unbalanced data
# ============================================================================


def test_unweighted_means_with_unbalanced_cells():
    """
    VAS marginal means must be unweighted (mean of cell means), not
    observation-weighted. This matters when cell sizes vary.

    With unbalanced data, observation-weighted averages give more weight to
    larger cells. Bishop VAS treats each experimental condition equally.
    """
    import numpy as np

    # Unbalanced: cell (A,1) has 3 obs, cell (A,2) has 2 obs,
    # cell (B,1) has 2 obs, cell (B,2) has 3 obs
    df = pd.DataFrame(
        {
            'lane': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
            'time': [1, 1, 1, 2, 2, 1, 1, 2, 2, 2],
            'weight': [10.0, 11.0, 12.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 42.0],
        }
    )
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        rsg_var_name='rsg',
        time_var='time',
    )
    prepared = _prepare_for_vas(df, spec)
    result = calculate_vas_residuals(prepared, spec, r2_method='exact')

    # Cell means (observation-weighted within each cell):
    # (A,1): mean(10,11,12) = 11.0
    # (A,2): mean(20,21) = 20.5
    # (B,1): mean(30,31) = 30.5
    # (B,2): mean(40,41,42) = 41.0
    cell_means = [11.0, 20.5, 30.5, 41.0]

    # Unweighted grand mean = mean of cell means
    expected_grand = np.mean(cell_means)  # 25.75
    assert np.isclose(result['Ybar'].iloc[0], expected_grand, atol=1e-10), (
        f'Grand mean {result["Ybar"].iloc[0]} != expected {expected_grand}'
    )

    # Observation-weighted grand mean would be different
    obs_weighted_grand = df['weight'].mean()  # 25.8
    assert not np.isclose(expected_grand, obs_weighted_grand, atol=1e-10), (
        'Test data should produce different weighted vs unweighted grand means'
    )

    # Unweighted factor means = mean of cell means per factor
    expected_factor_A = np.mean([11.0, 20.5])  # 15.75
    expected_factor_B = np.mean([30.5, 41.0])  # 35.75
    a_rows = result[result['rsg'] == 'A']
    b_rows = result[result['rsg'] == 'B']
    assert np.isclose(a_rows['Ybar_k'].iloc[0], expected_factor_A, atol=1e-10)
    assert np.isclose(b_rows['Ybar_k'].iloc[0], expected_factor_B, atol=1e-10)

    # Unweighted time means = mean of cell means per time
    expected_time_1 = np.mean([11.0, 30.5])  # 20.75
    expected_time_2 = np.mean([20.5, 41.0])  # 30.75
    t1_rows = result[result['time'] == 1]
    t2_rows = result[result['time'] == 2]
    assert np.isclose(t1_rows['Ybar_t'].iloc[0], expected_time_1, atol=1e-10)
    assert np.isclose(t2_rows['Ybar_t'].iloc[0], expected_time_2, atol=1e-10)

    # Verify residual algebra still holds
    assert np.allclose(result['R1'], result['weight'] - result['Ybar'], atol=1e-10)
    assert np.allclose(result['R2'], result['weight'] - result['Ybar_kt'], atol=1e-10)
    assert np.allclose(
        result['R3'],
        result['weight'] - result['Ybar_k'] - result['Ybar_t'] + result['Ybar'],
        atol=1e-10,
    )
    assert np.allclose(result['R4'], result['Ybar_t'] - result['Ybar'] + result['R2'], atol=1e-10)
    assert np.allclose(result['R5'], result['Ybar_k'] - result['Ybar'] + result['R2'], atol=1e-10)
