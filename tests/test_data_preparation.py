"""
Unit tests for DataPreparation class.

Tests cover:
- Column validation (helpful errors)
- Grouping variable creation
- Small group filtering
- Key generation
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from data_preparation import DataPreparation
from analysis_dataset import AnalysisSpecification


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def prep():
    """Create DataPreparation instance."""
    return DataPreparation()


@pytest.fixture
def simple_df():
    """Simple DataFrame with factors, time, and response."""
    return pd.DataFrame({
        'lane': ['A', 'A', 'A', 'B', 'B', 'B'],
        'pull': [1, 2, 3, 1, 2, 3],
        'weight': [10.1, 10.3, 10.2, 9.9, 9.8, 10.0]
    })


@pytest.fixture
def multi_factor_df():
    """DataFrame with multiple factors - ensure n>1 per RSG."""
    return pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'head': [1, 1, 2, 2, 1, 1, 2, 2],
        'pull': [1, 2, 1, 2, 1, 2, 1, 2],
        'weight': [10.1, 10.2, 10.3, 10.4, 9.9, 9.8, 10.0, 9.7]
    })


@pytest.fixture
def small_groups_df():
    """DataFrame with some groups having n=1."""
    return pd.DataFrame({
        'lane': ['A', 'A', 'B', 'C', 'C'],  # A:2, B:1, C:2
        'weight': [10.1, 10.2, 9.9, 9.5, 9.6]
    })


@pytest.fixture
def spec_xbar():
    """Xbar analysis specification."""
    return AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'time_var': 'pull',
        'response_var': 'weight'
    })


@pytest.fixture
def spec_multi_factor():
    """Multi-factor specification."""
    return AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane', 'head'],
        'time_var': 'pull',
        'response_var': 'weight'
    })


# ============================================================================
# Test: validate_columns
# ============================================================================

def test_validate_columns_passes_with_valid_data(prep, simple_df, spec_xbar):
    """Should pass validation when all required columns present."""
    # Should not raise
    prep.validate_columns(simple_df, spec_xbar)


def test_validate_columns_raises_on_missing_response(prep, simple_df):
    """Should raise helpful error if response variable missing."""
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'missing_column'
    })

    with pytest.raises(ValueError, match="Response variable 'missing_column' not found"):
        prep.validate_columns(simple_df, spec)


def test_validate_columns_raises_on_missing_grouping_var(prep, simple_df):
    """Should raise helpful error if grouping variable missing."""
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['missing_lane'],
        'response_var': 'weight'
    })

    with pytest.raises(ValueError, match="grouping variables not found"):
        prep.validate_columns(simple_df, spec)


def test_validate_columns_raises_on_missing_time(prep, simple_df):
    """Should raise helpful error if time variable missing."""
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'time_var': 'missing_time',
        'response_var': 'weight'
    })

    with pytest.raises(ValueError, match="Time variable 'missing_time' not found"):
        prep.validate_columns(simple_df, spec)


def test_validate_columns_raises_on_non_numeric_response(prep):
    """Should raise helpful error if response variable is not numeric."""
    df = pd.DataFrame({
        'lane': ['A', 'B'],
        'weight': ['ten', 'nine']  # Strings!
    })
    spec = AnalysisSpecification('Imr', {
        'response_var': 'weight'
    })

    with pytest.raises(ValueError, match="must be numeric"):
        prep.validate_columns(df, spec)


def test_validate_columns_error_suggests_fix(prep, simple_df):
    """Error messages should suggest how to fix the problem."""
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'weigth'  # Typo!
    })

    with pytest.raises(ValueError, match="Fix:"):
        prep.validate_columns(simple_df, spec)


# ============================================================================
# Test: prepare_dataset
# ============================================================================

def test_prepare_dataset_creates_rsg_column(prep, simple_df, spec_xbar):
    """Should create 'rsg' column from grouping variable."""
    result = prep.prepare_dataset(simple_df, spec_xbar)

    assert 'rsg' in result.columns
    assert result['rsg'].tolist() == ['A', 'A', 'A', 'B', 'B', 'B']


def test_prepare_dataset_creates_n_column(prep, simple_df, spec_xbar):
    """Should create 'n' column with group sizes."""
    result = prep.prepare_dataset(simple_df, spec_xbar)

    assert 'n' in result.columns
    assert all(result['n'] > 1)  # All groups should have n > 1


def test_prepare_dataset_removes_small_groups(prep, small_groups_df):
    """Should remove groups with n ≤ 1."""
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'weight'
    })

    result = prep.prepare_dataset(small_groups_df, spec)

    # Only A and C should remain (both have n=2)
    # B should be removed (n=1)
    assert 'A' in result['rsg'].values
    assert 'B' not in result['rsg'].values
    assert 'C' in result['rsg'].values


def test_prepare_dataset_raises_if_all_groups_small(prep):
    """Should raise helpful error if all groups have n ≤ 1."""
    df = pd.DataFrame({
        'lane': ['A', 'B', 'C'],  # All have n=1
        'weight': [10.1, 9.9, 10.0]
    })
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'weight'
    })

    with pytest.raises(ValueError, match="All subgroups have 1 or fewer observations"):
        prep.prepare_dataset(df, spec)


def test_prepare_dataset_sorts_when_needed(prep, simple_df, spec_xbar):
    """Should sort by grouping and time when both present."""
    # Scramble the data
    scrambled = simple_df.sample(frac=1.0, random_state=42)

    result = prep.prepare_dataset(scrambled, spec_xbar)

    # Check sorted by rsg then time
    expected_order = result.sort_values(['rsg', 'pull'])
    pd.testing.assert_frame_equal(
        result[['rsg', 'pull']],
        expected_order[['rsg', 'pull']]
    )


def test_prepare_dataset_drops_na_rows(prep):
    """Should drop rows with missing values."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'weight': [10.1, np.nan, 9.9, 10.0]  # One NaN
    })
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'weight'
    })

    result = prep.prepare_dataset(df, spec)

    assert len(result) == 3  # 4 - 1 NaN = 3


def test_prepare_dataset_keeps_only_requested_columns(prep, simple_df, spec_xbar):
    """Should only keep columns specified in spec.data_prep_output_cols."""
    # Add extra junk columns
    df_with_junk = simple_df.copy()
    df_with_junk['junk1'] = 999
    df_with_junk['junk2'] = 'garbage'

    result = prep.prepare_dataset(df_with_junk, spec_xbar)

    # Should not have junk columns
    assert 'junk1' not in result.columns
    assert 'junk2' not in result.columns


def test_prepare_dataset_creates_composite_grouping(prep, multi_factor_df, spec_multi_factor):
    """Should create composite grouping variable from multiple factors."""
    result = prep.prepare_dataset(multi_factor_df, spec_multi_factor)

    assert 'rsg' in result.columns
    # Should combine lane and head with delimiter
    assert 'A_1' in result['rsg'].values
    assert 'A_2' in result['rsg'].values
    assert 'B_1' in result['rsg'].values


# ============================================================================
# Test: build_keys
# ============================================================================

def test_build_keys_adds_obs_id(prep, simple_df, spec_xbar):
    """Should add unique obs_id to each row."""
    result = prep.build_keys(simple_df, spec_xbar)

    assert 'obs_id' in result.columns
    assert len(result['obs_id'].unique()) == len(result)
    assert result['obs_id'].dtype == np.int64


def test_build_keys_adds_rsg_key_tuples(prep, simple_df, spec_xbar):
    """Should add rsg_key as tuples of factor values."""
    result = prep.build_keys(simple_df, spec_xbar)

    assert 'rsg_key' in result.columns
    # Keys should be tuples
    assert all(isinstance(k, tuple) for k in result['rsg_key'])


def test_build_keys_adds_cell_key(prep, simple_df, spec_xbar):
    """Should add cell_key as tuples of (factor + time)."""
    result = prep.build_keys(simple_df, spec_xbar)

    assert 'cell_key' in result.columns
    # First row should be ('A', 1) or similar
    assert isinstance(result['cell_key'].iloc[0], tuple)


def test_build_keys_empty_rsg_key_when_no_factors(prep):
    """Should use empty tuples for rsg_key when no grouping."""
    df = pd.DataFrame({'weight': [10.1, 10.2]})
    spec = AnalysisSpecification('Imr', {
        'response_var': 'weight'
    })

    result = prep.build_keys(df, spec)

    assert 'rsg_key' in result.columns
    assert all(k == () for k in result['rsg_key'])


# ============================================================================
# Test: Private helper methods
# ============================================================================

def test_add_composite_column_combines_multiple_cols(prep):
    """Should combine multiple columns with delimiter."""
    df = pd.DataFrame({
        'col1': ['A', 'B'],
        'col2': ['X', 'Y']
    })

    result = prep._add_composite_column(
        df,
        cols_to_combine=['col1', 'col2'],
        col_name='combined',
        col_delim='_'
    )

    assert 'combined' in result.columns
    assert result['combined'].tolist() == ['A_X', 'B_Y']


def test_add_composite_column_custom_delimiter(prep):
    """Should use custom delimiter."""
    df = pd.DataFrame({
        'col1': ['A', 'B'],
        'col2': ['X', 'Y']
    })

    result = prep._add_composite_column(
        df,
        cols_to_combine=['col1', 'col2'],
        col_name='combined',
        col_delim='|'
    )

    assert result['combined'].tolist() == ['A|X', 'B|Y']


def test_add_composite_column_raises_on_missing(prep):
    """Should raise helpful error if column missing."""
    df = pd.DataFrame({'col1': ['A']})

    with pytest.raises(ValueError, match="missing"):
        prep._add_composite_column(
            df,
            cols_to_combine=['col1', 'missing_col'],
            col_name='combined'
        )


def test_add_column_copies_existing(prep):
    """Should copy existing column with new name."""
    df = pd.DataFrame({'old_name': [1, 2, 3]})

    result = prep._add_column(df, 'new_name', 'old_name')

    assert 'new_name' in result.columns
    assert result['new_name'].tolist() == [1, 2, 3]


def test_filter_small_groups_logs_count(prep, small_groups_df, caplog):
    """Should log how many groups were filtered."""
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'weight'
    })

    # Need to add RSG column first (what _add_grouping_column does)
    df_with_rsg = prep._add_grouping_column(small_groups_df, spec)

    with caplog.at_level('DEBUG'):
        result = prep._filter_small_groups(df_with_rsg, spec)

    # Should log starting and ending counts
    assert 'Starting with' in caplog.text
    assert 'remaining' in caplog.text.lower()


# ============================================================================
# Test: Edge cases
# ============================================================================

def test_prepare_dataset_with_single_factor(prep):
    """Should work with single factor (no composite needed)."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'weight': [10.1, 10.2, 9.9, 10.0]
    })
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'weight'
    })

    result = prep.prepare_dataset(df, spec)

    assert 'rsg' in result.columns
    # Should just copy lane values
    assert result['rsg'].tolist() == ['A', 'A', 'B', 'B']


def test_prepare_dataset_without_grouping(prep):
    """Should work without grouping variables (IMR case)."""
    df = pd.DataFrame({
        'time': [1, 2, 3],
        'weight': [10.1, 10.2, 10.3]
    })
    spec = AnalysisSpecification('Imr', {
        'time_var': 'time',
        'response_var': 'weight'
    })

    result = prep.prepare_dataset(df, spec)

    # Should not have rsg or n columns
    assert 'rsg' not in result.columns
    assert 'n' not in result.columns


def test_prepare_dataset_without_time(prep):
    """Should work without time variable."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'weight': [10.1, 10.2, 9.9, 10.0]
    })
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'weight'
    })

    result = prep.prepare_dataset(df, spec)

    # Should still work, just no sorting by time
    assert 'rsg' in result.columns


def test_prepare_dataset_preserves_data_types(prep, simple_df, spec_xbar):
    """Should preserve data types of response variable."""
    result = prep.prepare_dataset(simple_df, spec_xbar)

    assert pd.api.types.is_numeric_dtype(result['weight'])
    assert result['weight'].dtype == simple_df['weight'].dtype


def test_validate_columns_with_categorical_grouping(prep):
    """Should accept categorical data types for grouping variables."""
    df = pd.DataFrame({
        'lane': pd.Categorical(['A', 'A', 'B', 'B']),
        'weight': [10.1, 10.2, 9.9, 10.0]
    })
    spec = AnalysisSpecification('Xbar', {
        'rsg_vars': ['lane'],
        'response_var': 'weight'
    })

    # Should not raise
    prep.validate_columns(df, spec)


# ============================================================================
# Test: Integration scenarios
# ============================================================================

def test_full_preparation_pipeline(prep, simple_df, spec_xbar):
    """Test complete preparation pipeline."""
    # Step 1: Validate
    prep.validate_columns(simple_df, spec_xbar)

    # Step 2: Prepare
    result = prep.prepare_dataset(simple_df, spec_xbar)

    # Step 3: Add keys
    result = prep.build_keys(result, spec_xbar)

    # Verify all expected columns present
    expected_cols = {'rsg', 'pull', 'weight', 'n', 'obs_id', 'rsg_key', 'cell_key'}
    assert expected_cols.issubset(result.columns)

    # Verify data integrity
    assert len(result) == 6  # All rows kept
    assert result['weight'].notna().all()
    assert (result['n'] > 1).all()
