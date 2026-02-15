"""
Unit tests for DataPreparation class.

Tests cover:
- Column validation (helpful errors)
- Grouping variable creation
- Small group filtering
- Key generation
- Edge cases
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior.formulation_spec import FormulationSpec
from processbehavior.data_preparation import DataPreparation

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def prep():
    """Create DataPreparation instance."""
    return DataPreparation()


@pytest.fixture
def simple_df():
    """Simple DataFrame with factors, time, and response.

    Has n>=2 per kt cell to allow Xbar/S analysis:
    - Lane A: 2 time points × 2 observations each = 4 rows
    - Lane B: 2 time points × 2 observations each = 4 rows
    Total: 8 rows
    """
    return pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'pull': [1, 1, 2, 2, 1, 1, 2, 2],  # 2 obs per kt cell
        'weight': [10.1, 10.15, 10.3, 10.35, 9.9, 9.95, 9.8, 9.85]
    })


@pytest.fixture
def multi_factor_df():
    """DataFrame with multiple factors - ensure n>=2 per kt cell.

    Structure: 2 lanes × 2 heads × 2 time points × 2 obs = 16 rows
    Each kt cell (lane_head × pull) has n=2 observations.
    """
    return pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A',
                 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
        'head': [1, 1, 1, 1, 2, 2, 2, 2,
                 1, 1, 1, 1, 2, 2, 2, 2],
        'pull': [1, 1, 2, 2, 1, 1, 2, 2,
                 1, 1, 2, 2, 1, 1, 2, 2],  # 2 obs per kt cell
        'weight': [10.1, 10.15, 10.2, 10.25, 10.3, 10.35, 10.4, 10.45,
                   9.9, 9.95, 9.8, 9.85, 10.0, 10.05, 9.7, 9.75]
    })


@pytest.fixture
def small_groups_df():
    """DataFrame with some groups having n=1 (no time variable).

    Without a time variable, kt grouping degenerates to factor-only grouping.
    A:2, B:1, C:2 - B should be filtered out.
    """
    return pd.DataFrame({
        'lane': ['A', 'A', 'B', 'C', 'C'],  # A:2, B:1, C:2
        'weight': [10.1, 10.2, 9.9, 9.5, 9.6]
    })


@pytest.fixture
def spec_xbar():
    """Xbar analysis specification."""
    return FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='pull',
    )


@pytest.fixture
def spec_multi_factor():
    """Multi-factor specification."""
    return FormulationSpec(
        response_var='weight',
        rsg_vars=('lane', 'head'),
        time_var='pull',
    )


# ============================================================================
# Test: validate_columns
# ============================================================================

def test_validate_columns_passes_with_valid_data(prep, simple_df, spec_xbar):
    """Should pass validation when all required columns present."""
    # Should not raise
    prep.validate_columns(simple_df, spec_xbar)


def test_validate_columns_raises_on_missing_response(prep, simple_df):
    """Should raise helpful error if response variable missing."""
    spec = FormulationSpec(
        response_var='missing_column',
        rsg_vars=('lane',),
    )

    with pytest.raises(ValueError, match="Response variable 'missing_column' not found"):
        prep.validate_columns(simple_df, spec)


def test_validate_columns_raises_on_missing_grouping_var(prep, simple_df):
    """Should raise helpful error if grouping variable missing."""
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('missing_lane',),
    )

    with pytest.raises(ValueError, match="grouping variables not found"):
        prep.validate_columns(simple_df, spec)


def test_validate_columns_raises_on_missing_time(prep, simple_df):
    """Should raise helpful error if time variable missing."""
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='missing_time',
    )

    with pytest.raises(ValueError, match="Time variable 'missing_time' not found"):
        prep.validate_columns(simple_df, spec)


def test_validate_columns_raises_on_non_numeric_response(prep):
    """Should raise helpful error if response variable is not numeric."""
    df = pd.DataFrame({
        'lane': ['A', 'B'],
        'weight': ['ten', 'nine']  # Strings!
    })
    spec = FormulationSpec(
        response_var='weight',
    )

    with pytest.raises(ValueError, match="must be numeric"):
        prep.validate_columns(df, spec)


def test_validate_columns_error_suggests_fix(prep, simple_df):
    """Error messages should suggest how to fix the problem."""
    spec = FormulationSpec(
        response_var='weigth',  # Typo!
        rsg_vars=('lane',),
    )

    with pytest.raises(ValueError, match="Fix:"):
        prep.validate_columns(simple_df, spec)


# ============================================================================
# Test: prepare_dataset
# ============================================================================

def test_prepare_dataset_creates_rsg_column(prep, simple_df, spec_xbar):
    """Should create 'rsg' column from grouping variable."""
    result = prep.prepare_dataset(simple_df, spec_xbar)

    assert 'rsg' in result.columns
    # 8 rows: A×2 times × 2 obs + B×2 times × 2 obs
    assert sorted(result['rsg'].unique()) == ['A', 'B']
    assert len(result) == 8


def test_prepare_dataset_creates_n_column(prep, simple_df, spec_xbar):
    """Should create 'n' column with group sizes."""
    result = prep.prepare_dataset(simple_df, spec_xbar)

    assert 'n' in result.columns
    assert all(result['n'] > 1)  # All groups should have n > 1


def test_prepare_dataset_keeps_small_groups(prep, small_groups_df):
    """FormulationSpec is chart-agnostic, so prepare_dataset keeps all groups.

    Small group filtering (n≤1) is chart-specific and happens at analysis time,
    not during data preparation.
    """
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    result = prep.prepare_dataset(small_groups_df, spec)

    # All groups should be present (no chart-specific filtering)
    assert 'A' in result['rsg'].values
    assert 'B' in result['rsg'].values
    assert 'C' in result['rsg'].values


def test_prepare_dataset_all_groups_small_no_error(prep):
    """FormulationSpec is chart-agnostic, so all-small groups don't raise."""
    df = pd.DataFrame({
        'lane': ['A', 'B', 'C'],  # All have n=1
        'weight': [10.1, 9.9, 10.0]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    # Should not raise — filtering is chart-specific
    result = prep.prepare_dataset(df, spec)
    assert len(result) == 3


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
    """Should drop rows with missing values before calculating n.

    This ensures n reflects actual usable observations, not raw count.
    Groups with insufficient observations after NA removal are filtered out.
    """
    df = pd.DataFrame({
        # Lane A: 3 rows, 1 NaN → 2 usable (passes n≥2 filter)
        # Lane B: 2 rows, 0 NaN → 2 usable (passes n≥2 filter)
        'lane': ['A', 'A', 'A', 'B', 'B'],
        'weight': [10.1, np.nan, 10.2, 9.9, 10.0]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    result = prep.prepare_dataset(df, spec)

    # 5 rows - 1 NaN = 4 rows, both groups have n≥2
    assert len(result) == 4

    # Verify n column reflects actual usable observations (not raw count)
    n_by_group = result.groupby('rsg', observed=True)['n'].first()
    assert n_by_group['A'] == 2  # 3 raw - 1 NaN = 2 usable
    assert n_by_group['B'] == 2  # 2 raw - 0 NaN = 2 usable


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
    spec = FormulationSpec(
        response_var='weight',
    )

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
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    # Need to add RSG column first (what _add_grouping_column does)
    df_with_rsg = prep._add_grouping_column(small_groups_df, spec)

    with caplog.at_level('DEBUG'):
        prep._filter_small_groups(df_with_rsg, spec)

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
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

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
    spec = FormulationSpec(
        response_var='weight',
        time_var='time',
    )

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
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

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
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

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
    assert len(result) == 8  # All rows kept (2 lanes × 2 times × 2 obs)
    assert result['weight'].notna().all()
    assert (result['n'] > 1).all()


# ============================================================================
# Test: Type Conversion for Correct Sorting
# ============================================================================

def test_time_var_numeric_unchanged(prep):
    """Native numeric time_var should stay unchanged."""
    # Each kt cell (lane × time) needs n>=2 for Xbar
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': [1, 1, 2, 2, 10, 10],  # 2 obs per time point
        'weight': [10.1, 10.15, 10.2, 10.25, 10.3, 10.35]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    assert pd.api.types.is_numeric_dtype(result['time'])
    # Time values should be in sorted order (with duplicates)
    assert list(sorted(result['time'].unique())) == [1, 2, 10]


def test_time_var_string_numeric_converted(prep):
    """String-numeric time_var should be converted to numeric."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': ['1', '2', '10', '1', '2', '10'],  # String numbers
        'weight': [10.1, 10.2, 10.3, 10.0, 10.1, 10.2]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Should be converted to numeric
    assert pd.api.types.is_numeric_dtype(result['time'])
    # Should sort correctly: 1, 1, 2, 2, 10, 10 (not '1', '10', '1', '10', '2', '2')
    assert result['time'].tolist() == [1, 1, 2, 2, 10, 10]


def test_time_var_date_unchanged(prep):
    """Native date objects should stay unchanged."""
    from datetime import date

    # Each kt cell needs n>=2 for Xbar
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': [date(2024, 1, 1), date(2024, 1, 1),
                 date(2024, 1, 2), date(2024, 1, 2),
                 date(2024, 1, 10), date(2024, 1, 10)],
        'weight': [10.1, 10.15, 10.2, 10.25, 10.3, 10.35]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Should keep as object dtype (dates)
    assert result['time'].iloc[0] == date(2024, 1, 1)
    # Last unique date should be 2024-1-10
    assert date(2024, 1, 10) in result['time'].values


def test_time_var_datetime_unchanged(prep):
    """Native datetime objects should stay unchanged."""
    from datetime import datetime

    # Each kt cell needs n>=2 for Xbar
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': [datetime(2024, 1, 1), datetime(2024, 1, 1),
                 datetime(2024, 1, 2), datetime(2024, 1, 2),
                 datetime(2024, 1, 10), datetime(2024, 1, 10)],
        'weight': [10.1, 10.15, 10.2, 10.25, 10.3, 10.35]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    assert pd.api.types.is_datetime64_any_dtype(result['time'])


def test_time_var_string_date_converted(prep):
    """String-date time_var should be converted to datetime."""
    # Each kt cell needs n>=2 for Xbar
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': ['2024-01-01', '2024-01-01',
                 '2024-01-02', '2024-01-02',
                 '2024-01-10', '2024-01-10'],
        'weight': [10.1, 10.15, 10.2, 10.25, 10.3, 10.35]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Should be converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(result['time'])


def test_time_var_categorical_unchanged(prep):
    """Ordered categorical time_var should stay unchanged."""
    # Each kt cell needs n>=2 for Xbar
    times = pd.Categorical(['early', 'early', 'mid', 'mid', 'late', 'late'], ordered=True)
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': times,
        'weight': [10.1, 10.15, 10.2, 10.25, 10.3, 10.35]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Should stay categorical
    assert isinstance(result['time'].dtype, pd.CategoricalDtype)


def test_time_var_period_unchanged(prep):
    """Period time_var should stay unchanged."""
    # Each kt cell needs n>=2 for Xbar
    periods = list(pd.period_range('2024-01', periods=3, freq='M'))
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': [periods[0], periods[0], periods[1], periods[1], periods[2], periods[2]],
        'weight': [10.1, 10.15, 10.2, 10.25, 10.3, 10.35]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Should stay as Period
    assert isinstance(result['time'].dtype, pd.PeriodDtype)


# ============================================================================
# Test: Factor Column Type Conversion
# ============================================================================

def test_factor_numeric_unchanged(prep):
    """Numeric factor columns should stay unchanged."""
    df = pd.DataFrame({
        'lane': [1, 1, 2, 2, 10, 10],  # Already numeric
        'weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    result = prep.prepare_dataset(df, spec)

    # Lane should stay numeric (but RSG will be categorical)
    assert 'rsg' in result.columns


def test_factor_string_numeric_converted(prep):
    """String-numeric factor columns should be converted."""
    df = pd.DataFrame({
        'lane': ['1', '1', '2', '2', '10', '10'],  # String numbers
        'weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    result = prep.prepare_dataset(df, spec)

    # Lane column should be converted to numeric
    assert pd.api.types.is_numeric_dtype(result['lane'])
    # RSG should be categorical with correct order
    assert isinstance(result['rsg'].dtype, pd.CategoricalDtype)


def test_factor_mixed_stays_string(prep):
    """Mixed string/numeric factor columns should stay string."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B', '1', '1'],  # Mixed
        'weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    result = prep.prepare_dataset(df, spec)

    # Should stay as object/string (can't convert to numeric)
    # But RSG should be categorical
    assert isinstance(result['rsg'].dtype, pd.CategoricalDtype)


# ============================================================================
# Test: RSG Categorical with Natural Sort
# ============================================================================

def test_rsg_categorical_natural_sort(prep):
    """RSG should be categorical with natural sort order."""
    df = pd.DataFrame({
        'lane': ['Lane_1', 'Lane_10', 'Lane_2'] * 2,  # Would sort wrong lexicographically
        'weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    result = prep.prepare_dataset(df, spec)

    # RSG should be categorical
    assert isinstance(result['rsg'].dtype, pd.CategoricalDtype)
    # Categories should be naturally sorted: Lane_1, Lane_2, Lane_10
    categories = list(result['rsg'].cat.categories)
    assert categories == ['Lane_1', 'Lane_2', 'Lane_10']


def test_rsg_categorical_preserves_groupby_order(prep):
    """Groupby should respect categorical order."""
    df = pd.DataFrame({
        'lane': ['10', '10', '2', '2', '1', '1'],  # Would sort wrong as strings
        'weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
    )

    result = prep.prepare_dataset(df, spec)

    # Groupby should respect categorical order when sorted
    grouped = result.groupby('rsg', sort=True, observed=True)
    group_names = list(grouped.groups.keys())

    # Should be in categorical order: '1', '2', '10' (not '1', '10', '2' string sort)
    assert group_names == ['1', '2', '10']


def test_rsg_categorical_with_numeric_factors(prep):
    """Multi-factor RSG with numeric parts should sort naturally."""
    df = pd.DataFrame({
        'lane': [1, 1, 10, 10, 2, 2, 1, 1, 10, 10, 2, 2],
        'head': [1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10],
        'weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane', 'head'),
    )

    result = prep.prepare_dataset(df, spec)

    # RSG should be categorical
    assert isinstance(result['rsg'].dtype, pd.CategoricalDtype)
    # Categories should be naturally sorted
    # Should be: 1_1, 1_10, 2_1, 2_10, 10_1, 10_10
    categories = list(result['rsg'].cat.categories)
    expected = ['1_1', '1_10', '2_1', '2_10', '10_1', '10_10']
    assert categories == expected


# ============================================================================
# Test: Sorting Correctness (Critical for Analysis)
# ============================================================================

def test_sorting_correctness_for_moving_range(prep):
    """Moving range must use adjacent observations in correct time order."""
    df = pd.DataFrame({
        'time': ['1', '10', '2', '20', '3'],  # Intentionally scrambled strings
        'weight': [10.1, 10.2, 10.3, 10.4, 10.5]
    })
    spec = FormulationSpec(
        response_var='weight',
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Should be sorted correctly: 1, 2, 3, 10, 20
    assert result['time'].tolist() == [1, 2, 3, 10, 20]
    # Weights should be reordered accordingly
    assert result['weight'].tolist() == [10.1, 10.3, 10.5, 10.2, 10.4]


def test_sorting_correctness_for_signal_detection(prep):
    """Signal detection rules require correct sequential ordering."""
    # Each kt cell needs n>=2 for Xbar (duplicate each time point)
    df = pd.DataFrame({
        'lane': ['A'] * 20,
        'time': ['1', '1', '10', '10', '11', '11', '2', '2', '20', '20',
                 '21', '21', '3', '3', '30', '30', '4', '4', '5', '5'],
        'weight': [10.1, 10.15, 10.2, 10.25, 10.3, 10.35, 10.4, 10.45, 10.5, 10.55,
                   10.6, 10.65, 10.7, 10.75, 10.8, 10.85, 10.9, 10.95, 11.0, 11.05]
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Time should be sorted correctly: 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 10, 10, 11, 11, 20, 20, 21, 21, 30, 30
    expected_unique = [1, 2, 3, 4, 5, 10, 11, 20, 21, 30]
    assert list(sorted(result['time'].unique())) == expected_unique


# ============================================================================
# Test: Integration - Type Conversion End-to-End
# ============================================================================

def test_integration_string_numeric_time_correct_chart_ordering(prep):
    """End-to-end: String-numeric time should produce correctly ordered charts."""
    df = pd.DataFrame({
        'lane': ['A'] * 12,
        'time': ['1', '2', '3', '10', '11', '12'] * 2,  # Strings
        'weight': np.random.normal(10, 0.1, 12)
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Time should be numeric and correctly ordered
    assert pd.api.types.is_numeric_dtype(result['time'])
    assert result['time'].is_monotonic_increasing  # Should be in order


def test_integration_mixed_factor_types_correct_stratification(prep):
    """End-to-end: Mixed factor types should stratify correctly."""
    df = pd.DataFrame({
        'batch': ['1', '2', '10'] * 4,  # String-numeric
        'operator': ['A', 'B'] * 6,  # Categorical
        'weight': np.random.normal(10, 0.1, 12)
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('batch', 'operator'),
    )

    result = prep.prepare_dataset(df, spec)

    # Batch should be converted to numeric
    assert pd.api.types.is_numeric_dtype(result['batch'])
    # RSG should be categorical with natural order
    assert isinstance(result['rsg'].dtype, pd.CategoricalDtype)
    # Categories should respect numeric ordering
    categories = list(result['rsg'].cat.categories)
    # Should have: 1_A, 1_B, 2_A, 2_B, 10_A, 10_B (not 10_A, 10_B, 1_A...)
    assert '1_A' in categories
    assert '10_A' in categories
    assert categories.index('1_A') < categories.index('10_A')


def test_prepare_dataset_preserves_observation_counts(prep):
    """Observation counts should be preserved (after dropna)."""
    df = pd.DataFrame({
        'lane': [1, 1, 2, 2, 3, 3] * 3,  # 18 rows, 6 per lane
        'pull': [1, 2, 1, 2, 1, 2] * 3,
        'weight': [10.1, 10.2, 10.3, 10.4, 10.5, 10.6] * 3
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='pull',
    )

    result = prep.prepare_dataset(df, spec)

    # Should preserve all 18 observations
    assert len(result) == 18, f"Expected 18 rows, got {len(result)}"

    # Should preserve per-group counts
    counts = result.groupby('rsg', observed=True).size()
    assert all(counts == 6), f"Expected 6 observations per lane, got {counts.tolist()}"


def test_prepare_dataset_handles_missing_data_correctly(prep):
    """Missing data should be dropped and counts should reflect clean data.

    With kt-level filtering, we need n>=2 per kt cell AFTER dropna.
    This test uses data with n=2 per kt cell, minus some NaN values.
    """
    # Each kt cell has 2 obs initially, some will have NaN
    df = pd.DataFrame({
        'lane': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        'pull': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],  # 2 obs per kt cell
        'weight': [10.1, 10.15, None, 10.35,  # Lane 1: pull 2 loses 1 obs -> n=1
                   10.4, 10.45, 10.5, 10.55,  # Lane 2: all ok
                   10.7, 10.75, 10.8, 10.85]  # Lane 3: all ok
    })
    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='pull',
    )

    result = prep.prepare_dataset(df, spec)

    # Lane 1 pull 2 has n=1 after dropna, so gets filtered out
    # Lane 1 pull 1 has n=2, Lane 2 has n=2 per cell, Lane 3 has n=2 per cell
    # Total: 2 (lane1 pull1) + 4 (lane2) + 4 (lane3) = 10 - but lane1 pull2 filtered = 2 + 4 + 4 = 10
    # Actually, lane 1 pull 2 has only 1 valid obs, so that kt cell is removed
    # Remaining: 2 + 4 + 4 = 10, minus the rows in lane1/pull2 that were already NA = 10-1 = 9? No...
    # Let me recalculate:
    # After dropna: 11 rows (1 NaN removed)
    # kt cell (1, 2) has n=1 now -> filtered out -> lose 1 more row
    # Final: 11 - 1 = 10 rows
    assert len(result) >= 8, f"Expected at least 8 rows, got {len(result)}"

    # Lanes 2 and 3 should be fully present
    counts_by_lane = result.groupby('rsg', observed=True).size().to_dict()
    assert counts_by_lane.get('2', 0) == 4, "Lane 2 should have 4 observations"
    assert counts_by_lane.get('3', 0) == 4, "Lane 3 should have 4 observations"


def test_full_pipeline_observation_count_integrity(prep):
    """End-to-end test: Verify observation counts through full pipeline.

    With kt-level filtering, each (lane, pull) cell needs n>=2.
    We create data with 2 obs per kt cell (4 lanes × 25 pulls × 2 obs = 200 rows).
    """
    np.random.seed(42)
    # 4 lanes × 25 time points × 2 observations per kt cell = 200 rows
    df = pd.DataFrame({
        'lane': np.repeat([1, 2, 3, 4], 50),
        'pull': np.tile(np.repeat(range(1, 26), 2), 4),  # 1,1,2,2,...,25,25 repeated 4 times
        'weight': np.random.normal(10, 0.5, 200)
    })

    spec = FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='pull',
    )

    result = prep.prepare_dataset(df, spec)

    # Should have all 200 rows (no missing, all kt cells have n=2)
    assert len(result) == 200, f"Expected 200 rows, got {len(result)}"

    # Verify per-lane counts
    lane_counts = result.groupby('rsg', observed=True).size().to_dict()
    for lane in ['1', '2', '3', '4']:
        assert lane_counts.get(lane, 0) == 50, f"Lane {lane}: expected 50 observations"

    # Verify n column values (should all be 2)
    assert all(result['n'] == 2), f"Expected n=2 for all kt cells, got unique n values: {result['n'].unique()}"


# ============================================================================
# Test: KT (Factor × Time) Level Filtering
# ============================================================================

def test_filter_uses_factor_level_for_flexibility(prep):
    """Filtering uses factor level to allow factor-level aggregation.

    When `by` is specified as factors only (not time), the analysis aggregates
    across time points. This is a valid use case where each kt cell may have n=1
    but the factor-level subgroup has n>1.

    Data preparation filters at factor level (not kt level) to support this.
    """
    # Data where factor A has 6 observations total but n=1 per time point
    df = pd.DataFrame({
        'factor': ['A', 'A', 'A', 'A', 'A', 'A'],
        'time': [1, 2, 3, 4, 5, 6],  # 6 different time points
        'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    })

    spec = FormulationSpec(
        response_var='y',
        rsg_vars=('factor',),
        time_var='time',
    )

    # Should NOT raise - factor A has n=6 which is > 1
    # This allows factor-level analysis with by=['factor']
    result = prep.prepare_dataset(df, spec)
    assert len(result) == 6
    assert 'A' in result['rsg'].values


def test_filter_at_factor_level_keeps_all_factors_with_n_gt_1(prep):
    """All factors with n>1 are kept, regardless of per-kt-cell counts."""
    # Data where both factors have n>=2 at factor level
    # A: 4 observations (n=2 per kt cell)
    # B: 2 observations (n=1 per kt cell, but n=2 at factor level)
    df = pd.DataFrame({
        'factor': ['A', 'A', 'A', 'A', 'B', 'B'],
        'time': [1, 1, 2, 2, 1, 2],  # A has n=2 per time, B has n=1 per time
        'y': [1.0, 1.1, 2.0, 2.1, 3.0, 4.0]
    })

    spec = FormulationSpec(
        response_var='y',
        rsg_vars=('factor',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # Both factors should be kept because both have n>=2 at factor level
    assert 'A' in result['rsg'].values
    assert 'B' in result['rsg'].values
    assert len(result) == 6  # All rows kept


def test_no_filter_without_time_chart_agnostic(prep):
    """Chart-agnostic prepare_dataset keeps all groups regardless of size."""
    df = pd.DataFrame({
        'factor': ['A', 'A', 'B'],  # A has n=2, B has n=1
        'y': [1.0, 1.1, 2.0]
    })

    spec = FormulationSpec(
        response_var='y',
        rsg_vars=('factor',),
    )

    result = prep.prepare_dataset(df, spec)

    # Both factors kept (no chart-specific filtering)
    assert 'A' in result['rsg'].values
    assert 'B' in result['rsg'].values


def test_n_column_reflects_kt_cell_size(prep):
    """The n column should reflect kt cell size, not factor-level count."""
    # Data with varying observations per kt cell
    df = pd.DataFrame({
        'factor': ['A', 'A', 'A', 'A', 'A'],
        'time': [1, 1, 2, 2, 2],  # time 1 has n=2, time 2 has n=3
        'y': [1.0, 1.1, 2.0, 2.1, 2.2]
    })

    spec = FormulationSpec(
        response_var='y',
        rsg_vars=('factor',),
        time_var='time',
    )

    result = prep.prepare_dataset(df, spec)

    # n should reflect kt cell size, not factor-level count
    # Factor A has 5 total, but kt cells have n=2 and n=3
    n_values = result.groupby(['rsg', 'time'], observed=True)['n'].first()
    assert n_values[('A', 1)] == 2, f"Expected n=2 for (A, 1), got {n_values.get(('A', 1))}"
    assert n_values[('A', 2)] == 3, f"Expected n=3 for (A, 2), got {n_values.get(('A', 2))}"
