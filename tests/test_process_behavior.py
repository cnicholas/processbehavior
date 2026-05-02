"""
Unit tests for ProcessBehavior - the user-friendly wrapper with auto-completion.

Tests cover:
- Column accessor with auto-completion
- SDS detection via formulate()
- Simple series (SDS 4) → XmR chart
- Grouped data → Xbar/S charts
- User-friendly output and explanations

Note: Uses shared fixtures from conftest.py for common test data patterns.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from processbehavior.exceptions import ColumnNotFoundError, FactorNotFoundError
from processbehavior.process_behavior import ColumnAccessor, ProcessBehavior

# ============================================================================
# Test: ColumnAccessor - Auto-completion support
# ============================================================================

def test_column_accessor_basic():
    """ColumnAccessor should provide attribute access to column names."""
    df = pd.DataFrame({
        'Height': [1, 2, 3],
        'Width': [4, 5, 6],
        'Time': [1, 2, 3]
    })

    accessor = ColumnAccessor(df)

    # Should be able to access columns as attributes
    assert accessor.Height == 'Height'
    assert accessor.Width == 'Width'
    assert accessor.Time == 'Time'


def test_column_accessor_with_spaces():
    """ColumnAccessor should handle column names with spaces."""
    df = pd.DataFrame({
        'Production Time': [1, 2, 3],
        'Fill Weight': [4, 5, 6]
    })

    accessor = ColumnAccessor(df)

    # Spaces should be converted to underscores
    assert accessor.Production_Time == 'Production Time'
    assert accessor.Fill_Weight == 'Fill Weight'


def test_column_accessor_with_special_chars():
    """ColumnAccessor should sanitize special characters."""
    df = pd.DataFrame({
        'Factor-1': [1, 2, 3],
        'Factor 2': [4, 5, 6],
        'Value(%)': [7, 8, 9]
    })

    accessor = ColumnAccessor(df)

    # Special chars should be replaced
    assert accessor.Factor_1 == 'Factor-1'
    assert accessor.Factor_2 == 'Factor 2'
    assert accessor.Value___ == 'Value(%)'


def test_column_accessor_dir():
    """ColumnAccessor should support tab-completion via __dir__."""
    df = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4],
        'C': [5, 6]
    })

    accessor = ColumnAccessor(df)
    attrs = dir(accessor)

    # Should include all column names
    assert 'A' in attrs
    assert 'B' in attrs
    assert 'C' in attrs


# ============================================================================
# Test: ProcessBehavior - Basic initialization
# ============================================================================

def test_process_dataframe_init():
    """ProcessBehavior should initialize with a DataFrame."""
    df = pd.DataFrame({
        'X': [1, 2, 3],
        'Y': [4, 5, 6]
    })

    pdata = ProcessBehavior(df)

    assert len(pdata) == 3
    assert len(pdata.data.columns) == 2
    assert isinstance(pdata.cols, ColumnAccessor)


def test_process_dataframe_copies_data():
    """ProcessBehavior should copy the input DataFrame."""
    df = pd.DataFrame({'X': [1, 2, 3]})

    pdata = ProcessBehavior(df)

    # Modify original
    df['X'] = [9, 9, 9]

    # ProcessBehavior should have original values
    assert list(pdata.data['X']) == [1, 2, 3]


def test_input_dataframe_never_modified():
    """Input DataFrame should remain unchanged through full analysis pipeline."""
    df = pd.DataFrame({
        'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'Time': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        'Factor': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    original = df.copy()

    # Run full pipeline: formulate → execute → plot
    # Use factor-level aggregation since cells have n=1
    study = ProcessBehavior(df).formulate(response='Value', time='Time', factors=['Factor'])
    result = study.execute(chart='Xbar', by=['Factor'])
    _ = result.plot()

    # Original DataFrame should be identical
    pd.testing.assert_frame_equal(df, original)


def test_process_dataframe_rejects_non_dataframe():
    """ProcessBehavior should reject non-DataFrame input with ValidationError."""
    from processbehavior.exceptions import ValidationError

    with pytest.raises(ValidationError, match="Expected pandas DataFrame"):
        ProcessBehavior([1, 2, 3])

    with pytest.raises(ValidationError, match="Expected pandas DataFrame"):
        ProcessBehavior("not a dataframe")


# ============================================================================
# Test: ProcessBehavior.formulate() - Grouped data
# ============================================================================

def test_formulate_with_grouping():
    """Data with factors should trigger Xbar/S charts."""
    # Create grouped data with complete replication (SDS 1)
    # 2 operators × 2 machines × 3 time points × 3 replicates = 36 obs
    np.random.seed(42)

    operators = []
    machines = []
    times = []
    heights = []

    for op in ['A', 'B']:
        for machine in ['M1', 'M2']:
            for time in [1, 2, 3]:
                for _rep in range(3):  # 3 replicates per cell
                    operators.append(op)
                    machines.append(machine)
                    times.append(time)
                    heights.append(np.random.normal(100, 5))

    df = pd.DataFrame({
        'Height': heights,
        'Operator': operators,
        'Machine': machines,
        'Time': times
    })

    pdata = ProcessBehavior(df)

    study = pdata.formulate(
        response=pdata.cols.Height,
        time=pdata.cols.Time,
        factors=[pdata.cols.Operator, pdata.cols.Machine]
    )

    # Should detect grouped structure and recommend Xbar
    assert study.observed_design_state.sds == 1  # Full replication
    assert study.recommended_chart == 'Xbar'
    assert 'Xbar' in study.valid_charts
    assert 'S' in study.valid_charts


def test_formulate_with_single_grouping():
    """Single grouping factor should work."""
    # Create data with complete replication (SDS 1)
    # 2 batches × 5 time points × 3 replicates = 30 obs
    np.random.seed(42)

    batches = []
    sequences = []
    values = []

    for batch in ['A', 'B']:
        for seq in range(1, 6):
            for _rep in range(3):  # 3 replicates per cell
                batches.append(batch)
                sequences.append(seq)
                values.append(np.random.randint(1, 7))

    df = pd.DataFrame({
        'Value': values,
        'Batch': batches,
        'Sequence': sequences
    })

    pdata = ProcessBehavior(df)

    study = pdata.formulate(
        response=pdata.cols.Value,
        time=pdata.cols.Sequence,
        factors=[pdata.cols.Batch]
    )

    assert study.observed_design_state.sds == 1
    assert study.recommended_chart == 'Xbar'


def test_formulate_and_analyze_with_grouping():
    """formulate() with grouping followed by analyze() should work."""
    np.random.seed(42)

    # Create grouped data
    df = pd.DataFrame({
        'Height': np.random.normal(50, 3, 60),
        'Operator': ['Alice', 'Bob'] * 30,
        'Shift': ['Day', 'Night'] * 30,
        'ProductionTime': list(range(1, 16)) * 4
    })

    pdata = ProcessBehavior(df)

    study = pdata.formulate(
        response=pdata.cols.Height,
        time=pdata.cols.ProductionTime,
        factors=[pdata.cols.Operator, pdata.cols.Shift]
    )

    # Should be able to analyze
    result = study.execute()
    assert result is not None


# ============================================================================
# Test: formulate() parameter validation
# ============================================================================

def test_formulate_requires_response():
    """formulate() should require response parameter."""
    df = pd.DataFrame({'X': [1, 2, 3]})
    pdata = ProcessBehavior(df)

    with pytest.raises(TypeError):
        pdata.formulate()  # Missing required 'response' parameter


def test_formulate_validates_response_column():
    """formulate() should validate that response column exists."""
    df = pd.DataFrame({'X': [1, 2, 3], 'Factor': ['A', 'A', 'B']})
    pdata = ProcessBehavior(df)

    with pytest.raises(ColumnNotFoundError, match="not found"):
        pdata.formulate(response='NonExistent', factors=['Factor'])


# ============================================================================
# Test: Study object properties
# ============================================================================

def test_study_has_sds(simple_values):
    """Study should expose detected SDS via design state properties."""
    pdata = ProcessBehavior(simple_values)

    study = pdata.formulate(response='Value', factors=['Factor'], time='Time')

    assert hasattr(study, 'observed_design_state')
    assert hasattr(study, 'analytical_design_state')
    assert isinstance(study.observed_design_state.sds, int)


def test_study_has_valid_charts(simple_values):
    """Study should expose valid chart types."""
    pdata = ProcessBehavior(simple_values)

    study = pdata.formulate(response='Value', factors=['Factor'], time='Time')

    assert hasattr(study, 'valid_charts')
    assert isinstance(study.valid_charts, list)


def test_study_has_recommended_chart(simple_values):
    """Study should expose recommended chart type."""
    pdata = ProcessBehavior(simple_values)

    study = pdata.formulate(response='Value', factors=['Factor'], time='Time')

    assert hasattr(study, 'recommended_chart')
    assert study.recommended_chart in study.valid_charts


def test_study_has_charts_accessor(simple_values):
    """Study should have charts accessor for IDE auto-completion."""
    pdata = ProcessBehavior(simple_values)

    study = pdata.formulate(response='Value', factors=['Factor'], time='Time')

    assert hasattr(study, 'charts')
    # Should be able to access valid chart types as attributes
    if 'X' in study.valid_charts:
        assert study.charts.X == 'X'


# ============================================================================
# Test: String representations
# ============================================================================

def test_process_dataframe_repr():
    """ProcessBehavior should have informative repr."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    pdata = ProcessBehavior(df)
    repr_str = repr(pdata)

    assert 'ProcessBehavior' in repr_str
    assert '3 rows' in repr_str
    assert '2 columns' in repr_str


def test_process_dataframe_len():
    """ProcessBehavior should support len()."""
    df = pd.DataFrame({'X': range(100)})
    pdata = ProcessBehavior(df)

    assert len(pdata) == 100


def test_study_repr(simple_values):
    """Study should have informative repr/str."""
    pdata = ProcessBehavior(simple_values)

    study = pdata.formulate(response='Value', factors=['Factor'], time='Time')
    study_str = str(study)

    # Should contain useful information
    assert 'SDS' in study_str or 'sds' in study_str.lower()


# ============================================================================
# Test: Integration - End-to-end workflow
# ============================================================================

def test_full_workflow_grouped_data():
    """Test complete workflow with grouped data."""
    # Create grouped data with replication
    np.random.seed(42)

    data_rows = []
    for op in ['Alice', 'Bob']:
        for shift in ['Day', 'Night']:
            for time in range(1, 11):
                for _rep in range(3):
                    data_rows.append({
                        'Height': np.random.normal(50, 3),
                        'Operator': op,
                        'Shift': shift,
                        'ProductionTime': time
                    })

    df = pd.DataFrame(data_rows)

    # Wrap and formulate
    data = ProcessBehavior(df)

    study = data.formulate(
        response=data.cols.Height,
        time=data.cols.ProductionTime,
        factors=[data.cols.Operator, data.cols.Shift]
    )

    # Verify study
    assert study.observed_design_state.sds == 1  # Full replication
    assert study.recommended_chart == 'Xbar'

    # Analyze
    result = study.execute()
    assert result is not None


# ============================================================================
# Test: Edge cases
# ============================================================================

def test_formulate_with_precision():
    """Should pass precision parameter through."""
    df = pd.DataFrame({
        'Value': [100.123456, 102.654321, 101.111111, 103.999999, 100.5, 102.2],
        'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
        'Time': [1, 2, 3, 1, 2, 3]
    })

    pdata = ProcessBehavior(df)

    study = pdata.formulate(
        response=pdata.cols.Value,
        factors=[pdata.cols.Factor],
        time=pdata.cols.Time,
        precision=5
    )

    # Verify study was created with the precision
    assert study is not None


def test_column_accessor_with_numeric_start():
    """Column names starting with numbers should be sanitized."""
    df = pd.DataFrame({
        '1st_value': [1, 2, 3],
        '2nd_value': [4, 5, 6]
    })

    accessor = ColumnAccessor(df)

    # Should prefix with 'col_'
    assert accessor.col_1st_value == '1st_value'
    assert accessor.col_2nd_value == '2nd_value'


def test_formulate_with_chart_selection():
    """Should be able to analyze with specific chart type."""
    np.random.seed(42)

    # Create grouped data
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 11):
            for _rep in range(3):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Batch': batch,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    pdata = ProcessBehavior(df)

    study = pdata.formulate(
        response=pdata.cols.Value,
        time=pdata.cols.Time,
        factors=[pdata.cols.Batch]
    )

    # Should be able to analyze with different valid chart types
    result_xbar = study.execute(chart='Xbar')
    assert result_xbar is not None

    # XmR with factors requires explicit 'by' parameter
    result_xmr = study.execute(chart='X', by=['Batch'])
    assert result_xmr is not None


# ============================================================================
# Test: Study properties - SDS information
# ============================================================================

def test_study_sds_name():
    """Study should expose human-readable SDS name."""
    np.random.seed(42)

    df = pd.DataFrame({
        'Value': np.random.normal(100, 5, 30),
        'Factor': np.repeat(['A', 'B', 'C'], 10),
        'Time': np.tile(range(1, 11), 3)
    })
    study = ProcessBehavior(df).formulate(response='Value', factors=['Factor'], time='Time')

    assert hasattr(study, 'ads_reason')
    assert isinstance(study.ads_reason, str)
    assert len(study.ads_reason) > 0


def test_study_ads_description():
    """Study should expose ADS description explaining the data structure."""
    np.random.seed(42)
    df = pd.DataFrame({
        'Value': np.random.normal(100, 5, 30),
        'Factor': np.repeat(['A', 'B', 'C'], 10),
        'Time': np.tile(range(1, 11), 3)
    })
    study = ProcessBehavior(df).formulate(response='Value', factors=['Factor'], time='Time')

    assert hasattr(study, 'ads_description')
    assert isinstance(study.ads_description, str)
    assert len(study.ads_description) > 0


def test_study_response_property():
    """Study should expose the response variable name."""
    df = pd.DataFrame({
        'Measurement': [1, 2, 3, 4, 5, 6],
        'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
        'Time': [1, 2, 3, 1, 2, 3]
    })
    study = ProcessBehavior(df).formulate(response='Measurement', factors=['Factor'], time='Time')

    assert study.response == 'Measurement'


def test_study_factors_property():
    """Study should expose the factors list."""
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({'Value': np.random.normal(50, 3), 'Batch': batch, 'Time': time})

    df = pd.DataFrame(data_rows)
    study = ProcessBehavior(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    assert study.factors == ['Batch']


def test_study_time_property():
    """Study should expose the time variable name."""
    df = pd.DataFrame({
        'Value': [1, 2, 3, 4, 5, 6],
        'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
        'Sequence': [1, 2, 3, 1, 2, 3]
    })
    study = ProcessBehavior(df).formulate(response='Value', factors=['Factor'], time='Sequence')

    assert study.time == 'Sequence'


def test_study_precision_property(simple_values):
    """Study should expose precision setting."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time', precision=5)

    assert study.precision == 5


def test_study_precision_default(simple_values):
    """Study precision should default to 3."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    assert study.precision == 3


# ============================================================================
# Test: Study.dataset - Pre-calculated analysis data
# ============================================================================

def test_study_dataset_exists(simple_values):
    """Study should expose the analysis dataset."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    assert hasattr(study, 'dataset')
    assert study.dataset is not None


def test_study_dataset_is_dataframe(simple_values):
    """Study.dataset should be a pandas DataFrame."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    assert isinstance(study.dataset, pd.DataFrame)


def test_study_dataset_returns_copy(simple_values):
    """Study.dataset should return a copy (immutability)."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    dataset1 = study.dataset
    dataset2 = study.dataset

    # Modify dataset1
    dataset1['test_col'] = 999

    # dataset2 should not have the modification
    assert 'test_col' not in dataset2.columns


def test_study_dataset_has_ybar_for_grouped_data(grouped_single_factor):
    """Study.dataset should contain Ybar for grouped data."""
    study = ProcessBehavior(grouped_single_factor).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    dataset = study.dataset
    assert 'Ybar' in dataset.columns


def test_study_dataset_has_residuals_for_sds1(grouped_single_factor):
    """Study.dataset should contain VAS residuals (R1-R5) for SDS 1."""
    study = ProcessBehavior(grouped_single_factor).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    dataset = study.dataset
    # SDS 1 should have all residuals
    for r in ['R1', 'R2', 'R3', 'R4', 'R5']:
        assert r in dataset.columns, f"Missing residual {r}"


def test_study_dataset_has_rsg_column(grouped_single_factor):
    """Study.dataset should contain rsg (rational subgroup) column."""
    study = ProcessBehavior(grouped_single_factor).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    dataset = study.dataset
    assert 'rsg' in dataset.columns


# ============================================================================
# Test: Study.residual_charts
# ============================================================================

def test_study_residual_charts_property(grouped_single_factor):
    """Study should expose available residual chart types."""
    study = ProcessBehavior(grouped_single_factor).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    assert hasattr(study, 'residual_charts')
    assert isinstance(study.residual_charts, list)


def test_study_residual_charts_sds1_has_all(grouped_single_factor):
    """SDS 1 should have all residual charts available."""
    study = ProcessBehavior(grouped_single_factor).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    residual_charts = study.residual_charts
    assert ('S', 'R2') in residual_charts or ('X', 'R2') in residual_charts
    assert any(v == 'R3' for _, v in residual_charts)
    assert any(v == 'R4' for _, v in residual_charts)
    assert any(v == 'R5' for _, v in residual_charts)


# ============================================================================
# Test: Study.why_not() - Teaching method
# ============================================================================

def test_study_why_not_valid_chart(simple_values):
    """why_not() should confirm valid charts."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    # X is valid for SDS 4
    result = study.why_not('X')
    assert 'IS available' in result or 'available' in result.lower()


def test_study_why_not_invalid_chart(simple_values):
    """why_not() should explain why a chart is invalid."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    # S chart requires grouping variables (which SDS 4 implicit doesn't have)
    result = study.why_not('S')
    assert isinstance(result, str)
    assert len(result) > 0


def test_study_why_not_unknown_chart(simple_values):
    """why_not() should handle unknown chart types."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    result = study.why_not('NonExistentChart')
    assert 'not a recognized' in result.lower() or 'valid types' in result.lower()


# ============================================================================
# Test: Study.charts accessor - IDE auto-completion
# ============================================================================

def test_study_charts_accessor_has_valid_charts(simple_values):
    """Study.charts should have attributes for each valid chart."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    # For SDS 4, X should be available
    assert hasattr(study.charts, 'X')
    assert study.charts.X == 'X'


def test_study_charts_accessor_dir(simple_values):
    """Study.charts should support tab-completion via __dir__."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    chart_attrs = dir(study.charts)
    assert 'X' in chart_attrs


def test_study_charts_accessor_xbar_for_grouped(grouped_single_factor):
    """Study.charts should have Xbar for grouped data."""
    study = ProcessBehavior(grouped_single_factor).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    assert hasattr(study.charts, 'Xbar')
    assert hasattr(study.charts, 'S')


def test_study_charts_accessor_repr(simple_values):
    """Study.charts should have informative repr."""
    study = ProcessBehavior(simple_values).formulate(response='Value', factors=['Factor'], time='Time')

    repr_str = repr(study.charts)
    assert 'StudyChartAccessor' in repr_str


# ============================================================================
# Test: Study.execute() - Error handling
# ============================================================================

# ============================================================================
# Test: Study.execute() - Residual charts
# ============================================================================

def test_study_analyze_residual_chart(grouped_single_factor):
    """analyze() should work with residual chart types using value parameter."""
    study = ProcessBehavior(grouped_single_factor).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    # Get first available residual chart and execute it
    if study.residual_charts:
        chart_type, residual_id = study.residual_charts[0]
        result = study.execute(chart=chart_type, value=residual_id)
        assert result is not None


# ============================================================================
# Test: formulate() validation
# ============================================================================

def test_formulate_invalid_factor_column():
    """formulate() should raise for non-existent factor column."""
    df = pd.DataFrame({'Value': [1, 2, 3]})
    pdata = ProcessBehavior(df)

    with pytest.raises(FactorNotFoundError, match="not found"):
        pdata.formulate(response='Value', factors=['NonExistent'])


# ============================================================================
# Test: R4/R5 Xbar/S Charts (GitHub Issues #51 & #52)
# ============================================================================

def test_r5_xbar_chart_calculation(grouped_for_residuals):
    """R5 Xbar should use factor-based subgrouping."""
    study = ProcessBehavior(grouped_for_residuals).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    assert 'R5' in study.residuals

    # Analyze R5 on Xbar chart - aggregate by factor to get factor effects
    result = study.execute(chart='Xbar', value='R5', by=['Factor'])
    assert result is not None
    assert 'Xbar' in result.charts

    chart_data = result.charts['Xbar']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R5 Xbar should have one point per factor (3 factors)
    data_df = chart_data['data']
    assert len(data_df) == 3


def test_r5_s_chart_calculation(grouped_for_residuals):
    """R5 S chart should use factor-based subgrouping."""
    study = ProcessBehavior(grouped_for_residuals).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    assert 'R5' in study.residuals

    # Analyze R5 on S chart - aggregate by factor to get factor effects
    result = study.execute(chart='S', value='R5', by=['Factor'])
    assert result is not None
    assert 'S' in result.charts

    chart_data = result.charts['S']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R5 S should have one point per factor (3 factors)
    data_df = chart_data['data']
    assert len(data_df) == 3


def test_r4_r5_xbar_s_control_limits_structure(grouped_for_residuals):
    """R4 and R5 Xbar/S charts should have proper control limit structure."""
    study = ProcessBehavior(grouped_for_residuals).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    # Test R4 and R5 residuals on Xbar and S charts
    for residual in ['R4', 'R5']:
        for chart in ['Xbar', 'S']:
            result = study.execute(chart=chart, value=residual)
            chart_data = result.charts[chart]

            # Should have statistics with control limits
            stats = chart_data['statistics']
            assert 'center' in stats, f"{chart} missing center"
            assert 'upl' in stats, f"{chart} missing upl"
            assert 'lpl' in stats, f"{chart} missing lpl"

            # Control limits should be numeric
            assert isinstance(stats['center'], (int, float, np.number))
            assert isinstance(stats['upl'], (int, float, np.number))
            assert isinstance(stats['lpl'], (int, float, np.number))

            # UPL > center > LPL (for properly behaved data)
            assert stats['upl'] >= stats['center'], f"{chart}: UPL should be >= center"
            assert stats['center'] >= stats['lpl'], f"{chart}: center should be >= LPL"


# ============================================================================
# Test: R3 Xbar/S Charts (GitHub Issue #50)
# ============================================================================

def test_r3_xbar_chart_calculation(grouped_for_residuals):
    """R3 Xbar should use factor-based subgrouping (same as R2)."""
    study = ProcessBehavior(grouped_for_residuals).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    # Should be SDS 1 (all cells have n≥2)
    assert study.observed_design_state.sds == 1
    assert 'R3' in study.residuals

    # Analyze R3 on Xbar chart - aggregate by factor
    result = study.execute(chart='Xbar', value='R3', by=['Factor'])
    assert result is not None
    assert 'Xbar' in result.charts

    chart_data = result.charts['Xbar']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R3 Xbar uses factor-based subgrouping (same as R5), so 3 factors
    data_df = chart_data['data']
    assert len(data_df) == 3


def test_r3_s_chart_calculation(grouped_for_residuals):
    """R3 S chart should use factor-based subgrouping (same as R2)."""
    study = ProcessBehavior(grouped_for_residuals).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    assert 'R3' in study.residuals

    # Analyze R3 on S chart - aggregate by factor
    result = study.execute(chart='S', value='R3', by=['Factor'])
    assert result is not None
    assert 'S' in result.charts

    chart_data = result.charts['S']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R3 S uses factor-based subgrouping, so 3 factors
    data_df = chart_data['data']
    assert len(data_df) == 3


def test_r3_xbar_s_control_limits_structure(grouped_for_residuals):
    """R3 Xbar/S charts should have proper control limit structure."""
    study = ProcessBehavior(grouped_for_residuals).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    for chart in ['Xbar', 'S']:
        result = study.execute(chart=chart, value='R3')
        chart_data = result.charts[chart]

        # Should have statistics with control limits
        stats = chart_data['statistics']
        assert 'center' in stats, f"{chart} missing center"
        assert 'upl' in stats, f"{chart} missing upl"
        assert 'lpl' in stats, f"{chart} missing lpl"

        # Control limits should be numeric
        assert isinstance(stats['center'], (int, float, np.number))
        assert isinstance(stats['upl'], (int, float, np.number))
        assert isinstance(stats['lpl'], (int, float, np.number))


# ============================================================================
# Test: ColumnAccessor - Collision detection and dict-style access
# ============================================================================

def test_column_accessor_collision_warning(caplog):
    """Columns that sanitize to same name should warn."""
    df = pd.DataFrame({
        'A-B': [1, 2, 3],
        'A B': [4, 5, 6],  # Collides with A-B → A_B
        'Normal': [7, 8, 9]
    })

    with caplog.at_level(logging.WARNING):
        pb = ProcessBehavior(df)

    # Warning should be logged
    assert 'collision' in caplog.text.lower()

    # First column wins for attribute access
    # (alphabetically: 'A B' before 'A-B' since space 0x20 < hyphen 0x2D)
    assert pb.cols.A_B == 'A B'

    # Both accessible via dict-style
    assert pb.cols['A-B'] == 'A-B'
    assert pb.cols['A B'] == 'A B'
    assert pb.cols['Normal'] == 'Normal'


def test_column_accessor_getitem():
    """Dict-style access works for all columns."""
    df = pd.DataFrame({
        'Column With Spaces': [1, 2],
        '123_starts_with_number': [3, 4]
    })

    pb = ProcessBehavior(df)

    # Dict-style access works
    assert pb.cols['Column With Spaces'] == 'Column With Spaces'
    assert pb.cols['123_starts_with_number'] == '123_starts_with_number'

    # ColumnNotFoundError for missing columns
    with pytest.raises(ColumnNotFoundError):
        pb.cols['nonexistent']


def test_column_accessor_getitem_keyerror_message():
    """Dict-style access should show available columns in error message."""
    df = pd.DataFrame({
        'Alpha': [1, 2],
        'Beta': [3, 4]
    })

    pb = ProcessBehavior(df)

    with pytest.raises(ColumnNotFoundError) as exc_info:
        pb.cols['missing']

    # Error message should mention available columns
    assert 'Alpha' in str(exc_info.value)
    assert 'Beta' in str(exc_info.value)


# ============================================================================
# Test: Factory Methods (read_csv, read_excel, read_parquet, read_clipboard)
# ============================================================================

def test_read_csv(tmp_path):
    """read_csv should load data and create ProcessBehavior."""
    # Create test CSV
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        'time': [1, 2, 3, 4, 5],
        'value': [10.1, 10.2, 10.0, 10.3, 10.1]
    })
    df.to_csv(csv_path, index=False)

    # Load via factory method
    pb = ProcessBehavior.read_csv(csv_path)

    assert len(pb) == 5
    assert 'time' in pb.data.columns
    assert 'value' in pb.data.columns
    assert pb.cols.time == 'time'
    assert pb.cols.value == 'value'


def test_read_csv_with_na_values(tmp_path):
    """read_csv should handle na_values parameter."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        'time': [1, 2, 3],
        'value': ['10.1', 'MISSING', '10.3']
    })
    df.to_csv(csv_path, index=False)

    # Load with custom NA value
    pb = ProcessBehavior.read_csv(csv_path, na_values=['MISSING'])

    assert len(pb) == 3
    assert pd.isna(pb.data['value'].iloc[1])


def test_read_csv_with_kwargs(tmp_path):
    """read_csv should pass kwargs to pandas."""
    csv_path = tmp_path / "test_data.csv"
    with open(csv_path, 'w') as f:
        f.write("a;b;c\n1;2;3\n4;5;6\n")

    # Use sep kwarg
    pb = ProcessBehavior.read_csv(csv_path, sep=';')

    assert len(pb) == 2
    assert list(pb.data.columns) == ['a', 'b', 'c']


def test_read_excel(tmp_path):
    """read_excel should load data and create ProcessBehavior."""
    excel_path = tmp_path / "test_data.xlsx"
    df = pd.DataFrame({
        'time': [1, 2, 3],
        'value': [10.1, 10.2, 10.3]
    })
    df.to_excel(excel_path, index=False)

    pb = ProcessBehavior.read_excel(excel_path)

    assert len(pb) == 3
    assert pb.cols.time == 'time'
    assert pb.cols.value == 'value'


def test_read_parquet(tmp_path):
    """read_parquet should load data and create ProcessBehavior."""
    pytest.importorskip("pyarrow")

    parquet_path = tmp_path / "test_data.parquet"
    df = pd.DataFrame({
        'time': [1, 2, 3],
        'value': [10.1, 10.2, 10.3]
    })
    df.to_parquet(parquet_path, index=False)

    pb = ProcessBehavior.read_parquet(parquet_path)

    assert len(pb) == 3
    assert pb.cols.time == 'time'
    assert pb.cols.value == 'value'
