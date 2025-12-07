"""
Unit tests for ProcessDataFrame - the user-friendly wrapper with auto-completion.

Tests cover:
- Column accessor with auto-completion
- SDS detection via formulate()
- Simple series (SDS 0) → IMR chart
- Grouped data → Xbar/S charts
- User-friendly output and explanations
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior.process_dataframe import ColumnAccessor, ProcessDataFrame

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
# Test: ProcessDataFrame - Basic initialization
# ============================================================================

def test_process_dataframe_init():
    """ProcessDataFrame should initialize with a DataFrame."""
    df = pd.DataFrame({
        'X': [1, 2, 3],
        'Y': [4, 5, 6]
    })

    pdata = ProcessDataFrame(df)

    assert len(pdata) == 3
    assert len(pdata.data.columns) == 2
    assert isinstance(pdata.columns, ColumnAccessor)


def test_process_dataframe_copies_data():
    """ProcessDataFrame should copy the input DataFrame."""
    df = pd.DataFrame({'X': [1, 2, 3]})

    pdata = ProcessDataFrame(df)

    # Modify original
    df['X'] = [9, 9, 9]

    # ProcessDataFrame should have original values
    assert list(pdata.data['X']) == [1, 2, 3]


def test_process_dataframe_rejects_non_dataframe():
    """ProcessDataFrame should reject non-DataFrame input."""
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        ProcessDataFrame([1, 2, 3])

    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        ProcessDataFrame("not a dataframe")


# ============================================================================
# Test: ProcessDataFrame.formulate() - Simple series (SDS 0)
# ============================================================================

def test_formulate_simple_series():
    """Simple series should detect SDS 0 and recommend IMR chart."""
    # Create simple series
    np.random.seed(42)
    df = pd.DataFrame({
        'Measurement': np.random.normal(100, 5, 30),
        'Time': range(1, 31)
    })

    pdata = ProcessDataFrame(df)

    # Formulate without factors → should get SDS 0
    study = pdata.formulate(
        response=pdata.columns.Measurement,
        time=pdata.columns.Time
    )

    # Should have detected SDS 0 and recommend IMR
    assert study.sds == 0
    assert study.recommended_chart == 'Imr'
    assert 'Imr' in study.valid_charts


def test_formulate_simple_series_no_time():
    """Simple series without time variable should still work."""
    df = pd.DataFrame({
        'Value': [10, 12, 11, 13, 12, 14, 13, 15]
    })

    pdata = ProcessDataFrame(df)

    study = pdata.formulate(response=pdata.columns.Value)

    assert study.sds == 0
    assert study.recommended_chart == 'Imr'


def test_formulate_and_analyze_simple_series():
    """formulate() followed by analyze() should produce valid Analysis."""
    np.random.seed(42)
    df = pd.DataFrame({
        'Measurement': np.random.normal(100, 5, 30),
        'Time': range(1, 31)
    })

    pdata = ProcessDataFrame(df)

    study = pdata.formulate(
        response=pdata.columns.Measurement,
        time=pdata.columns.Time
    )

    # Analyze using recommended chart
    result = study.analyze()

    assert result is not None
    assert 'all' in result.charts


# ============================================================================
# Test: ProcessDataFrame.formulate() - Grouped data
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

    pdata = ProcessDataFrame(df)

    study = pdata.formulate(
        response=pdata.columns.Height,
        time=pdata.columns.Time,
        factors=[pdata.columns.Operator, pdata.columns.Machine]
    )

    # Should detect grouped structure and recommend Xbar
    assert study.sds == 1  # Full replication
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

    pdata = ProcessDataFrame(df)

    study = pdata.formulate(
        response=pdata.columns.Value,
        time=pdata.columns.Sequence,
        factors=[pdata.columns.Batch]
    )

    assert study.sds == 1
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

    pdata = ProcessDataFrame(df)

    study = pdata.formulate(
        response=pdata.columns.Height,
        time=pdata.columns.ProductionTime,
        factors=[pdata.columns.Operator, pdata.columns.Shift]
    )

    # Should be able to analyze
    result = study.analyze()
    assert result is not None


# ============================================================================
# Test: formulate() parameter validation
# ============================================================================

def test_formulate_requires_response():
    """formulate() should require response parameter."""
    df = pd.DataFrame({'X': [1, 2, 3]})
    pdata = ProcessDataFrame(df)

    with pytest.raises(TypeError):
        pdata.formulate()  # Missing required 'response' parameter


def test_formulate_validates_response_column():
    """formulate() should validate that response column exists."""
    df = pd.DataFrame({'X': [1, 2, 3]})
    pdata = ProcessDataFrame(df)

    with pytest.raises(ValueError, match="not found"):
        pdata.formulate(response='NonExistent')


# ============================================================================
# Test: Study object properties
# ============================================================================

def test_study_has_sds():
    """Study should expose detected SDS."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    pdata = ProcessDataFrame(df)

    study = pdata.formulate(response='Value')

    assert hasattr(study, 'sds')
    assert isinstance(study.sds, int)


def test_study_has_valid_charts():
    """Study should expose valid chart types."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    pdata = ProcessDataFrame(df)

    study = pdata.formulate(response='Value')

    assert hasattr(study, 'valid_charts')
    assert isinstance(study.valid_charts, list)


def test_study_has_recommended_chart():
    """Study should expose recommended chart type."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    pdata = ProcessDataFrame(df)

    study = pdata.formulate(response='Value')

    assert hasattr(study, 'recommended_chart')
    assert study.recommended_chart in study.valid_charts


def test_study_has_charts_accessor():
    """Study should have charts accessor for IDE auto-completion."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    pdata = ProcessDataFrame(df)

    study = pdata.formulate(response='Value')

    assert hasattr(study, 'charts')
    # Should be able to access valid chart types as attributes
    if 'Imr' in study.valid_charts:
        assert study.charts.Imr == 'Imr'


# ============================================================================
# Test: String representations
# ============================================================================

def test_process_dataframe_repr():
    """ProcessDataFrame should have informative repr."""
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    pdata = ProcessDataFrame(df)
    repr_str = repr(pdata)

    assert 'ProcessDataFrame' in repr_str
    assert 'rows=3' in repr_str
    assert 'columns=2' in repr_str


def test_process_dataframe_len():
    """ProcessDataFrame should support len()."""
    df = pd.DataFrame({'X': range(100)})
    pdata = ProcessDataFrame(df)

    assert len(pdata) == 100


def test_study_repr():
    """Study should have informative repr/str."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    pdata = ProcessDataFrame(df)

    study = pdata.formulate(response='Value')
    study_str = str(study)

    # Should contain useful information
    assert 'SDS' in study_str or 'sds' in study_str.lower()


# ============================================================================
# Test: Integration - End-to-end workflow
# ============================================================================

def test_full_workflow_simple_series():
    """Test complete workflow from DataFrame to Analysis (simple series)."""
    # Create data
    np.random.seed(42)
    df = pd.DataFrame({
        'Measurement': np.random.normal(100, 2, 20),
        'Time': range(1, 21)
    })

    # Wrap in ProcessDataFrame
    data = ProcessDataFrame(df)

    # Formulate study
    study = data.formulate(
        response=data.columns.Measurement,
        time=data.columns.Time
    )

    # Check study properties
    assert study.sds == 0
    assert study.recommended_chart == 'Imr'

    # Analyze
    result = study.analyze()

    assert result is not None
    assert 'all' in result.charts


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
    data = ProcessDataFrame(df)

    study = data.formulate(
        response=data.columns.Height,
        time=data.columns.ProductionTime,
        factors=[data.columns.Operator, data.columns.Shift]
    )

    # Verify study
    assert study.sds == 1  # Full replication
    assert study.recommended_chart == 'Xbar'

    # Analyze
    result = study.analyze()
    assert result is not None


# ============================================================================
# Test: Edge cases
# ============================================================================

def test_formulate_with_precision():
    """Should pass precision parameter through."""
    df = pd.DataFrame({
        'Value': [100.123456, 102.654321, 101.111111, 103.999999],
        'Time': [1, 2, 3, 4]
    })

    pdata = ProcessDataFrame(df)

    study = pdata.formulate(
        response=pdata.columns.Value,
        time=pdata.columns.Time,
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
    pdata = ProcessDataFrame(df)

    study = pdata.formulate(
        response=pdata.columns.Value,
        time=pdata.columns.Time,
        factors=[pdata.columns.Batch]
    )

    # Should be able to analyze with different valid chart types
    result_xbar = study.analyze(chart='Xbar')
    assert result_xbar is not None

    result_imr = study.analyze(chart='Imr')
    assert result_imr is not None
