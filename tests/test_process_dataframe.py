"""
Unit tests for ProcessDataFrame - the user-friendly wrapper with auto-completion.

Tests cover:
- Column accessor with auto-completion
- SDS detection and analysis type selection
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
# Test: ProcessDataFrame.analyze() - Simple series (SDS 0)
# ============================================================================

def test_analyze_simple_series():
    """Simple series should trigger IMR chart (SDS 0)."""
    # Create simple series
    np.random.seed(42)
    df = pd.DataFrame({
        'Measurement': np.random.normal(100, 5, 30),
        'Time': range(1, 31)
    })

    pdata = ProcessDataFrame(df)

    # Analyze without grouping → should get IMR
    analysis = pdata.analyze(
        response_var=pdata.columns.Measurement,
        time_var=pdata.columns.Time
    )

    # Should have created an Analysis object
    assert analysis is not None
    assert analysis.spec.analysis_type == 'Imr'


def test_analyze_simple_series_no_time():
    """Simple series without time variable should still work."""
    df = pd.DataFrame({
        'Value': [10, 12, 11, 13, 12, 14, 13, 15]
    })

    pdata = ProcessDataFrame(df)

    analysis = pdata.analyze(response_var=pdata.columns.Value)

    assert analysis.spec.analysis_type == 'Imr'


# ============================================================================
# Test: ProcessDataFrame.analyze() - Grouped data
# ============================================================================

def test_analyze_with_grouping():
    """Data with grouping variables should trigger Xbar/S charts."""
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

    analysis = pdata.analyze(
        response_var=pdata.columns.Height,
        time_var=pdata.columns.Time,
        grouping_vars=[pdata.columns.Operator, pdata.columns.Machine]
    )

    assert analysis.spec.analysis_type == 'Xbar'
    assert analysis.spec.has_grouping is True


def test_analyze_with_single_grouping():
    """Single grouping variable should work."""
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

    analysis = pdata.analyze(
        response_var=pdata.columns.Value,
        time_var=pdata.columns.Sequence,
        grouping_vars=[pdata.columns.Batch]
    )

    assert analysis.spec.analysis_type == 'Xbar'
    assert analysis.spec.rsg_vars == ['Batch']


# ============================================================================
# Test: ProcessDataFrame.analyze() - Parameter validation
# ============================================================================

def test_analyze_requires_response_var():
    """analyze() should require response_var or response_vars."""
    df = pd.DataFrame({'X': [1, 2, 3]})
    pdata = ProcessDataFrame(df)

    with pytest.raises(ValueError, match="Must provide either response_var or response_vars"):
        pdata.analyze(time_var='X')


def test_analyze_rejects_both_response_params():
    """analyze() should reject both response_var and response_vars."""
    df = pd.DataFrame({'X': [1, 2, 3], 'Y': [4, 5, 6]})
    pdata = ProcessDataFrame(df)

    with pytest.raises(ValueError, match="Provide either response_var OR response_vars"):
        pdata.analyze(response_var='X', response_vars=['Y'])


# ============================================================================
# Test: Analysis type selection logic
# ============================================================================

def test_determine_analysis_type_sds0():
    """SDS 0 should always select IMR."""
    df = pd.DataFrame({'X': [1, 2, 3]})
    pdata = ProcessDataFrame(df)

    analysis_type = pdata._determine_analysis_type(sds=0, grouping_vars=None)
    assert analysis_type == 'Imr'


def test_determine_analysis_type_with_grouping():
    """Any SDS with grouping variables should select Xbar."""
    df = pd.DataFrame({'X': [1, 2, 3]})
    pdata = ProcessDataFrame(df)

    for sds in [1, 2, 3, 4, 5, 6]:
        analysis_type = pdata._determine_analysis_type(
            sds=sds,
            grouping_vars=['Group']
        )
        assert analysis_type == 'Xbar'


def test_determine_analysis_type_no_grouping():
    """SDS > 0 without grouping should select IMR."""
    df = pd.DataFrame({'X': [1, 2, 3]})
    pdata = ProcessDataFrame(df)

    analysis_type = pdata._determine_analysis_type(sds=1, grouping_vars=None)
    assert analysis_type == 'Imr'


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


# ============================================================================
# Test: Integration - End-to-end workflow
# ============================================================================

def test_full_workflow_simple_series(capsys):
    """Test complete workflow from DataFrame to Analysis (simple series)."""
    # Create data
    np.random.seed(42)
    df = pd.DataFrame({
        'Measurement': np.random.normal(100, 2, 20),
        'Time': range(1, 21)
    })

    # Wrap in ProcessDataFrame
    data = ProcessDataFrame(df)

    # Auto-complete column names and run analysis
    analysis = data.analyze(
        response_var=data.columns.Measurement,
        time_var=data.columns.Time
    )

    # Should create valid analysis
    assert analysis is not None
    assert analysis.spec.analysis_type == 'Imr'
    assert analysis.spec.response_var == 'Measurement'
    assert analysis.spec.time_var == 'Time'

    # Should have printed explanation
    captured = capsys.readouterr()
    assert 'SDS 0' in captured.out or 'IMR' in captured.out


def test_full_workflow_grouped_data(capsys):
    """Test complete workflow with grouped data."""
    # Create grouped data
    np.random.seed(42)
    df = pd.DataFrame({
        'Height': np.random.normal(50, 3, 60),
        'Width': np.random.normal(25, 1, 60),
        'Operator': ['Alice', 'Bob'] * 30,
        'Shift': ['Day', 'Night'] * 30,
        'ProductionTime': range(1, 61)
    })

    # Wrap and analyze
    data = ProcessDataFrame(df)

    analysis = data.analyze(
        response_var=data.columns.Height,
        time_var=data.columns.ProductionTime,
        grouping_vars=[data.columns.Operator, data.columns.Shift]
    )

    # Verify analysis
    assert analysis.spec.analysis_type == 'Xbar'
    assert analysis.spec.has_grouping is True
    assert 'Operator' in analysis.spec.rsg_vars
    assert 'Shift' in analysis.spec.rsg_vars

    # Should have printed explanation
    captured = capsys.readouterr()
    assert 'Xbar' in captured.out or 'ANALYSIS' in captured.out


# ============================================================================
# Test: Edge cases
# ============================================================================

def test_analyze_with_zero_center():
    """Should pass zero_center parameter through."""
    df = pd.DataFrame({
        'Value': [100, 102, 101, 103],
        'Time': [1, 2, 3, 4]
    })

    pdata = ProcessDataFrame(df)

    analysis = pdata.analyze(
        response_var=pdata.columns.Value,
        time_var=pdata.columns.Time,
        zero_center=True
    )

    assert analysis.spec.zero_center is True


def test_analyze_with_custom_rsg_name():
    """Should pass custom rsg_var_name through."""
    df = pd.DataFrame({
        'Y': [1, 2, 3, 4],
        'Group': ['A', 'B', 'A', 'B']
    })

    pdata = ProcessDataFrame(df)

    analysis = pdata.analyze(
        response_var=pdata.columns.Y,
        grouping_vars=[pdata.columns.Group],
        rsg_var_name='CustomRSG'
    )

    assert analysis.spec.rsg_var_name == 'CustomRSG'


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
