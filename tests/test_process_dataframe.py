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


# ============================================================================
# Test: Study properties - SDS information
# ============================================================================

def test_study_sds_name():
    """Study should expose human-readable SDS name."""
    np.random.seed(42)

    # SDS 0 - Simple series
    df = pd.DataFrame({'Value': np.random.normal(100, 5, 30)})
    study = ProcessDataFrame(df).formulate(response='Value')

    assert hasattr(study, 'sds_name')
    assert isinstance(study.sds_name, str)
    assert len(study.sds_name) > 0


def test_study_sds_description():
    """Study should expose SDS description explaining the data structure."""
    np.random.seed(42)
    df = pd.DataFrame({'Value': np.random.normal(100, 5, 30)})
    study = ProcessDataFrame(df).formulate(response='Value')

    assert hasattr(study, 'sds_description')
    assert isinstance(study.sds_description, str)
    assert len(study.sds_description) > 0


def test_study_response_property():
    """Study should expose the response variable name."""
    df = pd.DataFrame({'Measurement': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Measurement')

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
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    assert study.factors == ['Batch']


def test_study_factors_none_when_no_grouping():
    """Study.factors should be None when no grouping specified."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    assert study.factors is None


def test_study_time_property():
    """Study should expose the time variable name."""
    df = pd.DataFrame({
        'Value': [1, 2, 3, 4, 5],
        'Sequence': [1, 2, 3, 4, 5]
    })
    study = ProcessDataFrame(df).formulate(response='Value', time='Sequence')

    assert study.time == 'Sequence'


def test_study_time_none_when_not_specified():
    """Study.time should be None when no time specified."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    assert study.time is None


def test_study_precision_property():
    """Study should expose precision setting."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value', precision=5)

    assert study.precision == 5


def test_study_precision_default():
    """Study precision should default to 3."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    assert study.precision == 3


# ============================================================================
# Test: Study.dataset - Pre-calculated analysis data
# ============================================================================

def test_study_dataset_exists():
    """Study should expose the analysis dataset."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    assert hasattr(study, 'dataset')
    assert study.dataset is not None


def test_study_dataset_is_dataframe():
    """Study.dataset should be a pandas DataFrame."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    assert isinstance(study.dataset, pd.DataFrame)


def test_study_dataset_returns_copy():
    """Study.dataset should return a copy (immutability)."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    dataset1 = study.dataset
    dataset2 = study.dataset

    # Modify dataset1
    dataset1['test_col'] = 999

    # dataset2 should not have the modification
    assert 'test_col' not in dataset2.columns


def test_study_dataset_has_ybar_for_grouped_data():
    """Study.dataset should contain Ybar for grouped data."""
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({'Value': np.random.normal(50, 3), 'Batch': batch, 'Time': time})

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    dataset = study.dataset
    assert 'Ybar' in dataset.columns


def test_study_dataset_has_residuals_for_sds1():
    """Study.dataset should contain VAS residuals (R1-R5) for SDS 1."""
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({'Value': np.random.normal(50, 3), 'Batch': batch, 'Time': time})

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    dataset = study.dataset
    # SDS 1 should have all residuals
    for r in ['R1', 'R2', 'R3', 'R4', 'R5']:
        assert r in dataset.columns, f"Missing residual {r}"


def test_study_dataset_has_rsg_column():
    """Study.dataset should contain rsg (rational subgroup) column."""
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({'Value': np.random.normal(50, 3), 'Batch': batch, 'Time': time})

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    dataset = study.dataset
    assert 'rsg' in dataset.columns


# ============================================================================
# Test: Study.residual_charts
# ============================================================================

def test_study_residual_charts_property():
    """Study should expose available residual chart types."""
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({'Value': np.random.normal(50, 3), 'Batch': batch, 'Time': time})

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    assert hasattr(study, 'residual_charts')
    assert isinstance(study.residual_charts, list)


def test_study_residual_charts_sds1_has_all():
    """SDS 1 should have all residual charts available."""
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({'Value': np.random.normal(50, 3), 'Batch': batch, 'Time': time})

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    # SDS 1 should have R2_S, R3_Imr, R4_Imr, R5_Imr
    residual_charts = study.residual_charts
    assert 'R2_S' in residual_charts or 'R2_Imr' in residual_charts
    assert any('R3' in r for r in residual_charts)
    assert any('R4' in r for r in residual_charts)
    assert any('R5' in r for r in residual_charts)


def test_study_residual_charts_empty_for_sds0():
    """SDS 0 should have no residual charts (no factors)."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    # SDS 0 has no residual charts
    assert study.residual_charts == [] or len(study.residual_charts) == 0


# ============================================================================
# Test: Study.why_not() - Teaching method
# ============================================================================

def test_study_why_not_valid_chart():
    """why_not() should confirm valid charts."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    # Imr is valid for SDS 0
    result = study.why_not('Imr')
    assert 'IS valid' in result or 'valid' in result.lower()


def test_study_why_not_invalid_chart():
    """why_not() should explain why a chart is invalid."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    # S chart requires n≥2, not valid for SDS 0
    result = study.why_not('S')
    assert isinstance(result, str)
    assert len(result) > 0


def test_study_why_not_unknown_chart():
    """why_not() should handle unknown chart types."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    result = study.why_not('NonExistentChart')
    assert 'not a recognized' in result.lower() or 'valid types' in result.lower()


# ============================================================================
# Test: Study.charts accessor - IDE auto-completion
# ============================================================================

def test_study_charts_accessor_has_valid_charts():
    """Study.charts should have attributes for each valid chart."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    # For SDS 0, Imr should be available
    assert hasattr(study.charts, 'Imr')
    assert study.charts.Imr == 'Imr'


def test_study_charts_accessor_dir():
    """Study.charts should support tab-completion via __dir__."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    chart_attrs = dir(study.charts)
    assert 'Imr' in chart_attrs


def test_study_charts_accessor_xbar_for_grouped():
    """Study.charts should have Xbar for grouped data."""
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({'Value': np.random.normal(50, 3), 'Batch': batch, 'Time': time})

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    assert hasattr(study.charts, 'Xbar')
    assert hasattr(study.charts, 'S')


def test_study_charts_accessor_repr():
    """Study.charts should have informative repr."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    repr_str = repr(study.charts)
    assert 'StudyChartAccessor' in repr_str


# ============================================================================
# Test: Study.analyze() - Error handling
# ============================================================================

def test_study_analyze_invalid_chart_raises():
    """analyze() should raise ValueError for invalid chart type."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    with pytest.raises(ValueError, match="not valid"):
        study.analyze(chart='S')  # S not valid for SDS 0


def test_study_analyze_invalid_chart_shows_valid():
    """analyze() error should show valid chart options."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    try:
        study.analyze(chart='S')
    except ValueError as e:
        assert 'Imr' in str(e)  # Should mention valid charts


def test_study_analyze_with_charts_accessor():
    """analyze() should work with charts accessor."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    study = ProcessDataFrame(df).formulate(response='Value')

    result = study.analyze(chart=study.charts.Imr)
    assert result is not None


# ============================================================================
# Test: Study.analyze() - Residual charts
# ============================================================================

def test_study_analyze_residual_chart():
    """analyze() should work with residual chart types."""
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({'Value': np.random.normal(50, 3), 'Batch': batch, 'Time': time})

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Batch'],
        time='Time'
    )

    # Get first available residual chart
    if study.residual_charts:
        residual_chart = study.residual_charts[0]
        result = study.analyze(chart=residual_chart)
        assert result is not None


def test_study_analyze_invalid_residual_chart_raises():
    """analyze() should raise for invalid residual chart."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5]})
    study = ProcessDataFrame(df).formulate(response='Value')

    # SDS 0 has no residual charts
    with pytest.raises(ValueError, match="not available"):
        study.analyze(chart='R4_Imr')


# ============================================================================
# Test: AnalysisResult integration
# ============================================================================

def test_study_analyze_returns_analysis_result():
    """analyze() should return an AnalysisResult object."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    study = ProcessDataFrame(df).formulate(response='Value')

    result = study.analyze()

    # Should have expected attributes
    assert hasattr(result, 'charts')
    assert hasattr(result, 'summary')


def test_study_analyze_result_has_charts():
    """AnalysisResult should contain chart data."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    study = ProcessDataFrame(df).formulate(response='Value')

    result = study.analyze()

    assert isinstance(result.charts, dict)
    assert len(result.charts) > 0


def test_study_analyze_result_has_plot():
    """AnalysisResult should have plot() method."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    study = ProcessDataFrame(df).formulate(response='Value')

    result = study.analyze()

    assert hasattr(result, 'plot')
    assert callable(result.plot)


def test_study_analyze_result_has_detect_signals():
    """AnalysisResult should have detect_signals() method."""
    df = pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    study = ProcessDataFrame(df).formulate(response='Value')

    result = study.analyze()

    assert hasattr(result, 'detect_signals')
    assert callable(result.detect_signals)


# ============================================================================
# Test: formulate() validation
# ============================================================================

def test_formulate_invalid_factor_column():
    """formulate() should raise for non-existent factor column."""
    df = pd.DataFrame({'Value': [1, 2, 3]})
    pdata = ProcessDataFrame(df)

    with pytest.raises(ValueError, match="not found"):
        pdata.formulate(response='Value', factors=['NonExistent'])


def test_formulate_invalid_time_column():
    """formulate() should raise for non-existent time column."""
    df = pd.DataFrame({'Value': [1, 2, 3]})
    pdata = ProcessDataFrame(df)

    with pytest.raises(ValueError, match="not found"):
        pdata.formulate(response='Value', time='NonExistent')


# ============================================================================
# Test: Edge cases
# ============================================================================

def test_formulate_single_observation():
    """formulate() should handle single observation gracefully."""
    df = pd.DataFrame({'Value': [100.0]})
    pdata = ProcessDataFrame(df)

    # Should not crash
    study = pdata.formulate(response='Value')
    assert study.sds == 0


def test_formulate_with_nan_values():
    """formulate() should handle NaN values in response."""
    df = pd.DataFrame({
        'Value': [1.0, 2.0, np.nan, 4.0, 5.0],
        'Time': [1, 2, 3, 4, 5]
    })
    pdata = ProcessDataFrame(df)

    # Should handle NaN gracefully
    study = pdata.formulate(response='Value', time='Time')
    assert study is not None


# ============================================================================
# Test: R4/R5 Xbar/S Charts (GitHub Issues #51 & #52)
# ============================================================================

def test_r4_xbar_chart_calculation():
    """R4_Xbar should use time-based subgrouping."""
    np.random.seed(42)
    data_rows = []
    # Create SDS 1 data: 3 factors × 5 time points × 2 replicates
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    # Should be SDS 1 (all cells have n≥2)
    assert study.sds == 1
    assert 'R4_Xbar' in study.residual_charts

    # Analyze R4_Xbar
    result = study.analyze(chart='R4_Xbar')
    assert result is not None
    assert 'R4_Xbar' in result.charts

    chart_data = result.charts['R4_Xbar']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R4_Xbar should have one point per time (5 time points)
    data_df = chart_data['data']
    assert len(data_df) == 5


def test_r4_s_chart_calculation():
    """R4_S should use time-based subgrouping."""
    np.random.seed(42)
    data_rows = []
    # Create SDS 1 data: 3 factors × 5 time points × 2 replicates
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    assert 'R4_S' in study.residual_charts

    # Analyze R4_S
    result = study.analyze(chart='R4_S')
    assert result is not None
    assert 'R4_S' in result.charts

    chart_data = result.charts['R4_S']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R4_S should have one point per time (5 time points)
    data_df = chart_data['data']
    assert len(data_df) == 5


def test_r5_xbar_chart_calculation():
    """R5_Xbar should use factor-based subgrouping."""
    np.random.seed(42)
    data_rows = []
    # Create SDS 1 data: 3 factors × 5 time points × 2 replicates
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    assert 'R5_Xbar' in study.residual_charts

    # Analyze R5_Xbar
    result = study.analyze(chart='R5_Xbar')
    assert result is not None
    assert 'R5_Xbar' in result.charts

    chart_data = result.charts['R5_Xbar']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R5_Xbar should have one point per factor (3 factors)
    data_df = chart_data['data']
    assert len(data_df) == 3


def test_r5_s_chart_calculation():
    """R5_S should use factor-based subgrouping."""
    np.random.seed(42)
    data_rows = []
    # Create SDS 1 data: 3 factors × 5 time points × 2 replicates
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    assert 'R5_S' in study.residual_charts

    # Analyze R5_S
    result = study.analyze(chart='R5_S')
    assert result is not None
    assert 'R5_S' in result.charts

    chart_data = result.charts['R5_S']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R5_S should have one point per factor (3 factors)
    data_df = chart_data['data']
    assert len(data_df) == 3


def test_r4_xbar_subgrouping_different_from_r5():
    """R4 and R5 should have different subgroup counts."""
    np.random.seed(42)
    data_rows = []
    # Create SDS 1 data: 4 factors × 6 time points × 2 replicates
    for factor in ['A', 'B', 'C', 'D']:
        for time in range(1, 7):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    # Analyze both R4 and R5 Xbar charts
    r4_result = study.analyze(chart='R4_Xbar')
    r5_result = study.analyze(chart='R5_Xbar')

    r4_data = r4_result.charts['R4_Xbar']['data']
    r5_data = r5_result.charts['R5_Xbar']['data']

    # R4 subgroups by time: 6 time points
    # R5 subgroups by factor: 4 factors
    assert len(r4_data) == 6, f"R4 should have 6 subgroups (time), got {len(r4_data)}"
    assert len(r5_data) == 4, f"R5 should have 4 subgroups (factor), got {len(r5_data)}"


def test_r4_r5_xbar_s_control_limits_structure():
    """R4 and R5 Xbar/S charts should have proper control limit structure."""
    np.random.seed(42)
    data_rows = []
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    for chart_type in ['R4_Xbar', 'R4_S', 'R5_Xbar', 'R5_S']:
        result = study.analyze(chart=chart_type)
        chart_data = result.charts[chart_type]

        # Should have statistics with control limits
        stats = chart_data['statistics']
        assert 'center' in stats, f"{chart_type} missing center"
        assert 'upl' in stats, f"{chart_type} missing upl"
        assert 'lpl' in stats, f"{chart_type} missing lpl"

        # Control limits should be numeric
        assert isinstance(stats['center'], (int, float, np.number))
        assert isinstance(stats['upl'], (int, float, np.number))
        assert isinstance(stats['lpl'], (int, float, np.number))

        # UPL > center > LPL (for properly behaved data)
        assert stats['upl'] >= stats['center'], f"{chart_type}: UPL should be >= center"
        assert stats['center'] >= stats['lpl'], f"{chart_type}: center should be >= LPL"


# ============================================================================
# Test: R3 Xbar/S Charts (GitHub Issue #50)
# ============================================================================

def test_r3_xbar_chart_calculation():
    """R3_Xbar should use factor-based subgrouping (same as R2)."""
    np.random.seed(42)
    data_rows = []
    # Create SDS 1 data: 3 factors × 5 time points × 2 replicates
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    # Should be SDS 1 (all cells have n≥2)
    assert study.sds == 1
    assert 'R3_Xbar' in study.residual_charts

    # Analyze R3_Xbar
    result = study.analyze(chart='R3_Xbar')
    assert result is not None
    assert 'R3_Xbar' in result.charts

    chart_data = result.charts['R3_Xbar']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R3_Xbar uses factor-based subgrouping (same as R5), so 3 factors
    data_df = chart_data['data']
    assert len(data_df) == 3


def test_r3_s_chart_calculation():
    """R3_S should use factor-based subgrouping (same as R2)."""
    np.random.seed(42)
    data_rows = []
    # Create SDS 1 data: 3 factors × 5 time points × 2 replicates
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    assert 'R3_S' in study.residual_charts

    # Analyze R3_S
    result = study.analyze(chart='R3_S')
    assert result is not None
    assert 'R3_S' in result.charts

    chart_data = result.charts['R3_S']
    assert 'data' in chart_data
    assert 'statistics' in chart_data

    # R3_S uses factor-based subgrouping, so 3 factors
    data_df = chart_data['data']
    assert len(data_df) == 3


def test_r3_xbar_s_control_limits_structure():
    """R3 Xbar/S charts should have proper control limit structure."""
    np.random.seed(42)
    data_rows = []
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })

    df = pd.DataFrame(data_rows)
    study = ProcessDataFrame(df).formulate(
        response='Value',
        factors=['Factor'],
        time='Time'
    )

    for chart_type in ['R3_Xbar', 'R3_S']:
        result = study.analyze(chart=chart_type)
        chart_data = result.charts[chart_type]

        # Should have statistics with control limits
        stats = chart_data['statistics']
        assert 'center' in stats, f"{chart_type} missing center"
        assert 'upl' in stats, f"{chart_type} missing upl"
        assert 'lpl' in stats, f"{chart_type} missing lpl"

        # Control limits should be numeric
        assert isinstance(stats['center'], (int, float, np.number))
        assert isinstance(stats['upl'], (int, float, np.number))
        assert isinstance(stats['lpl'], (int, float, np.number))
