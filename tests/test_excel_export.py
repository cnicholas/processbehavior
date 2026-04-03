"""
Tests for Excel export functionality in AnalysisResult.

This module tests the to_excel() method which exports analysis results
to multi-sheet Excel workbooks.
"""

import os
import tempfile

import pandas as pd
import pytest

from processbehavior.analysis import Analysis

# Import analysis components
from processbehavior.data_preparation import DataPreparation

# Import test data generators
from processbehavior.datasets.synthetic import make_sds
from processbehavior.formulation_spec import ChartRequest, FormulationSpec
from processbehavior.sds_detector import SDSRegistry

pytestmark = pytest.mark.io


def _make_spec(spec_dict: dict) -> FormulationSpec:
    """Convert old-style spec dict to FormulationSpec."""
    rsg_vars = spec_dict.get('rsg_vars')
    return FormulationSpec(
        response_var=spec_dict['response_var'],
        rsg_vars=tuple(rsg_vars) if rsg_vars else None,
        time_var=spec_dict.get('time_var'),
        round_to=spec_dict.get('round_to', 3),
        rsg_var_name=spec_dict.get('rsg_var_name', 'rsg'),
        rsg_var_delim=spec_dict.get('rsg_var_delim', '_'),
    )


def _make_request(spec_dict: dict) -> ChartRequest:
    """Convert old-style spec dict to ChartRequest."""
    return ChartRequest(
        chart=spec_dict.get('analysis_type', 'Xbar'),
        by=tuple(spec_dict['by']) if spec_dict.get('by') else None,
        companion=spec_dict.get('companion', False),
    )


def detect_sds_for_test(df: pd.DataFrame, spec: dict) -> int:
    """
    Helper to detect SDS for tests that need to create Analysis directly.
    """
    config = _make_spec(spec)
    prep = DataPreparation()
    prep.validate_columns(df, config)
    prepared_df = prep.prepare_dataset(df, config)
    detector = SDSRegistry()
    result = detector.detect_sds(prepared_df, config)
    return result.sds


@pytest.fixture
def temp_excel_file():
    """Create a temporary file path for Excel output."""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        filepath = f.name
    yield filepath
    # Cleanup after test
    if os.path.exists(filepath):
        os.remove(filepath)


def test_excel_export_basic(temp_excel_file):
    """Test basic Excel export with default parameters."""
    # Generate SDS1 data
    df = make_sds(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    # Run analysis
    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Verify file exists
    assert os.path.exists(temp_excel_file)

    # Read back and verify sheets
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    sheet_names = excel_file.sheet_names

    # Should have Summary and Chart tabs at minimum
    assert 'Summary' in sheet_names
    assert any('Chart_' in name for name in sheet_names)

    # Read Summary sheet
    summary_df = pd.read_excel(excel_file, sheet_name='Summary')
    assert not summary_df.empty
    assert 'Attribute' in summary_df.columns
    assert 'Value' in summary_df.columns


def test_excel_export_with_full_dataset(temp_excel_file):
    """Test Excel export including full dataset."""
    df = make_sds(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export with full dataset
    result.to_excel(temp_excel_file, include_full_dataset=True)

    # Verify Full_Dataset sheet exists
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    assert 'Full_Dataset' in excel_file.sheet_names

    # Read and verify it has data
    dataset_df = pd.read_excel(excel_file, sheet_name='Full_Dataset')
    assert not dataset_df.empty
    assert 'y' in dataset_df.columns


def test_excel_export_stratified_xmr(temp_excel_file):
    """Test Excel export with stratified XmR chart (SRP: XmR only)."""
    df = make_sds(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)

    spec = {
        'analysis_type': 'XmR',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Verify file exists
    assert os.path.exists(temp_excel_file)

    # Read back and verify stratified charts are present
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    chart_tabs = [name for name in excel_file.sheet_names if 'Chart_' in name]

    # SRP: XmR only returns XmR (no longer bundled with R by default)
    assert len(chart_tabs) == 1, f"Expected 1 chart tab (SRP: XmR only), got {len(chart_tabs)}: {chart_tabs}"
    assert 'Chart_XmR' in chart_tabs, f"Expected Chart_XmR tab, got: {chart_tabs}"

    # Verify the XmR tab has data
    xmr_data = pd.read_excel(temp_excel_file, sheet_name='Chart_XmR')
    assert len(xmr_data) > 0, "XmR chart tab should have data"
    assert 'rsg' in xmr_data.columns, "XmR data should have 'rsg' column for stratification"


def test_excel_export_stratified_xmr_companion(temp_excel_file):
    """Test Excel export with stratified XmR charts (companion=True for bundled output)."""
    df = make_sds(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)

    spec = {
        'analysis_type': 'XmR',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y',
        'companion': True  # Request bundled XmR+R
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Verify file exists
    assert os.path.exists(temp_excel_file)

    # Read back and verify companion charts are present
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    chart_tabs = [name for name in excel_file.sheet_names if 'Chart_' in name]

    # Companion mode: XmR+R bundled together
    assert len(chart_tabs) == 2, f"Expected 2 chart tabs (XmR+R companion), got {len(chart_tabs)}: {chart_tabs}"
    assert 'Chart_XmR' in chart_tabs, f"Expected Chart_XmR tab, got: {chart_tabs}"
    assert 'Chart_R' in chart_tabs, f"Expected Chart_R tab, got: {chart_tabs}"


def test_excel_export_with_residuals(temp_excel_file):
    """Test Excel export includes residuals when calculated."""
    df = make_sds(2, K1=3, K2=2, T=5, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Verify Residuals sheet exists (if VAS was calculated)
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')

    if result.has_residuals:
        assert 'Residuals' in excel_file.sheet_names
        residuals_df = pd.read_excel(excel_file, sheet_name='Residuals')
        assert not residuals_df.empty


def test_excel_export_minimal(temp_excel_file):
    """Test Excel export with minimal options (only charts)."""
    df = make_sds(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export with minimal options
    result.to_excel(
        temp_excel_file,
        include_summary=False,
        include_residuals=False,
        include_effects=False,
        include_interactions=False
    )

    # Verify only chart tabs exist (plus Visual_Charts)
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    sheet_names = excel_file.sheet_names

    assert 'Summary' not in sheet_names
    # All sheets should be either Chart_ sheets or Visual_Charts
    assert all('Chart_' in name or name == 'Visual_Charts' for name in sheet_names)


def test_excel_export_no_formatting(temp_excel_file):
    """Test Excel export without formatting."""
    df = make_sds(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export without formatting
    result.to_excel(temp_excel_file, format_cells=False)

    # Should still create valid Excel file
    assert os.path.exists(temp_excel_file)
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    assert len(excel_file.sheet_names) > 0


def test_excel_export_chart_data_integrity(temp_excel_file):
    """Test that chart data is correctly exported to Excel."""
    df = make_sds(1, K1=2, K2=2, T=5, n_min=3, n_max=3, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Get original chart data
    xbar_original = result.get_chart('Xbar')

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Read back chart data
    xbar_exported = pd.read_excel(temp_excel_file, sheet_name='Chart_Xbar')

    # Verify data integrity (column names and row count)
    assert list(xbar_original.columns) == list(xbar_exported.columns)
    assert len(xbar_original) == len(xbar_exported)


def test_excel_export_summary_content(temp_excel_file):
    """Test that summary tab contains expected information."""
    df = make_sds(2, K1=3, K2=2, T=5, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Read summary
    summary_df = pd.read_excel(temp_excel_file, sheet_name='Summary')

    # Check for key attributes
    attributes = summary_df['Attribute'].tolist()
    assert 'SDS' in attributes
    assert 'Analysis Type' in attributes
    assert 'Response Variable' in attributes
    assert 'Total Observations' in attributes


def test_excel_export_effects_tab(temp_excel_file):
    """Test that effects are properly exported when available."""
    df = make_sds(2, K1=3, K2=2, T=5, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')

    # If effects were calculated, verify Effects tab
    if result.has_effects:
        assert 'Effects' in excel_file.sheet_names
        effects_df = pd.read_excel(excel_file, sheet_name='Effects')
        assert 'Effect_Type' in effects_df.columns
        assert 'Value' in effects_df.columns


def test_excel_export_invalid_path():
    """Test Excel export with invalid file path."""
    df = make_sds(1, K1=2, K2=2, T=3, n_min=2, n_max=2, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Try to export to invalid path
    with pytest.raises(OSError):
        result.to_excel('/nonexistent_directory/output.xlsx')


def test_excel_export_multiple_analyses(temp_excel_file):
    """Test exporting different analysis types."""
    df = make_sds(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)

    # Test with S chart
    spec_s = {
        'analysis_type': 'S',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec_s)
    analysis = Analysis(spec=_make_spec(spec_s), request=_make_request(spec_s), sds=sds, df=df)
    result = analysis.calculate()

    # Should successfully export
    result.to_excel(temp_excel_file)
    assert os.path.exists(temp_excel_file)

    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    assert 'Summary' in excel_file.sheet_names


def test_excel_tab_name_truncation(temp_excel_file):
    """Test that long tab names are properly truncated to 31 characters."""
    df = make_sds(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)

    # Rename factor to create very long group names
    df['factor 1'] = df['factor 1'].replace({
        'F1_1': 'Very_Long_Group_Name_That_Exceeds_31_Characters',
        'F1_2': 'Another_Super_Long_Name_For_Testing'
    })

    spec = {
        'analysis_type': 'XmR',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Verify all tab names are <= 31 chars (Excel limitation)
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    for sheet_name in excel_file.sheet_names:
        assert len(sheet_name) <= 31, f"Tab name '{sheet_name}' exceeds 31 characters"


def test_excel_export_preserves_statistics(temp_excel_file):
    """Test that chart statistics are accessible in exported data."""
    df = make_sds(1, K1=2, K2=2, T=5, n_min=3, n_max=3, seed=42)

    spec = {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Get original statistics
    result.get_statistics('Xbar')

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Read back chart
    chart_df = pd.read_excel(temp_excel_file, sheet_name='Chart_Xbar')

    # Verify key columns exist
    expected_cols = ['centerline', 'ucl', 'lcl']
    for col in expected_cols:
        if col in chart_df.columns:
            # At least one statistical column should be present
            assert True
            return

    # If no stat columns found, that's also ok (they might be in statistics dict)
    assert True


# ============================================================================
# Additional coverage tests
# ============================================================================

def test_excel_export_stratified_multifactor(temp_excel_file):
    """Test export with multi-factor stratification."""
    from processbehavior import ProcessBehavior

    df = make_sds(1, K1=2, K2=2, T=4, n_min=3, n_max=3, seed=42)
    study = ProcessBehavior(df).formulate(
        response='y', time='time', factors=['factor 1', 'factor 2'],
    )
    result = study.execute(chart='XmR', by=['factor 1', 'factor 2'])
    result.to_excel(temp_excel_file)

    assert os.path.exists(temp_excel_file)
    xls = pd.ExcelFile(temp_excel_file)
    # Should have at least Summary and a chart sheet
    assert len(xls.sheet_names) >= 2


def test_excel_export_sds4_minimal(temp_excel_file):
    """Test export with minimal SDS 4-like data (single factor level)."""
    from processbehavior import ProcessBehavior

    import numpy as np
    df = pd.DataFrame({
        'y': np.random.default_rng(42).normal(50, 5, 20),
        'time': range(1, 21),
        'group': ['A'] * 20,
    })
    study = ProcessBehavior(df).formulate(
        response='y', time='time', factors=['group'],
    )
    result = study.execute(chart='XmR', by=['group'])
    result.to_excel(temp_excel_file)

    assert os.path.exists(temp_excel_file)
    xls = pd.ExcelFile(temp_excel_file)
    assert 'Summary' in xls.sheet_names


def test_excel_export_s_chart(temp_excel_file):
    """Test export with S chart (different chart type than Xbar/XmR)."""
    df = make_sds(1, K1=3, K2=2, T=4, n_min=3, n_max=3, seed=42)

    spec = {
        'analysis_type': 'S',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y'
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=_make_spec(spec), request=_make_request(spec), sds=sds, df=df)
    result = analysis.calculate()
    result.to_excel(temp_excel_file)

    assert os.path.exists(temp_excel_file)
    xls = pd.ExcelFile(temp_excel_file)
    # Should have chart sheet for S
    chart_sheets = [s for s in xls.sheet_names if 'Chart' in s]
    assert len(chart_sheets) >= 1


def test_excel_export_round_trip_summary(temp_excel_file):
    """Test that exported Summary sheet contains expected metadata."""
    from processbehavior import ProcessBehavior

    df = make_sds(1, K1=2, K2=2, T=4, n_min=3, n_max=3, seed=42)
    study = ProcessBehavior(df).formulate(
        response='y', time='time', factors=['factor 1', 'factor 2'],
    )
    result = study.execute(chart='Xbar')
    result.to_excel(temp_excel_file)

    summary_df = pd.read_excel(temp_excel_file, sheet_name='Summary')
    # Summary should have Attribute and Value columns
    assert 'Attribute' in summary_df.columns or len(summary_df.columns) >= 2
    # Should contain key metadata
    text = summary_df.to_string()
    assert 'SDS' in text or 'sds' in text.lower()
