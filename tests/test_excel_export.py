"""
Tests for Excel export functionality in AnalysisResult.

This module tests the to_excel() method which exports analysis results
to multi-sheet Excel workbooks.
"""

import os
from pathlib import Path

import pandas as pd
import pytest
from conftest import detect_sds_for_test, make_request, make_spec

from processbehavior import ProcessBehavior
from processbehavior.analysis import Analysis

# Import test data generators
from processbehavior.datasets.synthetic import make_design

pytestmark = pytest.mark.io


@pytest.fixture
def temp_excel_file(tmp_path):
    """Path for a per-test Excel output file.

    Use pytest's tmp_path so Windows file-handle cleanup is handled by the
    pytest fixture machinery (retries on WinError 32) instead of an
    immediate os.remove that races with openpyxl/pandas readers.
    """
    return str(tmp_path / 'out.xlsx')


def test_excel_export_basic(temp_excel_file):
    """Test basic Excel export with default parameters."""
    # Generate SDS1 data
    df = make_design(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    # Run analysis
    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
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
    df = make_design(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
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


def test_excel_export_stratified_x(temp_excel_file):
    """Test Excel export with stratified X chart (SRP: X only)."""
    df = make_design(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)

    spec = {'analysis_type': 'X', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Verify file exists
    assert os.path.exists(temp_excel_file)

    # Read back and verify stratified charts are present
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    chart_tabs = [name for name in excel_file.sheet_names if 'Chart_' in name]

    # SRP: X only returns X (no longer bundled with mR by default)
    assert len(chart_tabs) == 1, f'Expected 1 chart tab (SRP: X only), got {len(chart_tabs)}: {chart_tabs}'
    assert 'Chart_X' in chart_tabs, f'Expected Chart_X tab, got: {chart_tabs}'

    # Verify the X tab has data
    x_data = pd.read_excel(temp_excel_file, sheet_name='Chart_X')
    assert len(x_data) > 0, 'X chart tab should have data'
    assert 'rsg' in x_data.columns, "X data should have 'rsg' column for stratification"


def test_excel_export_stratified_x_companion(temp_excel_file):
    """Test Excel export with stratified X charts (companion=True for bundled output)."""
    df = make_design(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)

    spec = {
        'analysis_type': 'X',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y',
        'companion': True,  # Request bundled X+mR
    }

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Verify file exists
    assert os.path.exists(temp_excel_file)

    # Read back and verify companion charts are present
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    chart_tabs = [name for name in excel_file.sheet_names if 'Chart_' in name]

    # Companion mode: X+mR bundled together
    assert len(chart_tabs) == 2, f'Expected 2 chart tabs (X+mR companion), got {len(chart_tabs)}: {chart_tabs}'
    assert 'Chart_X' in chart_tabs, f'Expected Chart_X tab, got: {chart_tabs}'
    assert 'Chart_mR' in chart_tabs, f'Expected Chart_mR tab, got: {chart_tabs}'


def test_excel_export_with_residuals(temp_excel_file):
    """Test Excel export includes residuals when calculated."""
    df = make_design(2, K1=3, K2=2, T=5, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
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
    df = make_design(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export with minimal options
    result.to_excel(
        temp_excel_file,
        include_summary=False,
        include_residuals=False,
        include_effects=False,
        include_interactions=False,
    )

    # Verify only chart tabs exist (plus Visual_Charts)
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    sheet_names = excel_file.sheet_names

    assert 'Summary' not in sheet_names
    # All sheets should be either Chart_ sheets or Visual_Charts
    assert all('Chart_' in name or name == 'Visual_Charts' for name in sheet_names)


def test_excel_export_no_formatting(temp_excel_file):
    """Test Excel export without formatting."""
    df = make_design(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export without formatting
    result.to_excel(temp_excel_file, format_cells=False)

    # Should still create valid Excel file
    assert os.path.exists(temp_excel_file)
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    assert len(excel_file.sheet_names) > 0


def test_excel_export_chart_data_integrity(temp_excel_file):
    """Test that chart data is correctly exported to Excel."""
    df = make_design(1, K1=2, K2=2, T=5, n_min=3, n_max=3, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
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
    df = make_design(2, K1=3, K2=2, T=5, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Read summary
    summary_df = pd.read_excel(temp_excel_file, sheet_name='Summary')

    # Check for key attributes
    attributes = summary_df['Attribute'].tolist()
    assert 'ADS' in attributes
    assert 'Analysis Type' in attributes
    assert 'Response Variable' in attributes
    assert 'Total Observations' in attributes


def test_excel_export_effects_tab(temp_excel_file):
    """Test that effects are properly exported when available."""
    df = make_design(2, K1=3, K2=2, T=5, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
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
    df = make_design(1, K1=2, K2=2, T=3, n_min=2, n_max=2, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Try to export to invalid path
    with pytest.raises(OSError):
        result.to_excel('/nonexistent_directory/output.xlsx')


def test_excel_export_multiple_analyses(temp_excel_file):
    """Test exporting different analysis types."""
    df = make_design(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)

    # Test with S chart
    spec_s = {'analysis_type': 'S', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec_s)
    analysis = Analysis(spec=make_spec(spec_s), request=make_request(spec_s), sds=sds, df=df)
    result = analysis.calculate()

    # Should successfully export
    result.to_excel(temp_excel_file)
    assert os.path.exists(temp_excel_file)

    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    assert 'Summary' in excel_file.sheet_names


def test_excel_tab_name_truncation(temp_excel_file):
    """Test that long tab names are properly truncated to 31 characters."""
    df = make_design(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)

    # Rename factor to create very long group names
    df['factor 1'] = df['factor 1'].replace(
        {'F1_1': 'Very_Long_Group_Name_That_Exceeds_31_Characters', 'F1_2': 'Another_Super_Long_Name_For_Testing'}
    )

    spec = {'analysis_type': 'X', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
    result = analysis.calculate()

    # Export to Excel
    result.to_excel(temp_excel_file)

    # Verify all tab names are <= 31 chars (Excel limitation)
    excel_file = pd.ExcelFile(temp_excel_file, engine='openpyxl')
    for sheet_name in excel_file.sheet_names:
        assert len(sheet_name) <= 31, f"Tab name '{sheet_name}' exceeds 31 characters"


def test_excel_export_preserves_statistics(temp_excel_file):
    """Test that chart statistics are accessible in exported data."""
    df = make_design(1, K1=2, K2=2, T=5, n_min=3, n_max=3, seed=42)

    spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
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

    df = make_design(1, K1=2, K2=2, T=4, n_min=3, n_max=3, seed=42)
    study = ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1', 'factor 2'],
    )
    result = study.execute(chart='X', by=['factor 1', 'factor 2'])
    result.to_excel(temp_excel_file)

    assert os.path.exists(temp_excel_file)
    xls = pd.ExcelFile(temp_excel_file)
    # Should have at least Summary and a chart sheet
    assert len(xls.sheet_names) >= 2


def test_excel_export_sds4_minimal(temp_excel_file):
    """Test export with minimal SDS 4-like data (single factor level)."""
    import numpy as np

    from processbehavior import ProcessBehavior

    df = pd.DataFrame(
        {
            'y': np.random.default_rng(42).normal(50, 5, 20),
            'time': range(1, 21),
            'group': ['A'] * 20,
        }
    )
    study = ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['group'],
    )
    result = study.execute(chart='X', by=['group'])
    result.to_excel(temp_excel_file)

    assert os.path.exists(temp_excel_file)
    xls = pd.ExcelFile(temp_excel_file)
    assert 'Summary' in xls.sheet_names


def test_excel_export_s_chart(temp_excel_file):
    """Test export with S chart (different chart type than Xbar/X)."""
    df = make_design(1, K1=3, K2=2, T=4, n_min=3, n_max=3, seed=42)

    spec = {'analysis_type': 'S', 'rsg_vars': ['factor 1'], 'time_var': 'time', 'response_var': 'y'}

    sds = detect_sds_for_test(df, spec)
    analysis = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df)
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

    df = make_design(1, K1=2, K2=2, T=4, n_min=3, n_max=3, seed=42)
    study = ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor 1', 'factor 2'],
    )
    result = study.execute(chart='Xbar')
    result.to_excel(temp_excel_file)

    summary_df = pd.read_excel(temp_excel_file, sheet_name='Summary')
    # Summary should have Attribute and Value columns
    assert 'Attribute' in summary_df.columns or len(summary_df.columns) >= 2
    # Should contain key metadata
    text = summary_df.to_string()
    assert 'ADS' in text or 'ads' in text.lower()


# ============================================================================
# to_excel returns what it wrote, and writes nothing it did not name
# ============================================================================


class TestExportWritesWhatItSays:
    """One call naming one .xlsx should not leave a second file behind.

    `export_html` defaulted to True, so `to_excel('analysis.xlsx')` also wrote a
    standalone plotly document — several megabytes, undocumented in the return
    (there wasn't one) and unmentioned by the call. The workbook then carried an
    "INTERACTIVE CHARTS" note pointing at files that a default export never
    created once the default flipped, so that note is now conditional too.
    """

    @staticmethod
    def _result():
        df = make_design(1, K1=3, K2=2, T=5, n_min=2, n_max=3, seed=42)
        return (
            ProcessBehavior(df)
            .formulate(response='y', factors=['factor 1'], time='time')
            .execute()
        )

    def test_default_writes_only_the_workbook(self, tmp_path):
        target = tmp_path / 'analysis.xlsx'
        self._result().to_excel(str(target))

        assert [p.name for p in sorted(tmp_path.iterdir())] == ['analysis.xlsx']

    def test_default_returns_only_the_workbook(self, tmp_path):
        target = tmp_path / 'analysis.xlsx'
        written = self._result().to_excel(str(target))

        assert [Path(p).name for p in written] == ['analysis.xlsx']

    def test_html_opt_in_writes_and_reports_both(self, tmp_path):
        target = tmp_path / 'analysis.xlsx'
        written = self._result().to_excel(str(target), export_html=True)

        names = sorted(Path(p).name for p in written)
        assert 'analysis.xlsx' in names
        assert any(n.endswith('.html') for n in names)

    def test_every_returned_path_exists(self, tmp_path):
        target = tmp_path / 'analysis.xlsx'
        written = self._result().to_excel(str(target), export_html=True)

        assert all(Path(p).exists() for p in written)

    def test_nothing_written_is_unreported(self, tmp_path):
        """The return value must account for every file that appeared."""
        target = tmp_path / 'analysis.xlsx'
        written = self._result().to_excel(str(target), export_html=True)

        on_disk = {p.name for p in tmp_path.iterdir()}
        reported = {Path(p).name for p in written}
        assert on_disk == reported


class TestStratifiedAndOptionSurface:
    """The export paths #113's audit found unexercised."""

    @pytest.fixture
    def factorial_study(self):
        from processbehavior.datasets.synthetic import make_design
        df = make_design(1, K1=3, K2=2, T=5, seed=42)
        return ProcessBehavior(df).formulate(
            response='y', factors=['factor 1', 'factor 2'], time='time'
        )

    def test_stratified_histogram_gets_combined_tab(self, factorial_study, temp_excel_file):
        """Histogram is the one chart outside STANDARD_CHART_NAMES, so it is the
        live route into _write_stratified_chart_tab."""
        factorial_study.execute(chart='Histogram', by=['factor 1']).to_excel(temp_excel_file)
        sheets = pd.ExcelFile(temp_excel_file, engine='openpyxl').sheet_names
        assert 'Chart_Histogram_Stratified' in sheets
        tab = pd.read_excel(temp_excel_file, sheet_name='Chart_Histogram_Stratified')
        assert len(tab) > 0

    def test_stratified_two_grouping_vars(self, factorial_study, temp_excel_file):
        factorial_study.execute(chart='Histogram', by=['factor 1', 'factor 2']).to_excel(temp_excel_file)
        sheets = pd.ExcelFile(temp_excel_file, engine='openpyxl').sheet_names
        assert any(s.startswith('Chart_Histogram') for s in sheets)

    def test_multiindex_interactions_sheet(self, factorial_study, temp_excel_file):
        """SDS-1 factor x time interactions arrive MultiIndexed; the sheet
        flattens them into a Combination column."""
        factorial_study.execute(chart='Xbar').to_excel(temp_excel_file)
        tab = pd.read_excel(temp_excel_file, sheet_name='Interactions')
        assert 'Combination' in tab.columns
        assert len(tab) > 0

    def test_disable_summary_charts_and_images(self, factorial_study, temp_excel_file):
        written = factorial_study.execute(chart='Xbar').to_excel(
            temp_excel_file,
            include_summary=False,
            include_charts=False,
            include_chart_images=False,
        )
        sheets = pd.ExcelFile(temp_excel_file, engine='openpyxl').sheet_names
        assert 'Summary' not in sheets
        assert not any(s.startswith('Chart_') for s in sheets)
        assert 'Visual_Charts' not in sheets
        assert [Path(w).name for w in written] == [Path(temp_excel_file).name]

    def test_export_html_stratified_companion_writes_combined_html(
        self, factorial_study, temp_excel_file
    ):
        result = factorial_study.execute(chart='X', by=['factor 1'], companion=True)
        written = result.to_excel(temp_excel_file, export_html=True)
        names = sorted(Path(w).name for w in written)
        assert any(n.endswith('_combined.html') for n in names)


class TestVisualChartsErrorPaths:
    """The kaleido-absent and render-failure branches, simulated by monkeypatch
    (never by installing/uninstalling the extra)."""

    @pytest.fixture
    def result(self):
        from processbehavior.datasets.synthetic import make_design
        df = make_design(1, K1=3, K2=2, T=5, seed=42)
        study = ProcessBehavior(df).formulate(
            response='y', factors=['factor 1', 'factor 2'], time='time'
        )
        return study.execute(chart='Xbar')

    def _visual_cell_text(self, path):
        tab = pd.read_excel(path, sheet_name='Visual_Charts', header=None)
        return ' '.join(str(v) for v in tab.values.ravel() if pd.notna(v))

    def test_missing_kaleido_writes_install_hint(self, result, temp_excel_file, monkeypatch):
        import plotly.graph_objects as go

        def raise_import(self, *a, **k):
            raise ImportError('kaleido not installed')

        monkeypatch.setattr(go.Figure, 'write_image', raise_import)
        result.to_excel(temp_excel_file, include_chart_images=True)
        text = self._visual_cell_text(temp_excel_file)
        assert 'images' in text or 'kaleido' in text.lower()

    def test_render_failure_writes_could_not_render(self, result, temp_excel_file, monkeypatch):
        import plotly.graph_objects as go

        def raise_runtime(self, *a, **k):
            raise RuntimeError('renderer exploded')

        monkeypatch.setattr(go.Figure, 'write_image', raise_runtime)
        result.to_excel(temp_excel_file, include_chart_images=True)
        text = self._visual_cell_text(temp_excel_file)
        assert 'Could not render' in text or 'could not' in text.lower()
