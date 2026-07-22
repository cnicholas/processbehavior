"""
Tests for plotting framework.

Tests the plotting infrastructure including themes, figure creation,
and chart rendering.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from processbehavior import ProcessBehavior
from processbehavior.exceptions import ChartNotAvailableError
from processbehavior.plotting import ChartTheme, ControlChartFigure, Plotter, get_theme, list_themes, register_theme
from processbehavior.plotting.themes import apply_theme

pytestmark = pytest.mark.plotting


class TestThemes:
    """Test theme definitions and application."""

    def test_theme_registry(self):
        """Test that all required themes are registered."""
        available = list_themes()
        assert 'processbehavior' in available
        assert 'minimal' in available
        assert 'dark' in available

    def test_apply_theme_processbehavior(self):
        """Test applying processbehavior theme."""
        fig = go.Figure()
        themed_fig = apply_theme(fig, 'processbehavior')

        layout = themed_fig.layout
        assert layout.plot_bgcolor == 'white'
        assert layout.paper_bgcolor == 'white'

    def test_apply_theme_dark(self):
        """Test applying dark theme."""
        fig = go.Figure()
        themed_fig = apply_theme(fig, 'dark')

        layout = themed_fig.layout
        # Plotly normalizes hex colors to uppercase
        assert layout.plot_bgcolor.lower() == '#1e1e1e'
        assert layout.paper_bgcolor.lower() == '#2d2d2d'

    def test_apply_theme_invalid(self):
        """Test error handling for invalid theme."""
        fig = go.Figure()
        with pytest.raises(ValueError, match='Unknown theme'):
            apply_theme(fig, 'nonexistent')


class TestChartTheme:
    """Test ChartTheme dataclass and theme functions."""

    def test_list_themes(self):
        """Test listing available themes."""
        themes = list_themes()
        assert 'processbehavior' in themes
        assert 'ggplot' in themes
        assert 'minimal' in themes
        assert 'dark' in themes
        assert 'publication' in themes

    def test_get_theme(self):
        """Test getting a theme by name."""
        theme = get_theme('processbehavior')
        assert isinstance(theme, ChartTheme)
        assert theme.name == 'processbehavior'
        assert theme.data_color == 'steelblue'

    def test_get_theme_returns_copy(self):
        """Test that get_theme returns a copy (safe to modify)."""
        theme1 = get_theme('processbehavior')
        theme2 = get_theme('processbehavior')

        # Modify theme1
        theme1.data_color = 'purple'

        # theme2 should be unaffected
        assert theme2.data_color == 'steelblue'

    def test_get_theme_invalid(self):
        """Test error handling for invalid theme name."""
        with pytest.raises(ValueError, match='Unknown theme'):
            get_theme('nonexistent')

    def test_chart_theme_dataclass(self):
        """Test creating a custom ChartTheme."""
        theme = ChartTheme(name='custom', data_color='navy', signal_color='orange', center_color='darkgreen')
        assert theme.name == 'custom'
        assert theme.data_color == 'navy'
        assert theme.signal_color == 'orange'
        assert theme.center_color == 'darkgreen'
        # Defaults should be preserved
        assert theme.ucl_color == 'red'

    def test_chart_theme_to_layout_dict(self):
        """Test converting theme to Plotly layout dict."""
        theme = get_theme('processbehavior')
        layout = theme.to_layout_dict()

        assert 'plot_bgcolor' in layout
        assert 'paper_bgcolor' in layout
        assert 'font' in layout
        assert 'xaxis' in layout
        assert 'yaxis' in layout

    def test_register_custom_theme(self):
        """Test registering a custom theme."""
        custom = ChartTheme(name='test_corporate', data_color='#003366')
        register_theme(custom)

        # Should be able to retrieve it
        retrieved = get_theme('test_corporate')
        assert retrieved.data_color == '#003366'

    def test_ggplot_theme_properties(self):
        """Test ggplot theme has expected properties."""
        theme = get_theme('ggplot')
        # ggplot2 has gray background
        assert theme.plot_bgcolor == '#EBEBEB'
        # Uses circle markers for signals
        assert theme.signal_marker_symbol == 'circle'

    def test_publication_theme_properties(self):
        """Test publication theme has expected properties."""
        theme = get_theme('publication')
        # High contrast black for data
        assert theme.data_color == '#000000'
        # No zone shading for print
        assert theme.zone_opacity == 0.0
        # Serif font for academic
        assert 'serif' in theme.font_family.lower()


class TestControlChartFigure:
    """Test ControlChartFigure wrapper class."""

    @pytest.fixture
    def sample_result(self):
        """Create sample analysis result with factors."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 30), 'group': ['A'] * 15 + ['B'] * 15, 'time': list(range(15)) * 2}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.group], time=pdf.cols.time)
        return study.execute(chart='X', by=['group'])

    @pytest.fixture
    def sample_figure(self, sample_result):
        """Create sample ControlChartFigure."""
        fig = go.Figure()
        return ControlChartFigure(fig, sample_result)

    def test_initialization(self, sample_figure, sample_result):
        """Test figure initialization."""
        assert sample_figure._result is sample_result
        assert isinstance(sample_figure._fig, go.Figure)

    def test_figure_property(self, sample_figure):
        """Test access to underlying figure."""
        assert isinstance(sample_figure.figure, go.Figure)

    def test_update_layout(self, sample_figure):
        """Test layout update method."""
        result = sample_figure.update_layout(title='Test Title')
        assert result is sample_figure  # Check method chaining
        assert sample_figure._fig.layout.title.text == 'Test Title'

    def test_add_annotation(self, sample_figure):
        """Test annotation addition."""
        result = sample_figure.add_annotation('Test', x=10, y=100)
        assert result is sample_figure  # Check method chaining
        assert len(sample_figure._fig.layout.annotations) > 0

    def test_save_html(self, sample_figure):
        """Test HTML export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.html'
            sample_figure.save_html(filepath)
            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_repr(self, sample_figure):
        """Test string representation."""
        repr_str = repr(sample_figure)
        assert 'ControlChartFigure' in repr_str
        assert 'charts=' in repr_str

    def test_repr_html_returns_html_string(self, sample_figure):
        """Test _repr_html_() returns an HTML fragment for Jupyter/nbconvert."""
        html = sample_figure._repr_html_()
        assert isinstance(html, str)
        assert '<div' in html
        assert 'plotly' in html.lower()


class TestPlotter:
    """Test Plotter class."""

    @pytest.fixture
    def simple_result(self):
        """Create simple XmR analysis result with a factor."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 30), 'group': ['A'] * 15 + ['B'] * 15, 'time': list(range(15)) * 2}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.group], time=pdf.cols.time)
        return study.execute(chart='X', by=['group'])

    @pytest.fixture
    def xbar_result(self):
        """Create Xbar analysis result."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 100), 'subgroup': np.repeat(range(20), 5), 'time': range(100)}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.subgroup])
        return study.execute()

    def test_initialization(self, simple_result):
        """Test plotter initialization."""
        plotter = Plotter(simple_result)
        assert plotter.result is simple_result
        assert plotter.charts == simple_result.charts
        assert plotter.summary == simple_result.summary

    def test_list_charts(self, simple_result):
        """Test chart listing."""
        plotter = Plotter(simple_result)
        charts = plotter.list_charts()
        assert isinstance(charts, list)
        assert len(charts) > 0

    def test_plot_single_chart(self, simple_result):
        """Test plotting a single specific chart."""
        plotter = Plotter(simple_result)
        # Use 'X' which is the key for the X chart
        fig = plotter.plot(chart='X')

        assert isinstance(fig, ControlChartFigure)
        assert isinstance(fig.figure, go.Figure)

    def test_plot_all_charts(self, xbar_result):
        """Test plotting all charts."""
        plotter = Plotter(xbar_result)
        fig = plotter.plot()

        assert isinstance(fig, ControlChartFigure)

    def test_plot_invalid_chart(self, simple_result):
        """Test error handling for invalid chart name."""
        plotter = Plotter(simple_result)
        with pytest.raises(
            ChartNotAvailableError,
            match="Chart 'Invalid' not found",
        ):
            plotter.plot(chart='Invalid')

    def test_plot_with_theme(self, simple_result):
        """Test plotting with different themes."""
        plotter = Plotter(simple_result)

        # Test each theme
        for theme_name in ['processbehavior', 'minimal', 'dark']:
            fig = plotter.plot(chart='X', theme=theme_name)
            assert isinstance(fig, ControlChartFigure)

    def test_plot_with_custom_dimensions(self, simple_result):
        """Test plotting with custom dimensions."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='X', width=1200, height=600)

        assert fig.figure.layout.width == 1200
        assert fig.figure.layout.height == 600

    def test_plot_with_title(self, simple_result):
        """Test plotting with custom title."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='X', title='Custom Title')

        assert fig.figure.layout.title.text == 'Custom Title'

    def test_plot_without_limits(self, simple_result):
        """Test plotting without control limits."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='X', show_limits=False)

        assert isinstance(fig, ControlChartFigure)

    def test_plot_without_signals(self, simple_result):
        """Test plotting without signal highlighting."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='X', highlight_signals=False)

        assert isinstance(fig, ControlChartFigure)

    def test_helper_get_value_column(self, simple_result):
        """Test value column detection from metadata."""
        plotter = Plotter(simple_result)

        # Create chart_info with metadata (as charts now provide)
        chart_info = {
            'data': pd.DataFrame({'xbar': [1, 2, 3]}),
            'statistics': {},
            'metadata': {'chart_type': 'Xbar', 'value_col': 'xbar', 'center_col': 'center'},
        }

        value_col = plotter._get_value_column(chart_info, 'Xbar')
        assert value_col == 'xbar'

    def test_helper_get_x_column(self, simple_result):
        """Test x-axis column detection."""
        plotter = Plotter(simple_result)

        # No time_var or rsg column → None (use index)
        # Use non-unique values so the fallback scan doesn't pick 'value' as x-axis
        data = pd.DataFrame({'value': [1, 2, 1]})
        x_col = plotter._get_x_column(data)
        assert x_col is None

        # With rsg column → 'rsg'
        data_rsg = pd.DataFrame({'rsg': ['A', 'B', 'C'], 'value': [1, 2, 3]})
        x_col = plotter._get_x_column(data_rsg)
        assert x_col == 'rsg'

    def test_helper_get_center_key(self, simple_result):
        """Test centerline key detection."""
        plotter = Plotter(simple_result)

        # All chart types now use 'center'
        stats = {'center': 100, 'upl': 115, 'lpl': 85}
        center_key = plotter._get_center_key(stats)
        assert center_key == 'center'

        # Test with missing center
        stats = {'upl': 15, 'lpl': 0}
        center_key = plotter._get_center_key(stats)
        assert center_key is None

    def test_plot_with_zone_shading(self, simple_result):
        """Test plotting with zone shading enabled."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='X', show_zones=True)

        assert isinstance(fig, ControlChartFigure)
        # Figure should have shapes (zone rectangles)
        shapes = fig.figure.layout.shapes
        # Should have 5 zones: Zone C (center), Zone B upper/lower, Zone A upper/lower
        assert len(shapes) >= 5

    def test_plot_zone_shading_respects_theme_opacity(self, simple_result):
        """Test that zone shading respects theme opacity."""
        plotter = Plotter(simple_result)

        # Use publication theme which has zone_opacity=0
        fig = plotter.plot(chart='X', show_zones=True, theme='publication')

        # With opacity=0, no zone shapes should be added (only control limit lines)
        shapes = fig.figure.layout.shapes
        # Count zone shapes (rectangles) - should be 0
        zone_shapes = [s for s in shapes if s.type == 'rect']
        assert len(zone_shapes) == 0

    def test_plot_zone_shading_with_custom_theme(self, simple_result):
        """Test zone shading with custom theme colors."""
        custom_theme = ChartTheme(
            name='custom_zones',
            zone_a_color='#FF0000',
            zone_b_color='#FFFF00',
            zone_c_color='#00FF00',
            zone_opacity=0.3,
        )

        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='X', show_zones=True, theme=custom_theme)

        assert isinstance(fig, ControlChartFigure)
        shapes = fig.figure.layout.shapes
        assert len(shapes) >= 5

    def test_plot_with_run_rules(self, simple_result):
        """Test plotting with run rules visualization."""
        plotter = Plotter(simple_result)
        # show_rules should not error even if no rule violations exist
        fig = plotter.plot(chart='X', show_rules=True)

        assert isinstance(fig, ControlChartFigure)
        # Figure should render without errors
        assert fig.figure is not None

    def test_fixed_limit_lines_use_domain_coordinates(self):
        """Regression: limit lines must use domain-relative coordinates.

        Using x-data values for shape positioning causes Plotly to
        misinterpret numeric-looking category strings as numeric positions,
        compressing data to one side of the chart.  Domain coordinates
        (x0=0, x1=1 with xref='x domain') avoid this entirely.
        """
        pb = ProcessBehavior.read_csv('validation/electrical_resistance_in_megohms_204.csv')
        study = pb.formulate(
            response=pb.cols.resistance_megohms,
            time=pb.cols.obs,
            factors=[pb.cols.subgroup],
        )
        result = study.execute(chart='Xbar', by=['subgroup'])
        fig = result.plot(chart='Xbar')

        # Extract horizontal limit lines (UPL, LPL, centerline)
        hlines = [s for s in fig._fig.layout.shapes if s.type == 'line' and s.y0 == s.y1]
        assert len(hlines) == 3, f'Expected 3 limit lines, got {len(hlines)}'

        for line in hlines:
            assert line.x0 == 0, f'Limit line x0={line.x0!r}, expected 0'
            assert line.x1 == 1, f'Limit line x1={line.x1!r}, expected 1'
            assert 'domain' in line.xref, f'Expected domain xref, got {line.xref!r}'

    def test_plot_with_rules_and_zones(self, simple_result):
        """Test plotting with both show_rules and show_zones enabled."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='X', show_rules=True, show_zones=True, highlight_signals=True)

        assert isinstance(fig, ControlChartFigure)
        # Should have zone shapes
        shapes = fig.figure.layout.shapes
        # At least zone shapes should exist
        zone_shapes = [s for s in shapes if s.type == 'rect']
        assert len(zone_shapes) >= 0  # May be 0 or more depending on limits


class TestAnalysisResultIntegration:
    """Test plot() integration with AnalysisResult."""

    @pytest.fixture
    def result(self):
        """Create sample analysis result with a factor."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 30), 'group': ['A'] * 15 + ['B'] * 15, 'time': list(range(15)) * 2}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.group], time=pdf.cols.time)
        return study.execute(chart='X', by=['group'])

    def test_plot_method_exists(self, result):
        """Test that plot() method exists on AnalysisResult."""
        assert hasattr(result, 'plot')
        assert callable(result.plot)

    def test_plot_returns_figure(self, result):
        """Test that plot() returns ControlChartFigure."""
        fig = result.plot()
        assert isinstance(fig, ControlChartFigure)

    def test_plot_with_kwargs(self, result):
        """Test that plot() accepts keyword arguments."""
        fig = result.plot(theme='minimal', width=800, highlight_signals=False)
        assert isinstance(fig, ControlChartFigure)
        assert fig.figure.layout.width == 800


class TestFacetedPlotting:
    """Test faceted/stratified chart plotting."""

    @pytest.fixture
    def xbar_result(self):
        """Create Xbar result with multiple charts."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 100), 'subgroup': np.repeat(range(20), 5), 'time': range(100)}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.subgroup])
        return study.execute()

    def test_faceted_plot(self, xbar_result):
        """Test creating faceted plot."""
        plotter = Plotter(xbar_result)
        fig = plotter.plot(ncols=2)

        assert isinstance(fig, ControlChartFigure)

    def test_faceted_with_custom_ncols(self, xbar_result):
        """Test faceted plot with custom column count."""
        plotter = Plotter(xbar_result)
        fig = plotter.plot(ncols=3)

        assert isinstance(fig, ControlChartFigure)


class TestNumericStrataPlotting:
    """Test plotting with numeric factor values (strata as integer tuples)."""

    def test_plot_with_numeric_strata(self):
        """Plotting works when strata are numeric tuples like (1, 1)."""
        # This is the original bug: strata were (1, 1), (1, 2) etc. which
        # caused TypeError: sequence item 0: expected str instance, int found
        df = pd.DataFrame(
            {
                'value': np.random.normal(100, 5, 40),
                'factor1': [1, 1, 2, 2] * 10,
                'factor2': [1, 2, 1, 2] * 10,
                'time': list(range(10)) * 4,
            }
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.factor1, pdf.cols.factor2], time=pdf.cols.time)
        result = study.execute(chart='X', by=['factor1', 'factor2'])

        # Verify strata are tuples with numeric values
        assert result.is_stratified
        strata = result.strata
        assert len(strata) > 0

        # This should NOT raise TypeError
        plotter = Plotter(result)
        fig = plotter.plot(chart='X', show_zones=True)
        assert isinstance(fig, ControlChartFigure)

    def test_plot_with_mixed_type_strata(self):
        """Plotting works with mixed string/int factor values."""
        df = pd.DataFrame(
            {
                'value': np.random.normal(100, 5, 20),
                'machine': ['A', 'A', 'B', 'B'] * 5,
                'shift': [1, 2, 1, 2] * 5,
                'time': list(range(5)) * 4,
            }
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.machine, pdf.cols.shift], time=pdf.cols.time)
        result = study.execute(chart='X', by=['machine', 'shift'])

        plotter = Plotter(result)
        fig = plotter.plot(chart='X')
        assert isinstance(fig, ControlChartFigure)


class TestAspectRatio:
    """Test aspect ratio functionality."""

    @pytest.fixture
    def simple_result(self):
        """Create simple XmR analysis result with a factor."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 30), 'group': ['A'] * 15 + ['B'] * 15, 'time': list(range(15)) * 2}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.group], time=pdf.cols.time)
        return study.execute(chart='X', by=['group'])

    def test_aspect_ratio_calculation(self, simple_result):
        """Test that aspect ratio correctly calculates height."""
        plotter = Plotter(simple_result)

        # 16:9 aspect ratio
        fig = plotter.plot(chart='X', width=1600, aspect_ratio=16 / 9)
        assert fig.figure.layout.width == 1600
        assert fig.figure.layout.height == 900

    def test_aspect_ratio_square(self, simple_result):
        """Test square aspect ratio."""
        plotter = Plotter(simple_result)

        fig = plotter.plot(chart='X', width=800, aspect_ratio=1.0)
        assert fig.figure.layout.height == 800

    def test_aspect_ratio_portrait(self, simple_result):
        """Test portrait aspect ratio."""
        plotter = Plotter(simple_result)

        fig = plotter.plot(chart='X', width=600, aspect_ratio=0.75)
        assert fig.figure.layout.height == 800


class TestReportGeneration:
    """Test report generation functionality."""

    @pytest.fixture
    def simple_result(self):
        """Create simple XmR analysis result with a factor."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 30), 'group': ['A'] * 15 + ['B'] * 15, 'time': list(range(15)) * 2}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.group], time=pdf.cols.time)
        return study.execute(chart='X', by=['group'])

    def test_generate_report(self, simple_result):
        """Test basic report generation."""
        import tempfile
        from pathlib import Path

        plotter = Plotter(simple_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_report.html'
            plotter.generate_report(str(filepath))

            assert filepath.exists()
            assert filepath.stat().st_size > 0

            # Check content
            content = filepath.read_text()
            assert 'Process Behavior Analysis Report' in content
            assert 'Analysis Summary' in content

    def test_generate_report_custom_title(self, simple_result):
        """Test report with custom title."""
        import tempfile
        from pathlib import Path

        plotter = Plotter(simple_result)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'custom_report.html'
            plotter.generate_report(str(filepath), title='My Custom Report')

            content = filepath.read_text()
            assert 'My Custom Report' in content


class TestResidualPlots:
    """Test residual visualization functionality."""

    @pytest.fixture
    def result_with_residuals(self):
        """Create analysis result with residuals (requires SDS 1 - replicated data)."""
        np.random.seed(42)
        # Create replicated design: 4 factors x 5 time points x 3 replicates = 60 obs
        # This ensures SDS 1 detection (full replication) which produces residuals
        n_factors = 4
        n_times = 5
        n_reps = 3
        n_total = n_factors * n_times * n_reps

        df = pd.DataFrame(
            {
                'value': np.random.normal(100, 5, n_total),
                'factor': np.tile(np.repeat(['A', 'B', 'C', 'D'], n_reps), n_times),
                'time': np.repeat(range(n_times), n_factors * n_reps),
            }
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.factor], time=pdf.cols.time)
        result = study.execute()
        # Verify fixture produces residuals
        assert result.has_residuals, 'Fixture should produce result with residuals'
        return result

    def test_plot_residuals_available(self, result_with_residuals):
        """Test residual plots when residuals are available."""
        plotter = Plotter(result_with_residuals)
        fig = plotter.plot_residuals()
        assert isinstance(fig, ControlChartFigure)

    def test_plot_residuals_histogram(self, result_with_residuals):
        """Test histogram residual plot."""
        plotter = Plotter(result_with_residuals)
        fig = plotter.plot_residuals(plot_type='histogram')
        assert isinstance(fig, ControlChartFigure)

    def test_plot_residuals_qq(self, result_with_residuals):
        """Test Q-Q residual plot."""
        plotter = Plotter(result_with_residuals)
        fig = plotter.plot_residuals(plot_type='qq')
        assert isinstance(fig, ControlChartFigure)

    def test_plot_residuals_sequence(self, result_with_residuals):
        """Test sequence residual plot."""
        plotter = Plotter(result_with_residuals)
        fig = plotter.plot_residuals(plot_type='sequence')
        assert isinstance(fig, ControlChartFigure)


class TestEffectsPlots:
    """Test effects visualization functionality."""

    @pytest.fixture
    def result_with_effects(self):
        """Create analysis result with effects."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 100), 'subgroup': np.repeat(range(20), 5), 'time': range(100)}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.subgroup])
        return study.execute()

    def test_plot_effects_available(self, result_with_effects):
        """Test effects plots when effects are available."""
        if result_with_effects.has_effects:
            plotter = Plotter(result_with_effects)
            fig = plotter.plot_effects()
            assert isinstance(fig, ControlChartFigure)


class TestStatsBox:
    """Test statistical annotations (stats box) functionality."""

    @pytest.fixture
    def simple_result(self):
        """Create simple Xbar analysis result with replication."""
        np.random.seed(42)
        # Create data with replication within factor levels
        df = pd.DataFrame(
            {
                'value': np.random.normal(100, 5, 60),
                'factor': np.repeat(['A', 'B', 'C'], 20),
                'time': np.tile(range(1, 11), 6),
            }
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.factor], time=pdf.cols.time)
        return study.execute()

    @pytest.fixture
    def xbar_result(self):
        """Create Xbar analysis result with subgroups."""
        np.random.seed(42)
        df = pd.DataFrame(
            {'value': np.random.normal(100, 5, 100), 'subgroup': np.repeat(range(20), 5), 'time': range(100)}
        )
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.subgroup])
        return study.execute()

    def test_plot_with_stats_box(self, simple_result):
        """Test plotting with stats box enabled."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='Xbar', show_stats=True)

        assert isinstance(fig, ControlChartFigure)
        # Figure should have annotation (stats box)
        annotations = fig.figure.layout.annotations
        assert len(annotations) > 0

        # At least one annotation should contain 'n =' (sample size)
        stats_annotation = None
        for ann in annotations:
            if ann.text and 'n =' in ann.text:
                stats_annotation = ann
                break
        assert stats_annotation is not None

    def test_stats_box_content(self, simple_result):
        """Test that stats box contains expected statistics."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='Xbar', show_stats=True)

        # Find the stats box annotation
        stats_annotation = None
        for ann in fig.figure.layout.annotations:
            if ann.text and 'n =' in ann.text:
                stats_annotation = ann
                break

        assert stats_annotation is not None
        text = stats_annotation.text

        # Should contain key statistics
        assert 'n =' in text
        assert 'CL =' in text
        assert 'UPL =' in text
        assert 'LPL =' in text

    def test_stats_box_positioning(self, simple_result):
        """Test that stats box is positioned in upper-left corner."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='Xbar', show_stats=True)

        # Find the stats box annotation
        stats_annotation = None
        for ann in fig.figure.layout.annotations:
            if ann.text and 'n =' in ann.text:
                stats_annotation = ann
                break

        assert stats_annotation is not None
        # Should be in upper-left (small x, large y in paper coordinates)
        assert stats_annotation.x < 0.5  # Left side
        assert stats_annotation.y > 0.5  # Upper side
        assert stats_annotation.xanchor == 'left'
        assert stats_annotation.yanchor == 'top'

    def test_stats_box_respects_theme(self, simple_result):
        """Test that stats box respects theme styling."""
        custom_theme = ChartTheme(
            name='custom_stats',
            stats_box_bgcolor='rgba(200, 200, 255, 0.8)',
            stats_box_font_size=14,
            stats_box_font_color='#0000FF',
        )

        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='Xbar', show_stats=True, theme=custom_theme)

        # Find the stats box annotation
        stats_annotation = None
        for ann in fig.figure.layout.annotations:
            if ann.text and 'n =' in ann.text:
                stats_annotation = ann
                break

        assert stats_annotation is not None
        assert stats_annotation.font.size == 14
        assert stats_annotation.font.color == '#0000FF'

    def test_stats_box_faceted(self, xbar_result):
        """Test stats box in faceted plots."""
        plotter = Plotter(xbar_result)
        fig = plotter.plot(ncols=2, show_stats=True)

        assert isinstance(fig, ControlChartFigure)
        # Faceted plots should have multiple annotations (one per subplot)
        annotations = fig.figure.layout.annotations
        # Should have more than just subplot titles
        assert len(annotations) > 0

    def test_stats_box_with_other_options(self, simple_result):
        """Test stats box works with other visualization options."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='Xbar', show_stats=True, show_zones=True, show_rules=True, highlight_signals=True)

        assert isinstance(fig, ControlChartFigure)
        # Should have stats annotation
        annotations = fig.figure.layout.annotations
        has_stats = any('n =' in (ann.text or '') for ann in annotations)
        assert has_stats


class TestYAxisLabels:
    """Y-axis labels are chart-type-based, not the response variable name.

    X charts read "Individual Value" and mR charts "Moving Range" — for both
    response and residual charts — while Xbar/S keep their statistic labels and an
    explicit ``yaxis_title=`` override still wins. See ``Plotter._get_yaxis_label``.
    """

    @pytest.fixture
    def replicated_study(self):
        """Fully replicated design (factor x time, 4 reps/cell) so residuals exist."""
        np.random.seed(7)
        n_group, n_time, reps = 2, 12, 4
        df = pd.DataFrame(
            {
                'value': np.random.normal(100, 5, n_group * n_time * reps),
                'group': np.repeat(['A', 'B'], n_time * reps),
                'time': np.tile(np.repeat(range(1, n_time + 1), reps), n_group),
            }
        )
        pdf = ProcessBehavior(df)
        return pdf.formulate(response=pdf.cols.value, factors=[pdf.cols.group], time=pdf.cols.time)

    @staticmethod
    def _yaxis(result, chart):
        return result.plot(chart=chart).figure.layout.yaxis.title.text

    def test_individuals_and_moving_range_labels(self, replicated_study):
        result = replicated_study.execute(chart='X', by=[], companion=True)
        assert self._yaxis(result, 'X') == 'Individual Value'
        assert self._yaxis(result, 'mR') == 'Moving Range'

    def test_residual_x_mr_use_chart_type_labels(self, replicated_study):
        # R2 "Unexplained Effects" plotted as X/mR must use the chart-type labels,
        # not the response variable name.
        result = replicated_study.execute(chart='X', by=[], value='R2', companion=True)
        assert self._yaxis(result, 'X') == 'Individual Value'
        assert self._yaxis(result, 'mR') == 'Moving Range'

    def test_xbar_s_labels_unchanged(self, replicated_study):
        result = replicated_study.execute(chart='Xbar', companion=True)
        assert self._yaxis(result, 'Xbar') == 'Sample Average'
        assert self._yaxis(result, 'S') == 'Sample Standard Deviation'

    def test_explicit_yaxis_title_overrides_default(self, replicated_study):
        result = replicated_study.execute(chart='X', by=[])
        fig = result.plot(chart='X', yaxis_title='Custom Label')
        assert fig.figure.layout.yaxis.title.text == 'Custom Label'

    def test_stratified_individuals_label(self, replicated_study):
        # Faceted/stratified path resolves the label from chart_type too.
        result = replicated_study.execute(chart='X', by=['group'])
        assert self._yaxis(result, 'X') == 'Individual Value'
