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

from processbehavior import ProcessDataFrame
from processbehavior.plotting import ControlChartFigure, Plotter
from processbehavior.plotting.themes import THEMES, apply_theme


class TestThemes:
    """Test theme definitions and application."""

    def test_theme_registry(self):
        """Test that all required themes are registered."""
        assert 'processbehavior' in THEMES
        assert 'minimal' in THEMES
        assert 'dark' in THEMES

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
        assert layout.plot_bgcolor == '#1e1e1e'
        assert layout.paper_bgcolor == '#2d2d2d'

    def test_apply_theme_invalid(self):
        """Test error handling for invalid theme."""
        fig = go.Figure()
        with pytest.raises(ValueError, match="Unknown theme"):
            apply_theme(fig, 'nonexistent')


class TestControlChartFigure:
    """Test ControlChartFigure wrapper class."""

    @pytest.fixture
    def sample_result(self):
        """Create sample analysis result."""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(100, 5, 30),
            'time': range(30)
        })
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(response_var='value')
        return analysis.calculate()

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


class TestPlotter:
    """Test Plotter class."""

    @pytest.fixture
    def simple_result(self):
        """Create simple I-mR analysis result."""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(100, 5, 30),
            'time': range(30)
        })
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(response_var='value')
        return analysis.calculate()

    @pytest.fixture
    def xbar_result(self):
        """Create Xbar analysis result."""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(100, 5, 100),
            'subgroup': np.repeat(range(20), 5),
            'time': range(100)
        })
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(response_var='value', grouping_vars=['subgroup'])
        return analysis.calculate()

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
        # Use 'all' which is the key for combined IMR chart
        fig = plotter.plot(chart='all')

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
        with pytest.raises(ValueError, match="Chart 'Invalid' not found"):
            plotter.plot(chart='Invalid')

    def test_plot_with_template(self, simple_result):
        """Test plotting with different templates."""
        plotter = Plotter(simple_result)

        # Test each template
        for template in ['processbehavior', 'minimal', 'dark']:
            fig = plotter.plot(chart='all', template=template)
            assert isinstance(fig, ControlChartFigure)

    def test_plot_with_custom_dimensions(self, simple_result):
        """Test plotting with custom dimensions."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='all', width=1200, height=600)

        assert fig.figure.layout.width == 1200
        assert fig.figure.layout.height == 600

    def test_plot_with_title(self, simple_result):
        """Test plotting with custom title."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='all', title='Custom Title')

        assert fig.figure.layout.title.text == 'Custom Title'

    def test_plot_without_limits(self, simple_result):
        """Test plotting without control limits."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='all', show_limits=False)

        assert isinstance(fig, ControlChartFigure)

    def test_plot_without_signals(self, simple_result):
        """Test plotting without signal highlighting."""
        plotter = Plotter(simple_result)
        fig = plotter.plot(chart='all', highlight_signals=False)

        assert isinstance(fig, ControlChartFigure)

    def test_helper_get_value_column(self, simple_result):
        """Test value column detection from metadata."""
        plotter = Plotter(simple_result)

        # Create chart_info with metadata (as charts now provide)
        chart_info = {
            'data': pd.DataFrame({'xbar': [1, 2, 3]}),
            'statistics': {},
            'metadata': {
                'chart_type': 'Xbar',
                'value_col': 'xbar',
                'center_col': 'center'
            }
        }

        value_col = plotter._get_value_column(chart_info, 'Xbar')
        assert value_col == 'xbar'

    def test_helper_get_x_column(self, simple_result):
        """Test x-axis column detection."""
        plotter = Plotter(simple_result)
        data = pd.DataFrame({'x': [1, 2, 3]})

        x_col = plotter._get_x_column(data)
        assert x_col == 'x'

    def test_helper_get_center_key(self, simple_result):
        """Test centerline key detection."""
        plotter = Plotter(simple_result)

        # All chart types now use 'center'
        stats = {'center': 100, 'ucl': 115, 'lcl': 85}
        center_key = plotter._get_center_key(stats)
        assert center_key == 'center'

        # Test with missing center
        stats = {'ucl': 15, 'lcl': 0}
        center_key = plotter._get_center_key(stats)
        assert center_key is None


class TestAnalysisResultIntegration:
    """Test plot() integration with AnalysisResult."""

    @pytest.fixture
    def result(self):
        """Create sample analysis result."""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(100, 5, 30),
            'time': range(30)
        })
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(response_var='value')
        return analysis.calculate()

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
        fig = result.plot(
            template='minimal',
            width=800,
            highlight_signals=False
        )
        assert isinstance(fig, ControlChartFigure)
        assert fig.figure.layout.width == 800


class TestFacetedPlotting:
    """Test faceted/stratified chart plotting."""

    @pytest.fixture
    def xbar_result(self):
        """Create Xbar result with multiple charts."""
        np.random.seed(42)
        df = pd.DataFrame({
            'value': np.random.normal(100, 5, 100),
            'subgroup': np.repeat(range(20), 5),
            'time': range(100)
        })
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(response_var='value', grouping_vars=['subgroup'])
        return analysis.calculate()

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
