"""
Tests for extracted plotting decoration modules.

Tests limits, zones, lane_boundaries, stats_box, and run_rules_viz
modules in isolation using lightweight Plotly figures.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from processbehavior.plotting import get_theme
from processbehavior.plotting.lane_boundaries import add_lane_boundaries
from processbehavior.plotting.limits import (
    add_stepped_limit_line,
    build_stepped_coordinates,
    format_limit_label,
)
from processbehavior.plotting.run_rules_viz import add_run_rules_visualization
from processbehavior.plotting.stats_box import (
    add_stats_box,
    build_stats_text,
)
from processbehavior.plotting.zones import (
    add_zone_shading,
    calculate_zone_boundaries,
)


@pytest.fixture
def theme():
    return get_theme('processbehavior')


# =========================================================================
# Limits
# =========================================================================

class TestFormatLimitLabel:

    def test_with_value_large(self):
        assert format_limit_label('UPL', 152.3456, True) == 'UPL = 152.3'

    def test_with_value_medium(self):
        assert format_limit_label('CL', 52.3456, True) == 'CL = 52.35'

    def test_with_value_small(self):
        assert format_limit_label('LPL', 5.12345, True) == 'LPL = 5.123'

    def test_without_value(self):
        assert format_limit_label('UPL', 999.0, False) == 'UPL'

    def test_negative_value(self):
        assert format_limit_label('LPL', -25.678, True) == 'LPL = -25.68'


class TestBuildSteppedCoordinates:

    def test_single_point(self):
        x, y = build_stepped_coordinates([1], [10.0])
        assert x == [1]
        assert y == [10.0]

    def test_two_points(self):
        x, y = build_stepped_coordinates([1, 2], [10.0, 12.0])
        # Expected: (1,10), (2,10), (2,12)
        assert x == [1, 2, 2]
        assert y == [10.0, 10.0, 12.0]

    def test_three_points(self):
        x, y = build_stepped_coordinates([1, 2, 3], [10.0, 12.0, 11.0])
        # Each point + horizontal segment to next x
        assert x == [1, 2, 2, 3, 3]
        assert y == [10.0, 10.0, 12.0, 12.0, 11.0]

    def test_constant_limits(self):
        x, y = build_stepped_coordinates([1, 2, 3], [5.0, 5.0, 5.0])
        assert y == [5.0, 5.0, 5.0, 5.0, 5.0]


class TestAddSteppedLimitLine:

    def test_adds_trace_to_single_figure(self, theme):
        fig = go.Figure()
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'upl': [10.0, 11.0, 10.5]
        })
        add_stepped_limit_line(
            fig, data, 'x', 'upl',
            'red', 'dash', 1.5, 'UPL', theme
        )
        assert len(fig.data) == 1
        assert fig.data[0].mode == 'lines'
        assert fig.data[0].name == 'UPL (varies)'

    def test_adds_trace_to_faceted_subplot(self, theme):
        fig = make_subplots(rows=1, cols=2)
        data = pd.DataFrame({
            'x': [1, 2, 3],
            'lpl': [2.0, 1.5, 2.5]
        })
        add_stepped_limit_line(
            fig, data, 'x', 'lpl',
            'blue', 'dot', 1.0, 'LPL', theme,
            row=1, col=2
        )
        assert len(fig.data) == 1
        assert fig.data[0].name == 'LPL (varies)'

    def test_noop_when_column_missing(self, theme):
        fig = go.Figure()
        data = pd.DataFrame({'x': [1, 2, 3]})
        add_stepped_limit_line(
            fig, data, 'x', 'upl',
            'red', 'dash', 1.5, 'UPL', theme
        )
        assert len(fig.data) == 0

    def test_hover_uses_limit_name(self, theme):
        fig = go.Figure()
        data = pd.DataFrame({'x': [1, 2], 'upl': [10.0, 11.0]})
        add_stepped_limit_line(
            fig, data, 'x', 'upl',
            'red', 'dash', 1.5, 'UPL', theme
        )
        assert 'UPL' in fig.data[0].hovertemplate


# =========================================================================
# Zones
# =========================================================================

class TestCalculateZoneBoundaries:

    def test_returns_five_zones(self, theme):
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        zones = calculate_zone_boundaries(stats, theme)
        assert zones is not None
        assert len(zones) == 5

    def test_zone_values_correct(self, theme):
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        zones = calculate_zone_boundaries(stats, theme)
        # sigma = (65-50)/3 = 5
        # Zone C: 45-55
        assert zones[0] == pytest.approx((45.0, 55.0, theme.zone_c_color))
        # Zone B upper: 55-60
        assert zones[1] == pytest.approx((55.0, 60.0, theme.zone_b_color))
        # Zone B lower: 40-45
        assert zones[2] == pytest.approx((40.0, 45.0, theme.zone_b_color))
        # Zone A upper: 60-65
        assert zones[3] == pytest.approx((60.0, 65.0, theme.zone_a_color))
        # Zone A lower: 35-40
        assert zones[4] == pytest.approx((35.0, 40.0, theme.zone_a_color))

    def test_returns_none_for_varying_limits(self, theme):
        stats = {'center': 50.0, 'upl': 'Varies', 'lpl': 35.0}
        assert calculate_zone_boundaries(stats, theme) is None

    def test_returns_none_for_missing_keys(self, theme):
        assert calculate_zone_boundaries({}, theme) is None
        assert calculate_zone_boundaries({'center': 50.0}, theme) is None


class TestAddZoneShading:

    def test_adds_shapes_to_single_figure(self, theme):
        fig = go.Figure()
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        add_zone_shading(fig, stats, theme)
        # add_hrect adds shapes
        assert len(fig.layout.shapes) == 5

    def test_adds_shapes_to_faceted_figure(self, theme):
        fig = make_subplots(rows=1, cols=2)
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        add_zone_shading(fig, stats, theme, row=1, col=2, ncols=2)
        assert len(fig.layout.shapes) == 5
        # Check axis refs for subplot 2
        shape = fig.layout.shapes[0]
        assert shape.xref == 'x2 domain'
        assert shape.yref == 'y2'

    def test_first_subplot_uses_bare_refs(self, theme):
        fig = make_subplots(rows=1, cols=2)
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        add_zone_shading(fig, stats, theme, row=1, col=1, ncols=2)
        shape = fig.layout.shapes[0]
        assert shape.xref == 'x domain'
        assert shape.yref == 'y'

    def test_noop_for_varying_limits(self, theme):
        fig = go.Figure()
        stats = {'center': 50.0, 'upl': 'Varies', 'lpl': 35.0}
        add_zone_shading(fig, stats, theme)
        assert len(fig.layout.shapes) == 0


# =========================================================================
# Lane Boundaries
# =========================================================================

class TestAddLaneBoundaries:

    def test_noop_on_none(self, theme):
        fig = go.Figure()
        add_lane_boundaries(fig, None, (0, 100), theme)
        assert len(fig.layout.shapes) == 0

    def test_noop_on_empty_list(self, theme):
        fig = go.Figure()
        add_lane_boundaries(fig, [], (0, 100), theme)
        assert len(fig.layout.shapes) == 0

    def test_adds_shapes_and_annotations(self, theme):
        fig = go.Figure()
        boundaries = [
            {'position': 5, 'label': 'A'},
            {'position': 10, 'label': 'B'},
        ]
        add_lane_boundaries(fig, boundaries, (0, 100), theme)
        assert len(fig.layout.shapes) == 2
        assert len(fig.layout.annotations) == 2

    def test_no_labels_when_disabled(self, theme):
        fig = go.Figure()
        boundaries = [{'position': 5, 'label': 'A'}]
        add_lane_boundaries(fig, boundaries, (0, 100), theme, show_labels=False)
        assert len(fig.layout.shapes) == 1
        assert len(fig.layout.annotations) == 0

    def test_faceted_passes_row_col(self, theme):
        fig = make_subplots(rows=1, cols=2)
        boundaries = [{'position': 5, 'label': 'A'}]
        add_lane_boundaries(fig, boundaries, (0, 100), theme, row=1, col=2)
        assert len(fig.layout.shapes) == 1
        assert len(fig.layout.annotations) == 1

    def test_boundary_without_label(self, theme):
        fig = go.Figure()
        boundaries = [{'position': 5}]
        add_lane_boundaries(fig, boundaries, (0, 100), theme)
        assert len(fig.layout.shapes) == 1
        assert len(fig.layout.annotations) == 0


# =========================================================================
# Stats Box
# =========================================================================

class TestBuildStatsText:

    def test_full_format(self):
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        data = pd.DataFrame({'val': range(20)})
        text = build_stats_text(stats, data, compact=False)
        assert 'n = 20' in text
        assert 'CL = ' in text
        assert 'UPL = ' in text
        assert 'LPL = ' in text
        assert '<br>' in text

    def test_compact_format(self):
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        data = pd.DataFrame({'val': range(10)})
        text = build_stats_text(stats, data, compact=True)
        assert 'n=10' in text
        assert 'CL=' in text
        assert '|' in text
        # Compact should not include UPL/LPL
        assert 'UPL' not in text

    def test_varies_excluded(self):
        stats = {'center': 'Varies', 'upl': 65.0, 'lpl': 35.0}
        data = pd.DataFrame({'val': range(5)})
        text = build_stats_text(stats, data, compact=False)
        assert 'CL' not in text.split('n = 5')[1] if 'n = 5' in text else True


class TestAddStatsBox:

    def test_single_chart_position(self, theme):
        fig = go.Figure()
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        data = pd.DataFrame({'val': range(20)})
        add_stats_box(fig, stats, data, theme)
        assert len(fig.layout.annotations) == 1
        ann = fig.layout.annotations[0]
        assert ann.x == 0.02
        assert ann.y == 0.98
        assert ann.borderpad == 6

    def test_faceted_chart_position(self, theme):
        fig = go.Figure()
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        data = pd.DataFrame({'val': range(20)})
        add_stats_box(fig, stats, data, theme, row=1, col=2, nrows=2, ncols=2)
        assert len(fig.layout.annotations) == 1
        ann = fig.layout.annotations[0]
        # For row=1, col=2 in 2x2 grid: x = 0.5 * (1/2) + 0.02 * (1/2) = 0.51
        assert ann.x == pytest.approx(0.5 * 0.02 + 0.5)  # (col-1)*col_width + 0.02*col_width
        assert ann.borderpad == 3
        assert ann.font.size == theme.stats_box_font_size - 1

    def test_noop_empty_stats(self, theme):
        fig = go.Figure()
        stats = {}
        data = pd.DataFrame({'val': range(5)})
        add_stats_box(fig, stats, data, theme)
        # Should still add something because n is always present
        assert len(fig.layout.annotations) == 1


# =========================================================================
# Run Rules Visualization
# =========================================================================

class TestAddRunRulesVisualization:

    def test_function_signature(self):
        """Verify function accepts explicit result parameter."""
        import inspect
        sig = inspect.signature(add_run_rules_visualization)
        params = list(sig.parameters.keys())
        assert 'result' in params
        assert 'row' in params
        assert 'col' in params

    def test_integration_with_real_result(self):
        """Integration test: run rules with a real analysis result."""
        np.random.seed(42)
        # Create data with factors (required by formulate)
        values = list(np.arange(1, 21, dtype=float)) * 2
        df = pd.DataFrame({
            'value': values,
            'group': ['A'] * 20 + ['B'] * 20,
            'time': list(range(20)) * 2

        })

        from processbehavior import ProcessBehavior
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.value,
            factors=[pdf.cols.group],
            time=pdf.cols.time
        )
        result = study.execute(chart='X', by=['group'])

        # Pick the first available chart
        chart_name = list(result.charts.keys())[0]
        chart_info = result.charts[chart_name]
        fig = go.Figure()
        data = chart_info['data']
        stats = chart_info['statistics']

        theme = get_theme('processbehavior')
        value_col = chart_info['metadata']['value_col']

        # Should not raise, even if no rules are violated
        add_run_rules_visualization(
            fig, data, stats, chart_name, value_col, None, theme,
            result=result
        )
