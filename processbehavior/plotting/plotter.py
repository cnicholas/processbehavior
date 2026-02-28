"""
Main plotting interface for ProcessBehavior analysis results.

Orchestrates chart creation with built-in support for single and faceted charts.
"""

from __future__ import annotations

import html
import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data_preparation import encode_rsg
from ..spc_constants import normalize_chart_name
from .control_chart import ControlChartFigure
from .lane_boundaries import add_lane_boundaries
from .limits import add_stepped_limit_line, format_limit_label
from .run_rules_viz import add_run_rules_visualization
from .stats_box import add_stats_box
from .themes import ChartTheme, apply_theme, get_theme
from .zones import add_zone_shading

if TYPE_CHECKING:
    from processbehavior.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

_EFFECTS_CHART_TYPES = frozenset({
    'Effects', 'MainEffects', 'TimeEffects',
    'TimeInteraction', 'FactorInteraction',
})


# ============================================================================
# Pure Python Normal Distribution Functions (scipy replacement)
# ============================================================================

def _normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Calculate normal probability density function.

    Pure Python implementation to avoid scipy dependency.

    Parameters
    ----------
    x : float
        Point at which to evaluate the PDF
    mu : float, default 0.0
        Mean of the distribution
    sigma : float, default 1.0
        Standard deviation of the distribution

    Returns
    -------
    float
        PDF value at x
    """
    coefficient = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * math.exp(exponent)


def _normal_ppf(p: float) -> float:
    """
    Calculate normal percent point function (inverse CDF / quantile function).

    Uses Acklam's algorithm for rational approximation to the
    inverse cumulative normal distribution. Accurate to approximately
    1.15e-9 in absolute error for p in (0, 1).

    Parameters
    ----------
    p : float
        Probability value in (0, 1)

    Returns
    -------
    float
        Quantile (z-score) corresponding to probability p

    References
    ----------
    Acklam, P. J. (2010). An algorithm for computing the inverse normal
    cumulative distribution function.
    https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
    """
    # Coefficients for rational approximation
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    # Define break-points
    p_low = 0.02425
    p_high = 1 - p_low

    if p <= 0 or p >= 1:
        if p == 0:
            return float('-inf')
        elif p == 1:
            return float('inf')
        else:
            raise ValueError(f"p must be in (0, 1), got {p}")

    # Rational approximation for lower region
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    # Rational approximation for upper region
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    # Rational approximation for central region
    q = p - 0.5
    r = q * q
    return (
        ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
    ) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


class Plotter:
    """
    Unified plotting interface for ProcessBehavior analysis results.

    This class orchestrates chart creation with built-in support for:
    - Single and faceted control charts
    - Automatic detection of chart types
    - Residual and effects visualization
    - Interactive and static export

    Examples
    --------
    Simple plotting:

    >>> plotter = Plotter(result)
    >>> fig = plotter.plot()
    >>> fig.show()

    Faceted charts:

    >>> fig = plotter.plot(facet_by='Operator')

    Customize appearance:

    >>> fig = plotter.plot(
    ...     chart='Xbar',
    ...     theme='minimal',
    ...     highlight_signals=True
    ... )
    """

    def __init__(self, analysis_result: AnalysisResult):
        """
        Initialize plotter with an AnalysisResult.

        Parameters
        ----------
        analysis_result : AnalysisResult
            The result object from analysis
        """
        self.result = analysis_result
        self.charts = analysis_result.charts
        self.summary = analysis_result.summary
        self._theme: ChartTheme | None = None  # Set during plot()

    def plot(  # noqa: C901
        self,
        chart: str | None = None,
        facet: bool = False,
        facet_by: str | None = None,
        ncols: int = 2,
        highlight_signals: bool = True,
        show_limits: bool = True,
        show_limit_values: bool = True,
        show_zones: bool = False,
        show_rules: bool = False,
        show_stats: bool = False,
        theme: str | ChartTheme = 'processbehavior',
        width: int = 1000,
        height: int | None = None,
        aspect_ratio: float | None = None,
        title: str | None = None,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
        shared_yaxis: bool = True,
        yaxis_padding: float = 0.05,
        vertical_spacing: float = 0.15
    ) -> ControlChartFigure:
        """
        Create control chart visualization.

        This is the main entry point for plotting. It automatically
        determines the best visualization based on your data structure.

        Parameters
        ----------
        chart : str, optional
            Specific chart to plot ('Xbar', 'S', 'XmR', etc.)
            If None, plots all available charts
        facet : bool, default False
            Whether to create faceted plot for stratified data
        facet_by : str, optional
            Variable to facet by (overrides auto-detection)
        ncols : int, default 2
            Number of columns in faceted layout
        highlight_signals : bool, default True
            Whether to highlight points beyond control limits
        show_limits : bool, default True
            Whether to show control limit lines
        show_limit_values : bool, default True
            Whether to show numeric values in limit labels (e.g., "UPL = 52.34")
        show_zones : bool, default False
            Whether to show zone shading (A/B/C zones at ±1σ, ±2σ, ±3σ).
            Zone colors are controlled by the theme's zone_a_color, zone_b_color,
            zone_c_color, and zone_opacity properties.
        show_rules : bool, default False
            Whether to show additional run rules (Western Electric Rules 2-8)
        show_stats : bool, default False
            Whether to show a statistics box with CL, UPL, LPL, and n values
        theme : str or ChartTheme, default 'processbehavior'
            Visual theme. Can be a theme name string ('processbehavior', 'ggplot',
            'minimal', 'dark', 'publication') or a ChartTheme instance for full
            customization of colors, sizes, and fonts.
        width : int, default 1000
            Figure width in pixels
        height : int, optional
            Figure height in pixels (auto-calculated if None)
        aspect_ratio : float, optional
            Width-to-height ratio (e.g., 16/9 = 1.78, 4/3 = 1.33).
            If specified, overrides height calculation.
            Common presets: 1.5 (landscape), 1.0 (square), 0.75 (portrait)
        title : str, optional
            Custom title for the figure. If None, auto-generates a descriptive
            title like "Xbar Chart of {response_var} by {grouping_var}"
        xaxis_title : str, optional
            Custom x-axis label. If None, uses time variable name or "Observation"
        yaxis_title : str, optional
            Custom y-axis label. If None, uses response variable name
        shared_yaxis : bool, default True
            Whether faceted charts share the same y-axis range. When True,
            all facets use a common scale for honest cross-facet comparison.
            When False, each facet scales independently to maximize visibility.
        yaxis_padding : float, default 0.05
            Padding added above and below the y-axis range as a fraction of
            the data range (e.g., 0.05 = 5% padding on each side).
        vertical_spacing : float, default 0.15
            Vertical spacing between rows in faceted layouts, as a fraction
            of the figure height (0.15 = 15%). Increase if subplot titles
            overlap with x-axis labels from the row above.

        Returns
        -------
        ControlChartFigure
            Interactive figure object with .show(), .save_html(), etc.

        Raises
        ------
        ValueError
            If ``chart`` is specified but not found in available charts.
            Use ``list_charts()`` to see available options.

        Examples
        --------
        Auto-plot everything:

        >>> fig = plotter.plot()

        Specific chart:

        >>> fig = plotter.plot(chart='Xbar')

        Faceted by operator:

        >>> fig = plotter.plot(facet_by='Operator', ncols=3)

        Custom styling:

        >>> fig = plotter.plot(
        ...     theme='dark',
        ...     highlight_signals=True,
        ...     show_rules=True
        ... )
        """
        # Resolve theme
        if isinstance(theme, str):
            self._theme = get_theme(theme)
        else:
            self._theme = theme

        # Normalize chart name for case-insensitive lookup
        if chart:
            chart = normalize_chart_name(chart)

        # Handle effects charts (special chart types not in self.charts)
        if chart in _EFFECTS_CHART_TYPES:
            return self._plot_effects_chart(
                chart_type=chart,
                width=width,
                height=height,
                title=title
            )

        # Validate inputs
        if chart and chart not in self.charts:
            available = list(self.charts.keys())
            raise ValueError(
                f"Chart '{chart}' not found.\n"
                f"Available charts: {available}\n"
                f"Hint: Use plotter.list_charts() to see all options"
            )

        # Determine what to plot
        if chart:
            chart_info = self.charts[chart]
            # Check if this is a stratified chart that needs expansion
            if 'strata' in chart_info and chart_info['strata']:
                # Expand stratified chart into per-stratum charts
                charts_to_plot = self._get_stratified_charts()
                # Filter to only the requested chart's strata
                charts_to_plot = {
                    k: v for k, v in charts_to_plot.items()
                    if v.get('metadata', {}).get('original_chart') == chart
                }
            else:
                charts_to_plot = {chart: chart_info}
        elif facet or facet_by or self.summary.get('is_stratified', False):
            # Auto-detect stratified charts
            charts_to_plot = self._get_stratified_charts()
        else:
            # Plot all standard charts
            charts_to_plot = {
                k: v for k, v in self.charts.items()
                if k in ['Xbar', 'S', 'XmR', 'R', 'Histogram']
            }

            # If no standard charts found but only one chart exists, use it
            if not charts_to_plot and len(self.charts) == 1:
                charts_to_plot = dict(self.charts)

        if not charts_to_plot:
            raise ValueError(
                "No charts available to plot.\n"
                "Hint: Check result.charts to see available charts"
            )

        # Calculate height
        if height is None:
            if aspect_ratio is not None:
                # Use aspect ratio to calculate height
                height = int(width / aspect_ratio)
            else:
                # Auto-calculate based on number of charts
                # Use more height per row for faceted plots to accommodate titles + labels
                n_charts = len(charts_to_plot)
                nrows = (n_charts + ncols - 1) // ncols
                if n_charts == 1:
                    height = 400
                else:
                    # Faceted plots need more height per row for titles and spacing
                    height_per_row = 450 if nrows <= 3 else 400
                    height = nrows * height_per_row

        # Create figure
        if len(charts_to_plot) == 1:
            fig = self._plot_single_chart(
                list(charts_to_plot.values())[0],
                list(charts_to_plot.keys())[0],
                highlight_signals=highlight_signals,
                show_limits=show_limits,
                show_limit_values=show_limit_values,
                show_zones=show_zones,
                show_rules=show_rules,
                show_stats=show_stats,
                width=width,
                height=height,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title
            )
        else:
            # Auto-detect if charts are different types (e.g., Xbar+S)
            # Different chart types have fundamentally different scales
            # and should not share y-axis
            effective_shared_yaxis = shared_yaxis
            if shared_yaxis:
                base_types = set(self._get_base_chart_type(name) for name in charts_to_plot)
                if len(base_types) > 1:
                    # Different chart types - disable y-axis sharing
                    effective_shared_yaxis = False

            fig = self._plot_faceted(
                charts_to_plot,
                ncols=ncols,
                highlight_signals=highlight_signals,
                show_limits=show_limits,
                show_limit_values=show_limit_values,
                show_zones=show_zones,
                show_rules=show_rules,
                show_stats=show_stats,
                width=width,
                height=height,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                shared_yaxis=effective_shared_yaxis,
                yaxis_padding=yaxis_padding,
                vertical_spacing=vertical_spacing
            )

        # Apply theme
        fig = apply_theme(fig, self._theme)

        # Set title
        if title:
            fig.update_layout(title=title)
        elif len(charts_to_plot) == 1:
            chart_name = list(charts_to_plot.keys())[0]
            auto_title = self._generate_title(chart_name)
            fig.update_layout(title=auto_title)

        # Wrap in our custom figure class
        return ControlChartFigure(fig, self.result)

    def _plot_single_chart(  # noqa: C901
        self,
        chart_info: dict,
        chart_name: str,
        highlight_signals: bool,
        show_limits: bool,
        show_limit_values: bool,
        show_zones: bool,
        show_rules: bool,
        show_stats: bool,
        width: int,
        height: int,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None
    ) -> go.Figure:
        """Create a single control chart.

        Renders one chart panel with data trace, control limits, optional zone
        shading, signal highlighting, run-rules markers, and stats box.

        Parameters are grouped by concept:
        - Data columns: ``chart_info`` carries data, statistics, and metadata
          including value_col and x_col (auto-detected).
        - Layout: ``width``, ``height``, ``xaxis_title``, ``yaxis_title``.
        - Decoration toggles: ``show_limits``, ``show_zones``, ``show_rules``,
          ``show_stats``, ``highlight_signals``, ``show_limit_values``.

        Edge cases:
        - When limits vary (stratified stats), stepped lines are drawn instead
          of horizontal lines, and a text annotation explains the variation.
        - When ``chart_info`` metadata indicates a histogram, rendering is
          delegated to ``_plot_histogram()``.
        - Per-stratum lanes get separate lane boundaries and labels.
        """
        # Check if this is a histogram - route to histogram-specific rendering
        metadata = chart_info.get('metadata', {})
        if metadata.get('chart_type') == 'Histogram':
            return self._plot_histogram(
                chart_info,
                chart_name,
                show_stats=show_stats,
                width=width,
                height=height,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title
            )

        data = chart_info['data']
        stats = chart_info['statistics']

        fig = go.Figure()

        # Determine value column from metadata
        value_col = self._get_value_column(chart_info, chart_name)
        x_col = self._get_x_column(data)

        # Determine axis labels
        x_label = xaxis_title or self._get_xaxis_label(x_col, chart_name)
        y_label = yaxis_title or self._get_yaxis_label(value_col)

        theme = self._theme

        # Zone shading (add first so it's behind other elements)
        if show_zones and theme.zone_opacity > 0:
            add_zone_shading(fig, stats, theme)

        # Main data trace
        fig.add_trace(go.Scatter(
            x=data[x_col] if x_col in data.columns else data.index,
            y=data[value_col],
            mode='lines+markers',
            name=self._get_data_legend_name(),
            marker=dict(size=theme.data_marker_size, color=theme.data_color),
            line=dict(color=theme.data_color, width=theme.data_line_width),
            opacity=theme.data_opacity,
            hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'
        ))

        # Process limits
        if show_limits:
            limits_vary = stats.get('upl') == 'Varies' or stats.get('lpl') == 'Varies'

            # UPL
            if 'upl' in stats:
                if stats['upl'] != 'Varies':
                    # Fixed limit - draw horizontal line
                    upl_label = format_limit_label('UPL', stats['upl'], show_limit_values)
                    fig.add_hline(
                        y=stats['upl'],
                        line_dash=theme.limit_line_dash,
                        line_color=theme.ucl_color,
                        line_width=theme.limit_line_width,
                        annotation_text=upl_label,
                        annotation_position='right',
                        annotation_font_size=theme.annotation_font_size
                    )
                elif 'upl' in data.columns:
                    # Varying limit - draw stepped line
                    add_stepped_limit_line(
                        fig, data, x_col, 'upl',
                        theme.ucl_color, theme.limit_line_dash, theme.limit_line_width,
                        'UPL', theme
                    )

            # LPL
            if 'lpl' in stats:
                if stats['lpl'] != 'Varies':
                    # Fixed limit - draw horizontal line
                    lpl_label = format_limit_label('LPL', stats['lpl'], show_limit_values)
                    fig.add_hline(
                        y=stats['lpl'],
                        line_dash=theme.limit_line_dash,
                        line_color=theme.lcl_color,
                        line_width=theme.limit_line_width,
                        annotation_text=lpl_label,
                        annotation_position='right',
                        annotation_font_size=theme.annotation_font_size
                    )
                elif 'lpl' in data.columns:
                    # Varying limit - draw stepped line
                    add_stepped_limit_line(
                        fig, data, x_col, 'lpl',
                        theme.lcl_color, theme.limit_line_dash, theme.limit_line_width,
                        'LPL', theme
                    )

            # Add annotation when limits vary
            if limits_vary:
                metadata = chart_info.get('metadata', {})
                if metadata.get('phased'):
                    vary_text = "Process limits computed per phase"
                else:
                    vary_text = "Process limits vary by subgroup size (n)"
                fig.add_annotation(
                    text=vary_text,
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    bgcolor="rgba(255,255,255,0.8)",
                    borderpad=3
                )

            # Centerline
            center_key = self._get_center_key(stats)
            if center_key and center_key in stats:
                if stats[center_key] == 'Varies':
                    # Stepped center line (phased limits)
                    if 'center' in data.columns:
                        add_stepped_limit_line(
                            fig, data, x_col, 'center',
                            theme.center_color, 'solid',
                            theme.center_line_width,
                            'CL', theme
                        )
                else:
                    center_label = format_limit_label(
                        'CL', stats[center_key], show_limit_values
                    )
                    fig.add_hline(
                        y=stats[center_key],
                        line_color=theme.center_color,
                        line_width=theme.center_line_width,
                        annotation_text=center_label,
                        annotation_position='right',
                        annotation_font_size=theme.annotation_font_size
                    )

        # Highlight signals (Rule 1 - beyond limits)
        # Uses same marker style/size as data, just red color - no legend entry
        if highlight_signals and 'beyond_limits' in data.columns:
            signals = data[data['beyond_limits'] != 0]
            if not signals.empty:
                fig.add_trace(go.Scatter(
                    x=signals[x_col] if x_col in data.columns else signals.index,
                    y=signals[value_col],
                    mode='markers',
                    name='Beyond Limits',
                    marker=dict(
                        size=theme.data_marker_size,  # Same size as data
                        color=theme.signal_color,
                        symbol=theme.signal_marker_symbol,
                        line=dict(
                            width=theme.signal_marker_line_width,
                            color=theme.signal_marker_line_color
                        )
                    ),
                    hovertemplate='Beyond Limits<br>%{x}<br>%{y:.3f}<extra></extra>',
                    showlegend=False  # Color is enough, no legend clutter
                ))

        # Run rules visualization (Rules 2-8)
        metadata = chart_info.get('metadata', {})
        if show_rules and metadata.get('run_rules_applicable', True):
            add_run_rules_visualization(
                fig, data, stats, chart_name, value_col, x_col, theme,
                result=self.result
            )

        # Stats box
        if show_stats:
            add_stats_box(fig, stats, data, theme)

        # Lane boundaries (vertical separators for collapsed factors)
        metadata = chart_info.get('metadata', {})
        lane_boundaries = metadata.get('lane_boundaries')
        if lane_boundaries:
            # Calculate y-range for vertical lines
            y_min = data[value_col].min()
            y_max = data[value_col].max()
            y_padding = (y_max - y_min) * 0.05
            y_range = (y_min - y_padding, y_max + y_padding)
            add_lane_boundaries(fig, lane_boundaries, y_range, theme)

        # Time tick labels (replace integer indices with time values)
        self._apply_time_tick_labels(fig, data, metadata)

        # Layout
        fig.update_layout(
            width=width,
            height=height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            hovermode='x unified',
            showlegend=True
        )

        return fig

    def _plot_faceted(  # noqa: C901
        self,
        charts: dict,
        ncols: int,
        highlight_signals: bool,
        show_limits: bool,
        show_limit_values: bool,
        show_zones: bool,
        show_rules: bool,
        show_stats: bool,
        width: int,
        height: int,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
        shared_yaxis: bool = True,
        yaxis_padding: float = 0.05,
        vertical_spacing: float = 0.15
    ) -> go.Figure:
        """Create faceted control charts."""
        n_charts = len(charts)
        nrows = (n_charts + ncols - 1) // ncols

        # Generate descriptive subplot titles
        subplot_titles = [
            self._generate_subplot_title(name, charts[name]) for name in charts
        ]

        # Calculate spacing dynamically to avoid Plotly errors and overlaps
        # Constraint: spacing <= 1 / (n - 1) where n is rows or cols
        # Use more vertical spacing to accommodate subplot titles + x-axis labels
        if nrows > 1:
            max_v_spacing = 1.0 / (nrows - 1) - 0.01  # Small buffer
            # Use more spacing for fewer rows (more room per subplot)
            desired_v_spacing = 0.20 if nrows <= 3 else 0.18
            v_spacing = min(max(vertical_spacing, desired_v_spacing), max_v_spacing)
        else:
            v_spacing = vertical_spacing

        if ncols > 1:
            max_h_spacing = 1.0 / (ncols - 1) - 0.01
            h_spacing = min(0.10, max_h_spacing)
        else:
            h_spacing = 0.10

        # Create subplot grid
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subplot_titles,
            vertical_spacing=v_spacing,
            horizontal_spacing=h_spacing
        )

        # Check if all charts are histograms (needed for axis labels and binning)
        all_histograms = all(
            chart_info.get('metadata', {}).get('chart_type') == 'Histogram'
            for chart_info in charts.values()
        )

        # Determine axis labels (same for all subplots)
        first_chart_info = next(iter(charts.values()))
        first_chart_data = first_chart_info['data']

        if all_histograms:
            # For histograms, x-axis is the value column, y-axis is count
            value_col = first_chart_info.get('metadata', {}).get('value_col')
            x_label = xaxis_title or value_col
            y_label = yaxis_title or 'Count'
        else:
            # For control charts, use standard x-axis label
            x_col = self._get_x_column(first_chart_data)
            first_chart_name = next(iter(charts.keys()))
            x_label = xaxis_title or self._get_xaxis_label(x_col, first_chart_name)
            y_label = yaxis_title or self._get_yaxis_label(None)

        # Calculate global y-range for shared axis
        # Histograms need count-based y-range, control charts need value-based
        global_y_range = None
        histogram_y_range = None
        histogram_bin_edges = None

        if all_histograms:
            # Calculate shared bin edges for consistent binning across facets
            histogram_bin_edges, _ = self._calculate_histogram_bin_edges(charts)

        if shared_yaxis:
            if all_histograms:
                # For histograms, calculate y-range based on bin counts with shared edges
                histogram_y_range = self._calculate_histogram_yrange(
                    charts, histogram_bin_edges
                )
            else:
                # For control charts (or mixed), use data value range
                global_y_range = self._calculate_global_yrange(charts, yaxis_padding)

        # Get theme
        theme = self._theme

        # Plot each chart
        for idx, (chart_name, chart_info) in enumerate(charts.items()):
            row = idx // ncols + 1
            col = idx % ncols + 1

            data = chart_info['data']
            stats = chart_info['statistics']
            metadata = chart_info.get('metadata', {})

            # Check if this is a histogram
            is_histogram = metadata.get('chart_type') == 'Histogram'

            if is_histogram:
                # Histogram-specific rendering
                import numpy as np

                value_col = metadata.get('value_col')
                bins = metadata.get('bins', 10)

                # Use shared bin edges for consistent binning across facets
                if histogram_bin_edges is not None:
                    bin_width = histogram_bin_edges[1] - histogram_bin_edges[0]
                    fig.add_trace(
                        go.Histogram(
                            x=data[value_col],
                            xbins=dict(
                                start=histogram_bin_edges[0],
                                end=histogram_bin_edges[-1],
                                size=bin_width
                            ),
                            name=chart_name,
                            marker_color=theme.data_color,
                            opacity=0.75,
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )
                else:
                    fig.add_trace(
                        go.Histogram(
                            x=data[value_col],
                            nbinsx=bins,
                            name=chart_name,
                            marker_color=theme.data_color,
                            opacity=0.75,
                            showlegend=False
                        ),
                        row=row,
                        col=col
                    )

                # Add stats lines for histogram (use per-stratum stats)
                if show_stats:
                    mean = stats.get('mean')
                    std = stats.get('std')
                    n = stats.get('n', 0)

                    # Mean line - only if finite
                    if mean is not None and np.isfinite(mean):
                        fig.add_vline(
                            x=mean,
                            line_dash="solid",
                            line_color=theme.center_color,
                            line_width=2,
                            row=row, col=col
                        )

                        # Std deviation lines (±1, ±2, ±3) - only if n >= 2 and std is finite and > 0
                        if std is not None and n >= 2 and np.isfinite(std) and std > 0:
                            for mult in [1, 2, 3]:
                                fig.add_vline(
                                    x=mean + mult * std,
                                    line_dash="dash",
                                    line_color="orange",
                                    line_width=1,
                                    row=row, col=col
                                )
                                fig.add_vline(
                                    x=mean - mult * std,
                                    line_dash="dash",
                                    line_color="orange",
                                    line_width=1,
                                    row=row, col=col
                                )
                continue  # Skip control chart rendering for histograms

            # Zone shading for this subplot (add first so it's behind data)
            if show_zones and theme.zone_opacity > 0:
                add_zone_shading(fig, stats, theme, row=row, col=col, ncols=ncols)

            value_col = self._get_value_column(chart_info, chart_name)
            x_col = self._get_x_column(data)

            # Main trace
            x_data = data[x_col] if x_col in data.columns else data.index
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=data[value_col],
                    mode='lines+markers',
                    name=chart_name,
                    marker=dict(size=theme.facet_marker_size, color=theme.data_color),
                    line=dict(color=theme.data_color, width=theme.facet_line_width),
                    opacity=theme.data_opacity,
                    showlegend=False,
                    hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'
                ),
                row=row,
                col=col
            )

            # Control limits as shapes with annotations
            if show_limits:
                x_range = [x_data.min(), x_data.max()]

                # UPL
                if 'upl' in stats and stats['upl'] != 'Varies':
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats['upl'], y1=stats['upl'],
                        line=dict(
                            color=theme.ucl_color,
                            dash=theme.limit_line_dash,
                            width=theme.limit_line_width
                        ),
                        row=row, col=col
                    )
                    # Add UPL label annotation
                    upl_label = format_limit_label('UPL', stats['upl'], show_limit_values)
                    fig.add_annotation(
                        x=x_range[1],
                        y=stats['upl'],
                        text=upl_label,
                        showarrow=False,
                        xanchor='left',
                        font=dict(size=theme.annotation_font_size, color=theme.ucl_color),
                        row=row, col=col
                    )
                elif 'upl' in data.columns:
                    # Varying limit - draw stepped line
                    add_stepped_limit_line(
                        fig, data, x_col, 'upl',
                        theme.ucl_color, theme.limit_line_dash, theme.limit_line_width,
                        'UPL', theme, row=row, col=col
                    )

                # LPL
                if 'lpl' in stats and stats['lpl'] != 'Varies':
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats['lpl'], y1=stats['lpl'],
                        line=dict(
                            color=theme.lcl_color,
                            dash=theme.limit_line_dash,
                            width=theme.limit_line_width
                        ),
                        row=row, col=col
                    )
                    # Add LPL label annotation
                    lpl_label = format_limit_label('LPL', stats['lpl'], show_limit_values)
                    fig.add_annotation(
                        x=x_range[1],
                        y=stats['lpl'],
                        text=lpl_label,
                        showarrow=False,
                        xanchor='left',
                        font=dict(size=theme.annotation_font_size, color=theme.lcl_color),
                        row=row, col=col
                    )
                elif 'lpl' in data.columns:
                    # Varying limit - draw stepped line
                    add_stepped_limit_line(
                        fig, data, x_col, 'lpl',
                        theme.lcl_color, theme.limit_line_dash, theme.limit_line_width,
                        'LPL', theme, row=row, col=col
                    )

                # Centerline
                center_key = self._get_center_key(stats)
                if center_key and center_key in stats:
                    if stats[center_key] == 'Varies':
                        # Stepped center line (phased limits)
                        if 'center' in data.columns:
                            add_stepped_limit_line(
                                fig, data, x_col, 'center',
                                theme.center_color, 'solid',
                                theme.center_line_width,
                                'CL', theme, row=row, col=col
                            )
                    else:
                        fig.add_shape(
                            type='line',
                            x0=x_range[0], x1=x_range[1],
                            y0=stats[center_key], y1=stats[center_key],
                            line=dict(color=theme.center_color, width=theme.center_line_width),
                            row=row, col=col
                        )
                        # Add CL label annotation
                        cl_label = format_limit_label('CL', stats[center_key], show_limit_values)
                        fig.add_annotation(
                            x=x_range[1],
                            y=stats[center_key],
                            text=cl_label,
                            showarrow=False,
                            xanchor='left',
                            font=dict(size=theme.annotation_font_size, color=theme.center_color),
                            row=row, col=col
                        )

            # Signals (Rule 1 - beyond limits) - same size as facet data points
            if highlight_signals and 'beyond_limits' in data.columns:
                signals = data[data['beyond_limits'] != 0]
                if not signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=signals[x_col] if x_col in data.columns else signals.index,
                            y=signals[value_col],
                            mode='markers',
                            marker=dict(
                                size=theme.facet_marker_size,  # Same size as facet data
                                color=theme.signal_color,
                                symbol=theme.signal_marker_symbol
                            ),
                            showlegend=False,
                            hovertemplate='Signal<br>%{x}<br>%{y:.3f}<extra></extra>'
                        ),
                        row=row,
                        col=col
                    )

            # Run rules visualization (Rules 2-8)
            if show_rules and metadata.get('run_rules_applicable', True):
                add_run_rules_visualization(
                    fig, data, stats, chart_name, value_col, x_col, theme,
                    result=self.result, row=row, col=col
                )

            # Stats box for faceted charts
            if show_stats:
                add_stats_box(fig, stats, data, theme, row=row, col=col, nrows=nrows, ncols=ncols)

            # Lane boundaries for this facet
            metadata = chart_info.get('metadata', {})
            all_lane_boundaries = metadata.get('lane_boundaries')
            if all_lane_boundaries:
                # Lane boundaries are stored as dict keyed by stratum name
                if isinstance(all_lane_boundaries, dict):
                    lane_boundaries = all_lane_boundaries.get(chart_name)
                else:
                    lane_boundaries = all_lane_boundaries  # list for single chart

                if lane_boundaries:
                    # Calculate y-range for this facet
                    y_min = data[value_col].min()
                    y_max = data[value_col].max()
                    y_padding = (y_max - y_min) * 0.05
                    y_range = (y_min - y_padding, y_max + y_padding)
                    add_lane_boundaries(
                        fig, lane_boundaries, y_range, theme, row=row, col=col
                    )

            # Time tick labels for this facet
            self._apply_time_tick_labels(fig, data, metadata, row=row, col=col)

        # Update layout with axis labels
        fig.update_layout(
            width=width,
            height=height,
            hovermode='closest'
        )

        # Add axis labels and line styling to all subplots
        # (Template only applies to xaxis/yaxis, not xaxis2/yaxis2/etc.)
        fig.update_xaxes(
            title_text=x_label,
            showline=theme.show_axis_line,
            linecolor=theme.axis_line_color,
            linewidth=theme.axis_line_width,
            showgrid=theme.show_grid,
            gridcolor=theme.grid_color,
            gridwidth=theme.grid_width,
        )
        fig.update_yaxes(
            title_text=y_label,
            showline=theme.show_axis_line,
            linecolor=theme.axis_line_color,
            linewidth=theme.axis_line_width,
            showgrid=theme.show_grid,
            gridcolor=theme.grid_color,
            gridwidth=theme.grid_width,
        )

        # Apply shared y-axis range if calculated
        # Use histogram-specific range for histogram facets, otherwise use global range
        if histogram_y_range is not None:
            fig.update_yaxes(range=histogram_y_range)
        elif global_y_range is not None:
            fig.update_yaxes(range=global_y_range)

        return fig

    def list_charts(self) -> list[str]:
        """Get list of available charts."""
        return list(self.charts.keys())

    def _plot_histogram(
        self,
        chart_info: dict,
        chart_name: str,
        show_stats: bool = True,
        width: int = 1000,
        height: int = 400,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None
    ) -> go.Figure:
        """
        Render histogram with optional mean/std deviation lines.

        Parameters
        ----------
        chart_info : dict
            Chart info with 'data', 'statistics', and 'metadata'
        chart_name : str
            Chart display name
        show_stats : bool, default True
            Whether to show mean and std deviation lines
        width : int, default 1000
            Figure width in pixels
        height : int, default 400
            Figure height in pixels
        xaxis_title : str, optional
            Custom x-axis label
        yaxis_title : str, optional
            Custom y-axis label

        Returns
        -------
        go.Figure
            Plotly figure with histogram
        """
        import numpy as np

        data = chart_info['data']
        metadata = chart_info.get('metadata', {})
        stats = chart_info.get('statistics', {})

        value_col = metadata.get('value_col')
        bins = metadata.get('bins', 10)

        theme = self._theme

        fig = go.Figure()

        # Add histogram trace
        fig.add_trace(go.Histogram(
            x=data[value_col],
            nbinsx=bins,
            name=value_col,
            marker_color=theme.data_color,
            opacity=0.75
        ))

        if show_stats:
            mean = stats.get('mean')
            std = stats.get('std')
            n = stats.get('n', 0)

            # Mean line - only if finite
            if mean is not None and np.isfinite(mean):
                fig.add_vline(
                    x=mean,
                    line_dash="solid",
                    line_color=theme.center_color,
                    line_width=2,
                    annotation_text=f"Mean: {mean:.3f}",
                    annotation_position="top",
                    annotation_font_size=theme.annotation_font_size
                )

                # Std deviation lines (±1, ±2, ±3) - only if n >= 2 and std is finite and > 0
                if std is not None and n >= 2 and np.isfinite(std) and std > 0:
                    for mult in [1, 2, 3]:
                        fig.add_vline(
                            x=mean + mult * std,
                            line_dash="dash",
                            line_color="orange",
                            line_width=1,
                            annotation_text=f"+{mult}σ",
                            annotation_position="top",
                            annotation_font_size=theme.annotation_font_size
                        )
                        fig.add_vline(
                            x=mean - mult * std,
                            line_dash="dash",
                            line_color="orange",
                            line_width=1,
                            annotation_text=f"-{mult}σ",
                            annotation_position="top",
                            annotation_font_size=theme.annotation_font_size
                        )

        # Axis labels
        x_label = xaxis_title or value_col
        y_label = yaxis_title or "Count"

        fig.update_layout(
            width=width,
            height=height,
            xaxis_title=x_label,
            yaxis_title=y_label,
            showlegend=False,
            bargap=0.05
        )

        return fig

    # Helper methods
    def _get_value_column(self, chart_info: dict, chart_name: str) -> str:
        """
        Get the value column from chart metadata.

        Extracts the value column name from the chart's metadata dict,
        which is set during chart calculation. This follows the DECLARATIVE
        contract where each chart explicitly defines its output schema.

        Parameters
        ----------
        chart_info : dict
            Chart info dict with 'data', 'statistics', and 'metadata' keys
        chart_name : str
            Name of the chart (for error messages)

        Returns
        -------
        str
            Column name to use for y-axis values

        Raises
        ------
        ValueError
            If metadata is missing (indicates bug in chart calculation)

        Examples
        --------
        >>> chart_info = {
        ...     'data': xbar_df,
        ...     'statistics': {...},
        ...     'metadata': {'chart_type': 'Xbar', 'value_col': 'xbar', ...}
        ... }
        >>> value_col = self._get_value_column(chart_info, 'Xbar')
        >>> assert value_col == 'xbar'
        """
        if 'metadata' not in chart_info:
            raise ValueError(
                f"Chart '{chart_name}' missing metadata. "
                f"This indicates a bug in chart calculation. "
                f"All charts must have metadata with 'value_col'."
            )

        return chart_info['metadata']['value_col']

    def _get_base_chart_type(self, chart_name: str) -> str:
        """
        Extract base chart type from a chart name.

        Chart names may include suffixes (e.g., 'Xbar_Lane1', 'XmR_Group2').
        This extracts just the base type for scale comparison.

        Parameters
        ----------
        chart_name : str
            Full chart name (e.g., 'Xbar', 'Xbar_Lane1', 'S_Line2')

        Returns
        -------
        str
            Base chart type: 'Xbar', 'S', 'XmR', or 'R'
        """
        # Split on underscore and take first part
        base = chart_name.split('_')[0]
        return base

    def _get_x_column(self, data: pd.DataFrame) -> str | None:
        """
        Determine the x-axis column for plotting.

        Checks in order:
        1. time_var if unique AND present in data
        2. 'rsg' column (subgroup identifier for Xbar/S charts)
        3. None (signals caller to use DataFrame index)

        Returns
        -------
        str | None
            Column name to use for x-axis, or None to use DataFrame index
        """
        time_var = self.summary.get('time_var')

        if time_var and time_var in data.columns:
            # If time values repeat (collapsed chart), use index for positions
            # This aligns with lane boundary positions which use 0-based indices
            if not data[time_var].is_unique:
                return None
            return time_var

        # Use subgroup identifier for Xbar/S charts
        if 'subgroup' in data.columns:
            return 'subgroup'
        if 'rsg' in data.columns:
            return 'rsg'
        if 'group' in data.columns:
            return 'group'

        # Use index for positioning
        return None

    def _apply_time_tick_labels(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        metadata: dict,
        row: int | None = None,
        col: int | None = None,
        max_ticks: int = 30,
    ) -> None:
        """Replace integer x-axis positions with time values when applicable.

        Only applies when _get_x_column() returned None (integer index positions)
        and the time variable exists in the data. Ensures time values are displayed
        on the x-axis even when positions must be integers (for lane boundary
        alignment).
        """
        time_var = self.summary.get('time_var')
        if not time_var or time_var not in data.columns:
            return

        # Only apply when x-axis is using integer positions
        x_col = self._get_x_column(data)
        if x_col is not None:
            return

        n = len(data)
        if n == 0:
            return

        # Priority positions: endpoints + lane boundaries (always visible)
        priority = {0, n - 1}
        lane_boundaries = metadata.get('lane_boundaries')
        if lane_boundaries:
            if isinstance(lane_boundaries, list):
                priority |= {b['position'] for b in lane_boundaries}
            elif isinstance(lane_boundaries, dict):
                for bounds in lane_boundaries.values():
                    priority |= {b['position'] for b in bounds}

        # Fill remaining budget with evenly spaced regular ticks
        budget = max(0, max_ticks - len(priority))
        if budget > 0 and n > len(priority):
            step = max(1, n // budget)
            regular = set(range(0, n, step)) - priority
            if len(regular) > budget:
                regular = set(sorted(regular)[:budget])
            tick_positions = sorted(priority | regular)
        else:
            tick_positions = sorted(priority)

        # Filter to valid range
        tick_positions = [p for p in tick_positions if 0 <= p < n]

        tick_labels = data[time_var].iloc[tick_positions].astype(str).tolist()

        kwargs = {}
        if row is not None and col is not None:
            kwargs = {'row': row, 'col': col}
        fig.update_xaxes(tickvals=tick_positions, ticktext=tick_labels, **kwargs)

    def _get_center_key(self, stats: dict) -> str | None:
        """Get the centerline statistic key."""
        # All chart types now use 'center' for the centerline column
        if 'center' in stats:
            return 'center'
        return None

    def _get_stratified_charts(self) -> dict:
        """Get stratified charts expanded for faceted plotting.

        For charts with nested structure (strata list and nested statistics),
        this expands them into per-stratum charts suitable for faceted plotting.

        The expansion process:
        1. Detects charts with 'strata' key (stratified XmR/R charts)
        2. Splits combined DataFrame by RSG column
        3. Extracts per-stratum statistics from nested structure
        4. Creates expanded chart names as '{chart_type}_{stratum}'

        Returns
        -------
        dict
            Expanded charts with per-stratum data:
            - Keys: '{chart_type}_{stratum}' (e.g., 'XmR_Machine1_F2_1')
            - Values: {'data': DataFrame, 'statistics': dict, 'metadata': dict}
            - metadata includes 'original_chart' and 'stratum' for filtering
        """
        stratified = {}

        for name, chart_info in self.charts.items():
            # Check if this is a stratified chart with nested structure
            # New structure: 'strata' list and nested statistics by stratum
            if 'strata' in chart_info and chart_info['strata']:
                strata = chart_info['strata']
                combined_data = chart_info['data']
                nested_stats = chart_info['statistics']
                metadata = chart_info.get('metadata', {})

                # Get stratify column from metadata
                stratify_by = metadata.get('stratify_by', [])

                if len(stratify_by) == 1:
                    stratify_col = stratify_by[0]
                elif len(stratify_by) > 1:
                    # Multiple columns - create combined key using tuples
                    # Tuples avoid collision risk: ('A_B', 'C') != ('A', 'B_C')
                    combined_data = combined_data.copy()
                    combined_data['_stratify_key'] = combined_data[stratify_by].apply(tuple, axis=1)
                    stratify_col = '_stratify_key'
                else:
                    stratify_col = None

                if stratify_col is not None and stratify_col in combined_data.columns:
                    # Expand into per-stratum charts
                    for stratum in strata:
                        # Filter data for this stratum
                        # Direct comparison works for both scalar and tuple strata
                        stratum_mask = combined_data[stratify_col] == stratum
                        stratum_data = combined_data[stratum_mask].copy()
                        stratum_data = stratum_data.reset_index(drop=True)

                        # Get stratum-specific statistics
                        stratum_stats = nested_stats.get(stratum, {})

                        # Create expanded chart name (include chart type to avoid collision)
                        # Convert tuple/list strata to underscore-joined string for clean names
                        # Note: stratum identity assumes canonical factor ordering defined upstream
                        if isinstance(stratum, (tuple, list)):
                            stratum_str = encode_rsg(stratum)  # Use canonical encoding
                            # Display name uses space separator to preserve underscores in values
                            stratum_display = ' '.join(str(s) for s in stratum)
                        else:
                            stratum_str = encode_rsg(stratum)  # Handle scalar strata
                            stratum_display = str(stratum)
                        expanded_name = f"{name}_{stratum_str}"

                        # Extract lane boundaries for this specific stratum
                        all_lane_boundaries = metadata.get('lane_boundaries')
                        stratum_lane_boundaries = None
                        if isinstance(all_lane_boundaries, dict):
                            stratum_lane_boundaries = all_lane_boundaries.get(stratum)

                        stratified[expanded_name] = {
                            'data': stratum_data,
                            'statistics': stratum_stats,
                            'metadata': {
                                **metadata,
                                'original_chart': name,
                                'stratum': stratum,
                                'stratum_display': stratum_display,
                                'lane_boundaries': stratum_lane_boundaries
                            }
                        }
                else:
                    # Can't split - use as-is
                    stratified[name] = chart_info

            # Check for legacy stratified structure (underscore in name)
            elif '_' in str(name) or any(
                key in chart_info.get('metadata', {})
                for key in ['stratum', 'level', 'group']
            ):
                stratified[name] = chart_info

        # If no stratified charts found, return all charts
        return stratified if stratified else self.charts

    def _calculate_global_yrange(
        self,
        charts: dict,
        padding: float = 0.05
    ) -> list[float]:
        """
        Calculate global y-axis range across all charts.

        Examines all chart data and control limits to determine a common
        y-axis range that encompasses all values with appropriate padding.

        Parameters
        ----------
        charts : dict
            Dictionary of chart_name -> chart_info dicts
        padding : float, default 0.05
            Padding as fraction of data range (0.05 = 5% on each side)

        Returns
        -------
        list[float]
            [y_min, y_max] range for y-axis
        """
        global_min = float('inf')
        global_max = float('-inf')

        for chart_name, chart_info in charts.items():
            data = chart_info['data']
            stats = chart_info['statistics']

            # Get value column
            value_col = self._get_value_column(chart_info, chart_name)

            # Update bounds with data values
            data_min = data[value_col].min()
            data_max = data[value_col].max()
            global_min = min(global_min, data_min)
            global_max = max(global_max, data_max)

            # Include control limits in range
            ucl = stats.get('upl')
            lcl = stats.get('lpl')

            if ucl is not None and ucl != 'Varies':
                global_max = max(global_max, ucl)
            if lcl is not None and lcl != 'Varies':
                global_min = min(global_min, lcl)

        # Calculate padding amount
        data_range = global_max - global_min
        padding_amount = data_range * padding

        # Apply padding
        y_min = global_min - padding_amount
        y_max = global_max + padding_amount

        return [y_min, y_max]

    def _calculate_histogram_bin_edges(
        self,
        charts: dict
    ) -> tuple[np.ndarray, int]:
        """
        Calculate shared bin edges across all histogram facets.

        This ensures consistent binning across all faceted histograms so that
        numpy's histogram calculation matches Plotly's rendering.

        Parameters
        ----------
        charts : dict
            Dictionary of chart_name -> chart_info dicts

        Returns
        -------
        tuple[np.ndarray, int]
            (bin_edges, n_bins) - shared bin edges and number of bins
        """
        import numpy as np

        global_min = float('inf')
        global_max = float('-inf')
        n_bins = 10  # default

        for chart_info in charts.values():
            metadata = chart_info.get('metadata', {})
            if metadata.get('chart_type') != 'Histogram':
                continue

            data = chart_info['data']
            value_col = metadata.get('value_col')
            n_bins = metadata.get('bins', 10)

            if value_col is None or value_col not in data.columns:
                continue

            values = data[value_col].dropna()
            if len(values) == 0:
                continue

            global_min = min(global_min, values.min())
            global_max = max(global_max, values.max())

        # Fallback for empty data
        if global_min == float('inf') or global_max == float('-inf'):
            return np.linspace(0, 1, n_bins + 1), n_bins

        # Create shared bin edges
        bin_edges = np.linspace(global_min, global_max, n_bins + 1)
        return bin_edges, n_bins

    def _calculate_histogram_yrange(
        self,
        charts: dict,
        bin_edges: np.ndarray,
        padding: float = 0.20
    ) -> list[float]:
        """
        Calculate y-axis range for histogram facets based on bin counts.

        Unlike control charts which use data values on the y-axis, histograms
        display bin counts. This method calculates a shared y-range based on
        the maximum count across all histogram bins using shared bin edges.

        Parameters
        ----------
        charts : dict
            Dictionary of chart_name -> chart_info dicts
        bin_edges : np.ndarray
            Shared bin edges calculated by _calculate_histogram_bin_edges
        padding : float, default 0.20
            Padding as fraction above max count (0.20 = 20% above max)

        Returns
        -------
        list[float]
            [0, max_count * (1 + padding)] range for y-axis
        """
        import numpy as np

        max_count = 0

        for chart_info in charts.values():
            metadata = chart_info.get('metadata', {})
            if metadata.get('chart_type') != 'Histogram':
                continue

            data = chart_info['data']
            value_col = metadata.get('value_col')

            if value_col is None or value_col not in data.columns:
                continue

            # Calculate histogram to get counts using shared bin edges
            values = data[value_col].dropna()
            if len(values) == 0:
                continue

            counts, _ = np.histogram(values, bins=bin_edges)
            max_count = max(max_count, counts.max())

        # Add padding above max count
        if max_count == 0:
            return [0, 1]  # Fallback for empty data

        upper = max_count * (1 + padding)
        return [0, upper]

    # =========================================================================
    # Title and Label Generation (Smart Defaults)
    # =========================================================================

    def _generate_title(self, chart_name: str) -> str:
        """
        Generate a descriptive chart title.

        Creates contextual titles like:
        - "Xbar Chart of Weight"
        - "Xbar Chart of Weight by Operator"
        - "XmR Chart of Temperature - Line A"

        Parameters
        ----------
        chart_name : str
            Name of the chart being plotted

        Returns
        -------
        str
            Descriptive title for the chart
        """
        response_var = self.summary.get('response_var', '')
        grouping_vars = self.summary.get('grouping_vars', [])

        # Get chart type (clean up chart_name)
        chart_type = self._get_chart_type_display(chart_name)

        # Build title parts
        title_parts = [f"{chart_type} Chart"]

        # Add response variable if available
        if response_var:
            title_parts.append(f"of {response_var}")

        # Add grouping if single grouping variable
        if grouping_vars and len(grouping_vars) == 1:
            title_parts.append(f"by {grouping_vars[0]}")

        # Check if this is a stratified chart (has stratum in name)
        if '_' in chart_name and chart_name not in ['Xbar', 'S', 'XmR']:
            # Extract stratum name (e.g., "Operator_A" -> "Operator A")
            stratum = self._extract_stratum_name(chart_name)
            if stratum:
                title_parts.append(f"- {stratum}")

        return ' '.join(title_parts)

    def _generate_subplot_title(self, chart_name: str, chart_info: dict | None = None) -> str:
        """
        Generate a title for a subplot in faceted layout.

        Keeps subplot titles concise while still informative.
        Includes chart type and stratum name (e.g., "XmR F1_1 F2_1").

        Parameters
        ----------
        chart_name : str
            Name of the chart/stratum
        chart_info : dict, optional
            Chart info dict with metadata containing stratum_display

        Returns
        -------
        str
            Concise subplot title
        """
        # Get chart type display name
        chart_type = self._get_chart_type_display(chart_name)

        # Check for stratum_display in metadata (preserves underscores in factor values)
        if chart_info is not None:
            stratum_display = chart_info.get('metadata', {}).get('stratum_display')
            if stratum_display:
                return f"{chart_type} {stratum_display}"

        # Fallback: For stratified charts, extract stratum from name
        if '_' in chart_name:
            stratum = self._extract_stratum_name(chart_name)
            if stratum:
                return f"{chart_type} {stratum}"

        # For standard charts, just use the chart type
        return chart_type

    def _get_chart_type_display(self, chart_name: str) -> str:
        """
        Get display-friendly chart type name.

        Parameters
        ----------
        chart_name : str
            Raw chart name

        Returns
        -------
        str
            Display-friendly name (e.g., "Xbar" -> "X̄", "XmR" -> "XmR")
        """
        # Map technical names to display names
        # Valid chart types: Xbar, S, R, XmR, Histogram
        display_names = {
            'Xbar': 'X̄',
            'S': 'S',
            'XmR': 'XmR',
            'R': 'R',
            'Histogram': 'Histogram'
        }

        # Check for exact match first
        if chart_name in display_names:
            return display_names[chart_name]

        # Check if it starts with a known chart type (for stratified charts)
        for key, display in display_names.items():
            if chart_name.startswith(key + '_'):
                return display

        # No fallback - unknown chart types should raise an error
        raise ValueError(f"Unknown chart type: {chart_name}")

    def _extract_stratum_name(self, chart_name: str) -> str | None:
        """
        Extract stratum name from stratified chart name.

        Parameters
        ----------
        chart_name : str
            Chart name like "XmR_Operator_A" or "Lane_1"

        Returns
        -------
        str or None
            Cleaned stratum name like "Operator A" or "Lane 1"
        """
        # Known chart type prefixes to strip
        # Valid chart types: Xbar, S, R, XmR, Histogram
        prefixes = ['Xbar_', 'S_', 'XmR_', 'R_', 'Histogram_']

        result = chart_name
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix):]
                break

        # Replace underscores with spaces for readability
        result = result.replace('_', ' ')

        return result if result != chart_name else None

    def _get_xaxis_label(self, x_col: str | None = None, chart_name: str | None = None) -> str:
        """
        Get intelligent x-axis label.

        For XmR/R charts, 'rsg' is a stratification key, not a subgroup
        identifier. Only Xbar/S charts should label the x-axis as "Subgroup".

        Parameters
        ----------
        x_col : str, optional
            The x-axis column being used.
        chart_name : str, optional
            Chart name (e.g. 'XmR', 'XmR_residuals', 'Xbar'). Used to
            distinguish individual charts (XmR/R) from subgroup charts
            (Xbar/S) when x_col is a grouping column.

        Returns
        -------
        str
            X-axis label
        """
        # For grouping columns, the label depends on chart type
        if x_col in ('subgroup', 'rsg', 'group'):
            base_type = chart_name.split('_')[0] if chart_name else ''
            if base_type in ('XmR', 'R'):
                # Individual charts: use time variable name or "Observation"
                time_var = self.summary.get('time_var')
                if time_var:
                    return time_var.replace('_', ' ').title()
                return 'Observation'
            return 'Subgroup'

        # Time series charts use time variable
        time_var = self.summary.get('time_var')
        if time_var and (x_col is None or x_col == time_var):
            return time_var.replace('_', ' ').title()

        # Fallback for charts without explicit time variable
        if x_col is None:
            return 'Observation'

        return x_col.replace('_', ' ').title()

    def _get_yaxis_label(self, value_col: str | None) -> str:
        """
        Get intelligent y-axis label.

        Uses response variable name if available, otherwise falls back
        to the value column name.

        Parameters
        ----------
        value_col : str or None
            The value column name (fallback)

        Returns
        -------
        str
            Y-axis label
        """
        response_var = self.summary.get('response_var')
        if response_var:
            # Capitalize first letter of each word
            return response_var.replace('_', ' ').title()

        # Fallback to value column
        if value_col:
            return value_col.capitalize()

        return 'Value'

    def _get_data_legend_name(self) -> str:
        """
        Get legend name for the data trace.

        Returns
        -------
        str
            Legend entry name
        """
        response_var = self.summary.get('response_var')
        if response_var:
            return response_var.replace('_', ' ').title()
        return 'Data'

    # =========================================================================
    # Residual Visualization
    # =========================================================================

    def plot_residuals(
        self,
        residual_type: str = 'R1',
        plot_type: str = 'all',
        theme: str | ChartTheme = 'processbehavior',
        width: int = 1200,
        height: int = 400
    ) -> ControlChartFigure:
        """
        Create residual diagnostic plots.

        Visualizes VAS residuals to assess process behavior and identify
        patterns that may indicate non-random variation.

        Parameters
        ----------
        residual_type : str, default 'R1'
            Which residual to plot ('R1', 'R2', 'R3', 'R4', 'R5')
            - R1: Primary residual (within-cell)
            - R2: Time-level residual
            - R3: Factor-level residual
            - R4: Interaction residual
            - R5: Factor main effect residual
        plot_type : str, default 'all'
            Type of diagnostic plot:
            - 'histogram': Distribution of residuals
            - 'qq': Quantile-quantile plot for normality
            - 'sequence': Residuals vs. observation order
            - 'all': All three plots in subplots
        theme : str or ChartTheme, default 'processbehavior'
            Visual theme
        width : int, default 1200
            Figure width in pixels
        height : int, default 400
            Figure height in pixels (per row for 'all')

        Returns
        -------
        ControlChartFigure
            Interactive figure with residual diagnostics

        Raises
        ------
        ValueError
            If residuals not available or invalid residual type

        Examples
        --------
        >>> fig = plotter.plot_residuals()  # Default R1 residuals
        >>> fig = plotter.plot_residuals('R2', plot_type='histogram')
        >>> fig = plotter.plot_residuals('R5', plot_type='qq')
        """
        import numpy as np

        # Check residuals are available
        if not self.result.has_residuals:
            raise ValueError(
                "Residuals not available for this analysis.\n"
                "Residuals require SDS >= 1 (replicated observations)."
            )

        residuals = self.result.residuals
        if residual_type not in residuals.columns:
            available = list(residuals.columns)
            raise ValueError(
                f"Residual '{residual_type}' not found.\n"
                f"Available: {available}"
            )

        # Resolve theme
        theme = get_theme(theme) if isinstance(theme, str) else theme

        # Get residual data
        r_data = residuals[residual_type].dropna()

        if plot_type == 'all':
            # Create 3-panel diagnostic plot
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Histogram', 'Q-Q Plot', 'Sequence Plot'],
                horizontal_spacing=0.08
            )

            # 1. Histogram
            fig.add_trace(
                go.Histogram(
                    x=r_data,
                    name='Residuals',
                    marker_color=theme.data_color,
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=1
            )

            # Add normal curve overlay
            x_range = np.linspace(r_data.min(), r_data.max(), 100)
            mu, sigma = r_data.mean(), r_data.std()
            y_normal = np.array([_normal_pdf(x, mu, sigma) for x in x_range])
            y_normal = y_normal * len(r_data) * (r_data.max() - r_data.min()) / 20

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_normal,
                    mode='lines',
                    name='Normal',
                    line=dict(color=theme.center_color, width=2, dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )

            # 2. Q-Q Plot
            probs = np.linspace(0.01, 0.99, len(r_data))
            theoretical_q = np.array([_normal_ppf(p) for p in probs])
            sample_q = np.sort(r_data.values)

            fig.add_trace(
                go.Scatter(
                    x=theoretical_q,
                    y=sample_q,
                    mode='markers',
                    name='Q-Q',
                    marker=dict(color=theme.data_color, size=5),
                    showlegend=False
                ),
                row=1, col=2
            )

            # Add reference line
            qq_min = min(theoretical_q.min(), sample_q.min())
            qq_max = max(theoretical_q.max(), sample_q.max())
            fig.add_trace(
                go.Scatter(
                    x=[qq_min, qq_max],
                    y=[qq_min * r_data.std() + r_data.mean(),
                       qq_max * r_data.std() + r_data.mean()],
                    mode='lines',
                    name='Reference',
                    line=dict(color=theme.ucl_color, dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )

            # 3. Sequence plot
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(r_data))),
                    y=r_data.values,
                    mode='lines+markers',
                    name='Residuals',
                    marker=dict(size=4, color=theme.data_color),
                    line=dict(color=theme.data_color, width=1),
                    showlegend=False
                ),
                row=1, col=3
            )

            # Add zero line
            fig.add_hline(y=0, line_dash='dash', line_color=theme.center_color, row=1, col=3)

            # Update layout
            fig.update_layout(
                title=f'{residual_type} Residual Diagnostics',
                width=width,
                height=height,
                showlegend=False
            )
            fig.update_xaxes(title_text='Residual Value', row=1, col=1)
            fig.update_yaxes(title_text='Frequency', row=1, col=1)
            fig.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
            fig.update_yaxes(title_text='Sample Quantiles', row=1, col=2)
            fig.update_xaxes(title_text='Observation', row=1, col=3)
            fig.update_yaxes(title_text='Residual', row=1, col=3)

        elif plot_type == 'histogram':
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=r_data,
                    name='Residuals',
                    marker_color=theme.data_color,
                    opacity=0.7
                )
            )
            fig.update_layout(
                title=f'{residual_type} Residual Distribution',
                xaxis_title='Residual Value',
                yaxis_title='Frequency',
                width=width,
                height=height
            )

        elif plot_type == 'qq':
            probs = np.linspace(0.01, 0.99, len(r_data))
            theoretical_q = np.array([_normal_ppf(p) for p in probs])
            sample_q = np.sort(r_data.values)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=theoretical_q,
                    y=sample_q,
                    mode='markers',
                    name='Data',
                    marker=dict(color=theme.data_color, size=6)
                )
            )
            # Reference line
            qq_min = min(theoretical_q.min(), sample_q.min())
            qq_max = max(theoretical_q.max(), sample_q.max())
            fig.add_trace(
                go.Scatter(
                    x=[qq_min, qq_max],
                    y=[qq_min * r_data.std() + r_data.mean(),
                       qq_max * r_data.std() + r_data.mean()],
                    mode='lines',
                    name='Normal Reference',
                    line=dict(color=theme.ucl_color, dash='dash')
                )
            )
            fig.update_layout(
                title=f'{residual_type} Q-Q Plot',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                width=width,
                height=height
            )

        elif plot_type == 'sequence':
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(r_data))),
                    y=r_data.values,
                    mode='lines+markers',
                    name='Residuals',
                    marker=dict(size=5, color=theme.data_color),
                    line=dict(color=theme.data_color, width=1)
                )
            )
            fig.add_hline(y=0, line_dash='dash', line_color=theme.center_color)
            fig.update_layout(
                title=f'{residual_type} Residual Sequence Plot',
                xaxis_title='Observation',
                yaxis_title='Residual Value',
                width=width,
                height=height
            )

        else:
            raise ValueError(
                f"Invalid plot_type: '{plot_type}'.\n"
                f"Options: 'all', 'histogram', 'qq', 'sequence'"
            )

        # Apply theme
        fig = apply_theme(fig, theme)

        return ControlChartFigure(fig, self.result)

    def _plot_effects_chart(
        self,
        chart_type: str,
        width: int = 1000,
        height: int | None = None,
        title: str | None = None
    ) -> ControlChartFigure:
        """
        Route to appropriate effects chart based on chart_type.

        Called by plot() when chart is 'Effects', 'TimeInteraction', or 'FactorInteraction'.

        Parameters
        ----------
        chart_type : str
            One of: 'Effects', 'TimeInteraction', 'FactorInteraction'
        width : int
            Figure width in pixels
        height : int, optional
            Figure height in pixels
        title : str, optional
            Custom title

        Returns
        -------
        ControlChartFigure
            Interactive figure
        """
        from .effects_charts import (
            create_factor_effects_chart,
            create_factor_interaction_chart,
            create_main_effects_chart,
            create_time_effects_chart,
            create_time_interaction_chart,
        )

        theme = self._theme

        # Get effects and interactions from result
        effects = self.result.effects
        interactions = self.result.interactions

        if chart_type == 'Effects':
            # Main effects chart
            if not self.result.has_effects:
                raise ValueError(
                    "Effects not available for this analysis.\n"
                    "Effects require factors in the analysis specification."
                )

            fig = create_main_effects_chart(
                effects=effects,
                theme=theme,
                width=width,
                height=height
            )

        elif chart_type == 'MainEffects':
            # Factor main effects only (no time effects)
            if not self.result.has_effects:
                raise ValueError(
                    "Effects not available for this analysis.\n"
                    "Effects require factors in the analysis specification."
                )

            fig = create_factor_effects_chart(
                effects=effects,
                theme=theme,
                width=width,
                height=height or 500
            )

        elif chart_type == 'TimeEffects':
            # Time effects only (no factor effects)
            if 'time' not in effects and not any(
                isinstance(v, pd.DataFrame) and 'PT_ME' in v.columns
                for v in effects.values()
            ):
                raise ValueError(
                    "Time effects not available for this analysis.\n"
                    "This requires a time variable in the analysis."
                )

            fig = create_time_effects_chart(
                effects=effects,
                theme=theme,
                width=width,
                height=height
            )

        elif chart_type == 'TimeInteraction':
            # Factor × time interaction chart
            if 'factor_time' not in interactions:
                raise ValueError(
                    "Time interaction not available for this analysis.\n"
                    "This requires both factors and time variable in the analysis."
                )

            # Get factors and time_var from summary
            factors = self.summary.get('grouping_vars', [])
            time_var = self.summary.get('time_var')

            if not factors or not time_var:
                raise ValueError(
                    "Cannot create time interaction chart.\n"
                    "Requires both factors and time variable."
                )

            fig = create_time_interaction_chart(
                interactions=interactions,
                effects=effects,
                factors=factors,
                time_var=time_var,
                dataset=self.result.dataset,
                theme=theme,
                width=width,
                height=height or 500
            )

        elif chart_type == 'FactorInteraction':
            # Factor × factor interaction chart
            if 'factor_factor' not in interactions:
                raise ValueError(
                    "Factor interaction not available for this analysis.\n"
                    "This requires at least 2 factors in the analysis."
                )

            factors = self.summary.get('grouping_vars', [])
            if len(factors) < 2:
                raise ValueError(
                    f"Factor interaction requires at least 2 factors, "
                    f"got {len(factors)}."
                )

            fig = create_factor_interaction_chart(
                interactions=interactions,
                factors=factors,
                theme=theme,
                width=width,
                height=height or 600
            )

        else:
            raise ValueError(
                f"Unknown effects chart type: '{chart_type}'.\n"
                f"Options: 'Effects', 'MainEffects', 'TimeEffects', 'TimeInteraction', 'FactorInteraction'"
            )

        # Apply theme
        fig = apply_theme(fig, theme)

        # Set custom title if provided
        if title:
            fig.update_layout(title=title)

        return ControlChartFigure(fig, self.result)

    def plot_effects(  # noqa: C901
        self,
        effect_type: str = 'factor',
        theme: str | ChartTheme = 'processbehavior',
        width: int = 800,
        height: int = 500
    ) -> ControlChartFigure:
        """
        Create bar chart of main effects.

        Visualizes factor or time effects to identify which levels
        contribute most to process variation.

        Parameters
        ----------
        effect_type : str, default 'factor'
            Type of effects to plot:
            - 'factor': Factor main effects (mean R5 per level)
            - 'time': Time effects (mean R1 per time point)
            - 'all': Both effects in subplots
        theme : str or ChartTheme, default 'processbehavior'
            Visual theme
        width : int, default 800
            Figure width in pixels
        height : int, default 500
            Figure height in pixels

        Returns
        -------
        ControlChartFigure
            Interactive figure with effects visualization

        Raises
        ------
        ValueError
            If effects not available

        Examples
        --------
        >>> fig = plotter.plot_effects()  # Factor effects
        >>> fig = plotter.plot_effects('time')  # Time effects
        >>> fig = plotter.plot_effects('all')  # Both
        """
        # Check effects are available
        if not self.result.has_effects:
            raise ValueError(
                "Effects not available for this analysis.\n"
                "Effects require SDS >= 2 (multiple factor levels)."
            )

        effects = self.result.effects

        # Resolve theme
        theme = get_theme(theme) if isinstance(theme, str) else theme

        if effect_type == 'all':
            # Find factor and time effects
            factor_effects = []
            time_effects = None

            for name, data in effects.items():
                if 'PT_ME' in str(data.columns if hasattr(data, 'columns') else ''):
                    time_effects = data
                elif isinstance(data, pd.DataFrame) and 'Main_Effect' in data.columns:
                    factor_effects.append((name, data))

            n_plots = len(factor_effects) + (1 if time_effects is not None else 0)

            if n_plots == 0:
                raise ValueError("No effects data found to plot.")

            fig = make_subplots(
                rows=1, cols=n_plots,
                subplot_titles=[name for name, _ in factor_effects] +
                               (['Time Effects'] if time_effects is not None else []),
                horizontal_spacing=0.1
            )

            col = 1
            for name, data in factor_effects:
                factor_col = data.columns[0]
                fig.add_trace(
                    go.Bar(
                        x=data[factor_col].astype(str),
                        y=data['Main_Effect'],
                        name=name,
                        marker_color=theme.data_color,
                        showlegend=False
                    ),
                    row=1, col=col
                )
                fig.update_xaxes(title_text=factor_col, row=1, col=col)
                fig.update_yaxes(title_text='Main Effect', row=1, col=col)
                col += 1

            if time_effects is not None:
                time_col = time_effects.columns[0]
                fig.add_trace(
                    go.Bar(
                        x=time_effects[time_col].astype(str),
                        y=time_effects['PT_ME'],
                        name='Time',
                        marker_color=theme.center_color,
                        showlegend=False
                    ),
                    row=1, col=col
                )
                fig.update_xaxes(title_text=time_col, row=1, col=col)
                fig.update_yaxes(title_text='Time Effect', row=1, col=col)

            fig.update_layout(
                title='Main Effects Analysis',
                width=width,
                height=height
            )

        elif effect_type == 'factor':
            # Plot factor effects
            factor_effects = []
            for name, data in effects.items():
                if isinstance(data, pd.DataFrame) and 'Main_Effect' in data.columns:
                    factor_effects.append((name, data))

            if not factor_effects:
                raise ValueError("No factor effects found.")

            if len(factor_effects) == 1:
                name, data = factor_effects[0]
                factor_col = data.columns[0]

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=data[factor_col].astype(str),
                        y=data['Main_Effect'],
                        name='Effect',
                        marker_color=theme.data_color
                    )
                )
                fig.add_hline(y=0, line_dash='dash', line_color=theme.center_color)
                fig.update_layout(
                    title=f'Factor Main Effects: {name}',
                    xaxis_title=factor_col,
                    yaxis_title='Main Effect',
                    width=width,
                    height=height
                )

            else:
                # Multiple factors - subplot
                fig = make_subplots(
                    rows=1, cols=len(factor_effects),
                    subplot_titles=[name for name, _ in factor_effects],
                    horizontal_spacing=0.1
                )

                for col, (name, data) in enumerate(factor_effects, 1):
                    factor_col = data.columns[0]
                    fig.add_trace(
                        go.Bar(
                            x=data[factor_col].astype(str),
                            y=data['Main_Effect'],
                            name=name,
                            marker_color=theme.data_color,
                            showlegend=False
                        ),
                        row=1, col=col
                    )
                    fig.update_xaxes(title_text=factor_col, row=1, col=col)
                    fig.update_yaxes(title_text='Main Effect', row=1, col=col)

                fig.update_layout(
                    title='Factor Main Effects',
                    width=width,
                    height=height
                )

        elif effect_type == 'time':
            # Plot time effects
            time_effects = None
            for _name, data in effects.items():
                if isinstance(data, pd.DataFrame) and 'PT_ME' in data.columns:
                    time_effects = data
                    break

            if time_effects is None:
                raise ValueError(
                    "Time effects not found.\n"
                    "Time effects require time_var in analysis specification."
                )

            time_col = time_effects.columns[0]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=time_effects[time_col].astype(str),
                    y=time_effects['PT_ME'],
                    name='Time Effect',
                    marker_color=theme.center_color
                )
            )
            fig.add_hline(y=0, line_dash='dash', line_color=theme.data_color)
            fig.update_layout(
                title='Time Main Effects',
                xaxis_title=time_col,
                yaxis_title='Time Effect (PT_ME)',
                width=width,
                height=height
            )

        else:
            raise ValueError(
                f"Invalid effect_type: '{effect_type}'.\n"
                f"Options: 'factor', 'time', 'all'"
            )

        # Apply theme
        fig = apply_theme(fig, theme)

        return ControlChartFigure(fig, self.result)

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_report(
        self,
        filepath: str,
        include_charts: bool = True,
        include_residuals: bool = True,
        include_effects: bool = True,
        include_summary: bool = True,
        theme: str | ChartTheme = 'processbehavior',
        width: int = 1200,
        title: str | None = None
    ) -> None:
        """
        Generate a comprehensive HTML report with all visualizations.

        Creates a single-page HTML file with:
        - Analysis summary
        - Control charts
        - Residual diagnostics (if available)
        - Effects analysis (if available)
        - Interactive features (zoom, pan, hover)

        Parameters
        ----------
        filepath : str
            Output HTML file path (e.g., 'report.html')
        include_charts : bool, default True
            Include control charts
        include_residuals : bool, default True
            Include residual diagnostic plots (if available)
        include_effects : bool, default True
            Include effects bar charts (if available)
        include_summary : bool, default True
            Include analysis summary section
        theme : str or ChartTheme, default 'processbehavior'
            Visual theme for all charts
        width : int, default 1200
            Width of charts in pixels
        title : str, optional
            Report title (defaults to "Process Behavior Analysis Report")

        Examples
        --------
        >>> plotter.generate_report('analysis_report.html')
        >>> plotter.generate_report('dark_report.html', theme='dark')

        Notes
        -----
        The generated HTML file is self-contained and can be opened
        in any web browser without an internet connection.
        """
        from pathlib import Path

        # Build report sections
        sections = []
        report_title = title or "Process Behavior Analysis Report"
        esc = html.escape

        # Summary section
        if include_summary:
            summary = self.result.summary
            has_res = 'Yes' if summary['has_residuals'] else 'No'
            has_eff = 'Yes' if summary['has_effects'] else 'No'
            summary_html = f"""
            <div class="section">
                <h2>Analysis Summary</h2>
                <table class="summary-table">
                    <tr><td><strong>SDS</strong></td><td>{esc(str(summary['sds']))} - {
                        esc(str(summary['sds_description']))}</td></tr>
                    <tr><td><strong>Response Variable</strong></td><td>{
                        esc(str(summary['response_var']))}</td></tr>
                    <tr><td><strong>Observations</strong></td><td>{
                        esc(str(summary['n_observations']))}</td></tr>
                    <tr><td><strong>Charts</strong></td><td>{
                        esc(', '.join(str(c) for c in summary['chart_types']))}</td></tr>
                    <tr><td><strong>Signals Detected</strong></td><td>{esc(str(summary['n_signals_total']))}</td></tr>
                    <tr><td><strong>Has Residuals</strong></td><td>{has_res}</td></tr>
                    <tr><td><strong>Has Effects</strong></td><td>{has_eff}</td></tr>
                </table>
            </div>
            """
            sections.append(summary_html)

        # Control charts section
        if include_charts:
            fig = self.plot(theme=theme, width=width, height=500)
            chart_html = fig.figure.to_html(full_html=False, include_plotlyjs=False)
            sections.append(f"""
            <div class="section">
                <h2>Control Charts</h2>
                {chart_html}
            </div>
            """)

        # Residuals section
        if include_residuals and self.result.has_residuals:
            fig = self.plot_residuals(theme=theme, width=width, height=350)
            residual_html = fig.figure.to_html(full_html=False, include_plotlyjs=False)
            sections.append(f"""
            <div class="section">
                <h2>Residual Diagnostics</h2>
                <p>R1 residuals showing process variation over time.</p>
                {residual_html}
            </div>
            """)

        # Effects section
        if include_effects and self.result.has_effects:
            try:
                fig = self.plot_effects(effect_type='all', theme=theme, width=width, height=400)
                effects_html = fig.figure.to_html(full_html=False, include_plotlyjs=False)
                sections.append(f"""
                <div class="section">
                    <h2>Main Effects Analysis</h2>
                    <p>Factor and time effects showing contribution to process variation.</p>
                    {effects_html}
                </div>
                """)
            except ValueError:
                # No effects to plot
                pass

        # Build full HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{esc(report_title)}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4a90a4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4a90a4;
            margin-top: 30px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-table {{
            border-collapse: collapse;
            width: 100%;
            max-width: 600px;
        }}
        .summary-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
        }}
        .summary-table tr:last-child td {{
            border-bottom: none;
        }}
        .footer {{
            text-align: center;
            color: #888;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>{esc(report_title)}</h1>
    {''.join(sections)}
    <div class="footer">
        Generated with ProcessBehavior - Statistical Process Control for Python
    </div>
</body>
</html>
        """

        # Write file
        Path(filepath).write_text(html_content)
        logger.info(f"Report generated: {filepath}")
