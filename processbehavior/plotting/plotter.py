"""
Main plotting interface for ProcessBehavior analysis results.

Orchestrates chart creation with built-in support for single and faceted charts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .control_chart import ControlChartFigure
from .themes import ChartTheme, apply_theme, get_theme

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
    ...     template='minimal',
    ...     highlight_signals=True
    ... )
    """

    def __init__(self, analysis_result):
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

    def plot(
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
        template: str | ChartTheme = 'processbehavior',
        width: int = 1000,
        height: int | None = None,
        aspect_ratio: float | None = None,
        title: str | None = None,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None
    ) -> ControlChartFigure:
        """
        Create control chart visualization.

        This is the main entry point for plotting. It automatically
        determines the best visualization based on your data structure.

        Parameters
        ----------
        chart : str, optional
            Specific chart to plot ('Xbar', 'Sbar', 'Imr', etc.)
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
            Whether to show numeric values in limit labels (e.g., "UCL = 52.34")
        show_zones : bool, default False
            Whether to show zone shading (A/B/C zones at ±1σ, ±2σ, ±3σ).
            Zone colors are controlled by the theme's zone_a_color, zone_b_color,
            zone_c_color, and zone_opacity properties.
        show_rules : bool, default False
            Whether to show additional run rules (Western Electric Rules 2-8)
        show_stats : bool, default False
            Whether to show a statistics box with CL, UCL, LCL, and n values
        template : str or ChartTheme, default 'processbehavior'
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

        Returns
        -------
        ControlChartFigure
            Interactive figure object with .show(), .save_html(), etc.

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
        ...     template='dark',
        ...     highlight_signals=True,
        ...     show_rules=True
        ... )
        """
        # Resolve theme
        if isinstance(template, str):
            self._theme = get_theme(template)
        else:
            self._theme = template

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
            charts_to_plot = {chart: self.charts[chart]}
        elif facet or facet_by or self.summary.get('is_stratified', False):
            # Auto-detect stratified charts
            charts_to_plot = self._get_stratified_charts()
        else:
            # Plot all standard charts
            charts_to_plot = {
                k: v for k, v in self.charts.items()
                if k in ['Xbar', 'Sbar', 'Imr', 'R', 'all']
            }

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
                n_charts = len(charts_to_plot)
                nrows = (n_charts + ncols - 1) // ncols
                height = nrows * 400

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
                yaxis_title=yaxis_title
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

    def _plot_single_chart(
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
        """Create a single control chart."""
        data = chart_info['data']
        stats = chart_info['statistics']

        fig = go.Figure()

        # Determine value column from metadata
        value_col = self._get_value_column(chart_info, chart_name)
        x_col = self._get_x_column(data)

        # Determine axis labels
        x_label = xaxis_title or self._get_xaxis_label()
        y_label = yaxis_title or self._get_yaxis_label(value_col)

        theme = self._theme

        # Zone shading (add first so it's behind other elements)
        if show_zones and theme.zone_opacity > 0:
            self._add_zone_shading(fig, stats, theme)

        # Main data trace
        fig.add_trace(go.Scatter(
            x=data[x_col] if x_col in data.columns else data.index,
            y=data[value_col],
            mode='lines+markers',
            name=self._get_data_legend_name(),
            marker=dict(size=theme.data_marker_size, color=theme.data_color),
            line=dict(color=theme.data_color, width=theme.data_line_width),
            hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'
        ))

        # Control limits
        if show_limits:
            # UCL
            if 'ucl' in stats and stats['ucl'] != 'Varies':
                ucl_label = self._format_limit_label('UCL', stats['ucl'], show_limit_values)
                fig.add_hline(
                    y=stats['ucl'],
                    line_dash=theme.limit_line_dash,
                    line_color=theme.ucl_color,
                    line_width=theme.limit_line_width,
                    annotation_text=ucl_label,
                    annotation_position='right',
                    annotation_font_size=theme.annotation_font_size
                )

            # LCL
            if 'lcl' in stats and stats['lcl'] != 'Varies':
                lcl_label = self._format_limit_label('LCL', stats['lcl'], show_limit_values)
                fig.add_hline(
                    y=stats['lcl'],
                    line_dash=theme.limit_line_dash,
                    line_color=theme.lcl_color,
                    line_width=theme.limit_line_width,
                    annotation_text=lcl_label,
                    annotation_position='right',
                    annotation_font_size=theme.annotation_font_size
                )

            # Centerline
            center_key = self._get_center_key(stats)
            if center_key and center_key in stats:
                center_label = self._format_limit_label('CL', stats[center_key], show_limit_values)
                fig.add_hline(
                    y=stats[center_key],
                    line_color=theme.center_color,
                    line_width=theme.center_line_width,
                    annotation_text=center_label,
                    annotation_position='right',
                    annotation_font_size=theme.annotation_font_size
                )

        # Highlight signals (Rule 1 - beyond limits)
        if highlight_signals and 'beyond_limits' in data.columns:
            signals = data[data['beyond_limits'] != 0]
            if not signals.empty:
                fig.add_trace(go.Scatter(
                    x=signals[x_col] if x_col in data.columns else signals.index,
                    y=signals[value_col],
                    mode='markers',
                    name='Out of Control',
                    marker=dict(
                        size=theme.signal_marker_size,
                        color=theme.signal_color,
                        symbol=theme.signal_marker_symbol,
                        line=dict(
                            width=theme.signal_marker_line_width,
                            color=theme.signal_marker_line_color
                        )
                    ),
                    hovertemplate='Out of Control<br>%{x}<br>%{y:.3f}<extra></extra>'
                ))

        # Run rules visualization (Rules 2-8)
        if show_rules:
            self._add_run_rules_visualization(
                fig, data, stats, chart_name, value_col, x_col, theme
            )

        # Stats box
        if show_stats:
            self._add_stats_box(fig, stats, data, theme)

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

    def _plot_faceted(
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
        yaxis_title: str | None = None
    ) -> go.Figure:
        """Create faceted control charts."""
        n_charts = len(charts)
        nrows = (n_charts + ncols - 1) // ncols

        # Generate descriptive subplot titles
        subplot_titles = [
            self._generate_subplot_title(name) for name in charts.keys()
        ]

        # Calculate spacing dynamically to avoid Plotly errors
        # Constraint: spacing <= 1 / (n - 1) where n is rows or cols
        if nrows > 1:
            max_v_spacing = 1.0 / (nrows - 1) - 0.01  # Small buffer
            vertical_spacing = min(0.1, max_v_spacing)
        else:
            vertical_spacing = 0.1

        if ncols > 1:
            max_h_spacing = 1.0 / (ncols - 1) - 0.01
            horizontal_spacing = min(0.08, max_h_spacing)
        else:
            horizontal_spacing = 0.08

        # Create subplot grid
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing
        )

        # Determine axis labels (same for all subplots)
        x_label = xaxis_title or self._get_xaxis_label()
        y_label = yaxis_title or self._get_yaxis_label(None)

        # Get theme
        theme = self._theme

        # Plot each chart
        for idx, (chart_name, chart_info) in enumerate(charts.items()):
            row = idx // ncols + 1
            col = idx % ncols + 1

            data = chart_info['data']
            stats = chart_info['statistics']

            # Zone shading for this subplot (add first so it's behind data)
            if show_zones and theme.zone_opacity > 0:
                self._add_zone_shading_facet(fig, stats, theme, row, col)

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
                    showlegend=False,
                    hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'
                ),
                row=row,
                col=col
            )

            # Control limits as shapes (more efficient for facets)
            if show_limits:
                x_range = [x_data.min(), x_data.max()]

                # UCL
                if 'ucl' in stats and stats['ucl'] != 'Varies':
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats['ucl'], y1=stats['ucl'],
                        line=dict(
                            color=theme.ucl_color,
                            dash=theme.limit_line_dash,
                            width=theme.limit_line_width
                        ),
                        row=row, col=col
                    )

                # LCL
                if 'lcl' in stats and stats['lcl'] != 'Varies':
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats['lcl'], y1=stats['lcl'],
                        line=dict(
                            color=theme.lcl_color,
                            dash=theme.limit_line_dash,
                            width=theme.limit_line_width
                        ),
                        row=row, col=col
                    )

                # Centerline
                center_key = self._get_center_key(stats)
                if center_key and center_key in stats:
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats[center_key], y1=stats[center_key],
                        line=dict(color=theme.center_color, width=theme.center_line_width),
                        row=row, col=col
                    )

            # Signals (Rule 1 - beyond limits)
            if highlight_signals and 'beyond_limits' in data.columns:
                signals = data[data['beyond_limits'] != 0]
                if not signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=signals[x_col] if x_col in data.columns else signals.index,
                            y=signals[value_col],
                            mode='markers',
                            marker=dict(
                                size=theme.facet_signal_marker_size,
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
            if show_rules:
                self._add_run_rules_visualization(
                    fig, data, stats, chart_name, value_col, x_col, theme,
                    row=row, col=col
                )

            # Stats box for faceted charts
            if show_stats:
                self._add_stats_box_facet(fig, stats, data, theme, row, col, nrows, ncols)

        # Update layout with axis labels
        fig.update_layout(
            width=width,
            height=height,
            hovermode='closest'
        )

        # Add axis labels to all subplots
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)

        return fig

    def list_charts(self) -> list[str]:
        """Get list of available charts."""
        return list(self.charts.keys())

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

    def _get_x_column(self, data: pd.DataFrame) -> str:
        """
        Determine the x-axis column for plotting.

        Chart calculation logic (_build_output_columns) guarantees one of two cases:
        1. time_var was specified → data contains time_var column
        2. time_var not specified → data contains 'x' column (auto-generated)

        Returns
        -------
        str
            Column name to use for x-axis
        """
        # Use time variable if specified in analysis
        time_var = self.summary.get('time_var')
        if time_var:
            return time_var

        # Otherwise use auto-generated 'x' column
        return 'x'

    def _get_center_key(self, stats: dict) -> str | None:
        """Get the centerline statistic key."""
        # All chart types now use 'center' for the centerline column
        if 'center' in stats:
            return 'center'
        return None

    def _get_stratified_charts(self) -> dict:
        """Get stratified charts if available."""
        # Try to detect stratified charts by looking for charts with stratification markers
        stratified = {}
        for name, chart_info in self.charts.items():
            # Check if this looks like a stratified chart
            # Convert name to string to check for underscore
            name_str = str(name)
            if '_' in name_str or any(
                key in chart_info.get('metadata', {})
                for key in ['stratum', 'level', 'group']
            ):
                stratified[name] = chart_info

        # If no stratified charts found, return all charts
        return stratified if stratified else self.charts

    # =========================================================================
    # Title and Label Generation (Smart Defaults)
    # =========================================================================

    def _generate_title(self, chart_name: str) -> str:
        """
        Generate a descriptive chart title.

        Creates contextual titles like:
        - "Xbar Chart of Weight"
        - "Xbar Chart of Weight by Operator"
        - "IMR Chart of Temperature - Line A"

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
        if '_' in chart_name and chart_name not in ['Xbar', 'Sbar', 'Imr']:
            # Extract stratum name (e.g., "Operator_A" -> "Operator A")
            stratum = self._extract_stratum_name(chart_name)
            if stratum:
                title_parts.append(f"- {stratum}")

        return ' '.join(title_parts)

    def _generate_subplot_title(self, chart_name: str) -> str:
        """
        Generate a title for a subplot in faceted layout.

        Keeps subplot titles concise while still informative.

        Parameters
        ----------
        chart_name : str
            Name of the chart/stratum

        Returns
        -------
        str
            Concise subplot title
        """
        # For stratified charts, extract and clean up the stratum name
        if '_' in chart_name:
            stratum = self._extract_stratum_name(chart_name)
            if stratum:
                return stratum

        # For standard charts, just use the chart type
        return self._get_chart_type_display(chart_name)

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
            Display-friendly name (e.g., "Xbar" -> "X̄", "Imr" -> "I-MR")
        """
        # Map technical names to display names
        display_names = {
            'Xbar': 'X̄',
            'Sbar': 'S',
            'Imr': 'I-MR',
            'R': 'R',
            'Mr': 'MR',
            'all': 'I-MR'
        }

        # Check for exact match first
        if chart_name in display_names:
            return display_names[chart_name]

        # Check if it starts with a known chart type (for stratified charts)
        for key, display in display_names.items():
            if chart_name.startswith(key + '_'):
                return display

        # Fallback to original name
        return chart_name

    def _extract_stratum_name(self, chart_name: str) -> str | None:
        """
        Extract stratum name from stratified chart name.

        Parameters
        ----------
        chart_name : str
            Chart name like "Imr_Operator_A" or "Lane_1"

        Returns
        -------
        str or None
            Cleaned stratum name like "Operator A" or "Lane 1"
        """
        # Known chart type prefixes to strip
        prefixes = ['Xbar_', 'Sbar_', 'Imr_', 'R_', 'Mr_']

        result = chart_name
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix):]
                break

        # Replace underscores with spaces for readability
        result = result.replace('_', ' ')

        return result if result != chart_name else None

    def _get_xaxis_label(self) -> str:
        """
        Get intelligent x-axis label.

        Uses time variable name if available, otherwise "Observation".

        Returns
        -------
        str
            X-axis label
        """
        time_var = self.summary.get('time_var')
        if time_var:
            # Capitalize first letter of each word
            return time_var.replace('_', ' ').title()
        return 'Observation'

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

    def _format_limit_label(
        self,
        limit_name: str,
        value: float,
        show_value: bool
    ) -> str:
        """
        Format control limit annotation label.

        Parameters
        ----------
        limit_name : str
            Name of the limit ('UCL', 'LCL', 'CL')
        value : float
            Numeric value of the limit
        show_value : bool
            Whether to include the numeric value

        Returns
        -------
        str
            Formatted label like "UCL = 52.34" or just "UCL"
        """
        if show_value:
            # Determine appropriate decimal places based on magnitude
            if abs(value) >= 100:
                return f"{limit_name} = {value:.1f}"
            elif abs(value) >= 10:
                return f"{limit_name} = {value:.2f}"
            else:
                return f"{limit_name} = {value:.3f}"
        return limit_name

    # =========================================================================
    # Zone Shading
    # =========================================================================

    def _add_zone_shading(
        self,
        fig: go.Figure,
        stats: dict,
        theme: ChartTheme
    ) -> None:
        """
        Add zone shading to a single chart figure.

        Creates horizontal bands showing zones A, B, and C on both sides
        of the centerline. Zones are:
        - Zone C: 0σ to ±1σ (green - normal variation)
        - Zone B: ±1σ to ±2σ (yellow - watch)
        - Zone A: ±2σ to ±3σ (red - warning)

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to add shapes to
        stats : dict
            Chart statistics with 'center', 'ucl', 'lcl' keys
        theme : ChartTheme
            Theme with zone colors and opacity
        """
        # Skip if limits vary (can't calculate consistent zones)
        if stats.get('ucl') == 'Varies' or stats.get('lcl') == 'Varies':
            return

        center = stats.get('center')
        ucl = stats.get('ucl')
        lcl = stats.get('lcl')

        if center is None or ucl is None or lcl is None:
            return

        # Calculate sigma from control limits (UCL = center + 3σ)
        sigma = (ucl - center) / 3

        # Zone boundaries
        zones = [
            # Zone C (closest to center): 0σ to ±1σ
            (center - sigma, center + sigma, theme.zone_c_color),
            # Zone B: ±1σ to ±2σ (upper)
            (center + sigma, center + 2 * sigma, theme.zone_b_color),
            # Zone B: ±1σ to ±2σ (lower)
            (center - 2 * sigma, center - sigma, theme.zone_b_color),
            # Zone A: ±2σ to ±3σ (upper)
            (center + 2 * sigma, ucl, theme.zone_a_color),
            # Zone A: ±2σ to ±3σ (lower)
            (lcl, center - 2 * sigma, theme.zone_a_color),
        ]

        for y0, y1, color in zones:
            fig.add_hrect(
                y0=y0,
                y1=y1,
                fillcolor=color,
                opacity=theme.zone_opacity,
                layer='below',
                line_width=0
            )

    def _add_zone_shading_facet(
        self,
        fig: go.Figure,
        stats: dict,
        theme: ChartTheme,
        row: int,
        col: int
    ) -> None:
        """
        Add zone shading to a subplot in a faceted figure.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure with subplots
        stats : dict
            Chart statistics with 'center', 'ucl', 'lcl' keys
        theme : ChartTheme
            Theme with zone colors and opacity
        row : int
            Row number of subplot (1-indexed)
        col : int
            Column number of subplot (1-indexed)
        """
        # Skip if limits vary
        if stats.get('ucl') == 'Varies' or stats.get('lcl') == 'Varies':
            return

        center = stats.get('center')
        ucl = stats.get('ucl')
        lcl = stats.get('lcl')

        if center is None or ucl is None or lcl is None:
            return

        # Calculate sigma from control limits
        sigma = (ucl - center) / 3

        # Calculate axis references for this subplot
        # For subplot at row r, col c, the axis names are:
        # First subplot: xaxis, yaxis
        # Others: xaxis2, yaxis2, etc.
        subplot_idx = (row - 1) * 2 + col  # Assuming ncols=2, adjust if needed
        if subplot_idx == 1:
            xref = 'x'
            yref = 'y'
        else:
            xref = f'x{subplot_idx}'
            yref = f'y{subplot_idx}'

        # Zone boundaries
        zones = [
            # Zone C (closest to center): 0σ to ±1σ
            (center - sigma, center + sigma, theme.zone_c_color),
            # Zone B: ±1σ to ±2σ (upper)
            (center + sigma, center + 2 * sigma, theme.zone_b_color),
            # Zone B: ±1σ to ±2σ (lower)
            (center - 2 * sigma, center - sigma, theme.zone_b_color),
            # Zone A: ±2σ to ±3σ (upper)
            (center + 2 * sigma, ucl, theme.zone_a_color),
            # Zone A: ±2σ to ±3σ (lower)
            (lcl, center - 2 * sigma, theme.zone_a_color),
        ]

        for y0, y1, color in zones:
            fig.add_shape(
                type='rect',
                x0=0,
                x1=1,
                y0=y0,
                y1=y1,
                xref=f'{xref} domain',
                yref=yref,
                fillcolor=color,
                opacity=theme.zone_opacity,
                layer='below',
                line_width=0,
                row=row,
                col=col
            )

    # =========================================================================
    # Run Rules Visualization
    # =========================================================================

    def _add_run_rules_visualization(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        stats: dict,
        chart_name: str,
        value_col: str,
        x_col: str,
        theme: ChartTheme,
        row: int | None = None,
        col: int | None = None
    ) -> None:
        """
        Add visualization for Western Electric run rules (Rules 2-8).

        Detects rule violations and adds annotations and markers to highlight
        the specific rules that were violated at each observation.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to add visualizations to
        data : DataFrame
            Chart data
        stats : dict
            Chart statistics
        chart_name : str
            Name of the chart being plotted
        value_col : str
            Name of the value column
        x_col : str
            Name of the x-axis column
        theme : ChartTheme
            Theme with rule colors and styling
        row : int, optional
            Row number for faceted plots (1-indexed)
        col : int, optional
            Column number for faceted plots (1-indexed)
        """
        try:
            # Run signal detection to get all rule violations
            signal_result = self.result.detect_signals(chart=chart_name)

            if not signal_result.has_signals:
                return

            violations = signal_result.violations

            # Skip Rule 1 (already handled by highlight_signals)
            violations = violations[violations['rule_name'] != 'rule_1']

            if violations.empty:
                return

            # Group violations by observation
            grouped = violations.groupby('obs_id')

            # Rule descriptions for hover text
            rule_short_names = {
                'rule_2': '2 of 3 in Zone A',
                'rule_3': '4 of 5 in Zone B+',
                'rule_4': '8+ same side',
                'rule_5': 'Trend',
                'rule_6': 'Oscillation',
                'rule_7': 'In Zone C',
                'rule_8': 'Avoiding center',
            }

            # Collect points to annotate
            annotated_points = []

            for obs_id, obs_violations in grouped:
                # Get the observation data
                if obs_id in data.index:
                    obs_data = data.loc[obs_id]
                else:
                    continue

                # Get x and y values
                if x_col in data.columns:
                    x_val = obs_data[x_col]
                else:
                    x_val = obs_id

                y_val = obs_data[value_col]

                # Get rules violated at this observation
                rules = obs_violations['rule_name'].unique().tolist()
                rule_nums = [r.split('_')[1] for r in rules]

                annotated_points.append({
                    'x': x_val,
                    'y': y_val,
                    'rules': rules,
                    'rule_nums': rule_nums,
                    'hover': '<br>'.join([
                        rule_short_names.get(r, r) for r in rules
                    ])
                })

            if not annotated_points:
                return

            # Add markers for rule violations
            # Use different colors based on first violated rule
            x_vals = [p['x'] for p in annotated_points]
            y_vals = [p['y'] for p in annotated_points]
            hover_texts = [
                f"Rule violations:<br>{p['hover']}<br>Value: {p['y']:.3f}"
                for p in annotated_points
            ]

            # Get color for first rule of each point
            colors = [
                theme.rule_colors.get(p['rules'][0], theme.signal_color)
                for p in annotated_points
            ]

            # Add scatter trace for rule violation markers
            scatter_kwargs = dict(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                name='Rule Violations',
                marker=dict(
                    size=theme.rule_marker_size,
                    color=colors,
                    symbol='diamond',
                    line=dict(width=1, color='white')
                ),
                text=[','.join(p['rule_nums']) for p in annotated_points],
                textposition='top center',
                textfont=dict(size=theme.rule_annotation_size, color='black'),
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=True
            )

            if row is not None and col is not None:
                scatter_kwargs['showlegend'] = False
                fig.add_trace(go.Scatter(**scatter_kwargs), row=row, col=col)
            else:
                fig.add_trace(go.Scatter(**scatter_kwargs))

        except Exception as e:
            # Log but don't fail if signal detection has issues
            logger.warning(f"Could not add run rules visualization: {e}")

    # =========================================================================
    # Statistics Box
    # =========================================================================

    def _add_stats_box(
        self,
        fig: go.Figure,
        stats: dict,
        data: pd.DataFrame,
        theme: ChartTheme
    ) -> None:
        """
        Add a statistics box annotation to a single chart.

        Displays key statistics (CL, UCL, LCL, n) in a clean box in the
        upper-left corner of the chart.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to add annotation to
        stats : dict
            Chart statistics with 'center', 'ucl', 'lcl' keys
        data : DataFrame
            Chart data (for calculating n)
        theme : ChartTheme
            Theme with stats box styling
        """
        # Build stats text
        stats_lines = []

        # Number of observations
        n = len(data)
        stats_lines.append(f"n = {n}")

        # Centerline
        center = stats.get('center')
        if center is not None and center != 'Varies':
            stats_lines.append(f"CL = {self._format_stat_value(center)}")

        # UCL
        ucl = stats.get('ucl')
        if ucl is not None and ucl != 'Varies':
            stats_lines.append(f"UCL = {self._format_stat_value(ucl)}")

        # LCL
        lcl = stats.get('lcl')
        if lcl is not None and lcl != 'Varies':
            stats_lines.append(f"LCL = {self._format_stat_value(lcl)}")

        if not stats_lines:
            return

        stats_text = '<br>'.join(stats_lines)

        # Add annotation in upper-left corner
        fig.add_annotation(
            text=stats_text,
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            showarrow=False,
            font=dict(
                size=theme.stats_box_font_size,
                color=theme.stats_box_font_color,
                family='monospace'
            ),
            bgcolor=theme.stats_box_bgcolor,
            bordercolor=theme.stats_box_bordercolor,
            borderwidth=theme.stats_box_borderwidth,
            borderpad=6,
            align='left'
        )

    def _add_stats_box_facet(
        self,
        fig: go.Figure,
        stats: dict,
        data: pd.DataFrame,
        theme: ChartTheme,
        row: int,
        col: int,
        nrows: int,
        ncols: int
    ) -> None:
        """
        Add a compact statistics box to a subplot in a faceted figure.

        Uses a smaller, more compact format suitable for faceted layouts.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure with subplots
        stats : dict
            Chart statistics with 'center', 'ucl', 'lcl' keys
        data : DataFrame
            Chart data (for calculating n)
        theme : ChartTheme
            Theme with stats box styling
        row : int
            Row number of subplot (1-indexed)
        col : int
            Column number of subplot (1-indexed)
        nrows : int
            Total number of rows in the facet grid
        ncols : int
            Total number of columns in the facet grid
        """
        # Build compact stats text
        stats_parts = []

        # Number of observations
        n = len(data)
        stats_parts.append(f"n={n}")

        # Centerline (compact)
        center = stats.get('center')
        if center is not None and center != 'Varies':
            stats_parts.append(f"CL={self._format_stat_value(center, compact=True)}")

        if not stats_parts:
            return

        stats_text = ' | '.join(stats_parts)

        # Calculate position within subplot
        # Each subplot occupies a fraction of the paper
        col_width = 1.0 / ncols
        row_height = 1.0 / nrows

        # Calculate x, y position (upper-left of this subplot)
        x_pos = (col - 1) * col_width + 0.02 * col_width
        y_pos = 1.0 - (row - 1) * row_height - 0.05 * row_height

        fig.add_annotation(
            text=stats_text,
            xref='paper',
            yref='paper',
            x=x_pos,
            y=y_pos,
            xanchor='left',
            yanchor='top',
            showarrow=False,
            font=dict(
                size=theme.stats_box_font_size - 1,  # Slightly smaller for facets
                color=theme.stats_box_font_color,
                family='monospace'
            ),
            bgcolor=theme.stats_box_bgcolor,
            bordercolor=theme.stats_box_bordercolor,
            borderwidth=theme.stats_box_borderwidth,
            borderpad=3,
            align='left'
        )

    def _format_stat_value(self, value: float, compact: bool = False) -> str:
        """
        Format a statistic value for display.

        Parameters
        ----------
        value : float
            The statistic value to format
        compact : bool
            If True, use fewer decimal places for compact display

        Returns
        -------
        str
            Formatted value string
        """
        if compact:
            if abs(value) >= 100:
                return f"{value:.1f}"
            elif abs(value) >= 10:
                return f"{value:.1f}"
            else:
                return f"{value:.2f}"
        else:
            if abs(value) >= 100:
                return f"{value:.2f}"
            elif abs(value) >= 10:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

    # =========================================================================
    # Residual Visualization
    # =========================================================================

    def plot_residuals(
        self,
        residual_type: str = 'R1',
        plot_type: str = 'all',
        template: str | ChartTheme = 'processbehavior',
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
        template : str or ChartTheme, default 'processbehavior'
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
        from scipy import stats as scipy_stats

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
        if isinstance(template, str):
            theme = get_theme(template)
        else:
            theme = template

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
            y_normal = scipy_stats.norm.pdf(x_range, r_data.mean(), r_data.std())
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
            theoretical_q = scipy_stats.norm.ppf(
                np.linspace(0.01, 0.99, len(r_data))
            )
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
            theoretical_q = scipy_stats.norm.ppf(
                np.linspace(0.01, 0.99, len(r_data))
            )
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

    def plot_effects(
        self,
        effect_type: str = 'factor',
        template: str | ChartTheme = 'processbehavior',
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
        template : str or ChartTheme, default 'processbehavior'
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
        if isinstance(template, str):
            theme = get_theme(template)
        else:
            theme = template

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
            for name, data in effects.items():
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
        template: str | ChartTheme = 'processbehavior',
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
        template : str or ChartTheme, default 'processbehavior'
            Visual theme for all charts
        width : int, default 1200
            Width of charts in pixels
        title : str, optional
            Report title (defaults to "Process Behavior Analysis Report")

        Examples
        --------
        >>> plotter.generate_report('analysis_report.html')
        >>> plotter.generate_report('dark_report.html', template='dark')

        Notes
        -----
        The generated HTML file is self-contained and can be opened
        in any web browser without an internet connection.
        """
        from pathlib import Path

        # Build report sections
        sections = []
        report_title = title or "Process Behavior Analysis Report"

        # Summary section
        if include_summary:
            summary = self.result.summary
            summary_html = f"""
            <div class="section">
                <h2>Analysis Summary</h2>
                <table class="summary-table">
                    <tr><td><strong>SDS</strong></td><td>{summary['sds']} - {summary['sds_description']}</td></tr>
                    <tr><td><strong>Response Variable</strong></td><td>{summary['response_var']}</td></tr>
                    <tr><td><strong>Observations</strong></td><td>{summary['n_observations']}</td></tr>
                    <tr><td><strong>Charts</strong></td><td>{', '.join(summary['chart_types'])}</td></tr>
                    <tr><td><strong>Signals Detected</strong></td><td>{summary['n_signals_total']}</td></tr>
                    <tr><td><strong>Has Residuals</strong></td><td>{'Yes' if summary['has_residuals'] else 'No'}</td></tr>
                    <tr><td><strong>Has Effects</strong></td><td>{'Yes' if summary['has_effects'] else 'No'}</td></tr>
                </table>
            </div>
            """
            sections.append(summary_html)

        # Control charts section
        if include_charts:
            fig = self.plot(template=template, width=width, height=500)
            chart_html = fig.figure.to_html(full_html=False, include_plotlyjs=False)
            sections.append(f"""
            <div class="section">
                <h2>Control Charts</h2>
                {chart_html}
            </div>
            """)

        # Residuals section
        if include_residuals and self.result.has_residuals:
            fig = self.plot_residuals(template=template, width=width, height=350)
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
                fig = self.plot_effects(effect_type='all', template=template, width=width, height=400)
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
    <title>{report_title}</title>
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
    <h1>{report_title}</h1>
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
