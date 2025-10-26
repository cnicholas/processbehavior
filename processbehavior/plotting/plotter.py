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
from .themes import apply_theme

if TYPE_CHECKING:
    from typing import Literal, Optional

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

    def plot(
        self,
        chart: Optional[str] = None,
        facet: bool = False,
        facet_by: Optional[str] = None,
        ncols: int = 2,
        highlight_signals: bool = True,
        show_limits: bool = True,
        show_rules: bool = False,
        template: str = 'processbehavior',
        width: int = 1000,
        height: Optional[int] = None,
        title: Optional[str] = None
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
        show_rules : bool, default False
            Whether to show additional run rules
        template : str, default 'processbehavior'
            Visual theme ('processbehavior', 'minimal', 'dark')
        width : int, default 1000
            Figure width in pixels
        height : int, optional
            Figure height in pixels (auto-calculated if None)
        title : str, optional
            Custom title for the figure

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

        # Calculate height if not specified
        if height is None:
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
                width=width,
                height=height
            )
        else:
            fig = self._plot_faceted(
                charts_to_plot,
                ncols=ncols,
                highlight_signals=highlight_signals,
                show_limits=show_limits,
                width=width,
                height=height
            )

        # Apply theme
        fig = apply_theme(fig, template)

        # Set title
        if title:
            fig.update_layout(title=title)
        elif not title and len(charts_to_plot) == 1:
            chart_name = list(charts_to_plot.keys())[0]
            fig.update_layout(title=f"{chart_name} Control Chart")

        # Wrap in our custom figure class
        return ControlChartFigure(fig, self.result)

    def _plot_single_chart(
        self,
        chart_info: dict,
        chart_name: str,
        highlight_signals: bool,
        show_limits: bool,
        width: int,
        height: int
    ) -> go.Figure:
        """Create a single control chart."""
        data = chart_info['data']
        stats = chart_info['statistics']

        fig = go.Figure()

        # Determine value column
        value_col = self._get_value_column(data, chart_name)
        x_col = self._get_x_column(data)

        # Main data trace
        fig.add_trace(go.Scatter(
            x=data[x_col] if x_col in data.columns else data.index,
            y=data[value_col],
            mode='lines+markers',
            name='Data',
            marker=dict(size=8, color='steelblue'),
            line=dict(color='steelblue', width=2),
            hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'
        ))

        # Control limits
        if show_limits:
            # UCL
            if 'ucl' in stats and stats['ucl'] != 'Varies':
                fig.add_hline(
                    y=stats['ucl'],
                    line_dash='dash',
                    line_color='red',
                    annotation_text='UCL',
                    annotation_position='right'
                )

            # LCL
            if 'lcl' in stats and stats['lcl'] != 'Varies':
                fig.add_hline(
                    y=stats['lcl'],
                    line_dash='dash',
                    line_color='red',
                    annotation_text='LCL',
                    annotation_position='right'
                )

            # Centerline
            center_key = self._get_center_key(stats)
            if center_key and center_key in stats:
                fig.add_hline(
                    y=stats[center_key],
                    line_color='green',
                    annotation_text='Center',
                    annotation_position='right'
                )

        # Highlight signals
        if highlight_signals and 'beyond_limits' in data.columns:
            signals = data[data['beyond_limits'] != 0]
            if not signals.empty:
                fig.add_trace(go.Scatter(
                    x=signals[x_col] if x_col in data.columns else signals.index,
                    y=signals[value_col],
                    mode='markers',
                    name='Signal',
                    marker=dict(
                        size=14,
                        color='red',
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='Signal<br>%{x}<br>%{y:.3f}<extra></extra>'
                ))

        # Layout
        fig.update_layout(
            width=width,
            height=height,
            xaxis_title='Observation',
            yaxis_title=value_col.capitalize(),
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
        width: int,
        height: int
    ) -> go.Figure:
        """Create faceted control charts."""
        n_charts = len(charts)
        nrows = (n_charts + ncols - 1) // ncols

        # Create subplot grid
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=list(charts.keys()),
            vertical_spacing=0.1,
            horizontal_spacing=0.08
        )

        # Plot each chart
        for idx, (chart_name, chart_info) in enumerate(charts.items()):
            row = idx // ncols + 1
            col = idx % ncols + 1

            data = chart_info['data']
            stats = chart_info['statistics']

            value_col = self._get_value_column(data, chart_name)
            x_col = self._get_x_column(data)

            # Main trace
            x_data = data[x_col] if x_col in data.columns else data.index
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=data[value_col],
                    mode='lines+markers',
                    name=chart_name,
                    marker=dict(size=6, color='steelblue'),
                    line=dict(color='steelblue', width=1.5),
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
                        line=dict(color='red', dash='dash', width=1),
                        row=row, col=col
                    )

                # LCL
                if 'lcl' in stats and stats['lcl'] != 'Varies':
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats['lcl'], y1=stats['lcl'],
                        line=dict(color='red', dash='dash', width=1),
                        row=row, col=col
                    )

                # Centerline
                center_key = self._get_center_key(stats)
                if center_key and center_key in stats:
                    fig.add_shape(
                        type='line',
                        x0=x_range[0], x1=x_range[1],
                        y0=stats[center_key], y1=stats[center_key],
                        line=dict(color='green', width=1.5),
                        row=row, col=col
                    )

            # Signals
            if highlight_signals and 'beyond_limits' in data.columns:
                signals = data[data['beyond_limits'] != 0]
                if not signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=signals[x_col] if x_col in data.columns else signals.index,
                            y=signals[value_col],
                            mode='markers',
                            marker=dict(size=10, color='red', symbol='x'),
                            showlegend=False,
                            hovertemplate='Signal<br>%{x}<br>%{y:.3f}<extra></extra>'
                        ),
                        row=row,
                        col=col
                    )

        # Update layout
        fig.update_layout(
            width=width,
            height=height,
            hovermode='closest'
        )

        return fig

    def list_charts(self) -> list[str]:
        """Get list of available charts."""
        return list(self.charts.keys())

    # Helper methods
    def _get_value_column(self, data: pd.DataFrame, chart_name: str) -> str:
        """Determine the value column for a chart."""
        if 'mean' in data.columns:
            return 'mean'
        elif 's' in data.columns:
            return 's'
        elif 'mr' in data.columns:
            return 'mr'
        else:
            # Get first numeric column
            numeric_cols = data.select_dtypes(include='number').columns
            return numeric_cols[0] if len(numeric_cols) > 0 else data.columns[0]

    def _get_x_column(self, data: pd.DataFrame) -> str:
        """Determine the x-axis column."""
        if 'x' in data.columns:
            return 'x'
        else:
            return data.index.name or 'index'

    def _get_center_key(self, stats: dict) -> Optional[str]:
        """Get the centerline statistic key."""
        for key in ['Mean', 'mean', 'S', 'mR']:
            if key in stats:
                return key
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
