"""Main plotting coordinator for ProcessBehavior analysis results.

Orchestrates chart creation by delegating to specialised renderers.
This module is the routing/coordination layer — actual rendering lives
in ``renderers.py``, ``residuals.py``, ``report.py``, and
``effects_charts.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data_preparation import encode_rsg
from ..exceptions import ChartNotAvailableError, ValidationError
from ..spc_constants import normalize_chart_name
from .contracts import PlotError, build_render_context, build_render_spec
from .control_chart import ControlChartFigure
from .renderers import render_control_chart, render_histogram
from .themes import ChartTheme, apply_theme, get_theme

if TYPE_CHECKING:
    from processbehavior.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)

_EFFECTS_CHART_TYPES = frozenset({
    'Effects', 'MainEffects', 'TimeEffects',
    'TimeInteraction', 'FactorInteraction',
})

_RESIDUAL_LABELS = {
    'R1': 'Within-Subgroup',
    'R2': 'Within-Subgroup Variation',
    'R3': 'Interaction',
    'R4': 'Time Effects',
    'R5': 'Factor Effects',
}


class Plotter:
    """Unified plotting interface for ProcessBehavior analysis results.

    Coordinates chart creation with built-in support for single and
    faceted control charts, residual diagnostics, effects visualisation,
    and HTML report generation.

    Examples
    --------
    Simple plotting::

        plotter = Plotter(result)
        fig = plotter.plot()
        fig.show()

    Faceted charts::

        fig = plotter.plot(facet=True)

    Customize appearance::

        fig = plotter.plot(chart='Xbar', theme='minimal',
                           highlight_signals=True)
    """

    def __init__(self, analysis_result: AnalysisResult):
        self.result = analysis_result
        self.charts = analysis_result.charts
        self.summary = analysis_result.summary
        self._theme: ChartTheme | None = None

    # -----------------------------------------------------------------
    #  Main entry point
    # -----------------------------------------------------------------

    def plot(  # noqa: C901
        self,
        chart: str | None = None,
        facet: bool = False,
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
        vertical_spacing: float = 0.15,
    ) -> ControlChartFigure:
        """Create control chart visualisation.

        Parameters
        ----------
        chart : str, optional
            Specific chart to plot ('Xbar', 'S', 'XmR', etc.).
            Also accepts effects chart types: 'Effects', 'MainEffects',
            'TimeEffects', 'TimeInteraction', 'FactorInteraction'.
            If None, plots all available charts.
        facet : bool, default False
            Whether to create faceted plot for stratified data.
        ncols : int, default 2
            Number of columns in faceted layout.
        highlight_signals : bool, default True
            Highlight points beyond control limits.
        show_limits : bool, default True
            Show control limit lines.
        show_limit_values : bool, default True
            Show numeric values in limit labels.
        show_zones : bool, default False
            Show zone shading (A/B/C zones).
        show_rules : bool, default False
            Show additional run rules (Western Electric Rules 2-8).
        show_stats : bool, default False
            Show statistics box.
        theme : str or ChartTheme, default 'processbehavior'
            Visual theme.
        width : int, default 1000
            Figure width in pixels.
        height : int, optional
            Figure height in pixels (auto-calculated if None).
        aspect_ratio : float, optional
            Width-to-height ratio (overrides height if specified).
        title : str, optional
            Custom title.
        xaxis_title, yaxis_title : str, optional
            Custom axis labels.
        shared_yaxis : bool, default True
            Whether faceted charts share y-axis range.
        yaxis_padding : float, default 0.05
            Padding fraction for y-axis range.
        vertical_spacing : float, default 0.15
            Vertical spacing between rows in faceted layouts.

        Returns
        -------
        ControlChartFigure
            Interactive figure.
        """
        # Resolve theme
        if isinstance(theme, str):
            self._theme = get_theme(theme)
        else:
            self._theme = theme

        # Normalize chart name
        if chart:
            chart = normalize_chart_name(chart)

        # Handle effects charts
        if chart in _EFFECTS_CHART_TYPES:
            return self._plot_effects_chart(
                chart_type=chart, width=width, height=height, title=title,
            )

        # Validate
        if chart and chart not in self.charts:
            available = list(self.charts.keys())
            raise ChartNotAvailableError(
                f"Chart '{chart}' not found.\n"
                f"Available charts: {available}\n"
                f"Hint: Use list_charts() to see all options",
                chart=chart,
                available=available,
            )

        # Determine what to plot
        charts_to_plot = self._resolve_charts(chart, facet)

        if not charts_to_plot:
            raise PlotError(
                "No charts available to plot.\n"
                "Hint: Check result.charts to see available charts"
            )

        # Reorder stratified companion pairs for side-by-side layout
        charts_to_plot, forced_ncols = self._reorder_companion_pairs(charts_to_plot)
        if forced_ncols is not None:
            ncols = forced_ncols

        # Calculate height
        height = self._resolve_height(
            height, width, aspect_ratio, len(charts_to_plot), ncols,
        )

        # Build display options dict for reuse
        display_opts = dict(
            highlight_signals=highlight_signals,
            show_limits=show_limits,
            show_limit_values=show_limit_values,
            show_zones=show_zones,
            show_rules=show_rules,
            show_stats=show_stats,
        )

        # Create figure
        if len(charts_to_plot) == 1:
            fig = self._plot_single_chart(
                list(charts_to_plot.values())[0],
                list(charts_to_plot.keys())[0],
                width=width, height=height,
                xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                **display_opts,
            )
        else:
            effective_shared = shared_yaxis
            per_type_shared = False
            if shared_yaxis:
                base_types = {self._get_base_chart_type(n) for n in charts_to_plot}
                if len(base_types) > 1:
                    effective_shared = False
                    per_type_shared = True

            fig = self._plot_faceted(
                charts_to_plot, ncols=ncols,
                width=width, height=height,
                xaxis_title=xaxis_title, yaxis_title=yaxis_title,
                shared_yaxis=effective_shared,
                yaxis_padding=yaxis_padding,
                vertical_spacing=vertical_spacing,
                per_type_shared=per_type_shared,
                **display_opts,
            )

        # Apply theme
        fig = apply_theme(fig, self._theme)

        # Title
        if title:
            fig.update_layout(title=title)
        elif len(charts_to_plot) == 1:
            chart_name = list(charts_to_plot.keys())[0]
            chart_info = list(charts_to_plot.values())[0]
            fig.update_layout(title=self._generate_title(chart_name, chart_info))

        return ControlChartFigure(fig, self.result)

    # -----------------------------------------------------------------
    #  Single chart
    # -----------------------------------------------------------------

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
        yaxis_title: str | None = None,
    ) -> go.Figure:
        """Create a single control chart."""
        metadata = chart_info.get('metadata', {})

        # Histogram delegation
        if metadata.get('chart_type') == 'Histogram':
            return self._plot_histogram(
                chart_info, chart_name,
                show_stats=show_stats, width=width, height=height,
                xaxis_title=xaxis_title, yaxis_title=yaxis_title,
            )

        fig = go.Figure()

        # Build render context
        value_col = self._get_value_column(chart_info, chart_name)
        x_col = self._get_x_column(chart_info['data'])
        center_key = self._get_center_key(chart_info['statistics'])

        spec = build_render_spec(chart_info, chart_name, x_col, center_key)
        ctx = build_render_context(
            spec, chart_info, chart_name, self._theme, self.result,
            highlight_signals=highlight_signals,
            show_limits=show_limits, show_limit_values=show_limit_values,
            show_zones=show_zones, show_rules=show_rules,
            show_stats=show_stats, is_faceted=False,
        )

        # Render
        render_control_chart(fig, ctx)

        # Time tick labels
        self._apply_time_tick_labels(fig, chart_info['data'], metadata)

        # Force categorical axis — control chart x-axes are always ordinal
        # (evenly spaced in sequence order), never continuous numeric
        if x_col:
            fig.update_xaxes(type='category')

        # Layout
        x_label = xaxis_title or self._get_xaxis_label(x_col, chart_name)
        y_label = yaxis_title or self._get_yaxis_label(value_col)
        fig.update_layout(
            width=width, height=height,
            xaxis_title=x_label, yaxis_title=y_label,
            hovermode='x unified', showlegend=True,
        )

        return fig

    # -----------------------------------------------------------------
    #  Faceted charts
    # -----------------------------------------------------------------

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
        vertical_spacing: float = 0.15,
        per_type_shared: bool = False,
    ) -> go.Figure:
        """Create faceted control charts."""
        n_charts = len(charts)
        nrows = (n_charts + ncols - 1) // ncols

        subplot_titles = [
            self._generate_subplot_title(name, charts[name]) for name in charts
        ]

        # Dynamic spacing
        v_spacing, h_spacing = self._calculate_spacing(
            nrows, ncols, vertical_spacing,
        )

        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=subplot_titles,
            vertical_spacing=v_spacing,
            horizontal_spacing=h_spacing,
        )

        # Detect if all histograms
        all_histograms = all(
            ci.get('metadata', {}).get('chart_type') == 'Histogram'
            for ci in charts.values()
        )

        # Determine axis labels
        first_info = next(iter(charts.values()))
        first_data = first_info['data']

        if all_histograms:
            value_col = first_info.get('metadata', {}).get('value_col')
            x_label = xaxis_title or value_col
            y_label = yaxis_title or 'Count'
        else:
            x_col = self._get_x_column(first_data)
            first_name = next(iter(charts.keys()))
            x_label = xaxis_title or self._get_xaxis_label(x_col, first_name)
            y_label = yaxis_title or self._get_yaxis_label(None)

        # Shared binning / y-range
        global_y_range = None
        histogram_y_range = None
        histogram_bin_edges = None

        if all_histograms:
            histogram_bin_edges, _ = self._calculate_histogram_bin_edges(charts)

        if shared_yaxis:
            if all_histograms:
                histogram_y_range = self._calculate_histogram_yrange(
                    charts, histogram_bin_edges,
                )
            else:
                global_y_range = self._calculate_global_yrange(charts, yaxis_padding)

        theme = self._theme

        # Render each chart
        for idx, (chart_name, chart_info) in enumerate(charts.items()):
            row = idx // ncols + 1
            col = idx % ncols + 1
            metadata = chart_info.get('metadata', {})

            if metadata.get('chart_type') == 'Histogram':
                render_histogram(
                    fig, chart_info, theme,
                    show_stats=show_stats, row=row, col=col,
                    shared_bin_edges=histogram_bin_edges,
                    is_faceted=True,
                )
                continue

            # Build context for this panel
            data = chart_info['data']
            x_col = self._get_x_column(data)
            center_key = self._get_center_key(chart_info['statistics'])

            spec = build_render_spec(chart_info, chart_name, x_col, center_key)
            ctx = build_render_context(
                spec, chart_info, chart_name, theme, self.result,
                highlight_signals=highlight_signals,
                show_limits=show_limits, show_limit_values=show_limit_values,
                show_zones=show_zones, show_rules=show_rules,
                show_stats=show_stats, is_faceted=True,
            )

            render_control_chart(fig, ctx, row=row, col=col, nrows=nrows, ncols=ncols)

            # Time tick labels
            self._apply_time_tick_labels(fig, data, metadata, row=row, col=col)

        # Layout
        fig.update_layout(width=width, height=height, hovermode='closest')

        # Collect per-panel x_col to determine axis type per subplot
        time_var = self.summary.get('time_var')
        panel_x_cols: dict[int, str | None] = {}
        for idx, (_chart_name, chart_info) in enumerate(charts.items()):
            metadata = chart_info.get('metadata', {})
            if metadata.get('chart_type') == 'Histogram':
                panel_x_cols[idx] = None
            else:
                panel_x_cols[idx] = self._get_x_column(chart_info['data'])

        # Apply axis styling to all subplots, but only show titles on
        # leftmost column (y-axis) and bottom row (x-axis) to avoid clutter.
        axis_style_y = dict(
            showline=theme.show_axis_line,
            linecolor=theme.axis_line_color,
            linewidth=theme.axis_line_width,
            showgrid=theme.show_grid,
            gridcolor=theme.grid_color,
            gridwidth=theme.grid_width,
        )

        axis_style_x_base = dict(
            showline=theme.show_axis_line,
            linecolor=theme.axis_line_color,
            linewidth=theme.axis_line_width,
            showgrid=theme.show_grid,
            gridcolor=theme.grid_color,
            gridwidth=theme.grid_width,
        )

        for idx in range(n_charts):
            r = idx // ncols + 1
            c = idx % ncols + 1
            # Bottom row: last full row, or incomplete last row
            is_bottom = (r == nrows) or (idx + ncols >= n_charts)
            is_leftmost = (c == 1)
            panel_xcol = panel_x_cols.get(idx)
            fig.update_xaxes(
                title_text=x_label if is_bottom else None,
                **axis_style_x_base,
                **({"type": "category"} if panel_xcol else {}),
                row=r, col=c,
            )
            fig.update_yaxes(
                title_text=y_label if is_leftmost else None,
                **axis_style_y,
                row=r, col=c,
            )

        if histogram_y_range is not None:
            fig.update_yaxes(range=histogram_y_range, autorange=False)
        elif global_y_range is not None:
            fig.update_yaxes(range=global_y_range, autorange=False)
        elif per_type_shared:
            # Group charts by base type and share y-range within each group
            type_groups: dict[str, dict] = {}
            for chart_name, chart_info in charts.items():
                base = self._get_base_chart_type(chart_name)
                type_groups.setdefault(base, {})[chart_name] = chart_info

            chart_names_list = list(charts.keys())
            for _base_type, group_charts in type_groups.items():
                if all(
                    c.get('metadata', {}).get('chart_type') == 'Histogram'
                    for c in group_charts.values()
                ):
                    continue
                y_range = self._calculate_global_yrange(group_charts, yaxis_padding)
                if y_range is not None:
                    for cn in group_charts:
                        idx = chart_names_list.index(cn)
                        r = idx // ncols + 1
                        c = idx % ncols + 1
                        fig.update_yaxes(
                            range=y_range, autorange=False, row=r, col=c,
                        )

        return fig

    # -----------------------------------------------------------------
    #  Histogram (standalone)
    # -----------------------------------------------------------------

    def _plot_histogram(
        self,
        chart_info: dict,
        chart_name: str,
        show_stats: bool = True,
        width: int = 1000,
        height: int = 400,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,
    ) -> go.Figure:
        """Render standalone histogram with optional mean/std lines."""
        fig = go.Figure()
        render_histogram(
            fig, chart_info, self._theme,
            show_stats=show_stats, is_faceted=False,
        )

        metadata = chart_info.get('metadata', {})
        value_col = metadata.get('value_col')
        x_label = xaxis_title or value_col
        y_label = yaxis_title or "Count"

        fig.update_layout(
            width=width, height=height,
            xaxis_title=x_label, yaxis_title=y_label,
            showlegend=False, bargap=0.05,
        )
        return fig

    # -----------------------------------------------------------------
    #  Chart listing
    # -----------------------------------------------------------------

    def list_charts(self) -> list[str]:
        """Get list of available charts."""
        return list(self.charts.keys())

    # -----------------------------------------------------------------
    #  Effects charts
    # -----------------------------------------------------------------

    def _plot_effects_chart(
        self,
        chart_type: str,
        width: int = 1000,
        height: int | None = None,
        title: str | None = None,
    ) -> ControlChartFigure:
        """Route to appropriate effects chart function."""
        from .effects_charts import (
            create_factor_effects_chart,
            create_factor_interaction_chart,
            create_main_effects_chart,
            create_time_effects_chart,
            create_time_interaction_chart,
        )

        theme = self._theme
        effects = self.result.effects
        interactions = self.result.interactions

        if chart_type == 'Effects':
            if not self.result.has_effects:
                raise ValidationError(
                    "Effects not available for this analysis.\n"
                    "Effects require factors in the analysis specification."
                )
            fig = create_main_effects_chart(
                effects=effects, theme=theme, width=width, height=height,
            )

        elif chart_type == 'MainEffects':
            if not self.result.has_effects:
                raise ValidationError(
                    "Effects not available for this analysis.\n"
                    "Effects require factors in the analysis specification."
                )
            fig = create_factor_effects_chart(
                effects=effects, theme=theme, width=width, height=height or 500,
            )

        elif chart_type == 'TimeEffects':
            if 'time' not in effects and not any(
                isinstance(v, pd.DataFrame) and 'PT_ME' in v.columns
                for v in effects.values()
            ):
                raise ValidationError(
                    "Time effects not available for this analysis.\n"
                    "This requires a time variable in the analysis."
                )
            fig = create_time_effects_chart(
                effects=effects, theme=theme, width=width, height=height,
            )

        elif chart_type == 'TimeInteraction':
            if 'factor_time' not in interactions:
                raise ValidationError(
                    "Time interaction not available for this analysis.\n"
                    "This requires both factors and time variable."
                )
            factors = self.summary.get('grouping_vars', [])
            time_var = self.summary.get('time_var')
            if not factors or not time_var:
                raise ValidationError(
                    "Cannot create time interaction chart.\n"
                    "Requires both factors and time variable."
                )
            fig = create_time_interaction_chart(
                interactions=interactions, effects=effects,
                factors=factors, time_var=time_var,
                dataset=self.result.dataset,
                theme=theme, width=width, height=height or 500,
            )

        elif chart_type == 'FactorInteraction':
            if 'factor_factor' not in interactions:
                raise ValidationError(
                    "Factor interaction not available for this analysis.\n"
                    "This requires at least 2 factors in the analysis."
                )
            factors = self.summary.get('grouping_vars', [])
            if len(factors) < 2:
                raise ValidationError(
                    f"Factor interaction requires at least 2 factors, "
                    f"got {len(factors)}."
                )
            fig = create_factor_interaction_chart(
                interactions=interactions, factors=factors,
                theme=theme, width=width, height=height or 600,
            )

        else:
            raise ChartNotAvailableError(
                f"Unknown effects chart type: '{chart_type}'.\n"
                f"Options: 'Effects', 'MainEffects', 'TimeEffects', "
                f"'TimeInteraction', 'FactorInteraction'",
                chart=chart_type,
            )

        fig = apply_theme(fig, theme)
        if title:
            fig.update_layout(title=title)
        return ControlChartFigure(fig, self.result)

    # -----------------------------------------------------------------
    #  plot_effects (legacy bar-chart path)
    # -----------------------------------------------------------------

    def plot_effects(  # noqa: C901
        self,
        effect_type: str = 'factor',
        theme: str | ChartTheme = 'processbehavior',
        width: int = 800,
        height: int = 500,
    ) -> ControlChartFigure:
        """Create bar chart of main effects.

        Parameters
        ----------
        effect_type : str, default 'factor'
            Type of effects: 'factor', 'time', or 'all'.
        theme : str or ChartTheme
            Visual theme.
        width, height : int
            Figure dimensions.

        Returns
        -------
        ControlChartFigure
        """
        if not self.result.has_effects:
            raise ValidationError(
                "Effects not available for this analysis.\n"
                "Effects require SDS >= 2 (multiple factor levels)."
            )

        effects = self.result.effects
        theme = get_theme(theme) if isinstance(theme, str) else theme

        if effect_type == 'all':
            fig = self._effects_all(effects, theme, width, height)
        elif effect_type == 'factor':
            fig = self._effects_factor(effects, theme, width, height)
        elif effect_type == 'time':
            fig = self._effects_time(effects, theme, width, height)
        else:
            raise ValidationError(
                f"Invalid effect_type: '{effect_type}'.\n"
                f"Options: 'factor', 'time', 'all'"
            )

        fig = apply_theme(fig, theme)
        return ControlChartFigure(fig, self.result)

    # -----------------------------------------------------------------
    #  Residuals (delegate)
    # -----------------------------------------------------------------

    def plot_residuals(
        self,
        residual_type: str = 'R1',
        plot_type: str = 'all',
        theme: str | ChartTheme = 'processbehavior',
        width: int = 1200,
        height: int = 400,
    ) -> ControlChartFigure:
        """Create residual diagnostic plots.

        Parameters
        ----------
        residual_type : str, default 'R1'
            Which residual to plot ('R1'-'R5').
        plot_type : str, default 'all'
            'histogram', 'qq', 'sequence', or 'all'.
        theme : str or ChartTheme
            Visual theme.
        width, height : int
            Figure dimensions.

        Returns
        -------
        ControlChartFigure
        """
        from .residuals import plot_residuals as _plot_residuals

        return _plot_residuals(
            self.result,
            residual_type=residual_type,
            plot_type=plot_type,
            theme=theme,
            width=width,
            height=height,
        )

    # -----------------------------------------------------------------
    #  Report (delegate)
    # -----------------------------------------------------------------

    def generate_report(
        self,
        filepath: str,
        include_charts: bool = True,
        include_residuals: bool = True,
        include_effects: bool = True,
        include_summary: bool = True,
        theme: str | ChartTheme = 'processbehavior',
        width: int = 1200,
        title: str | None = None,
    ) -> None:
        """Generate comprehensive HTML report."""
        from .report import generate_report as _generate_report

        _generate_report(
            self.result,
            filepath=filepath,
            include_charts=include_charts,
            include_residuals=include_residuals,
            include_effects=include_effects,
            include_summary=include_summary,
            theme=theme,
            width=width,
            title=title,
        )

    # =================================================================
    #  Private helpers
    # =================================================================

    def _resolve_charts(
        self, chart: str | None, facet: bool,
    ) -> dict:
        """Determine which charts to plot."""
        if chart:
            chart_info = self.charts[chart]
            if 'strata' in chart_info and chart_info['strata']:
                charts = self._get_stratified_charts()
                return {
                    k: v for k, v in charts.items()
                    if v.get('metadata', {}).get('original_chart') == chart
                }
            return {chart: chart_info}

        if facet or self.summary.get('is_stratified', False):
            return self._get_stratified_charts()

        # Standard charts
        standard = {
            k: v for k, v in self.charts.items()
            if k in ['Xbar', 'S', 'XmR', 'R', 'Histogram']
        }
        if not standard and len(self.charts) == 1:
            standard = dict(self.charts)
        return standard

    @staticmethod
    def _resolve_height(
        height: int | None,
        width: int,
        aspect_ratio: float | None,
        n_charts: int,
        ncols: int,
    ) -> int:
        """Calculate figure height."""
        if height is not None:
            return height
        if aspect_ratio is not None:
            return int(width / aspect_ratio)
        nrows = (n_charts + ncols - 1) // ncols
        if n_charts == 1:
            return 400
        height_per_row = 450 if nrows <= 3 else 400
        return nrows * height_per_row

    @staticmethod
    def _calculate_spacing(
        nrows: int, ncols: int, vertical_spacing: float,
    ) -> tuple[float, float]:
        """Calculate dynamic subplot spacing."""
        if nrows > 1:
            # Cap total gap fraction to preserve plot area
            max_total_gap = 0.4  # at most 40% of figure for gaps
            max_v = max_total_gap / (nrows - 1)
            desired_v = 0.20 if nrows <= 3 else 0.12
            v_spacing = min(max(vertical_spacing, desired_v), max_v)
        else:
            v_spacing = vertical_spacing
        if ncols > 1:
            max_h = 1.0 / (ncols - 1) - 0.01
            h_spacing = min(0.10, max_h)
        else:
            h_spacing = 0.10
        return v_spacing, h_spacing

    # ---- Data column helpers ----

    def _get_value_column(self, chart_info: dict, chart_name: str) -> str:
        if 'metadata' not in chart_info:
            raise PlotError(
                f"Chart '{chart_name}' missing metadata. "
                f"All charts must have metadata with 'value_col'."
            )
        return chart_info['metadata']['value_col']

    @staticmethod
    def _get_base_chart_type(chart_name: str) -> str:
        return chart_name.split('_')[0]

    @staticmethod
    def _reorder_companion_pairs(
        charts: dict,
    ) -> tuple[dict, int | None]:
        """Reorder stratified companion charts into paired rows.

        Detects stratified companion pairs (Xbar+S or XmR+R with ≥ 4 charts)
        and interleaves them so each stratum's pair is side-by-side:
        ``[Xbar_1_1, S_1_1, Xbar_1_2, S_1_2, ...]``

        Returns ``(reordered_dict, 2)`` to force ncols=2, or
        ``(charts, None)`` if not applicable.
        """
        if len(charts) < 4:
            return charts, None

        # Determine base types present
        base_types: dict[str, list[str]] = {}
        for name in charts:
            base = name.split('_')[0]
            base_types.setdefault(base, []).append(name)

        if len(base_types) != 2:
            return charts, None

        type_set = set(base_types.keys())
        if type_set not in ({'Xbar', 'S'}, {'XmR', 'R'}):
            return charts, None

        # Extract stratum suffixes: everything after the first '_'
        # e.g. "Xbar_1_2" -> "1_2"
        def suffix(name: str) -> str:
            return name.split('_', 1)[1] if '_' in name else ''

        # Determine primary/secondary order
        if 'Xbar' in type_set:
            primary, secondary = 'Xbar', 'S'
        else:
            primary, secondary = 'XmR', 'R'

        # Build lookup by suffix for each type
        secondary_by_suffix = {suffix(n): n for n in base_types[secondary]}

        # Interleave using primary order (preserves original stratum ordering)
        reordered: dict = {}
        for name in base_types[primary]:
            sfx = suffix(name)
            reordered[name] = charts[name]
            sec_name = secondary_by_suffix.get(sfx)
            if sec_name:
                reordered[sec_name] = charts[sec_name]

        # Append any secondary keys without matching primary (shouldn't happen)
        for name in base_types[secondary]:
            if name not in reordered:
                reordered[name] = charts[name]

        return reordered, 2

    def _get_x_column(self, data: pd.DataFrame) -> str | None:
        time_var = self.summary.get('time_var')
        if time_var and time_var in data.columns:
            if not data[time_var].is_unique:
                return None
            return time_var
        for col in ('subgroup', 'rsg', 'group'):
            if col in data.columns and data[col].is_unique:
                return col
        # For explicit by=[single factor] in Xbar/S paths, the grouping
        # column may be the original factor name (e.g., "FACTOR 1").
        metric_cols = {
            'xbar', 's', 'mr', 'center', 'lpl', 'upl',
            'beyond_limits', 'obs_id', 'groups', 'n', 'N',
        }
        for col in data.columns:
            if col in metric_cols:
                continue
            if data[col].is_unique:
                return col
        return None

    @staticmethod
    def _get_center_key(stats: dict) -> str | None:
        return 'center' if 'center' in stats else None

    # ---- Stratification ----

    def _get_stratified_charts(self) -> dict:
        """Expand stratified charts into per-stratum charts for faceting."""
        stratified: dict = {}

        for name, chart_info in self.charts.items():
            if 'strata' in chart_info and chart_info['strata']:
                strata = chart_info['strata']
                combined_data = chart_info['data']
                nested_stats = chart_info['statistics']
                metadata = chart_info.get('metadata', {})
                stratify_by = metadata.get('stratify_by', [])

                if len(stratify_by) == 1:
                    stratify_col = stratify_by[0]
                elif len(stratify_by) > 1:
                    combined_data = combined_data.copy()
                    combined_data['_stratify_key'] = combined_data[
                        stratify_by
                    ].apply(tuple, axis=1)
                    stratify_col = '_stratify_key'
                else:
                    stratify_col = None

                if stratify_col is not None and stratify_col in combined_data.columns:
                    for stratum in strata:
                        mask = combined_data[stratify_col] == stratum
                        stratum_data = combined_data[mask].copy().reset_index(drop=True)
                        stratum_stats = nested_stats.get(stratum, {})

                        if isinstance(stratum, (tuple, list)):
                            stratum_str = encode_rsg(stratum)
                            stratum_display = ' '.join(str(s) for s in stratum)
                        else:
                            stratum_str = encode_rsg(stratum)
                            stratum_display = str(stratum)
                        expanded_name = f"{name}_{stratum_str}"

                        all_lb = metadata.get('lane_boundaries')
                        stratum_lb = (
                            all_lb.get(stratum) if isinstance(all_lb, dict) else None
                        )

                        stratified[expanded_name] = {
                            'data': stratum_data,
                            'statistics': stratum_stats,
                            'metadata': {
                                **metadata,
                                'original_chart': name,
                                'stratum': stratum,
                                'stratum_display': stratum_display,
                                'lane_boundaries': stratum_lb,
                            },
                        }
                else:
                    stratified[name] = chart_info

            elif '_' in str(name) or any(
                key in chart_info.get('metadata', {})
                for key in ['stratum', 'level', 'group']
            ):
                stratified[name] = chart_info

        return stratified if stratified else self.charts

    # ---- Time tick labels ----

    def _apply_time_tick_labels(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        metadata: dict,
        row: int | None = None,
        col: int | None = None,
        max_ticks: int = 20,
    ) -> None:
        """Replace integer x-axis positions with time values."""
        time_var = self.summary.get('time_var')
        if not time_var or time_var not in data.columns:
            return
        x_col = self._get_x_column(data)
        if x_col is not None:
            return

        n = len(data)
        if n == 0:
            return

        priority = {0, n - 1}
        lane_boundaries = metadata.get('lane_boundaries')
        if lane_boundaries:
            if isinstance(lane_boundaries, list):
                priority |= {b['position'] for b in lane_boundaries}
            elif isinstance(lane_boundaries, dict):
                for bounds in lane_boundaries.values():
                    priority |= {b['position'] for b in bounds}

        # Adaptive tick budget: reduce regular ticks when lane boundaries
        # consume many priority slots to avoid overcrowding
        effective_max = max_ticks
        if len(priority) > max_ticks // 2:
            effective_max = len(priority) + min(10, max_ticks // 3)

        budget = max(0, effective_max - len(priority))
        if budget > 0 and n > len(priority):
            step = max(1, n // budget)
            regular = set(range(0, n, step)) - priority
            if len(regular) > budget:
                regular = set(sorted(regular)[:budget])
            tick_positions = sorted(priority | regular)
        else:
            tick_positions = sorted(priority)

        tick_positions = [p for p in tick_positions if 0 <= p < n]
        tick_labels = data[time_var].iloc[tick_positions].astype(str).tolist()

        # Adaptive angle: rotate labels when total label footprint is large
        n_ticks = len(tick_positions)
        max_label_len = max((len(lbl) for lbl in tick_labels), default=1)
        label_footprint = n_ticks * max_label_len
        angle = -45 if label_footprint > 80 else 0

        kwargs = {}
        if row is not None and col is not None:
            kwargs = {'row': row, 'col': col}
        fig.update_xaxes(
            tickvals=tick_positions,
            ticktext=tick_labels,
            tickangle=angle,
            automargin=True,
            **kwargs,
        )

    # ---- Y-range / histogram helpers ----

    def _calculate_global_yrange(
        self, charts: dict, padding: float = 0.05,
    ) -> list[float]:
        """Global y-axis range across all charts."""
        global_min = float('inf')
        global_max = float('-inf')

        for chart_name, chart_info in charts.items():
            data = chart_info['data']
            stats = chart_info['statistics']
            value_col = self._get_value_column(chart_info, chart_name)

            global_min = min(global_min, data[value_col].min())
            global_max = max(global_max, data[value_col].max())

            ucl = stats.get('upl')
            lcl = stats.get('lpl')
            if ucl is not None and ucl != 'Varies':
                global_max = max(global_max, ucl)
            if lcl is not None and lcl != 'Varies':
                global_min = min(global_min, lcl)

        data_range = global_max - global_min
        pad = data_range * padding
        return [global_min - pad, global_max + pad]

    @staticmethod
    def _calculate_histogram_bin_edges(charts: dict) -> tuple[np.ndarray, int]:
        """Shared bin edges across all histogram facets."""
        global_min = float('inf')
        global_max = float('-inf')
        n_bins = 10

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

        if global_min == float('inf') or global_max == float('-inf'):
            return np.linspace(0, 1, n_bins + 1), n_bins
        return np.linspace(global_min, global_max, n_bins + 1), n_bins

    @staticmethod
    def _calculate_histogram_yrange(
        charts: dict,
        bin_edges: np.ndarray,
        padding: float = 0.20,
    ) -> list[float]:
        """Y-axis range for histogram facets."""
        max_count = 0
        for chart_info in charts.values():
            metadata = chart_info.get('metadata', {})
            if metadata.get('chart_type') != 'Histogram':
                continue
            data = chart_info['data']
            value_col = metadata.get('value_col')
            if value_col is None or value_col not in data.columns:
                continue
            values = data[value_col].dropna()
            if len(values) == 0:
                continue
            counts, _ = np.histogram(values, bins=bin_edges)
            max_count = max(max_count, counts.max())
        if max_count == 0:
            return [0, 1]
        return [0, max_count * (1 + padding)]

    # ---- Title / label generation ----

    def _generate_title(self, chart_name: str, chart_info: dict | None = None) -> str:
        response_var = self.summary.get('response_var', '')
        grouping_vars = self.summary.get('grouping_vars', [])
        chart_type = self._get_chart_type_display(chart_name)

        parts = [f"{chart_type} Chart"]

        # Use residual label if this is a residual chart
        metadata = chart_info.get('metadata', {}) if chart_info else {}
        residual_type = metadata.get('residual_type')
        if residual_type:
            recentered = metadata.get('recentered', False)
            label = _RESIDUAL_LABELS.get(residual_type, residual_type)
            suffix = f"{residual_type} ({label})"
            if recentered:
                suffix += " Recentered"
            parts.append(f"of {suffix}")
        elif response_var:
            parts.append(f"of {response_var}")

        if grouping_vars and len(grouping_vars) == 1:
            parts.append(f"by {grouping_vars[0]}")
        if '_' in chart_name and chart_name not in ['Xbar', 'S', 'XmR']:
            stratum = self._extract_stratum_name(chart_name)
            if stratum:
                parts.append(f"- {stratum}")
        return ' '.join(parts)

    def _generate_subplot_title(
        self, chart_name: str, chart_info: dict | None = None,
    ) -> str:
        chart_type = self._get_chart_type_display(chart_name)
        if chart_info is not None:
            sd = chart_info.get('metadata', {}).get('stratum_display')
            if sd:
                return f"{chart_type} {sd}"
        if '_' in chart_name:
            stratum = self._extract_stratum_name(chart_name)
            if stratum:
                return f"{chart_type} {stratum}"
        return chart_type

    @staticmethod
    def _get_chart_type_display(chart_name: str) -> str:
        display_names = {
            'Xbar': 'X\u0304', 'S': 'S', 'XmR': 'XmR',
            'R': 'R', 'Histogram': 'Histogram',
        }
        if chart_name in display_names:
            return display_names[chart_name]
        for key, display in display_names.items():
            if chart_name.startswith(key + '_'):
                return display
        raise PlotError(f"Unknown chart type: {chart_name}")

    @staticmethod
    def _extract_stratum_name(chart_name: str) -> str | None:
        prefixes = ['Xbar_', 'S_', 'XmR_', 'R_', 'Histogram_']
        result = chart_name
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix):]
                break
        result = result.replace('_', ' ')
        return result if result != chart_name else None

    def _get_xaxis_label(
        self, x_col: str | None = None, chart_name: str | None = None,
    ) -> str:
        if x_col in ('subgroup', 'rsg', 'group'):
            base_type = chart_name.split('_')[0] if chart_name else ''
            if base_type in ('XmR', 'R'):
                time_var = self.summary.get('time_var')
                if time_var:
                    return time_var.replace('_', ' ').title()
                return 'Observation'
            return 'Subgroup'
        time_var = self.summary.get('time_var')
        if time_var and (x_col is None or x_col == time_var):
            return time_var.replace('_', ' ').title()
        if x_col is None:
            return 'Observation'
        return x_col.replace('_', ' ').title()

    def _get_yaxis_label(self, value_col: str | None) -> str:
        response_var = self.summary.get('response_var')
        if response_var:
            return response_var.replace('_', ' ').title()
        if value_col:
            return value_col.capitalize()
        return 'Value'

    def _get_data_legend_name(self) -> str:
        response_var = self.summary.get('response_var')
        if response_var:
            return response_var.replace('_', ' ').title()
        return 'Data'

    # ---- Effects bar chart helpers (used by plot_effects) ----

    @staticmethod
    def _effects_all(effects, theme, width, height):
        from plotly.subplots import make_subplots as _ms

        factor_effects = []
        time_effects = None
        for name, data in effects.items():
            if 'PT_ME' in str(data.columns if hasattr(data, 'columns') else ''):
                time_effects = data
            elif isinstance(data, pd.DataFrame) and 'Main_Effect' in data.columns:
                factor_effects.append((name, data))

        n_plots = len(factor_effects) + (1 if time_effects is not None else 0)
        if n_plots == 0:
            raise ValidationError("No effects data found to plot.")

        fig = _ms(
            rows=1, cols=n_plots,
            subplot_titles=[n for n, _ in factor_effects]
            + (['Time Effects'] if time_effects is not None else []),
            horizontal_spacing=0.1,
        )
        c = 1
        for name, data in factor_effects:
            fc = data.columns[0]
            fig.add_trace(
                go.Bar(
                    x=data[fc].astype(str), y=data['Main_Effect'],
                    name=name, marker_color=theme.data_color, showlegend=False,
                ),
                row=1, col=c,
            )
            fig.update_xaxes(title_text=fc, row=1, col=c)
            fig.update_yaxes(title_text='Main Effect', row=1, col=c)
            c += 1

        if time_effects is not None:
            tc = time_effects.columns[0]
            fig.add_trace(
                go.Bar(
                    x=time_effects[tc].astype(str), y=time_effects['PT_ME'],
                    name='Time', marker_color=theme.center_color, showlegend=False,
                ),
                row=1, col=c,
            )
            fig.update_xaxes(title_text=tc, row=1, col=c)
            fig.update_yaxes(title_text='Time Effect', row=1, col=c)

        fig.update_layout(title='Main Effects Analysis', width=width, height=height)
        return fig

    @staticmethod
    def _effects_factor(effects, theme, width, height):
        from plotly.subplots import make_subplots as _ms

        factor_effects = [
            (n, d) for n, d in effects.items()
            if isinstance(d, pd.DataFrame) and 'Main_Effect' in d.columns
        ]
        if not factor_effects:
            raise ValidationError("No factor effects found.")

        if len(factor_effects) == 1:
            name, data = factor_effects[0]
            fc = data.columns[0]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=data[fc].astype(str), y=data['Main_Effect'],
                name='Effect', marker_color=theme.data_color,
            ))
            fig.add_hline(y=0, line_dash='dash', line_color=theme.center_color)
            fig.update_layout(
                title=f'Factor Main Effects: {name}',
                xaxis_title=fc, yaxis_title='Main Effect',
                width=width, height=height,
            )
        else:
            fig = _ms(
                rows=1, cols=len(factor_effects),
                subplot_titles=[n for n, _ in factor_effects],
                horizontal_spacing=0.1,
            )
            for c, (name, data) in enumerate(factor_effects, 1):
                fc = data.columns[0]
                fig.add_trace(
                    go.Bar(
                        x=data[fc].astype(str), y=data['Main_Effect'],
                        name=name, marker_color=theme.data_color, showlegend=False,
                    ),
                    row=1, col=c,
                )
                fig.update_xaxes(title_text=fc, row=1, col=c)
                fig.update_yaxes(title_text='Main Effect', row=1, col=c)
            fig.update_layout(
                title='Factor Main Effects', width=width, height=height,
            )
        return fig

    @staticmethod
    def _effects_time(effects, theme, width, height):
        time_effects = None
        for _name, data in effects.items():
            if isinstance(data, pd.DataFrame) and 'PT_ME' in data.columns:
                time_effects = data
                break
        if time_effects is None:
            raise ValidationError(
                "Time effects not found.\n"
                "Time effects require time_var in analysis specification."
            )
        tc = time_effects.columns[0]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=time_effects[tc].astype(str), y=time_effects['PT_ME'],
            name='Time Effect', marker_color=theme.center_color,
        ))
        fig.add_hline(y=0, line_dash='dash', line_color=theme.data_color)
        fig.update_layout(
            title='Time Main Effects',
            xaxis_title=tc, yaxis_title='Time Effect (PT_ME)',
            width=width, height=height,
        )
        return fig
