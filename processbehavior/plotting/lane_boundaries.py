"""Lane boundary rendering for control charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from .themes import ChartTheme


def add_lane_boundaries(
    fig: go.Figure,
    lane_boundaries: list[dict] | None,
    y_range: tuple[float, float],
    theme: ChartTheme,
    row: int | None = None,
    col: int | None = None,
    show_labels: bool = True,
) -> None:
    """
    Add vertical lane boundary lines to a chart figure.

    Unified function for both single charts and faceted subplots.
    Lane boundaries show where collapsed factors change within the chart,
    helping distinguish groups of observations from different factor levels.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add shapes to
    lane_boundaries : list[dict] or None
        List of boundary dicts with 'position' and 'label' keys
    y_range : tuple[float, float]
        (y_min, y_max) for vertical line extent
    theme : ChartTheme
        Theme with lane boundary styling
    row : int, optional
        Subplot row (1-indexed). None for single charts.
    col : int, optional
        Subplot column (1-indexed). None for single charts.
    show_labels : bool, default True
        Whether to show factor labels at boundary positions
    """
    if not lane_boundaries:
        return

    y_min, y_max = y_range

    subplot_kwargs = {}
    if row is not None and col is not None:
        subplot_kwargs = {'row': row, 'col': col}

    for boundary in lane_boundaries:
        x_pos = boundary['position']
        label = boundary.get('label', '')

        fig.add_shape(
            type='line',
            x0=x_pos,
            x1=x_pos,
            y0=y_min,
            y1=y_max,
            line=dict(color=theme.lane_boundary_color, dash=theme.lane_boundary_dash, width=theme.lane_boundary_width),
            **subplot_kwargs,
        )

        if show_labels and label:
            fig.add_annotation(
                x=x_pos,
                y=y_max,
                text=label,
                showarrow=False,
                yanchor='bottom',
                font=dict(size=theme.lane_boundary_annotation_size, color=theme.lane_boundary_color),
                **subplot_kwargs,
            )
