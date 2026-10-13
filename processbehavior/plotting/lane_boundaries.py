"""Lane boundary rendering for control charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from .themes import ChartTheme


# More dividers than this can't be told apart at any chart width: they merge into a solid band,
# and the browser still has to draw every one. Above it none are drawn. A chart with that many
# lanes (one per condition: 6,032 for a 6,033-gene study) is read from its points and limits.
MAX_LANE_DIVIDERS = 1_000


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
    Nothing is drawn above ``MAX_LANE_DIVIDERS`` boundaries.

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
    if not lane_boundaries or len(lane_boundaries) > MAX_LANE_DIVIDERS:
        return

    y_min, y_max = y_range

    subplot_kwargs = {}
    if row is not None and col is not None:
        subplot_kwargs = {'row': row, 'col': col}

    line = dict(color=theme.lane_boundary_color, dash=theme.lane_boundary_dash, width=theme.lane_boundary_width)
    font = dict(size=theme.lane_boundary_annotation_size, color=theme.lane_boundary_color)
    shapes, annotations = [], []
    for boundary in lane_boundaries:
        x_pos = boundary['position']
        label = boundary.get('label', '')
        shapes.append(dict(type='line', x0=x_pos, x1=x_pos, y0=y_min, y1=y_max, line=line))
        if show_labels and label:
            annotations.append(
                dict(x=x_pos, y=y_max, text=label, showarrow=False, yanchor='bottom', font=font)
            )

    _add_all(fig, 'shapes', fig.add_shape, shapes, subplot_kwargs)
    _add_all(fig, 'annotations', fig.add_annotation, annotations, subplot_kwargs)


def _add_all(fig: go.Figure, prop: str, add_one, items: list[dict], subplot_kwargs: dict) -> None:
    """Add layout shapes or annotations in one assignment instead of one call each.

    ``fig.add_shape`` re-validates every shape already on the figure, so adding N one at a
    time costs O(N^2): 800 dividers took 27s, and 6,032 would take about 25 minutes. The first
    item goes through ``add_one`` so plotly resolves the subplot's axis references for
    ``row``/``col``. The rest copy those references and are appended in one assignment. The
    figure is the same as one built call by call.
    """
    if not items:
        return
    add_one(**items[0], **subplot_kwargs)
    if len(items) == 1:
        return
    first = fig.layout[prop][-1]
    refs = {'xref': first.xref, 'yref': first.yref}
    fig.layout[prop] = fig.layout[prop] + tuple({**item, **refs} for item in items[1:])
