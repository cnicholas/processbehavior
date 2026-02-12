"""Statistics box rendering for control charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    import pandas as pd

    from .themes import ChartTheme


def format_stat_value(value: float, compact: bool = False) -> str:
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
        if abs(value) >= 100 or abs(value) >= 10:
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


def build_stats_text(
    stats: dict,
    data: pd.DataFrame,
    compact: bool = False
) -> str | None:
    """
    Build statistics text for display in a stats box.

    Parameters
    ----------
    stats : dict
        Chart statistics with 'center', 'upl', 'lpl' keys
    data : DataFrame
        Chart data (for calculating n)
    compact : bool, default False
        If True, use compact format (n=X | CL=Y) for faceted charts.
        If False, use full format with line breaks.

    Returns
    -------
    str or None
        Formatted stats text, or None if no stats available
    """
    n = len(data)

    if compact:
        parts = [f"n={n}"]
        center = stats.get('center')
        if center is not None and center != 'Varies':
            parts.append(f"CL={format_stat_value(center, compact=True)}")
        return ' | '.join(parts) if parts else None
    else:
        lines = [f"n = {n}"]

        center = stats.get('center')
        if center is not None and center != 'Varies':
            lines.append(f"CL = {format_stat_value(center)}")

        ucl = stats.get('upl')
        if ucl is not None and ucl != 'Varies':
            lines.append(f"UPL = {format_stat_value(ucl)}")

        lcl = stats.get('lpl')
        if lcl is not None and lcl != 'Varies':
            lines.append(f"LPL = {format_stat_value(lcl)}")

        return '<br>'.join(lines) if lines else None


def add_stats_box(
    fig: go.Figure,
    stats: dict,
    data: pd.DataFrame,
    theme: ChartTheme,
    row: int | None = None,
    col: int | None = None,
    nrows: int | None = None,
    ncols: int | None = None
) -> None:
    """
    Add a statistics box annotation to a chart.

    Unified function for both single charts and faceted subplots.
    When row/col are None, positions at paper (0.02, 0.98) with full text.
    When provided, calculates subplot-relative position with compact text.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add annotation to
    stats : dict
        Chart statistics with 'center', 'upl', 'lpl' keys
    data : DataFrame
        Chart data (for calculating n)
    theme : ChartTheme
        Theme with stats box styling
    row : int, optional
        Row number of subplot (1-indexed). None for single charts.
    col : int, optional
        Column number of subplot (1-indexed). None for single charts.
    nrows : int, optional
        Total number of rows in the facet grid. Required when row/col are set.
    ncols : int, optional
        Total number of columns in the facet grid. Required when row/col are set.
    """
    if row is not None and col is not None:
        # Faceted: compact text, subplot-relative position, smaller font
        stats_text = build_stats_text(stats, data, compact=True)
        if stats_text is None:
            return

        col_width = 1.0 / ncols
        row_height = 1.0 / nrows
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
                size=theme.stats_box_font_size - 1,
                color=theme.stats_box_font_color,
                family='monospace'
            ),
            bgcolor=theme.stats_box_bgcolor,
            bordercolor=theme.stats_box_bordercolor,
            borderwidth=theme.stats_box_borderwidth,
            borderpad=3,
            align='left'
        )
    else:
        # Single chart: full text, fixed position, standard font
        stats_text = build_stats_text(stats, data, compact=False)
        if stats_text is None:
            return

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
