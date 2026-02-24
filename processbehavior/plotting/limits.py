"""Limit line rendering for control charts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    import pandas as pd

    from .themes import ChartTheme

logger = logging.getLogger(__name__)


def format_limit_label(limit_name: str, value: float, show_value: bool) -> str:
    """
    Format control limit annotation label.

    Parameters
    ----------
    limit_name : str
        Name of the limit ('UPL', 'LPL', 'CL')
    value : float
        Numeric value of the limit
    show_value : bool
        Whether to include the numeric value

    Returns
    -------
    str
        Formatted label like "UPL = 52.34" or just "UPL"
    """
    if show_value:
        if abs(value) >= 100:
            return f"{limit_name} = {value:.1f}"
        elif abs(value) >= 10:
            return f"{limit_name} = {value:.2f}"
        else:
            return f"{limit_name} = {value:.3f}"
    return limit_name


def build_stepped_coordinates(
    x_vals: list,
    limit_vals: list
) -> tuple[list, list]:
    """
    Build stepped line coordinates from x values and limit values.

    For each point, draws a horizontal line to the next point's x position,
    then steps up/down to the next point's limit value.

    Parameters
    ----------
    x_vals : list
        X-axis values
    limit_vals : list
        Limit values at each x position

    Returns
    -------
    tuple[list, list]
        (x_stepped, y_stepped) coordinate lists for the stepped line
    """
    if len(x_vals) != len(limit_vals):
        raise ValueError(
            f"x_vals and limit_vals must have equal length: "
            f"{len(x_vals)} != {len(limit_vals)}"
        )

    x_stepped = []
    y_stepped = []

    for i in range(len(x_vals)):
        x_stepped.append(x_vals[i])
        y_stepped.append(limit_vals[i])

        if i < len(x_vals) - 1:
            x_stepped.append(x_vals[i + 1])
            y_stepped.append(limit_vals[i])

    return x_stepped, y_stepped


def add_stepped_limit_line(
    fig: go.Figure,
    data: pd.DataFrame,
    x_col: str,
    limit_col: str,
    line_color: str,
    line_dash: str,
    line_width: float,
    limit_name: str,
    theme: ChartTheme,
    row: int | None = None,
    col: int | None = None
) -> go.Figure:
    """
    Add a stepped limit line that follows varying per-row limits.

    Unified function for both single charts and faceted subplots.
    When row/col are None, adds to the main figure. When provided,
    adds to the specified subplot.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add the line to
    data : pd.DataFrame
        Chart data with limit column
    x_col : str
        Name of x-axis column
    limit_col : str
        Name of limit column ('upl' or 'lpl')
    line_color : str
        Color for the limit line
    line_dash : str
        Dash pattern for the line
    line_width : float
        Width of the line
    limit_name : str
        Name for hover label ('UPL' or 'LPL')
    theme : ChartTheme
        Chart theme for styling
    row : int, optional
        Subplot row (1-indexed). None for single charts.
    col : int, optional
        Subplot column (1-indexed). None for single charts.

    Returns
    -------
    go.Figure
        The figure, with the stepped limit line added (or unchanged if
        ``limit_col`` is not present in the data).
    """
    if limit_col not in data.columns:
        logger.debug("Limit column '%s' not found; skipping stepped limit line", limit_col)
        return fig

    x_vals = data[x_col].tolist() if x_col in data.columns else data.index.tolist()
    limit_vals = data[limit_col].tolist()

    x_stepped, y_stepped = build_stepped_coordinates(x_vals, limit_vals)

    scatter = go.Scatter(
        x=x_stepped,
        y=y_stepped,
        mode='lines',
        name=f'{limit_name} (varies)',
        line=dict(color=line_color, dash=line_dash, width=line_width),
        hovertemplate=f'{limit_name}: %{{y:.3f}}<extra></extra>',
        showlegend=False
    )

    if row is not None and col is not None:
        fig.add_trace(scatter, row=row, col=col)
    else:
        fig.add_trace(scatter)

    return fig
