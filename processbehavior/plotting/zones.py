"""Zone shading for control charts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from .themes import ChartTheme

logger = logging.getLogger(__name__)


def calculate_zone_boundaries(stats: dict, theme: ChartTheme) -> list[tuple[float, float, str]] | None:
    """
    Calculate zone boundaries for Western Electric rules visualization.

    Returns zone definitions as (y0, y1, color) tuples, or None if
    zones cannot be calculated (e.g., limits vary or are missing).

    Zones are:
    - Zone C: 0-sigma to +/-1-sigma (green - normal variation)
    - Zone B: +/-1-sigma to +/-2-sigma (yellow - watch)
    - Zone A: +/-2-sigma to +/-3-sigma (red - warning)

    Parameters
    ----------
    stats : dict
        Chart statistics with 'center', 'upl', 'lpl' keys
    theme : ChartTheme
        Theme with zone colors

    Returns
    -------
    list of tuple or None
        List of (y0, y1, color) tuples defining zone rectangles,
        or None if zones cannot be calculated
    """
    if stats.get('limits_vary'):
        logger.debug('Zone shading skipped: limits vary')
        return None

    center = stats.get('center')
    ucl = stats.get('upl')
    lcl = stats.get('lpl')

    if center is None or ucl is None or lcl is None:
        logger.debug('Zone shading skipped: missing center/upl/lpl in stats')
        return None

    sigma = (ucl - center) / 3

    return [
        (center - sigma, center + sigma, theme.zone_c_color),
        (center + sigma, center + 2 * sigma, theme.zone_b_color),
        (center - 2 * sigma, center - sigma, theme.zone_b_color),
        (center + 2 * sigma, ucl, theme.zone_a_color),
        (lcl, center - 2 * sigma, theme.zone_a_color),
    ]


def add_zone_shading(
    fig: go.Figure,
    stats: dict,
    theme: ChartTheme,
    row: int | None = None,
    col: int | None = None,
    ncols: int | None = None,
) -> None:
    """
    Add zone shading to a chart figure.

    Unified function for both single charts and faceted subplots.
    When row/col are None, adds zones spanning the full figure.
    When provided, targets the specific subplot.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add shapes to
    stats : dict
        Chart statistics with 'center', 'upl', 'lpl' keys
    theme : ChartTheme
        Theme with zone colors and opacity
    row : int, optional
        Row number of subplot (1-indexed). None for single charts.
    col : int, optional
        Column number of subplot (1-indexed). None for single charts.
    ncols : int, optional
        Number of columns in the faceted layout. Required when row/col are set.
    """
    zones = calculate_zone_boundaries(stats, theme)
    if zones is None:
        return

    if row is not None and col is not None:
        # Faceted: use add_shape with explicit axis references
        subplot_idx = (row - 1) * ncols + col
        if subplot_idx == 1:
            xref = 'x'
            yref = 'y'
        else:
            xref = f'x{subplot_idx}'
            yref = f'y{subplot_idx}'

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
                col=col,
            )
    else:
        # Single chart: use add_hrect
        for y0, y1, color in zones:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=theme.zone_opacity, layer='below', line_width=0)
