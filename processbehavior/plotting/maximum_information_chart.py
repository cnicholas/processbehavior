"""
Maximum Information Analysis visualization.

Combined XmR + percentage histogram layout for R2 residuals,
following the ``capability_chart.py`` pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from ..maximum_information import MaximumInformationResult
    from .themes import ChartTheme


def create_maximum_information_chart(
    result: MaximumInformationResult,
    *,
    view: str = 'combined',
    bins: int = 10,
    theme: ChartTheme | None = None,
    width: int = 900,
    height: int = 700,
    title: str | None = None,
) -> go.Figure:
    """
    Create a Maximum Information Analysis chart.

    Parameters
    ----------
    result : MaximumInformationResult
        Result from ``study.maximum_information()``.
    view : str, default ``'combined'``
        ``'combined'`` — Individuals chart + percentage histogram.
        ``'xmr'`` — Individuals chart of R2 only.
        ``'histogram'`` — Percentage histogram only.
    bins : int, default 10
        Number of histogram bins.
    theme : ChartTheme, optional
        Visual theme. Defaults to the ``processbehavior`` theme.
    width, height : int
        Figure dimensions in pixels.
    title : str, optional
        Override chart title.

    Returns
    -------
    go.Figure
    """
    if view not in ('combined', 'xmr', 'histogram'):
        raise ValueError(f"view must be 'combined', 'xmr', or 'histogram', got {view!r}")

    if theme is None or isinstance(theme, str):
        from .themes import get_theme

        theme = get_theme(theme if isinstance(theme, str) else 'processbehavior')

    default_title = 'Maximum Information Analysis of R2 Residuals'
    chart_title = title if title is not None else default_title

    if view == 'combined':
        return _render_combined(result, theme, chart_title, width, height, bins)
    elif view == 'xmr':
        return _render_xmr(result, theme, chart_title, width, height)
    else:
        return _render_histogram(result, theme, chart_title, width, height, bins)


# ---------------------------------------------------------------------------
# Combined: XmR (top two rows) + Histogram (bottom)
# ---------------------------------------------------------------------------


def _render_combined(
    result: MaximumInformationResult,
    theme: ChartTheme,
    title: str,
    width: int,
    height: int,
    bins: int = 10,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.5, 0.5],
        subplot_titles=['XmR', 'Distribution'],
        vertical_spacing=0.14,
    )

    _add_individuals_panel(fig, result, theme, row=1, col=1)
    _add_histogram_panel(fig, result, theme, row=2, col=1, bins=bins)

    fig.update_layout(**theme.to_layout_dict())
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------------
# XmR only (individuals chart)
# ---------------------------------------------------------------------------


def _render_xmr(
    result: MaximumInformationResult,
    theme: ChartTheme,
    title: str,
    width: int,
    height: int,
) -> go.Figure:
    fig = go.Figure()

    _add_individuals_panel(fig, result, theme)

    fig.update_layout(**theme.to_layout_dict())
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------------
# Histogram only
# ---------------------------------------------------------------------------


def _render_histogram(
    result: MaximumInformationResult,
    theme: ChartTheme,
    title: str,
    width: int,
    height: int,
    bins: int = 10,
) -> go.Figure:
    fig = go.Figure()

    _add_histogram_panel(fig, result, theme, bins=bins)

    fig.update_layout(**theme.to_layout_dict())
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def _add_individuals_panel(
    fig: go.Figure,
    result: MaximumInformationResult,
    theme: ChartTheme,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Render the X (individuals) panel of the XmR chart."""
    info = result._xmr_chart_info
    data = info['data']
    stats = info['statistics']
    x = data['obs']
    y = data['R2']

    subplot_kw = {}
    if row is not None and col is not None:
        subplot_kw = {'row': row, 'col': col}

    # Data trace
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name='R2',
            marker=dict(size=5, color=theme.data_color),
            line=dict(color=theme.data_color, width=1),
            hovertemplate='%{x}<br>%{y:.4f}<extra></extra>',
            showlegend=False,
        ),
        **subplot_kw,
    )

    # Signal markers
    signals = data[data['beyond_limits'] != 0]
    if not signals.empty:
        fig.add_trace(
            go.Scatter(
                x=signals['obs'],
                y=signals['R2'],
                mode='markers',
                name='Signal',
                marker=dict(
                    size=theme.signal_marker_size,
                    color=theme.signal_color,
                    symbol=theme.signal_marker_symbol,
                    line=dict(
                        width=theme.signal_marker_line_width,
                        color=theme.signal_marker_line_color,
                    ),
                ),
                showlegend=False,
                hovertemplate='Signal<br>%{x}<br>%{y:.4f}<extra></extra>',
            ),
            **subplot_kw,
        )

    if row is not None and col is not None:
        # Limit lines (subplot)
        _add_hline(fig, stats['x_mean'], theme.center_color, 'solid', 2, row, col)
        _add_hline(fig, stats['x_upl'], theme.ucl_color, 'dash', 1.5, row, col)
        _add_hline(fig, stats['x_lpl'], theme.lcl_color, 'dash', 1.5, row, col)

        # Limit annotations
        r = result.round_to
        _add_limit_annotation(fig, f'UPL={round(stats["x_upl"], r)}', stats['x_upl'], theme, row, col)
        _add_limit_annotation(fig, f'CL={round(stats["x_mean"], r)}', stats['x_mean'], theme, row, col)
        _add_limit_annotation(fig, f'LPL={round(stats["x_lpl"], r)}', stats['x_lpl'], theme, row, col)

        fig.update_yaxes(title_text='R2', row=row, col=col)
    else:
        # Standalone figure
        fig.add_hline(y=stats['x_mean'], line_color=theme.center_color, line_dash='solid', line_width=2)
        fig.add_hline(y=stats['x_upl'], line_color=theme.ucl_color, line_dash='dash', line_width=1.5)
        fig.add_hline(y=stats['x_lpl'], line_color=theme.lcl_color, line_dash='dash', line_width=1.5)

        r = result.round_to
        for text, yval in [
            (f'UPL={round(stats["x_upl"], r)}', stats['x_upl']),
            (f'CL={round(stats["x_mean"], r)}', stats['x_mean']),
            (f'LPL={round(stats["x_lpl"], r)}', stats['x_lpl']),
        ]:
            fig.add_annotation(
                x=1.0,
                xref='x domain',
                y=yval,
                yref='y',
                text=text,
                showarrow=False,
                xanchor='left',
                font=dict(size=theme.annotation_font_size, color='gray'),
            )

        fig.update_yaxes(title_text='R2')
        fig.update_xaxes(title_text='Observation')


def _add_histogram_panel(
    fig: go.Figure,
    result: MaximumInformationResult,
    theme: ChartTheme,
    row: int | None = None,
    col: int | None = None,
    bins: int = 10,
) -> None:
    """Render the percentage histogram of R2 with NPL x-axis extension."""
    info = result._histogram_chart_info
    data = info['data']
    r2_values = data['R2'].to_numpy()

    trace_kwargs = dict(
        x=r2_values,
        nbinsx=bins,
        histnorm='percent',
        marker_color=theme.data_color,
        opacity=0.75,
        name='R2',
        showlegend=False,
        hovertemplate='Range: %{x}<br>Percent: %{y:.1f}%<extra></extra>',
    )

    if row is not None and col is not None:
        fig.add_trace(go.Histogram(**trace_kwargs), row=row, col=col)
    else:
        fig.add_trace(go.Histogram(**trace_kwargs))

    # Mean line
    stats = result._xmr_chart_info['statistics']
    r2_mean = stats['x_mean']
    upl = stats['x_upl']
    lpl = stats['x_lpl']

    subplot_kw = {}
    if row is not None and col is not None:
        subplot_kw = {'row': row, 'col': col}

    fig.add_vline(
        x=r2_mean,
        line_color=theme.center_color,
        line_width=2,
        line_dash='solid',
        **subplot_kw,
    )

    # NPL lines on histogram
    fig.add_vline(
        x=upl,
        line_color=theme.ucl_color,
        line_width=1.5,
        line_dash='dash',
        **subplot_kw,
    )
    fig.add_vline(
        x=lpl,
        line_color=theme.lcl_color,
        line_width=1.5,
        line_dash='dash',
        **subplot_kw,
    )

    # Extend x-axis to NPL range
    npl_range = upl - lpl
    padding = npl_range * 0.05
    x_min = lpl - padding
    x_max = upl + padding

    if row is not None and col is not None:
        fig.update_xaxes(range=[x_min, x_max], row=row, col=col)
        fig.update_yaxes(title_text='Percent', row=row, col=col)
        fig.update_xaxes(title_text='R2', row=row, col=col)
    else:
        fig.update_xaxes(range=[x_min, x_max], title_text='R2')
        fig.update_yaxes(title_text='Percent')

    fig.update_layout(bargap=0.05)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_hline(
    fig: go.Figure,
    y: float,
    color: str,
    dash: str,
    width: float,
    row: int,
    col: int,
) -> None:
    """Add a horizontal line to a subplot."""
    fig.add_hline(
        y=y,
        line_color=color,
        line_dash=dash,
        line_width=width,
        row=row,
        col=col,
    )


def _add_limit_annotation(
    fig: go.Figure,
    text: str,
    y: float,
    theme: ChartTheme,
    row: int,
    col: int,
) -> None:
    """Add a right-aligned limit annotation."""
    # Determine axis refs for the subplot
    subplot_idx = row  # single-column layout: row index = subplot index
    yref = 'y' if subplot_idx == 1 else f'y{subplot_idx}'
    xref = 'x domain' if subplot_idx == 1 else f'x{subplot_idx} domain'

    fig.add_annotation(
        x=1.0,
        xref=xref,
        y=y,
        yref=yref,
        text=text,
        showarrow=False,
        xanchor='left',
        font=dict(size=theme.annotation_font_size, color='gray'),
    )
