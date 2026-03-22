"""Shared rendering pipeline for control charts and histograms.

Provides one code path used by both single and faceted chart layouts.
All rendering goes through ``render_control_chart()`` or
``render_histogram()``, eliminating the duplication that previously
existed between ``_plot_single_chart`` and ``_plot_faceted``.

The key insight: Plotly's ``add_hline()`` doesn't support subplot
targeting (``row``/``col``), so we use ``add_shape(type='line')`` +
``add_annotation()`` everywhere. This lets the same function handle
both single and faceted charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from .contracts import RenderContext
from .lane_boundaries import add_lane_boundaries
from .limits import add_stepped_limit_line, format_limit_label
from .run_rules_viz import add_run_rules_visualization
from .stats_box import add_stats_box
from .zones import add_zone_shading

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
#  Control chart rendering pipeline
# ---------------------------------------------------------------------------


def render_control_chart(
    fig: go.Figure,
    ctx: RenderContext,
    row: int | None = None,
    col: int | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
) -> None:
    """Render one control chart panel onto *fig*.

    This is the single rendering pipeline used by both single-chart and
    faceted layouts.  The ``row``/``col`` parameters determine subplot
    targeting; when both are ``None`` the chart is drawn on a plain
    ``go.Figure()``.

    Parameters
    ----------
    fig : go.Figure
        Target figure (may be a plain figure or a ``make_subplots`` grid).
    ctx : RenderContext
        Immutable rendering context carrying data, stats, theme and options.
    row, col : int | None
        1-indexed subplot coordinates, or ``None`` for single-chart mode.
    nrows, ncols : int | None
        Grid dimensions (needed by ``add_stats_box`` for position math).
    """
    data = ctx.data
    stats = ctx.stats
    theme = ctx.theme
    spec = ctx.spec
    value_col = spec.value_col
    x_col = spec.x_col

    x_data = data[x_col] if x_col and x_col in data.columns else data.index

    # 1. Zone shading (behind everything)
    if ctx.show_zones and theme.zone_opacity > 0:
        add_zone_shading(fig, stats, theme, row=row, col=col, ncols=ncols)

    # 2. Main data trace
    trace_kw = dict(
        x=x_data,
        y=data[value_col],
        mode='lines+markers',
        name=ctx.chart_name,
        marker=dict(size=ctx.marker_size, color=theme.data_color),
        line=dict(color=theme.data_color, width=ctx.line_width),
        opacity=theme.data_opacity,
        hovertemplate='%{x}<br>%{y:.3f}<extra></extra>',
    )
    if ctx.is_faceted:
        trace_kw['showlegend'] = False
    _add_trace(fig, go.Scatter(**trace_kw), row, col)

    # 3. Control limits and centerline
    if ctx.show_limits:
        _add_limits(fig, ctx, x_data, row, col, ncols)

    # 4. Signal highlighting (Rule 1 — beyond limits)
    if ctx.highlight_signals and 'beyond_limits' in data.columns:
        _add_signals(fig, ctx, x_data, row, col)

    # 5. Run rules (Rules 2-8)
    if ctx.show_rules and spec.run_rules_applicable:
        add_run_rules_visualization(
            fig, data, stats, ctx.chart_name,
            value_col, x_col, theme,
            result=ctx.result, row=row, col=col,
        )

    # 6. Stats box
    if ctx.show_stats:
        add_stats_box(
            fig, stats, data, theme,
            row=row, col=col, nrows=nrows, ncols=ncols,
        )

    # 7. Lane boundaries
    lane_boundaries = spec.lane_boundaries
    if lane_boundaries:
        # Handle dict-keyed boundaries (per-stratum in faceted)
        if isinstance(lane_boundaries, dict):
            lane_boundaries = lane_boundaries.get(ctx.chart_name)
        if lane_boundaries:
            y_min = data[value_col].min()
            y_max = data[value_col].max()
            y_pad = (y_max - y_min) * 0.05
            y_range = (y_min - y_pad, y_max + y_pad)
            add_lane_boundaries(fig, lane_boundaries, y_range, theme, row=row, col=col)


# ---------------------------------------------------------------------------
#  Histogram rendering pipeline
# ---------------------------------------------------------------------------


def render_histogram(
    fig: go.Figure,
    chart_info: dict,
    theme,
    *,
    show_stats: bool = True,
    row: int | None = None,
    col: int | None = None,
    shared_bin_edges: np.ndarray | None = None,
    is_faceted: bool = False,
    histnorm: str = "",
) -> None:
    """Render one histogram panel onto *fig*.

    Handles both standalone and faceted histograms with optional
    shared bin edges and mean/std overlay lines.

    Parameters
    ----------
    fig : go.Figure
        Target figure.
    chart_info : dict
        Chart info with 'data', 'statistics', and 'metadata'.
    theme : ChartTheme
        Visual theme.
    show_stats : bool
        Whether to overlay mean/std deviation lines.
    row, col : int | None
        Subplot coordinates (None for standalone).
    shared_bin_edges : np.ndarray | None
        Pre-computed bin edges for consistent cross-facet binning.
    is_faceted : bool
        Whether this is part of a faceted layout.
    histnorm : str, default ""
        Histogram normalization mode (``""`` for counts,
        ``"percent"`` for percentages). Also checked in
        ``chart_info['metadata']['histnorm']`` as fallback.
    """
    metadata = chart_info.get('metadata', {})
    stats = chart_info.get('statistics', {})
    data = chart_info['data']

    value_col = metadata.get('value_col')
    bins = metadata.get('bins', 10)
    chart_name = metadata.get('chart_type', 'Histogram')

    # Resolve histnorm: explicit parameter wins, then metadata fallback
    effective_histnorm = histnorm or metadata.get('histnorm', '')

    # Histogram trace
    if shared_bin_edges is not None:
        bin_width = shared_bin_edges[1] - shared_bin_edges[0]
        hist_trace = go.Histogram(
            x=data[value_col],
            xbins=dict(
                start=shared_bin_edges[0],
                end=shared_bin_edges[-1],
                size=bin_width,
            ),
            name=chart_name,
            marker_color=theme.data_color,
            opacity=0.75,
            showlegend=False,
            histnorm=effective_histnorm if effective_histnorm else None,
        )
    else:
        hist_trace = go.Histogram(
            x=data[value_col],
            nbinsx=bins,
            name=value_col if not is_faceted else chart_name,
            marker_color=theme.data_color,
            opacity=0.75,
            showlegend=is_faceted is False,  # standalone shows legend default
            histnorm=effective_histnorm if effective_histnorm else None,
        )
    _add_trace(fig, hist_trace, row, col)

    # Stats overlay lines
    if show_stats:
        _add_histogram_stats(fig, stats, theme, row, col, is_faceted)


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _subplot_kwargs(
    row: int | None, col: int | None
) -> dict:
    """Build ``**subplot_kwargs`` for Plotly calls that accept row/col."""
    if row is not None and col is not None:
        return {'row': row, 'col': col}
    return {}


def _add_trace(
    fig: go.Figure,
    trace: go.Scatter | go.Histogram,
    row: int | None,
    col: int | None,
) -> None:
    """Add a trace to fig, targeting subplot if row/col given."""
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _add_fixed_limit(
    fig: go.Figure,
    y: float,
    label: str,
    color: str,
    dash: str,
    width: float,
    font_size: int,
    row: int | None,
    col: int | None,
    ncols: int | None = None,
) -> None:
    """Draw a fixed horizontal limit line using add_shape + add_annotation.

    Uses domain-relative x-coordinates (0–1) so the line always spans the
    full subplot width regardless of x-data type.  This matches the pattern
    used by ``add_zone_shading``.
    """
    subplot_kw = _subplot_kwargs(row, col)

    # Build domain-relative xref (same logic as zones.py)
    if row is not None and col is not None:
        subplot_idx = (row - 1) * (ncols or 1) + col
        xref = 'x domain' if subplot_idx == 1 else f'x{subplot_idx} domain'
    else:
        xref = 'x domain'

    fig.add_shape(
        type='line',
        x0=0, x1=1,
        xref=xref,
        y0=y, y1=y,
        line=dict(color=color, dash=dash, width=width),
        **subplot_kw,
    )


def _add_limit_summary_annotation(
    fig: go.Figure,
    ctx: RenderContext,
    row: int | None,
    col: int | None,
    ncols: int | None = None,
) -> None:
    """Place a single summary annotation above a subplot with UPL/CL/LPL values.

    Replaces the per-line right-side annotations with a compact pipe-delimited
    string like ``UPL = 238.6 | CL = 237.4 | LPL = 236.2``.
    """
    stats = ctx.stats
    spec = ctx.spec
    theme = ctx.theme

    parts: list[str] = []

    # UPL
    if 'upl' in stats and stats['upl'] != 'Varies':
        parts.append(format_limit_label('UPL', stats['upl'], True))

    # CL
    center_key = spec.center_key
    if center_key and center_key in stats and stats[center_key] != 'Varies':
        parts.append(format_limit_label('CL', stats[center_key], True))

    # LPL
    if 'lpl' in stats and stats['lpl'] != 'Varies':
        parts.append(format_limit_label('LPL', stats['lpl'], True))

    if not parts:
        return

    text = ' | '.join(parts)

    # Build domain-relative refs
    subplot_kw = _subplot_kwargs(row, col)
    if row is not None and col is not None:
        subplot_idx = (row - 1) * (ncols or 1) + col
        xref = 'x domain' if subplot_idx == 1 else f'x{subplot_idx} domain'
        yref = 'y domain' if subplot_idx == 1 else f'y{subplot_idx} domain'
    else:
        xref = 'x domain'
        yref = 'y domain'

    fig.add_annotation(
        x=1.0, xref=xref,
        y=1.0, yref=yref,
        yanchor='top',
        xanchor='right',
        text=text,
        showarrow=False,
        font=dict(
            size=theme.annotation_font_size,
            color=theme.limit_summary_color,
        ),
        bgcolor=theme.limit_summary_bgcolor,
        **subplot_kw,
    )


def _add_limits(
    fig: go.Figure,
    ctx: RenderContext,
    x_data,
    row: int | None,
    col: int | None,
    ncols: int | None = None,
) -> None:
    """Add UPL, LPL, centerline (fixed or stepped) and vary annotation."""
    stats = ctx.stats
    theme = ctx.theme
    spec = ctx.spec
    data = ctx.data
    x_col = spec.x_col
    font_size = theme.annotation_font_size

    # ---- UPL ----
    if 'upl' in stats:
        if stats['upl'] != 'Varies':
            label = format_limit_label('UPL', stats['upl'], ctx.show_limit_values)
            _add_fixed_limit(
                fig, stats['upl'], label,
                theme.ucl_color, theme.limit_line_dash, theme.limit_line_width,
                font_size, row, col, ncols,
            )
        elif 'upl' in data.columns:
            add_stepped_limit_line(
                fig, data, x_col, 'upl',
                theme.ucl_color, theme.limit_line_dash, theme.limit_line_width,
                'UPL', theme, row=row, col=col,
            )

    # ---- LPL ----
    if 'lpl' in stats:
        if stats['lpl'] != 'Varies':
            label = format_limit_label('LPL', stats['lpl'], ctx.show_limit_values)
            _add_fixed_limit(
                fig, stats['lpl'], label,
                theme.lcl_color, theme.limit_line_dash, theme.limit_line_width,
                font_size, row, col, ncols,
            )
        elif 'lpl' in data.columns:
            add_stepped_limit_line(
                fig, data, x_col, 'lpl',
                theme.lcl_color, theme.limit_line_dash, theme.limit_line_width,
                'LPL', theme, row=row, col=col,
            )

    # ---- "Limits vary" annotation (single chart only) ----
    if spec.limits_vary and not ctx.is_faceted:
        vary_text = "Process limits computed per phase" if spec.phased else "Process limits vary by subgroup size (n)"
        fig.add_annotation(
            text=vary_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            borderpad=3,
        )

    # ---- Centerline ----
    center_key = spec.center_key
    if center_key and center_key in stats:
        if stats[center_key] == 'Varies':
            if 'center' in data.columns:
                add_stepped_limit_line(
                    fig, data, x_col, 'center',
                    theme.center_color, 'solid', theme.center_line_width,
                    'CL', theme, row=row, col=col,
                )
        else:
            label = format_limit_label('CL', stats[center_key], ctx.show_limit_values)
            _add_fixed_limit(
                fig, stats[center_key], label,
                theme.center_color, 'solid', theme.center_line_width,
                font_size, row, col, ncols,
            )

    # ---- Limit summary annotation ----
    if ctx.show_limit_values:
        _add_limit_summary_annotation(fig, ctx, row, col, ncols)


def _add_signals(
    fig: go.Figure,
    ctx: RenderContext,
    x_data,
    row: int | None,
    col: int | None,
) -> None:
    """Add signal markers (Rule 1 — beyond limits)."""
    data = ctx.data
    theme = ctx.theme
    value_col = ctx.spec.value_col
    x_col = ctx.spec.x_col

    signals = data[data['beyond_limits'] != 0]
    if signals.empty:
        return

    sig_x = signals[x_col] if x_col and x_col in data.columns else signals.index

    marker = dict(
        size=ctx.marker_size,
        color=theme.signal_color,
        symbol=theme.signal_marker_symbol,
    )
    # Single charts get the marker-line detail
    if not ctx.is_faceted:
        marker['line'] = dict(
            width=theme.signal_marker_line_width,
            color=theme.signal_marker_line_color,
        )

    hover_label = 'Signal' if ctx.is_faceted else 'Beyond Limits'
    trace = go.Scatter(
        x=sig_x,
        y=signals[value_col],
        mode='markers',
        name=hover_label,
        marker=marker,
        showlegend=False,
        hovertemplate=f'{hover_label}<br>%{{x}}<br>%{{y:.3f}}<extra></extra>',
    )
    _add_trace(fig, trace, row, col)


def _add_histogram_stats(
    fig: go.Figure,
    stats: dict,
    theme,
    row: int | None,
    col: int | None,
    is_faceted: bool,
) -> None:
    """Add mean/std overlay lines to a histogram."""
    mean = stats.get('mean')
    std = stats.get('std')
    n = stats.get('n', 0)
    subplot_kw = _subplot_kwargs(row, col)

    if mean is None or not np.isfinite(mean):
        return

    # Mean line
    vline_kw = dict(
        x=mean,
        line_dash="solid",
        line_color=theme.center_color,
        line_width=2,
    )
    if not is_faceted:
        vline_kw['annotation_text'] = f"Mean: {mean:.3f}"
        vline_kw['annotation_position'] = "top"
        vline_kw['annotation_font_size'] = theme.annotation_font_size
    fig.add_vline(**vline_kw, **subplot_kw)

    # Std deviation lines (±1σ, ±2σ, ±3σ)
    if std is None or n < 2 or not np.isfinite(std) or std <= 0:
        return

    for mult in [1, 2, 3]:
        for sign in [1, -1]:
            vline_kw = dict(
                x=mean + sign * mult * std,
                line_dash="dash",
                line_color="orange",
                line_width=1,
            )
            if not is_faceted:
                label = f"+{mult}σ" if sign > 0 else f"-{mult}σ"
                vline_kw['annotation_text'] = label
                vline_kw['annotation_position'] = "top"
                vline_kw['annotation_font_size'] = theme.annotation_font_size
            fig.add_vline(**vline_kw, **subplot_kw)
