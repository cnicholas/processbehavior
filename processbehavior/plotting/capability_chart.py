"""
Process Capability visualization chart.

Produces a Plotly histogram with specification limit lines, natural process
limit (NPL) overlay, capability index annotation box, and out-of-spec shading.

Supports ``view='current'`` (default), ``view='potential'`` (NPLs from R2 σ̂),
and ``paired=True`` (two-panel facet: Current | Potential).

Follows the ``effects_charts.py`` pattern: standalone function, takes theme,
returns ``go.Figure``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    import pandas as pd

    from ..capability import CapabilityResult, SpecLimits
    from .themes import ChartTheme

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local constants — spec-specific colors (NOT reused from control limits)
# ---------------------------------------------------------------------------

_SPEC_LINE_COLOR = '#DC143C'  # Crimson — distinct from control limit colors
_SPEC_SHADE_OPACITY = 0.10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_capability_chart(
    cap: CapabilityResult,
    values: Sequence[float] | np.ndarray | pd.Series,
    *,
    theme: ChartTheme | None = None,
    show_potential: bool = True,
    view: str = 'current',
    paired: bool = False,
    x_label: str | None = None,
    nbins: int | None = None,
    histnorm: str = '',
    width: int = 900,
    height: int = 500,
    title: str | None = None,
) -> go.Figure:
    """
    Create a process capability histogram with spec limits and index annotations.

    Parameters
    ----------
    cap : CapabilityResult
        Result from ``study.capability()`` or ``assess_capability()``.
    values : array-like
        Response values to histogram. NaN values are dropped silently.
    theme : ChartTheme, optional
        Visual theme. Defaults to the ``processbehavior`` theme.
    show_potential : bool, default True
        Show Cp/Cpk in the annotation box (when available).
    view : str, default ``'current'``
        Which capability view to plot: ``'current'`` (NPLs from overall σ̂,
        Pp/Ppk annotation) or ``'potential'`` (NPLs from R2 σ̂_R2,
        Cp/Cpk annotation).
    paired : bool, default False
        When ``True``, create a two-panel facet with Current on the left
        and Potential on the right.
    x_label : str, optional
        X-axis label. Defaults to ``"Value"``.
    nbins : int, optional
        Number of histogram bins. ``None`` lets Plotly auto-select.
    histnorm : str, default ""
        Histogram normalization (``""`` for counts, ``"probability density"``
        for density).
    width : int, default 900
        Figure width in pixels.
    height : int, default 500
        Figure height in pixels.
    title : str, optional
        Chart title. Auto-generated from spec limits when ``None``.

    Returns
    -------
    go.Figure
    """
    from ..exceptions import ValidationError
    from .themes import get_theme

    # --- Validation ---
    if view not in ('current', 'potential'):
        raise ValueError(f"view must be 'current' or 'potential', got {view!r}")

    if view == 'potential' and cap.sigma_hat_r2 is None:
        raise ValidationError(f'Cannot plot potential capability: {cap.potential_unavailable_reason}')

    if paired and cap.sigma_hat_r2 is None:
        warnings.warn(
            f'Potential capability unavailable ({cap.potential_unavailable_reason}); '
            f'falling back to current-only chart.',
            stacklevel=2,
        )
        paired = False

    if theme is None:
        theme = get_theme('processbehavior')
    elif isinstance(theme, str):
        theme = get_theme(theme)

    # Track whether caller explicitly set histnorm
    caller_histnorm = histnorm

    vals = _normalize_values(values)
    specs = cap.specs

    # --- Paired mode: two-panel facet ---
    if paired:
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=['Current Capability', 'Potential Capability'],
        )

        for col, v in enumerate(('current', 'potential'), start=1):
            # Default to "percent" for potential panel when caller didn't specify
            panel_histnorm = caller_histnorm
            if v == 'potential' and not caller_histnorm:
                panel_histnorm = 'percent'
            _render_single_capability(
                fig,
                cap,
                vals,
                specs,
                theme,
                view=v,
                show_potential_text=show_potential,
                nbins=nbins,
                histnorm=panel_histnorm,
                x_label=x_label,
                row=1,
                col=col,
            )

        fig.update_layout(**theme.to_layout_dict())
        fig.update_layout(
            title=title if title is not None else _auto_title(specs),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.18,
                xanchor='center',
                x=0.5,
            ),
            bargap=0.05,
        )

        x_lab = x_label or _default_x_label(cap)
        fig.update_xaxes(title_text=x_lab, row=1, col=1)
        fig.update_xaxes(title_text=x_lab, row=1, col=2)
        y_label_left = 'Count' if not caller_histnorm else caller_histnorm.title()
        y_label_right = 'Percent' if not caller_histnorm else caller_histnorm.title()
        fig.update_yaxes(title_text=y_label_left, row=1, col=1)
        fig.update_yaxes(title_text=y_label_right, row=1, col=2)

        return fig

    # --- Single chart mode ---
    # Default to "percent" for potential view when caller didn't specify
    if view == 'potential' and not caller_histnorm:
        histnorm = 'percent'

    fig = go.Figure()

    _render_single_capability(
        fig,
        cap,
        vals,
        specs,
        theme,
        view=view,
        show_potential_text=show_potential,
        nbins=nbins,
        histnorm=histnorm,
        x_label=x_label,
    )

    fig.update_layout(**theme.to_layout_dict())
    fig.update_layout(
        title=title if title is not None else _auto_title(specs, view=view),
        xaxis_title=x_label or _default_x_label(cap),
        yaxis_title='Count' if not histnorm else histnorm.title(),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.18,
            xanchor='center',
            x=0.5,
        ),
        bargap=0.05,
    )

    return fig


# ---------------------------------------------------------------------------
# Single-panel renderer
# ---------------------------------------------------------------------------


def _render_single_capability(
    fig: go.Figure,
    cap: CapabilityResult,
    vals: np.ndarray,
    specs: SpecLimits,
    theme: ChartTheme,
    *,
    view: str = 'current',
    show_potential_text: bool = True,
    nbins: int | None = None,
    histnorm: str = '',
    x_label: str | None = None,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Render histogram + spec lines + NPL + legend + annotation into fig.

    When *row*/*col* are ``None``, renders into a plain ``go.Figure``.
    When they are integers, renders into a subplot panel.
    """
    is_subplot = row is not None and col is not None

    # 1. Histogram — potential view uses recentered R2 values
    hist_data = cap.potential_values if view == 'potential' and cap.potential_values is not None else vals
    hist_kwargs: dict = dict(
        x=hist_data,
        marker_color=theme.data_color,
        opacity=0.75,
        histnorm=histnorm if histnorm else None,
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>',
        showlegend=not is_subplot or col == 1,
    )
    if nbins is not None:
        hist_kwargs['nbinsx'] = nbins
    if is_subplot:
        fig.add_trace(go.Histogram(**hist_kwargs), row=row, col=col)
    else:
        fig.add_trace(go.Histogram(**hist_kwargs))

    # 2. Out-of-spec shading
    _add_out_of_spec_shading(fig, vals, specs, row=row, col=col)

    # 3. Spec limit lines
    _add_spec_lines(fig, specs, row=row, col=col)

    # 4. NPL lines — skip for potential view (only spec lines + mean)
    if view != 'potential':
        _add_npl_lines(fig, cap, theme, view=view, row=row, col=col)
    else:
        # Still draw the mean line for potential view
        _add_mean_line(fig, cap, theme, row=row, col=col)

    # 5. Legend traces
    _add_legend_traces(fig, cap, theme, view=view, row=row, col=col)

    # 6. Annotation box
    index_text = _build_index_text(cap, show_potential_text, view=view)
    if is_subplot:
        # Position annotation relative to subplot axis
        xaxis_id = f'x{col}' if col > 1 else 'x'
        yaxis_id = f'y{col}' if col > 1 else 'y'
        fig.add_annotation(
            text=index_text,
            xref=f'{xaxis_id} domain',
            yref=f'{yaxis_id} domain',
            x=0.98,
            y=0.95,
            xanchor='right',
            yanchor='top',
            showarrow=False,
            font=dict(
                family='monospace',
                size=theme.stats_box_font_size,
                color=theme.stats_box_font_color,
            ),
            bgcolor=theme.stats_box_bgcolor,
            bordercolor=theme.stats_box_bordercolor,
            borderwidth=theme.stats_box_borderwidth,
            align='left',
        )
    else:
        fig.add_annotation(
            text=index_text,
            xref='paper',
            yref='paper',
            x=0.98,
            y=0.95,
            xanchor='right',
            yanchor='top',
            showarrow=False,
            font=dict(
                family='monospace',
                size=theme.stats_box_font_size,
                color=theme.stats_box_font_color,
            ),
            bgcolor=theme.stats_box_bgcolor,
            bordercolor=theme.stats_box_bordercolor,
            borderwidth=theme.stats_box_borderwidth,
            align='left',
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_x_label(cap: CapabilityResult) -> str:
    """Return response variable name (title-cased) or 'Value' as fallback."""
    if cap.response_var:
        return cap.response_var.replace('_', ' ').title()
    return 'Value'


def _normalize_values(
    values: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Convert array-like to clean float ndarray (drop NaN)."""
    import pandas as pd

    return pd.Series(values).astype(float).dropna().to_numpy()


def _build_index_text(
    cap: CapabilityResult,
    show_potential: bool,
    view: str = 'current',
) -> str:
    """Build capability index annotation text with deterministic ordering."""
    r = cap.round_to
    specs = cap.specs

    if view == 'potential':
        # Potential view: show σ̂(R2) and CP/CPL/CPU indices
        lines: list[str] = []
        lines.append(f'n = {cap.n}')
        lines.append(f'\u0232 = {round(cap.y_bar, r)}')
        lines.append(f'\u03c3\u0302(R2) = {round(cap.sigma_hat_r2, r)}')

        if specs.is_two_sided:
            lines.append('')
            lines.append(f'CP  Index = {_fmt(cap.cp, r)}')
            lines.append(f'CPL Index = {_fmt(cap.cpk_lower, r)}')
            lines.append(f'CPU Index = {_fmt(cap.cpk_upper, r)}')
            lines.append('')
            lines.append(f'Pct Below LSL = {_fmt(cap.potential_pct_below_lsl, 2)}%')
            lines.append(f'Pct Above USL = {_fmt(cap.potential_pct_above_usl, 2)}%')
        elif specs.usl is not None:
            lines.append('')
            lines.append(f'CPU Index = {_fmt(cap.cpk_upper, r)}')
            lines.append('')
            lines.append(f'Pct Above USL = {_fmt(cap.potential_pct_above_usl, 2)}%')
        else:
            lines.append('')
            lines.append(f'CPL Index = {_fmt(cap.cpk_lower, r)}')
            lines.append('')
            lines.append(f'Pct Below LSL = {_fmt(cap.potential_pct_below_lsl, 2)}%')

        return '<br>'.join(lines)

    # Current view
    lines = []
    lines.append(f'n = {cap.n}')
    lines.append(f'\u0232 = {round(cap.y_bar, r)}')
    lines.append(f'\u03c3\u0302 = {round(cap.sigma_hat, r)}')

    if specs.is_two_sided:
        lines.append('')
        lines.append(f'PP  Index = {_fmt(cap.pp, r)}')
        lines.append(f'PPL Index = {_fmt(cap.ppk_lower, r)}')
        lines.append(f'PPU Index = {_fmt(cap.ppk_upper, r)}')
        lines.append('')
        lines.append(f'Pct Below LSL = {_fmt(cap.pct_below_lsl, 2)}%')
        lines.append(f'Pct Above USL = {_fmt(cap.pct_above_usl, 2)}%')
    elif specs.usl is not None:
        lines.append('')
        lines.append(f'PPU Index = {_fmt(cap.ppk_upper, r)}')
        lines.append('')
        lines.append(f'Pct Above USL = {_fmt(cap.pct_above_usl, 2)}%')
    else:
        lines.append('')
        lines.append(f'PPL Index = {_fmt(cap.ppk_lower, r)}')
        lines.append('')
        lines.append(f'Pct Below LSL = {_fmt(cap.pct_below_lsl, 2)}%')

    return '<br>'.join(lines)


def _fmt(value: float | None, decimals: int) -> str:
    """Format a float for display, handling None and inf."""
    if value is None:
        return 'N/A'
    if not np.isfinite(value):
        return 'inf'
    return f'{value:.{decimals}f}'


def _get_axis_ref(row: int | None, col: int | None) -> tuple[str, str]:
    """Return (xref, yref) for shapes targeting a specific subplot axis."""
    if row is None or col is None:
        return 'x', 'y'
    # Plotly subplot axes: x, x2, x3... and y, y2, y3...
    idx = (row - 1) * 2 + col  # for 1-row layouts: col=1→1, col=2→2
    xref = 'x' if idx == 1 else f'x{idx}'
    yref = 'y' if idx == 1 else f'y{idx}'
    return xref, yref


def _add_spec_lines(
    fig: go.Figure,
    specs: SpecLimits,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add USL/LSL/Target vertical lines."""
    if row is not None and col is not None:
        xref, yref = _get_axis_ref(row, col)
        if specs.lsl is not None:
            fig.add_shape(
                type='line',
                x0=specs.lsl,
                x1=specs.lsl,
                y0=0,
                y1=1,
                xref=xref,
                yref=f'{yref} domain',
                line=dict(color=_SPEC_LINE_COLOR, width=2, dash='solid'),
                layer='above',
            )
        if specs.usl is not None:
            fig.add_shape(
                type='line',
                x0=specs.usl,
                x1=specs.usl,
                y0=0,
                y1=1,
                xref=xref,
                yref=f'{yref} domain',
                line=dict(color=_SPEC_LINE_COLOR, width=2, dash='solid'),
                layer='above',
            )
        if specs.target is not None:
            fig.add_shape(
                type='line',
                x0=specs.target,
                x1=specs.target,
                y0=0,
                y1=1,
                xref=xref,
                yref=f'{yref} domain',
                line=dict(color='#2E8B57', width=1.5, dash='dashdot'),
                layer='above',
            )
    else:
        if specs.lsl is not None:
            fig.add_vline(
                x=specs.lsl,
                line_color=_SPEC_LINE_COLOR,
                line_width=2,
                line_dash='solid',
                layer='above',
            )
        if specs.usl is not None:
            fig.add_vline(
                x=specs.usl,
                line_color=_SPEC_LINE_COLOR,
                line_width=2,
                line_dash='solid',
                layer='above',
            )
        if specs.target is not None:
            fig.add_vline(
                x=specs.target,
                line_color='#2E8B57',
                line_width=1.5,
                line_dash='dashdot',
                layer='above',
            )


def _add_npl_lines(
    fig: go.Figure,
    cap: CapabilityResult,
    theme: ChartTheme,
    *,
    view: str = 'current',
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add Natural Process Limit lines: mean, LNPL, UNPL."""
    center_color = theme.center_color

    # Select sigma based on view
    sigma = cap.sigma_hat_r2 if view == 'potential' else cap.sigma_hat

    is_subplot = row is not None and col is not None

    if is_subplot:
        xref, yref = _get_axis_ref(row, col)

        # Mean line — always drawn
        fig.add_shape(
            type='line',
            x0=cap.y_bar,
            x1=cap.y_bar,
            y0=0,
            y1=1,
            xref=xref,
            yref=f'{yref} domain',
            line=dict(color=center_color, width=2, dash='solid'),
            layer='above',
        )

        if sigma is None or sigma <= 0:
            if sigma is None:
                logger.warning('sigma is None for view=%s; NPL lines omitted', view)
            else:
                logger.warning('sigma <= 0 (sigma=%s); NPL lines omitted', sigma)
            return

        lnpl = cap.y_bar - 3 * sigma
        unpl = cap.y_bar + 3 * sigma

        fig.add_shape(
            type='line',
            x0=lnpl,
            x1=lnpl,
            y0=0,
            y1=1,
            xref=xref,
            yref=f'{yref} domain',
            line=dict(color=center_color, width=1.5, dash='dash'),
            layer='above',
        )
        fig.add_shape(
            type='line',
            x0=unpl,
            x1=unpl,
            y0=0,
            y1=1,
            xref=xref,
            yref=f'{yref} domain',
            line=dict(color=center_color, width=1.5, dash='dash'),
            layer='above',
        )
    else:
        # Mean line — always drawn
        fig.add_vline(
            x=cap.y_bar,
            line_color=center_color,
            line_width=2,
            line_dash='solid',
            layer='above',
        )

        if sigma is None or sigma <= 0:
            if sigma is None:
                logger.warning('sigma is None for view=%s; NPL lines omitted', view)
            else:
                logger.warning('sigma <= 0 (sigma=%s); NPL lines omitted', sigma)
            return

        lnpl = cap.y_bar - 3 * sigma
        unpl = cap.y_bar + 3 * sigma

        fig.add_vline(
            x=lnpl,
            line_color=center_color,
            line_width=1.5,
            line_dash='dash',
            layer='above',
        )
        fig.add_vline(
            x=unpl,
            line_color=center_color,
            line_width=1.5,
            line_dash='dash',
            layer='above',
        )


def _add_mean_line(
    fig: go.Figure,
    cap: CapabilityResult,
    theme: ChartTheme,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add only the mean (Y-bar) vertical line — used by potential view."""
    center_color = theme.center_color
    is_subplot = row is not None and col is not None

    if is_subplot:
        xref, yref = _get_axis_ref(row, col)
        fig.add_shape(
            type='line',
            x0=cap.y_bar,
            x1=cap.y_bar,
            y0=0,
            y1=1,
            xref=xref,
            yref=f'{yref} domain',
            line=dict(color=center_color, width=2, dash='solid'),
            layer='above',
        )
    else:
        fig.add_vline(
            x=cap.y_bar,
            line_color=center_color,
            line_width=2,
            line_dash='solid',
            layer='above',
        )


def _add_legend_traces(
    fig: go.Figure,
    cap: CapabilityResult,
    theme: ChartTheme,
    *,
    view: str = 'current',
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add invisible scatter traces that serve as legend entries for line styles."""
    specs = cap.specs
    center_color = theme.center_color
    is_subplot = row is not None and col is not None

    # In paired mode, only add legend traces for the first panel
    if is_subplot and col > 1:
        return

    # Select sigma based on view
    sigma = cap.sigma_hat_r2 if view == 'potential' else cap.sigma_hat

    # Spec limits (solid crimson)
    has_specs = specs.lsl is not None or specs.usl is not None
    if has_specs:
        spec_parts: list[str] = []
        if specs.lsl is not None:
            spec_parts.append(f'LSL={specs.lsl}')
        if specs.usl is not None:
            spec_parts.append(f'USL={specs.usl}')
        trace_kwargs = dict(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color=_SPEC_LINE_COLOR, width=2, dash='solid'),
            name=f'Spec Limits ({", ".join(spec_parts)})',
        )
        if is_subplot:
            fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**trace_kwargs))

    # Target (green dashdot)
    if specs.target is not None:
        trace_kwargs = dict(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color='#2E8B57', width=1.5, dash='dashdot'),
            name=f'Target ({specs.target})',
        )
        if is_subplot:
            fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**trace_kwargs))

    # Y-bar (solid green)
    trace_kwargs = dict(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color=center_color, width=2, dash='solid'),
        name=f'Y-bar ({round(cap.y_bar, cap.round_to)})',
    )
    if is_subplot:
        fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
    else:
        fig.add_trace(go.Scatter(**trace_kwargs))

    # NPL (dashed green) — uses view-dependent sigma; skip for potential view
    if view != 'potential' and sigma is not None and sigma > 0:
        lnpl = round(cap.y_bar - 3 * sigma, cap.round_to)
        unpl = round(cap.y_bar + 3 * sigma, cap.round_to)
        trace_kwargs = dict(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color=center_color, width=1.5, dash='dash'),
            name=f'NPL ({lnpl}, {unpl})',
        )
        if is_subplot:
            fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**trace_kwargs))


def _add_out_of_spec_shading(
    fig: go.Figure,
    vals: np.ndarray,
    specs: SpecLimits,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add padded vrects beyond spec limits."""
    if len(vals) == 0:
        return

    xmin, xmax = float(vals.min()), float(vals.max())
    pad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    xminp = xmin - pad
    xmaxp = xmax + pad

    is_subplot = row is not None and col is not None

    if is_subplot:
        xref, yref = _get_axis_ref(row, col)
        if specs.lsl is not None and specs.lsl > xminp:
            fig.add_shape(
                type='rect',
                x0=xminp,
                x1=specs.lsl,
                y0=0,
                y1=1,
                xref=xref,
                yref=f'{yref} domain',
                fillcolor=_SPEC_LINE_COLOR,
                opacity=_SPEC_SHADE_OPACITY,
                line_width=0,
                layer='below',
            )
        if specs.usl is not None and specs.usl < xmaxp:
            fig.add_shape(
                type='rect',
                x0=specs.usl,
                x1=xmaxp,
                y0=0,
                y1=1,
                xref=xref,
                yref=f'{yref} domain',
                fillcolor=_SPEC_LINE_COLOR,
                opacity=_SPEC_SHADE_OPACITY,
                line_width=0,
                layer='below',
            )
    else:
        if specs.lsl is not None and specs.lsl > xminp:
            fig.add_vrect(
                x0=xminp,
                x1=specs.lsl,
                fillcolor=_SPEC_LINE_COLOR,
                opacity=_SPEC_SHADE_OPACITY,
                line_width=0,
                layer='below',
            )
        if specs.usl is not None and specs.usl < xmaxp:
            fig.add_vrect(
                x0=specs.usl,
                x1=xmaxp,
                fillcolor=_SPEC_LINE_COLOR,
                opacity=_SPEC_SHADE_OPACITY,
                line_width=0,
                layer='below',
            )


def _auto_title(specs: SpecLimits, *, view: str = 'current') -> str:
    """Generate title from SpecLimits."""
    parts: list[str] = []
    if specs.lsl is not None:
        parts.append(f'LSL={specs.lsl}')
    if specs.target is not None:
        parts.append(f'Target={specs.target}')
    if specs.usl is not None:
        parts.append(f'USL={specs.usl}')
    prefix = 'Potential' if view == 'potential' else 'Process'
    return f'{prefix} Capability ({", ".join(parts)})'
