"""
Process Capability visualization chart.

Produces a Plotly histogram with specification limit lines, natural process
limit (NPL) overlay, capability index annotation box, and out-of-spec shading.

Follows the ``effects_charts.py`` pattern: standalone function, takes theme,
returns ``go.Figure``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    import pandas as pd

    from ..capability import CapabilityResult, SpecLimits
    from .themes import ChartTheme

# ---------------------------------------------------------------------------
# Local constants — spec-specific colors (NOT reused from control limits)
# ---------------------------------------------------------------------------

_SPEC_LINE_COLOR = "#DC143C"  # Crimson — distinct from control limit colors
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
    x_label: str | None = None,
    nbins: int | None = None,
    histnorm: str = "",
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
    from .themes import get_theme

    if theme is None:
        theme = get_theme("processbehavior")

    vals = _normalize_values(values)
    specs = cap.specs

    fig = go.Figure()

    # 1. Histogram
    hist_kwargs: dict = dict(
        x=vals,
        marker_color=theme.data_color,
        opacity=0.75,
        histnorm=histnorm if histnorm else None,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    )
    if nbins is not None:
        hist_kwargs["nbinsx"] = nbins
    fig.add_trace(go.Histogram(**hist_kwargs))

    # 2. Out-of-spec shading (layer="below" — under bars)
    _add_out_of_spec_shading(fig, vals, specs)

    # 3. Spec limit lines (layer="above" — on top of bars)
    _add_spec_lines(fig, specs)

    # 4. Natural Process Limits (layer="above")
    _add_npl_lines(fig, cap, theme)

    # 5. Legend traces — invisible markers to label the line styles
    _add_legend_traces(fig, cap, theme)

    # 6. Capability index annotation box
    index_text = _build_index_text(cap, show_potential)
    fig.add_annotation(
        text=index_text,
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.95,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        font=dict(
            family="monospace",
            size=theme.stats_box_font_size,
            color=theme.stats_box_font_color,
        ),
        bgcolor=theme.stats_box_bgcolor,
        bordercolor=theme.stats_box_bordercolor,
        borderwidth=theme.stats_box_borderwidth,
        align="left",
    )

    # 7. Layout — apply theme first, then override with chart-specific settings
    fig.update_layout(**theme.to_layout_dict())
    fig.update_layout(
        title=title if title is not None else _auto_title(specs),
        xaxis_title=x_label or "Value",
        yaxis_title="Count" if not histnorm else histnorm.title(),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
        ),
        bargap=0.05,
    )

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_values(
    values: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """Convert array-like to clean float ndarray (drop NaN)."""
    import pandas as pd

    return pd.Series(values).astype(float).dropna().to_numpy()


def _build_index_text(cap: CapabilityResult, show_potential: bool) -> str:
    """Build capability index annotation text with deterministic ordering."""
    r = cap.round_to
    lines: list[str] = []

    lines.append(f"n = {cap.n}")
    lines.append(f"Y-bar = {round(cap.y_bar, r)}")
    lines.append(f"sigma = {round(cap.sigma_hat, r)}")

    # Current capability
    specs = cap.specs
    if specs.is_two_sided:
        lines.append("")
        lines.append(f"Pp  = {_fmt(cap.pp, r)}")
        lines.append(f"Ppk = {_fmt(cap.ppk, r)}")
    elif specs.usl is not None:
        lines.append("")
        lines.append(f"Ppk(USL) = {_fmt(cap.ppk, r)}")
    else:
        lines.append("")
        lines.append(f"Ppk(LSL) = {_fmt(cap.ppk, r)}")

    # Potential capability
    if show_potential and cap.cp is not None:
        if specs.is_two_sided:
            lines.append("")
            lines.append(f"Cp  = {_fmt(cap.cp, r)}")
            lines.append(f"Cpk = {_fmt(cap.cpk, r)}")
        elif specs.usl is not None:
            lines.append("")
            lines.append(f"Cpk(USL) = {_fmt(cap.cpk, r)}")
        else:
            lines.append("")
            lines.append(f"Cpk(LSL) = {_fmt(cap.cpk, r)}")

    # Empirical outside
    lines.append("")
    if specs.is_two_sided:
        lines.append(f"Outside: {cap.n_outside} ({_fmt(cap.pct_outside, 2)}%)")
    elif specs.usl is not None:
        lines.append(f"Above USL: {cap.n_above_usl} ({_fmt(cap.pct_above_usl, 2)}%)")
    else:
        lines.append(f"Below LSL: {cap.n_below_lsl} ({_fmt(cap.pct_below_lsl, 2)}%)")

    return "<br>".join(lines)


def _fmt(value: float | None, decimals: int) -> str:
    """Format a float for display, handling None and inf."""
    if value is None:
        return "N/A"
    if not np.isfinite(value):
        return "inf"
    return f"{value:.{decimals}f}"


def _add_spec_lines(fig: go.Figure, specs: SpecLimits) -> None:
    """Add USL/LSL/Target vertical lines."""
    if specs.lsl is not None:
        fig.add_vline(
            x=specs.lsl,
            line_color=_SPEC_LINE_COLOR,
            line_width=2,
            line_dash="solid",
            layer="above",
        )

    if specs.usl is not None:
        fig.add_vline(
            x=specs.usl,
            line_color=_SPEC_LINE_COLOR,
            line_width=2,
            line_dash="solid",
            layer="above",
        )

    if specs.target is not None:
        fig.add_vline(
            x=specs.target,
            line_color="#2E8B57",
            line_width=1.5,
            line_dash="dashdot",
            layer="above",
        )


def _add_npl_lines(
    fig: go.Figure, cap: CapabilityResult, theme: ChartTheme
) -> None:
    """Add Natural Process Limit lines: mean, LNPL, UNPL."""
    center_color = theme.center_color

    # Mean line — always drawn
    fig.add_vline(
        x=cap.y_bar,
        line_color=center_color,
        line_width=2,
        line_dash="solid",
        layer="above",
    )

    # Guard: skip LNPL/UNPL when sigma is 0 (or negligible)
    if cap.sigma_hat <= 0:
        return

    lnpl = cap.y_bar - 3 * cap.sigma_hat
    unpl = cap.y_bar + 3 * cap.sigma_hat

    fig.add_vline(
        x=lnpl,
        line_color=center_color,
        line_width=1.5,
        line_dash="dash",
        layer="above",
    )

    fig.add_vline(
        x=unpl,
        line_color=center_color,
        line_width=1.5,
        line_dash="dash",
        layer="above",
    )


def _add_legend_traces(
    fig: go.Figure, cap: CapabilityResult, theme: ChartTheme
) -> None:
    """Add invisible scatter traces that serve as legend entries for line styles."""
    specs = cap.specs
    center_color = theme.center_color

    # Spec limits (solid crimson)
    has_specs = specs.lsl is not None or specs.usl is not None
    if has_specs:
        spec_parts: list[str] = []
        if specs.lsl is not None:
            spec_parts.append(f"LSL={specs.lsl}")
        if specs.usl is not None:
            spec_parts.append(f"USL={specs.usl}")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=_SPEC_LINE_COLOR, width=2, dash="solid"),
            name=f"Spec Limits ({', '.join(spec_parts)})",
        ))

    # Target (green dashdot)
    if specs.target is not None:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#2E8B57", width=1.5, dash="dashdot"),
            name=f"Target ({specs.target})",
        ))

    # Y-bar (solid green)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=center_color, width=2, dash="solid"),
        name=f"Y-bar ({round(cap.y_bar, cap.round_to)})",
    ))

    # NPL (dashed green)
    if cap.sigma_hat > 0:
        lnpl = round(cap.y_bar - 3 * cap.sigma_hat, cap.round_to)
        unpl = round(cap.y_bar + 3 * cap.sigma_hat, cap.round_to)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=center_color, width=1.5, dash="dash"),
            name=f"NPL ({lnpl}, {unpl})",
        ))


def _add_out_of_spec_shading(
    fig: go.Figure,
    vals: np.ndarray,
    specs: SpecLimits,
) -> None:
    """Add padded vrects beyond spec limits."""
    if len(vals) == 0:
        return

    xmin, xmax = float(vals.min()), float(vals.max())
    pad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    xminp = xmin - pad
    xmaxp = xmax + pad

    if specs.lsl is not None and specs.lsl > xminp:
        fig.add_vrect(
            x0=xminp,
            x1=specs.lsl,
            fillcolor=_SPEC_LINE_COLOR,
            opacity=_SPEC_SHADE_OPACITY,
            line_width=0,
            layer="below",
        )

    if specs.usl is not None and specs.usl < xmaxp:
        fig.add_vrect(
            x0=specs.usl,
            x1=xmaxp,
            fillcolor=_SPEC_LINE_COLOR,
            opacity=_SPEC_SHADE_OPACITY,
            line_width=0,
            layer="below",
        )


def _auto_title(specs: SpecLimits) -> str:
    """Generate title from SpecLimits."""
    parts: list[str] = []
    if specs.lsl is not None:
        parts.append(f"LSL={specs.lsl}")
    if specs.target is not None:
        parts.append(f"Target={specs.target}")
    if specs.usl is not None:
        parts.append(f"USL={specs.usl}")
    return f"Process Capability ({', '.join(parts)})"
