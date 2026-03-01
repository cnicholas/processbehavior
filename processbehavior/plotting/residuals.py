"""Residual diagnostic plots.

Extracted from plotter.py to isolate residual visualization from
control chart rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..exceptions import ValidationError
from ._math import normal_pdf, normal_ppf
from .control_chart import ControlChartFigure
from .themes import ChartTheme, apply_theme, get_theme

if TYPE_CHECKING:
    from ..analysis_result import AnalysisResult


def plot_residuals(
    result: AnalysisResult,
    residual_type: str = 'R1',
    plot_type: str = 'all',
    theme: str | ChartTheme = 'processbehavior',
    width: int = 1200,
    height: int = 400,
) -> ControlChartFigure:
    """Create residual diagnostic plots.

    Visualizes VAS residuals to assess process behavior and identify
    patterns that may indicate non-random variation.

    Parameters
    ----------
    result : AnalysisResult
        The analysis result containing residuals.
    residual_type : str, default 'R1'
        Which residual to plot ('R1', 'R2', 'R3', 'R4', 'R5').
    plot_type : str, default 'all'
        Type of diagnostic plot:
        - 'histogram': Distribution of residuals
        - 'qq': Quantile-quantile plot for normality
        - 'sequence': Residuals vs. observation order
        - 'all': All three plots in subplots
    theme : str or ChartTheme, default 'processbehavior'
        Visual theme.
    width : int, default 1200
        Figure width in pixels.
    height : int, default 400
        Figure height in pixels (per row for 'all').

    Returns
    -------
    ControlChartFigure
        Interactive figure with residual diagnostics.

    Raises
    ------
    ValidationError
        If residuals not available or invalid residual type.
    """
    if not result.has_residuals:
        raise ValidationError(
            "Residuals not available for this analysis.\n"
            "Residuals require SDS >= 1 (replicated observations)."
        )

    residuals = result.residuals
    if residual_type not in residuals.columns:
        available = list(residuals.columns)
        raise ValidationError(
            f"Residual '{residual_type}' not found.\n"
            f"Available: {available}"
        )

    theme = get_theme(theme) if isinstance(theme, str) else theme
    r_data = residuals[residual_type].dropna()

    if plot_type == 'all':
        fig = _plot_all_diagnostics(r_data, residual_type, theme, width, height)
    elif plot_type == 'histogram':
        fig = _plot_histogram(r_data, residual_type, theme, width, height)
    elif plot_type == 'qq':
        fig = _plot_qq(r_data, residual_type, theme, width, height)
    elif plot_type == 'sequence':
        fig = _plot_sequence(r_data, residual_type, theme, width, height)
    else:
        raise ValidationError(
            f"Invalid plot_type: '{plot_type}'.\n"
            f"Options: 'all', 'histogram', 'qq', 'sequence'"
        )

    fig = apply_theme(fig, theme)
    return ControlChartFigure(fig, result)


# ---------------------------------------------------------------------------
#  Internal plot builders
# ---------------------------------------------------------------------------


def _plot_all_diagnostics(r_data, residual_type, theme, width, height):
    """3-panel diagnostic: histogram + Q-Q + sequence."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Histogram', 'Q-Q Plot', 'Sequence Plot'],
        horizontal_spacing=0.08,
    )

    # 1. Histogram
    fig.add_trace(
        go.Histogram(
            x=r_data, name='Residuals',
            marker_color=theme.data_color, opacity=0.7,
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Normal curve overlay
    x_range = np.linspace(r_data.min(), r_data.max(), 100)
    mu, sigma = r_data.mean(), r_data.std()
    y_normal = np.array([normal_pdf(x, mu, sigma) for x in x_range])
    y_normal = y_normal * len(r_data) * (r_data.max() - r_data.min()) / 20

    fig.add_trace(
        go.Scatter(
            x=x_range, y=y_normal, mode='lines', name='Normal',
            line=dict(color=theme.center_color, width=2, dash='dash'),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # 2. Q-Q Plot
    probs = np.linspace(0.01, 0.99, len(r_data))
    theoretical_q = np.array([normal_ppf(p) for p in probs])
    sample_q = np.sort(r_data.values)

    fig.add_trace(
        go.Scatter(
            x=theoretical_q, y=sample_q, mode='markers', name='Q-Q',
            marker=dict(color=theme.data_color, size=5),
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Reference line
    qq_min = min(theoretical_q.min(), sample_q.min())
    qq_max = max(theoretical_q.max(), sample_q.max())
    fig.add_trace(
        go.Scatter(
            x=[qq_min, qq_max],
            y=[qq_min * r_data.std() + r_data.mean(),
               qq_max * r_data.std() + r_data.mean()],
            mode='lines', name='Reference',
            line=dict(color=theme.ucl_color, dash='dash'),
            showlegend=False,
        ),
        row=1, col=2,
    )

    # 3. Sequence plot
    fig.add_trace(
        go.Scatter(
            x=list(range(len(r_data))), y=r_data.values,
            mode='lines+markers', name='Residuals',
            marker=dict(size=4, color=theme.data_color),
            line=dict(color=theme.data_color, width=1),
            showlegend=False,
        ),
        row=1, col=3,
    )
    fig.add_hline(y=0, line_dash='dash', line_color=theme.center_color, row=1, col=3)

    fig.update_layout(
        title=f'{residual_type} Residual Diagnostics',
        width=width, height=height, showlegend=False,
    )
    fig.update_xaxes(title_text='Residual Value', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
    fig.update_yaxes(title_text='Sample Quantiles', row=1, col=2)
    fig.update_xaxes(title_text='Observation', row=1, col=3)
    fig.update_yaxes(title_text='Residual', row=1, col=3)

    return fig


def _plot_histogram(r_data, residual_type, theme, width, height):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r_data, name='Residuals',
        marker_color=theme.data_color, opacity=0.7,
    ))
    fig.update_layout(
        title=f'{residual_type} Residual Distribution',
        xaxis_title='Residual Value', yaxis_title='Frequency',
        width=width, height=height,
    )
    return fig


def _plot_qq(r_data, residual_type, theme, width, height):
    probs = np.linspace(0.01, 0.99, len(r_data))
    theoretical_q = np.array([normal_ppf(p) for p in probs])
    sample_q = np.sort(r_data.values)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical_q, y=sample_q, mode='markers', name='Data',
        marker=dict(color=theme.data_color, size=6),
    ))
    qq_min = min(theoretical_q.min(), sample_q.min())
    qq_max = max(theoretical_q.max(), sample_q.max())
    fig.add_trace(go.Scatter(
        x=[qq_min, qq_max],
        y=[qq_min * r_data.std() + r_data.mean(),
           qq_max * r_data.std() + r_data.mean()],
        mode='lines', name='Normal Reference',
        line=dict(color=theme.ucl_color, dash='dash'),
    ))
    fig.update_layout(
        title=f'{residual_type} Q-Q Plot',
        xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles',
        width=width, height=height,
    )
    return fig


def _plot_sequence(r_data, residual_type, theme, width, height):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(r_data))), y=r_data.values,
        mode='lines+markers', name='Residuals',
        marker=dict(size=5, color=theme.data_color),
        line=dict(color=theme.data_color, width=1),
    ))
    fig.add_hline(y=0, line_dash='dash', line_color=theme.center_color)
    fig.update_layout(
        title=f'{residual_type} Residual Sequence Plot',
        xaxis_title='Observation', yaxis_title='Residual Value',
        width=width, height=height,
    )
    return fig
