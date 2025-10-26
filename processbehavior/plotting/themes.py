"""
Theme definitions for control chart plotting.

Provides pre-configured visual themes and theme application utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go


# Default ProcessBehavior theme
PROCESSBEHAVIOR_THEME = {
    'layout': {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {
            'family': 'Arial, sans-serif',
            'size': 12,
            'color': '#333'
        },
        'title': {
            'font': {'size': 16, 'color': '#222'},
            'x': 0.5,
            'xanchor': 'center'
        },
        'xaxis': {
            'showgrid': True,
            'gridcolor': '#eee',
            'linecolor': '#999',
            'linewidth': 1
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': '#eee',
            'linecolor': '#999',
            'linewidth': 1
        }
    }
}

# Minimal theme
MINIMAL_THEME = {
    'layout': {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {'family': 'Helvetica, sans-serif', 'size': 11},
        'xaxis': {'showgrid': False, 'linecolor': '#ccc'},
        'yaxis': {'showgrid': True, 'gridcolor': '#f5f5f5', 'linecolor': '#ccc'}
    }
}

# Dark theme
DARK_THEME = {
    'layout': {
        'plot_bgcolor': '#1e1e1e',
        'paper_bgcolor': '#2d2d2d',
        'font': {'family': 'Arial, sans-serif', 'size': 12, 'color': '#e0e0e0'},
        'title': {'font': {'color': '#fff'}},
        'xaxis': {
            'showgrid': True,
            'gridcolor': '#444',
            'linecolor': '#666',
            'color': '#e0e0e0'
        },
        'yaxis': {
            'showgrid': True,
            'gridcolor': '#444',
            'linecolor': '#666',
            'color': '#e0e0e0'
        }
    }
}

# Theme registry
THEMES = {
    'processbehavior': PROCESSBEHAVIOR_THEME,
    'minimal': MINIMAL_THEME,
    'dark': DARK_THEME
}


def apply_theme(fig: go.Figure, template: str = 'processbehavior') -> go.Figure:
    """
    Apply a theme to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to apply theme to
    template : str, default 'processbehavior'
        Theme name ('processbehavior', 'minimal', 'dark')

    Returns
    -------
    go.Figure
        Figure with theme applied

    Examples
    --------
    >>> fig = go.Figure()
    >>> fig = apply_theme(fig, 'dark')
    """
    if template not in THEMES:
        available = list(THEMES.keys())
        raise ValueError(
            f"Unknown theme: '{template}'.\n"
            f"Available themes: {available}"
        )

    theme = THEMES[template]
    fig.update_layout(**theme['layout'])

    return fig
