"""
Theme definitions for control chart plotting.

Provides a comprehensive theme system with full customization of colors,
sizes, fonts, and layout properties. Inspired by ggplot2's theme system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go


@dataclass
class ChartTheme:
    """
    Complete theme specification for control charts.

    This dataclass defines all visual properties for control chart rendering,
    following ggplot2's philosophy of sensible defaults with full customization.

    Attributes
    ----------
    name : str
        Theme identifier

    # Data appearance
    data_color : str
        Color for data points and lines
    data_marker_size : int
        Size of data point markers
    data_line_width : float
        Width of data connecting lines

    # Control limits
    ucl_color : str
        Upper control limit line color
    lcl_color : str
        Lower control limit line color
    center_color : str
        Centerline color
    limit_line_dash : str
        Dash style for control limits ('dash', 'dot', 'solid')
    limit_line_width : float
        Width of control limit lines
    center_line_width : float
        Width of centerline

    # Signal/out-of-control markers
    signal_color : str
        Color for out-of-control points
    signal_marker_size : int
        Size of signal markers
    signal_marker_symbol : str
        Plotly marker symbol ('x', 'circle', 'diamond', etc.)
    signal_marker_line_width : float
        Line width of signal marker border
    signal_marker_line_color : str
        Border color of signal markers

    # Zone shading (for Western Electric rules visualization)
    zone_a_color : str
        Zone A (±2σ to ±3σ) fill color
    zone_b_color : str
        Zone B (±1σ to ±2σ) fill color
    zone_c_color : str
        Zone C (±0σ to ±1σ) fill color
    zone_opacity : float
        Opacity for zone shading (0-1)

    # Layout
    plot_bgcolor : str
        Plot area background color
    paper_bgcolor : str
        Figure background color
    grid_color : str
        Grid line color
    grid_width : float
        Grid line width
    show_grid : bool
        Whether to show grid lines
    axis_line_color : str
        Axis line color
    axis_line_width : float
        Axis line width

    # Typography
    font_family : str
        Font family for all text
    font_size : int
        Base font size
    font_color : str
        Default text color
    title_font_size : int
        Title font size
    title_font_color : str
        Title color
    axis_title_font_size : int
        Axis title font size
    annotation_font_size : int
        Annotation (limit labels) font size
    data_opacity : float
        Opacity for data points and lines (0-1)
    show_axis_line : bool
        Whether to show axis lines
    facet_col_spacing : float
        Horizontal spacing between facet columns (fraction of figure width)
    facet_row_spacing : float
        Vertical spacing between facet rows (fraction of figure height)
    lane_boundary_color : str
        Color for lane boundary lines (vertical separators for collapsed factors)
    lane_boundary_width : float
        Width of lane boundary lines
    lane_label_font_size : int
        Font size for lane boundary annotations
    stats_box_bgcolor : str
        Background color for statistics box annotations
    stats_box_bordercolor : str
        Border color for statistics box
    stats_box_borderwidth : int
        Border width for statistics box
    stats_box_font_size : int
        Font size for statistics box text
    stats_box_font_color : str
        Font color for statistics box text
    pattern_signal_color : str
        Color for pattern-based signal markers (Rules 2-8)

    Examples
    --------
    Create custom theme:

    >>> theme = ChartTheme(
    ...     name='custom',
    ...     data_color='navy',
    ...     signal_color='orange',
    ...     center_color='darkgreen'
    ... )

    Modify existing theme:

    >>> theme = get_theme('processbehavior')
    >>> theme.data_color = 'purple'
    """

    # Theme name
    name: str = 'processbehavior'

    # Data appearance
    data_color: str = 'steelblue'
    data_marker_size: int = 5
    data_line_width: float = 1.0
    data_opacity: float = 1.0

    # Control limits
    ucl_color: str = 'red'
    lcl_color: str = 'red'
    center_color: str = '#2E8B57'  # SeaGreen - professional green
    limit_line_dash: str = 'dash'
    limit_line_width: float = 1.5
    center_line_width: float = 1.5

    # Signal markers - Two-tier system:
    # Both tiers use same marker style/size as data (circle, size 5), differentiated only by color
    # Tier 1 (Rule 1): Red - points outside control limits
    signal_color: str = 'red'
    signal_marker_size: int = 5    # Same size as data points
    signal_marker_symbol: str = 'circle'  # Same as data points
    signal_marker_line_width: float = 0.5
    signal_marker_line_color: str = 'darkred'
    # Tier 2 (Rules 2-8): Orange - pattern-based signals
    pattern_signal_color: str = '#FF8C00'  # Dark Orange

    # Zone shading
    zone_a_color: str = '#FFB3B3'  # Light red
    zone_b_color: str = '#FFFFB3'  # Light yellow
    zone_c_color: str = '#B3FFB3'  # Light green
    zone_opacity: float = 0.15

    # Layout
    plot_bgcolor: str = 'white'
    paper_bgcolor: str = 'white'
    grid_color: str = '#E5E5E5'
    grid_width: float = 1.0
    show_grid: bool = True
    axis_line_color: str = '#999999'
    axis_line_width: float = 1.0
    show_axis_line: bool = True

    # Typography
    font_family: str = 'Arial, sans-serif'
    font_size: int = 12
    font_color: str = '#333333'
    title_font_size: int = 16
    title_font_color: str = '#222222'
    axis_title_font_size: int = 12
    annotation_font_size: int = 10

    # Faceted plot settings
    facet_marker_size: int = 5
    facet_line_width: float = 1.5

    # Lane boundaries (vertical separators for collapsed factors)
    lane_boundary_color: str = '#888888'  # Medium gray
    lane_boundary_dash: str = 'dot'
    lane_boundary_width: float = 1.0
    lane_boundary_annotation_size: int = 8

    # Stats box
    stats_box_bgcolor: str = 'rgba(255, 255, 255, 0.9)'
    stats_box_bordercolor: str = '#CCCCCC'
    stats_box_borderwidth: int = 1
    stats_box_font_size: int = 10
    stats_box_font_color: str = '#333333'

    def to_layout_dict(self) -> dict:
        """
        Convert theme to Plotly layout dictionary.

        Returns
        -------
        dict
            Layout properties for fig.update_layout()
        """
        return {
            'plot_bgcolor': self.plot_bgcolor,
            'paper_bgcolor': self.paper_bgcolor,
            'font': {
                'family': self.font_family,
                'size': self.font_size,
                'color': self.font_color
            },
            'title': {
                'font': {
                    'size': self.title_font_size,
                    'color': self.title_font_color
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'showgrid': self.show_grid,
                'gridcolor': self.grid_color,
                'gridwidth': self.grid_width,
                'showline': self.show_axis_line,
                'linecolor': self.axis_line_color,
                'linewidth': self.axis_line_width
            },
            'yaxis': {
                'showgrid': self.show_grid,
                'gridcolor': self.grid_color,
                'gridwidth': self.grid_width,
                'showline': self.show_axis_line,
                'linecolor': self.axis_line_color,
                'linewidth': self.axis_line_width
            }
        }


# =============================================================================
# Built-in Themes
# =============================================================================

def _create_processbehavior_theme() -> ChartTheme:
    """Default theme with professional, balanced appearance."""
    return ChartTheme(
        name='processbehavior',
        data_opacity=0.8,
    )


def _create_ggplot_theme() -> ChartTheme:
    """Theme inspired by ggplot2's default appearance."""
    return ChartTheme(
        name='ggplot',
        # Data - ggplot2's default blue
        data_color='#3366CC',
        # Control limits
        ucl_color='#E31A1C',  # ggplot2 red
        lcl_color='#E31A1C',
        center_color='#33A02C',  # ggplot2 green
        limit_line_width=1.2,
        center_line_width=1.2,
        # Signals - same marker as data, just red fill
        signal_color='#E31A1C',
        signal_marker_symbol='circle',
        signal_marker_line_color='#B2182B',
        # Zones
        zone_a_color='#FDAE61',
        zone_b_color='#FEE08B',
        zone_c_color='#D9EF8B',
        zone_opacity=0.2,
        # Layout - ggplot2's gray panel background
        plot_bgcolor='#EBEBEB',
        paper_bgcolor='white',
        grid_color='white',
        grid_width=1.0,
        axis_line_color='#EBEBEB',
        # Typography
        font_family='sans-serif',
        font_size=11,
        font_color='#333333',
        title_font_size=14,
    )


def _create_minimal_theme() -> ChartTheme:
    """Clean, minimalist theme with reduced visual elements."""
    return ChartTheme(
        name='minimal',
        # Data
        data_color='#2C3E50',  # Dark slate
        # Control limits - subtle
        ucl_color='#95A5A6',  # Gray
        lcl_color='#95A5A6',
        center_color='#7F8C8D',
        limit_line_dash='dot',
        limit_line_width=1.0,
        center_line_width=1.0,
        # Signals - same marker as data, just red fill
        signal_color='#E74C3C',  # Flat red
        signal_marker_symbol='circle',
        signal_marker_line_color='#C0392B',
        # Zones - very subtle
        zone_a_color='#FADBD8',
        zone_b_color='#FCF3CF',
        zone_c_color='#D5F5E3',
        zone_opacity=0.1,
        # Layout - minimal grid
        plot_bgcolor='white',
        paper_bgcolor='white',
        grid_color='#F5F5F5',
        show_grid=True,  # Only y-grid
        axis_line_color='#CCCCCC',
        # Typography
        font_family='Helvetica, sans-serif',
        font_size=11,
        font_color='#2C3E50',
        title_font_size=14,
        title_font_color='#2C3E50',
    )


def _create_dark_theme() -> ChartTheme:
    """Dark theme for presentations and dark mode UIs."""
    return ChartTheme(
        name='dark',
        # Data - bright for contrast
        data_color='#00D4AA',  # Teal/cyan
        # Control limits
        ucl_color='#FF6B6B',  # Soft red
        lcl_color='#FF6B6B',
        center_color='#4ECDC4',  # Bright teal
        limit_line_width=1.5,
        center_line_width=1.5,
        # Signals - same marker as data, just red fill
        signal_color='#FF4757',  # Bright red
        signal_marker_symbol='circle',
        signal_marker_line_color='#FF6B81',
        # Zones - semi-transparent on dark
        zone_a_color='#FF6B6B',
        zone_b_color='#FFA502',
        zone_c_color='#2ED573',
        zone_opacity=0.15,
        # Layout
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#2D2D2D',
        grid_color='#444444',
        axis_line_color='#666666',
        # Typography
        font_family='Arial, sans-serif',
        font_size=12,
        font_color='#E0E0E0',
        title_font_size=16,
        title_font_color='#FFFFFF',
    )


def _create_publication_theme() -> ChartTheme:
    """Theme optimized for academic publications and print."""
    return ChartTheme(
        name='publication',
        # Data - high contrast black
        data_color='#000000',
        data_marker_size=5,
        # Control limits - grayscale for B&W printing
        ucl_color='#666666',
        lcl_color='#666666',
        center_color='#333333',
        limit_line_dash='dash',
        limit_line_width=0.8,
        center_line_width=1.0,
        # Signals - red to stand out against black data
        signal_color='red',
        signal_marker_size=5,   # Same as data_marker_size
        signal_marker_symbol='circle',
        signal_marker_line_width=0.5,
        signal_marker_line_color='darkred',
        # Zones - not used in print (set opacity to 0)
        zone_opacity=0.0,
        # Layout - clean white
        plot_bgcolor='white',
        paper_bgcolor='white',
        grid_color='#CCCCCC',
        grid_width=0.5,
        axis_line_color='#000000',
        axis_line_width=1.0,
        # Typography - serif for academic
        font_family='Times New Roman, serif',
        font_size=10,
        font_color='#000000',
        title_font_size=12,
        title_font_color='#000000',
        axis_title_font_size=10,
        annotation_font_size=9,
        facet_line_width=0.8,
    )


# =============================================================================
# Theme Registry and Access Functions
# =============================================================================

# Registry of built-in themes (lazily populated)
_THEME_REGISTRY: dict[str, ChartTheme] = {}


def _ensure_registry():
    """Populate theme registry if empty."""
    global _THEME_REGISTRY
    if not _THEME_REGISTRY:
        _THEME_REGISTRY = {
            'processbehavior': _create_processbehavior_theme(),
            'ggplot': _create_ggplot_theme(),
            'minimal': _create_minimal_theme(),
            'dark': _create_dark_theme(),
            'publication': _create_publication_theme(),
        }


def get_theme(name: str) -> ChartTheme:
    """
    Get a theme by name.

    Parameters
    ----------
    name : str
        Theme name ('processbehavior', 'ggplot', 'minimal', 'dark', 'publication')

    Returns
    -------
    ChartTheme
        Copy of the requested theme (safe to modify)

    Raises
    ------
    ValueError
        If theme name is not found

    Examples
    --------
    >>> theme = get_theme('ggplot')
    >>> theme.data_color = 'purple'  # Safe to modify
    """
    _ensure_registry()

    if name not in _THEME_REGISTRY:
        available = list(_THEME_REGISTRY.keys())
        raise ValueError(
            f"Unknown theme: '{name}'.\n"
            f"Available themes: {available}"
        )

    # Return a copy so modifications don't affect the registry
    import copy
    return copy.copy(_THEME_REGISTRY[name])


def list_themes() -> list[str]:
    """
    List all available theme names.

    Returns
    -------
    list of str
        Available theme names
    """
    _ensure_registry()
    return list(_THEME_REGISTRY.keys())


def register_theme(theme: ChartTheme):
    """
    Register a custom theme.

    Parameters
    ----------
    theme : ChartTheme
        Theme to register (uses theme.name as key)

    Examples
    --------
    >>> custom = ChartTheme(name='corporate', data_color='#003366')
    >>> register_theme(custom)
    >>> fig = plotter.plot(theme='corporate')
    """
    _ensure_registry()
    _THEME_REGISTRY[theme.name] = theme


def apply_theme(fig: go.Figure, theme: str | ChartTheme = 'processbehavior') -> go.Figure:
    """
    Apply a theme to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to apply theme to
    theme : str or ChartTheme
        Theme name or ChartTheme instance

    Returns
    -------
    go.Figure
        Figure with theme applied

    Examples
    --------
    >>> fig = go.Figure()
    >>> fig = apply_theme(fig, 'dark')

    >>> custom_theme = ChartTheme(name='custom', plot_bgcolor='#F0F0F0')
    >>> fig = apply_theme(fig, custom_theme)
    """
    theme = get_theme(theme) if isinstance(theme, str) else theme

    fig.update_layout(**theme.to_layout_dict())

    return fig


# =============================================================================
# Backward Compatibility
# =============================================================================

# Legacy theme dictionaries for backward compatibility
PROCESSBEHAVIOR_THEME = {
    'layout': _create_processbehavior_theme().to_layout_dict()
}

MINIMAL_THEME = {
    'layout': _create_minimal_theme().to_layout_dict()
}

DARK_THEME = {
    'layout': _create_dark_theme().to_layout_dict()
}

# Legacy registry
THEMES = {
    'processbehavior': PROCESSBEHAVIOR_THEME,
    'minimal': MINIMAL_THEME,
    'dark': DARK_THEME
}
