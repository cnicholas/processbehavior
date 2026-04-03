"""
Effects visualization charts for ProcessBehavior analysis.

This module provides chart functions for visualizing main effects and interactions:
- create_main_effects_chart: Horizontal bar chart of all main effects (combined)
- create_factor_effects_chart: Vertical bar chart of factor main effects only
- create_time_effects_chart: Horizontal bar chart of time effects only
- create_time_interaction_chart: Line plot of factor × time interaction
- create_factor_interaction_chart: Line chart for factor × factor interaction
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go

from processbehavior.exceptions import ChartNotAvailableError, ValidationError

if TYPE_CHECKING:
    from .themes import ChartTheme

logger = logging.getLogger(__name__)

_FACTOR_COLORS = [
    '#FF6B6B',  # Coral
    '#4ECDC4',  # Teal
    '#45B7D1',  # Sky blue
    '#96CEB4',  # Sage
]

_INTERACTION_COLORS = _FACTOR_COLORS + [
    '#FFEAA7',  # Soft yellow
    '#DDA0DD',  # Plum
]


def _is_main_effect_key(name: str) -> bool:
    """Check if an effects dict key represents a main effect.

    Convention: main effect keys end with '_MEs' suffix or are
    exactly 'main_effect'. All other keys are filtered out when
    building main effects charts.

    See: EffectsCalculator output key naming.
    """
    return name.endswith('_MEs') or name == 'main_effect'


def create_main_effects_chart(
    effects: Mapping[str, Any],
    theme: ChartTheme,
    factors: list[str] | None = None,
    width: int = 1000,
    height: int | None = None
) -> go.Figure:
    """
    Create horizontal bar chart showing all main effects.

    Displays main effect magnitudes for each factor level, with factors
    grouped together. Includes a vertical reference line at 0.

    Parameters
    ----------
    effects : dict
        Effects dictionary from result.effects containing factor main effects.
        Expected keys: factor names (e.g., 'Lane', 'Phase') mapping to DataFrames
        with columns [factor_name, 'Main_Effect']
    theme : ChartTheme
        Visual theme for styling
    factors : list of str, optional
        Specific factors to include. If None, includes all factors.
    width : int, default 1000
        Figure width in pixels
    height : int, optional
        Figure height in pixels. Auto-calculated if None.

    Returns
    -------
    go.Figure
        Plotly figure with horizontal bar chart

    Raises
    ------
    ValueError
        If no main effects are found in the ``effects`` dictionary.

    Examples
    --------
    >>> fig = create_main_effects_chart(result.effects, theme)
    >>> fig.show()
    """
    # Find factor effects (DataFrames with 'Main_Effect' column)
    factor_effects = []
    time_effects = None

    for name, data in effects.items():
        if not isinstance(data, pd.DataFrame):
            continue

        # Skip non-effect DataFrames (MEs scores, etc.)
        if _is_main_effect_key(name):
            continue

        if 'Main_Effect' in data.columns and (factors is None or name in factors):
            factor_effects.append((name, data))
        elif 'PT_ME' in data.columns:
            time_effects = (name, data)

    if not factor_effects and time_effects is None:
        raise ChartNotAvailableError(
            "No main effects found to plot.\n"
            f"Available effects keys: {list(effects.keys())}",
            chart='main_effects',
            available=list(effects.keys())
        )

    # Build combined data for horizontal bar chart
    all_labels = []
    all_values = []
    all_colors = []

    # Color palette for factors
    factor_colors = [theme.data_color, theme.center_color] + _FACTOR_COLORS

    for i, (factor_name, data) in enumerate(factor_effects):
        factor_col = data.columns[0]
        color = factor_colors[i % len(factor_colors)]

        for _, row in data.iterrows():
            label = f"{factor_name}: {row[factor_col]}"
            all_labels.append(label)
            all_values.append(row['Main_Effect'])
            all_colors.append(color)

    # Add time effects if present
    if time_effects is not None:
        name, data = time_effects
        time_col = data.columns[0]
        for _, row in data.iterrows():
            label = f"Time: {row[time_col]}"
            all_labels.append(label)
            all_values.append(row['PT_ME'])
            all_colors.append(theme.pattern_signal_color)

    # Calculate height if not specified
    if height is None:
        height = max(400, 25 * len(all_labels) + 100)

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=all_labels,
        x=all_values,
        orientation='h',
        marker_color=all_colors,
        text=[f'{v:.3f}' for v in all_values],
        textposition='outside',
        hovertemplate='%{y}<br>Effect: %{x:.4f}<extra></extra>'
    ))

    # Add vertical line at 0
    fig.add_vline(
        x=0,
        line_dash='dash',
        line_color='gray',
        line_width=1
    )

    fig.update_layout(
        title='Main Effects',
        xaxis_title='Effect Magnitude',
        yaxis_title='',
        width=width,
        height=height,
        yaxis=dict(autorange='reversed'),  # Top-to-bottom order
        showlegend=False
    )

    return fig


def create_factor_effects_chart(
    effects: Mapping[str, Any],
    theme: ChartTheme,
    factors: list[str] | None = None,
    width: int = 1000,
    height: int = 500
) -> go.Figure:
    """
    Create vertical bar chart showing factor main effects only.

    Displays main effect magnitudes for each factor level, with factors
    grouped together. Excludes time effects. Includes a horizontal
    reference line at 0.

    Parameters
    ----------
    effects : dict
        Effects dictionary from result.effects containing factor main effects.
        Expected keys: factor names (e.g., 'Lane', 'Phase') mapping to DataFrames
        with columns [factor_name, 'Main_Effect']
    theme : ChartTheme
        Visual theme for styling
    factors : list of str, optional
        Specific factors to include. If None, includes all factors.
    width : int, default 1000
        Figure width in pixels
    height : int, default 500
        Figure height in pixels

    Returns
    -------
    go.Figure
        Plotly figure with vertical bar chart

    Raises
    ------
    ValueError
        If no factor main effects are found in the ``effects`` dictionary.

    Examples
    --------
    >>> fig = create_factor_effects_chart(result.effects, theme)
    >>> fig.show()
    """
    # Find factor effects (DataFrames with 'Main_Effect' column)
    factor_effects = []

    for name, data in effects.items():
        if not isinstance(data, pd.DataFrame):
            continue

        # Skip non-effect DataFrames (MEs scores, etc.)
        if _is_main_effect_key(name):
            continue

        if 'Main_Effect' in data.columns and (factors is None or name in factors):
            factor_effects.append((name, data))

    if not factor_effects:
        raise ChartNotAvailableError(
            "No factor main effects found to plot.\n"
            f"Available effects keys: {list(effects.keys())}",
            chart='factor_effects',
            available=list(effects.keys())
        )

    # Build combined data for vertical bar chart
    all_labels = []
    all_values = []
    all_colors = []

    # Color palette for factors
    factor_colors = [theme.data_color, theme.center_color] + _FACTOR_COLORS

    for i, (factor_name, data) in enumerate(factor_effects):
        factor_col = data.columns[0]
        color = factor_colors[i % len(factor_colors)]

        for _, row in data.iterrows():
            label = f"{factor_name}: {row[factor_col]}"
            all_labels.append(label)
            all_values.append(row['Main_Effect'])
            all_colors.append(color)

    # Create vertical bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=all_labels,
        y=all_values,
        marker_color=all_colors,
        text=[f'{v:.3f}' for v in all_values],
        textposition='outside',
        hovertemplate='%{x}<br>Effect: %{y:.4f}<extra></extra>'
    ))

    # Add horizontal line at 0
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='gray',
        line_width=1
    )

    fig.update_layout(
        title='Factor Main Effects',
        xaxis_title='Factor Level',
        yaxis_title='Effect Magnitude',
        width=width,
        height=height,
        showlegend=False
    )

    return fig


def create_time_effects_chart(
    effects: Mapping[str, Any],
    theme: ChartTheme,
    width: int = 1000,
    height: int | None = None
) -> go.Figure:
    """
    Create horizontal bar chart showing time effects only.

    Displays time effect magnitudes for each time period. Excludes factor
    main effects. Includes a vertical reference line at 0.

    Parameters
    ----------
    effects : dict
        Effects dictionary from result.effects containing time effects.
        Expected key: time variable name mapping to DataFrame with columns
        [time_var, 'PT_ME']
    theme : ChartTheme
        Visual theme for styling
    width : int, default 1000
        Figure width in pixels
    height : int, optional
        Figure height in pixels. Auto-calculated if None.

    Returns
    -------
    go.Figure
        Plotly figure with horizontal bar chart

    Raises
    ------
    ValueError
        If no time effects (PT_ME — Period-Time Main Effect) are found
        in the ``effects`` dictionary.

    Examples
    --------
    >>> fig = create_time_effects_chart(result.effects, theme)
    >>> fig.show()
    """
    # Find time effects (DataFrame with 'PT_ME' column)
    time_effects = None

    for name, data in effects.items():
        if not isinstance(data, pd.DataFrame):
            continue

        if 'PT_ME' in data.columns:
            time_effects = (name, data)
            break

    if time_effects is None:
        raise ChartNotAvailableError(
            "No time effects found to plot.\n"
            "This requires a time variable in the analysis.\n"
            f"Available effects keys: {list(effects.keys())}",
            chart='time_effects',
            available=list(effects.keys())
        )

    # Build data for horizontal bar chart
    all_labels = []
    all_values = []

    name, data = time_effects
    time_col = data.columns[0]
    for _, row in data.iterrows():
        label = f"Time: {row[time_col]}"
        all_labels.append(label)
        all_values.append(row['PT_ME'])

    # Calculate height if not specified
    if height is None:
        height = max(400, 25 * len(all_labels) + 100)

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=all_labels,
        x=all_values,
        orientation='h',
        marker_color=theme.pattern_signal_color,
        text=[f'{v:.3f}' for v in all_values],
        textposition='outside',
        hovertemplate='%{y}<br>Effect: %{x:.4f}<extra></extra>'
    ))

    # Add vertical line at 0
    fig.add_vline(
        x=0,
        line_dash='dash',
        line_color='gray',
        line_width=1
    )

    fig.update_layout(
        title='Time Effects',
        xaxis_title='Effect Magnitude',
        yaxis_title='',
        width=width,
        height=height,
        yaxis=dict(autorange='reversed'),  # Top-to-bottom order
        showlegend=False
    )

    return fig


def create_time_interaction_chart(
    interactions: Mapping[str, Any],
    effects: Mapping[str, Any],
    factors: list[str],
    time_var: str,
    dataset: pd.DataFrame,
    theme: ChartTheme,
    width: int = 1000,
    height: int = 500
) -> go.Figure:
    """
    Create line plot showing factor × time interaction.

    Displays one line per factor level combination, with x-axis as time
    and y-axis as the interaction effect. Non-parallel lines indicate
    interaction between factors and time.

    Parameters
    ----------
    interactions : dict
        Interactions dictionary from result.interactions.
        Expected key: 'factor_time' mapping to Series of PDC values.
    effects : dict
        Effects dictionary from result.effects.
    factors : list of str
        Factor variable names
    time_var : str
        Time variable name
    dataset : pd.DataFrame
        Full analysis dataset with factor and time columns
    theme : ChartTheme
        Visual theme for styling
    width : int, default 1000
        Figure width in pixels
    height : int, default 500
        Figure height in pixels

    Returns
    -------
    go.Figure
        Plotly figure with line plot

    Raises
    ------
    ValueError
        If ``'factor_time'`` key is missing from ``interactions``.
    ValueError
        If the PDC (Predictable Difference of Cell means) series length
        does not match the dataset length.

    Examples
    --------
    >>> fig = create_time_interaction_chart(
    ...     result.interactions, result.effects,
    ...     factors=['Lane'], time_var='Pull',
    ...     dataset=result.dataset, theme=theme
    ... )
    >>> fig.show()
    """
    if 'factor_time' not in interactions:
        raise ChartNotAvailableError(
            "Factor × time interaction not available.\n"
            "This requires both factors and time variable in the analysis.",
            chart='factor_time_interaction',
            available=list(interactions.keys())
        )

    pdc = interactions['factor_time']

    # Get unique factor levels and time points from dataset
    # Build aggregated data: mean PDC per (factor_combo, time)
    agg_cols = list(factors) + [time_var]

    # Add PDC to dataset for aggregation
    df = dataset.copy()
    if len(pdc) != len(df):
        raise ValueError(
            "Interaction chart requires PDC aligned to plotted data: "
            f"len(pdc)={len(pdc)} != len(df)={len(df)}"
        )
    df['_pdc'] = pdc.values

    # Aggregate to cell level
    agg_data = df.groupby(agg_cols, observed=True)['_pdc'].mean().reset_index()

    # Create combined factor key for grouping lines
    if len(factors) == 1:
        agg_data['_factor_key'] = agg_data[factors[0]].astype(str)
    else:
        agg_data['_factor_key'] = agg_data[list(factors)].apply(
            lambda x: '_'.join(str(v) for v in x), axis=1
        )

    # Sort by time
    agg_data = agg_data.sort_values(time_var)

    # Create figure
    fig = go.Figure()

    # Color palette for factor levels
    colors = [theme.data_color, theme.center_color] + _INTERACTION_COLORS

    factor_keys = agg_data['_factor_key'].unique()

    for i, factor_key in enumerate(factor_keys):
        mask = agg_data['_factor_key'] == factor_key
        subset = agg_data[mask]

        fig.add_trace(go.Scatter(
            x=subset[time_var],
            y=subset['_pdc'],
            mode='lines+markers',
            name=str(factor_key),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8, color=colors[i % len(colors)]),
            hovertemplate=f'{factor_key}<br>Time: %{{x}}<br>Effect: %{{y:.4f}}<extra></extra>'
        ))

    # Add horizontal line at 0
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='gray',
        line_width=1
    )

    factor_label = ', '.join(factors)
    fig.update_layout(
        title=f'Factor × Time Interaction ({factor_label})',
        xaxis_title=time_var,
        yaxis_title='Interaction Effect (PDC)',
        width=width,
        height=height,
        legend_title=factor_label,
        hovermode='closest'
    )

    return fig


def create_factor_interaction_chart(
    interactions: Mapping[str, Any],
    factors: list[str],
    theme: ChartTheme,
    width: int = 800,
    height: int = 600
) -> go.Figure:
    """
    Create line chart for factor × factor interaction.

    Displays one line per factor2 level, with x-axis as factor1 levels
    and y-axis as the interaction effect (Rx). Non-parallel lines indicate
    interaction between the two factors.

    Parameters
    ----------
    interactions : dict
        Interactions dictionary from result.interactions.
        Expected key: 'factor_factor' mapping to DataFrame with
        columns [factor1, factor2, 'Rx']
    factors : list of str
        Factor variable names (at least 2)
    theme : ChartTheme
        Visual theme for styling
    width : int, default 800
        Figure width in pixels
    height : int, default 600
        Figure height in pixels

    Returns
    -------
    go.Figure
        Plotly figure with line chart

    Raises
    ------
    ValueError
        If ``'factor_factor'`` key is missing from ``interactions``.
    ValueError
        If fewer than 2 factors are provided.
    ValueError
        If expected factor columns are not found in the interaction data.

    Examples
    --------
    >>> fig = create_factor_interaction_chart(
    ...     result.interactions,
    ...     factors=['Lane', 'Phase'],
    ...     theme=theme
    ... )
    >>> fig.show()
    """
    if 'factor_factor' not in interactions:
        raise ChartNotAvailableError(
            "Factor × factor interaction not available.\n"
            "This requires at least 2 factors in the analysis.",
            chart='factor_factor_interaction',
            available=list(interactions.keys())
        )

    if len(factors) < 2:
        raise ValidationError(
            f"Factor interaction requires at least 2 factors, got {len(factors)}."
        )

    fi = interactions['factor_factor']
    factor1, factor2 = factors[0], factors[1]

    if factor1 not in fi.columns or factor2 not in fi.columns:
        raise ValueError(
            f"Expected columns {factor1}, {factor2} in interaction data.\n"
            f"Found: {list(fi.columns)}"
        )

    fig = go.Figure()

    # Get unique levels, sorted for consistent ordering
    levels1 = sorted(fi[factor1].unique(), key=str)
    levels2 = sorted(fi[factor2].unique(), key=str)

    # Color palette for factor2 levels
    colors = [theme.data_color, theme.center_color] + _INTERACTION_COLORS

    # Create one line per factor2 level
    for i, level2 in enumerate(levels2):
        mask = fi[factor2] == level2
        subset = fi[mask].set_index(factor1)

        x_values = [str(lv) for lv in levels1]
        # Missing factor combinations are filled with 0: mathematically correct
        # for interaction residuals (Rx), since absence = no interaction effect.
        y_values = [subset.loc[lv, 'Rx'] if lv in subset.index else 0 for lv in levels1]

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=str(level2),
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8, color=colors[i % len(colors)]),
            hovertemplate=(
                f'{factor1}: %{{x}}<br>'
                f'{factor2}: {level2}<br>'
                'Interaction: %{y:.4f}<extra></extra>'
            )
        ))

    # Add horizontal reference line at 0
    fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)

    fig.update_layout(
        title=f'Factor Interaction: {factor1} × {factor2}',
        xaxis_title=factor1,
        yaxis_title='Interaction Effect (Rx)',
        width=width,
        height=height,
        legend_title=factor2,
        hovermode='closest'
    )

    return fig
