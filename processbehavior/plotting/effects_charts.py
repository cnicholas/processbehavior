"""
Effects visualization charts for ProcessBehavior analysis.

This module provides chart functions for visualizing main effects and interactions:
- create_main_effects_chart: Horizontal bar chart of all main effects
- create_time_interaction_chart: Line plot of factor × time interaction
- create_factor_interaction_chart: Heatmap/grouped bar for factor × factor interaction
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

if TYPE_CHECKING:
    from .themes import ChartTheme

logger = logging.getLogger(__name__)


def create_main_effects_chart(
    effects: dict,
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
        if '_MEs' in name or name in ['main_effect']:
            continue

        if 'Main_Effect' in data.columns:
            if factors is None or name in factors:
                factor_effects.append((name, data))
        elif 'PT_ME' in data.columns:
            time_effects = (name, data)

    if not factor_effects and time_effects is None:
        raise ValueError(
            "No main effects found to plot.\n"
            f"Available effects keys: {list(effects.keys())}"
        )

    # Build combined data for horizontal bar chart
    all_labels = []
    all_values = []
    all_colors = []

    # Color palette for factors
    factor_colors = [
        theme.data_color,
        theme.center_color,
        '#FF6B6B',  # Coral
        '#4ECDC4',  # Teal
        '#45B7D1',  # Sky blue
        '#96CEB4',  # Sage
    ]

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


def create_time_interaction_chart(
    interactions: dict,
    effects: dict,
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
        raise ValueError(
            "Factor × time interaction not available.\n"
            "This requires both factors and time variable in the analysis."
        )

    pdc = interactions['factor_time']

    # Get unique factor levels and time points from dataset
    # Build aggregated data: mean PDC per (factor_combo, time)
    agg_cols = factors + [time_var]

    # Add PDC to dataset for aggregation
    df = dataset.copy()
    if len(pdc) == len(df):
        df['_pdc'] = pdc.values
    else:
        # PDC might be cell-level, not row-level
        # Fall back to showing factor means over time
        logger.warning("PDC length mismatch - using factor means")
        if 'R3' in df.columns:
            df['_pdc'] = df['R3']
        else:
            raise ValueError("Cannot create interaction chart - R3 not in dataset")

    # Aggregate to cell level
    agg_data = df.groupby(agg_cols, observed=True)['_pdc'].mean().reset_index()

    # Create combined factor key for grouping lines
    if len(factors) == 1:
        agg_data['_factor_key'] = agg_data[factors[0]].astype(str)
    else:
        agg_data['_factor_key'] = agg_data[factors].apply(
            lambda x: '_'.join(str(v) for v in x), axis=1
        )

    # Sort by time
    agg_data = agg_data.sort_values(time_var)

    # Create figure
    fig = go.Figure()

    # Color palette for factor levels
    colors = [
        theme.data_color,
        theme.center_color,
        '#FF6B6B',
        '#4ECDC4',
        '#45B7D1',
        '#96CEB4',
        '#FFEAA7',
        '#DDA0DD',
    ]

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
    interactions: dict,
    factors: list[str],
    theme: ChartTheme,
    chart_type: str = 'heatmap',
    width: int = 800,
    height: int = 600
) -> go.Figure:
    """
    Create visualization for factor × factor interaction.

    Can display as either a heatmap or grouped bar chart.

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
    chart_type : str, default 'heatmap'
        Visualization type: 'heatmap' or 'bar'
    width : int, default 800
        Figure width in pixels
    height : int, default 600
        Figure height in pixels

    Returns
    -------
    go.Figure
        Plotly figure with heatmap or grouped bar chart

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
        raise ValueError(
            "Factor × factor interaction not available.\n"
            "This requires at least 2 factors in the analysis."
        )

    if len(factors) < 2:
        raise ValueError(
            f"Factor interaction requires at least 2 factors, got {len(factors)}."
        )

    fi = interactions['factor_factor']
    factor1, factor2 = factors[0], factors[1]

    if factor1 not in fi.columns or factor2 not in fi.columns:
        raise ValueError(
            f"Expected columns {factor1}, {factor2} in interaction data.\n"
            f"Found: {list(fi.columns)}"
        )

    if chart_type == 'heatmap':
        # Pivot to matrix form for heatmap
        pivot = fi.pivot(index=factor1, columns=factor2, values='Rx')

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale='RdBu_r',  # Diverging: red for negative, blue for positive
            zmid=0,  # Center colorscale at 0
            text=[[f'{v:.3f}' for v in row] for row in pivot.values],
            texttemplate='%{text}',
            textfont={'size': 10},
            hovertemplate=(
                f'{factor1}: %{{y}}<br>'
                f'{factor2}: %{{x}}<br>'
                'Interaction: %{z:.4f}<extra></extra>'
            )
        ))

        fig.update_layout(
            title=f'Factor Interaction: {factor1} × {factor2}',
            xaxis_title=factor2,
            yaxis_title=factor1,
            width=width,
            height=height
        )

    elif chart_type == 'bar':
        # Grouped bar chart
        fig = go.Figure()

        # Get unique levels
        levels1 = fi[factor1].unique()
        levels2 = fi[factor2].unique()

        # Color palette
        colors = [
            theme.data_color,
            theme.center_color,
            '#FF6B6B',
            '#4ECDC4',
            '#45B7D1',
            '#96CEB4',
        ]

        for i, level2 in enumerate(levels2):
            mask = fi[factor2] == level2
            subset = fi[mask].set_index(factor1)

            fig.add_trace(go.Bar(
                x=[str(lv) for lv in levels1],
                y=[subset.loc[lv, 'Rx'] if lv in subset.index else 0 for lv in levels1],
                name=str(level2),
                marker_color=colors[i % len(colors)],
                hovertemplate=(
                    f'{factor1}: %{{x}}<br>'
                    f'{factor2}: {level2}<br>'
                    'Interaction: %{y:.4f}<extra></extra>'
                )
            ))

        fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1)

        fig.update_layout(
            title=f'Factor Interaction: {factor1} × {factor2}',
            xaxis_title=factor1,
            yaxis_title='Interaction Effect (Rx)',
            barmode='group',
            width=width,
            height=height,
            legend_title=factor2
        )

    else:
        raise ValueError(
            f"Invalid chart_type: '{chart_type}'.\n"
            "Options: 'heatmap', 'bar'"
        )

    return fig
