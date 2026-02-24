"""Run rules visualization for control charts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from processbehavior.exceptions import ProcessBehaviorError

if TYPE_CHECKING:
    from processbehavior.analysis_result import AnalysisResult
    from .themes import ChartTheme

logger = logging.getLogger(__name__)

_RULE_SHORT_NAMES = {
    'rule_2': '2 of 3 in Zone A',
    'rule_3': '4 of 5 in Zone B+',
    'rule_4': '8+ same side',
    'rule_5': 'Trend',
    'rule_6': 'Oscillation',
    'rule_7': 'In Zone C',
    'rule_8': 'Avoiding center',
}


def add_run_rules_visualization(
    fig: go.Figure,
    data: pd.DataFrame,
    stats: dict,
    chart_name: str,
    value_col: str,
    x_col: str,
    theme: ChartTheme,
    result: "AnalysisResult",
    row: int | None = None,
    col: int | None = None
) -> go.Figure:
    """
    Add visualization for Western Electric run rules (Rules 2-8).

    Detects rule violations and adds annotations and markers to highlight
    the specific rules that were violated at each observation.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add visualizations to
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics
    chart_name : str
        Name of the chart being plotted
    value_col : str
        Name of the value column
    x_col : str
        Name of the x-axis column
    theme : ChartTheme
        Theme with rule colors and styling
    result : AnalysisResult
        Analysis result for signal detection
    row : int, optional
        Row number for faceted plots (1-indexed)
    col : int, optional
        Column number for faceted plots (1-indexed)
    """
    try:
        signal_result = result.detect_signals(chart=chart_name)

        if not signal_result.has_signals:
            return fig

        violations = signal_result.violations

        # Skip Rule 1 (already handled by highlight_signals)
        violations = violations[violations['rule_name'] != 'rule_1']

        if violations.empty:
            return fig

        grouped = violations.groupby('obs_id', observed=True)

        annotated_points = []

        for obs_id, obs_violations in grouped:
            if obs_id in data.index:
                obs_data = data.loc[obs_id]
                if isinstance(obs_data, pd.DataFrame):
                    logger.debug("Skipping obs_id=%s: MultiIndex expanded to DataFrame", obs_id)
                    continue
            else:
                continue

            x_val = obs_data[x_col] if x_col in data.columns else obs_id
            y_val = obs_data[value_col]

            rules = obs_violations['rule_name'].unique().tolist()
            rule_nums = [r.split('_')[1] if '_' in r else r for r in rules]

            annotated_points.append({
                'x': x_val,
                'y': y_val,
                'rules': rules,
                'rule_nums': rule_nums,
                'hover': '<br>'.join([
                    _RULE_SHORT_NAMES.get(r, r) for r in rules
                ])
            })

        if not annotated_points:
            return fig

        x_vals = [p['x'] for p in annotated_points]
        y_vals = [p['y'] for p in annotated_points]
        hover_texts = [
            f"Rule violations:<br>{p['hover']}<br>Value: {p['y']:.3f}"
            for p in annotated_points
        ]

        scatter_kwargs = dict(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name='Pattern Signals',
            marker=dict(
                size=theme.data_marker_size,
                color=theme.pattern_signal_color,
                symbol='circle',
                line=dict(width=1, color='darkorange')
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False
        )

        if row is not None and col is not None:
            scatter_kwargs['showlegend'] = False
            fig.add_trace(go.Scatter(**scatter_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**scatter_kwargs))

        return fig
    except (AttributeError, KeyError, ValueError, ProcessBehaviorError) as e:
        logger.info("Skipping run-rule annotations: %s", e)
        return fig
    except Exception:
        logger.exception("Run-rule annotation failed unexpectedly")
        raise
