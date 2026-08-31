"""
Plotting framework for ProcessBehavior.

Provides an intuitive, extensible API for creating control charts
with built-in support for faceting, interactivity, and export.
"""

from __future__ import annotations

from .capability_chart import create_capability_chart
from .contracts import PlotError
from .control_chart import ControlChartFigure
from .effects_charts import (
    create_factor_effects_chart,
    create_factor_interaction_chart,
    create_main_effects_chart,
    create_time_effects_chart,
    create_time_interaction_chart,
)

# Redundant alias marks this as a deliberate re-export that is intentionally
# absent from __all__ — the convention both ruff and type checkers understand.
from .plotter import Plotter as Plotter
from .themes import ChartTheme, get_theme, list_themes, register_theme

__all__ = [
    'PlotError',
    'ControlChartFigure',
    'ChartTheme',
    'get_theme',
    'list_themes',
    'register_theme',
    'create_capability_chart',
    'create_main_effects_chart',
    'create_factor_effects_chart',
    'create_time_effects_chart',
    'create_time_interaction_chart',
    'create_factor_interaction_chart',
]

# `Plotter` is imported above and stays importable for library internals
# (analysis_result, excel_exporter) and for anyone who reaches for it
# deliberately. It is deliberately absent from __all__: it was already labelled
# internal, and listing an internal class in the star-import surface is how it
# becomes a semver commitment by accident.

