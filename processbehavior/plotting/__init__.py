"""
Plotting framework for ProcessBehavior.

Provides an intuitive, extensible API for creating control charts
with built-in support for faceting, interactivity, and export.
"""

from __future__ import annotations

from .control_chart import ControlChartFigure
from .plotter import Plotter

__all__ = ['Plotter', 'ControlChartFigure']
