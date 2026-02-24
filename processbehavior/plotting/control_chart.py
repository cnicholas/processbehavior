"""
Custom control chart figure wrapper.

Extends Plotly figures with domain-specific methods for ProcessBehavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from processbehavior.analysis_result import AnalysisResult

logger = logging.getLogger(__name__)


class ControlChartFigure:
    """
    Wrapper around plotly Figure with domain-specific methods.

    This class extends Plotly figures with processbehavior-specific
    functionality while maintaining full access to Plotly's API.

    Examples
    --------
    >>> fig = result.plot()
    >>> fig.show()  # Display in browser/notebook
    >>> fig.save_html('report.html')  # Export to HTML
    >>> fig.save_image('chart.png')  # Export to image
    >>> fig.update_layout(title='Custom Title')  # Full Plotly API
    """

    def __init__(self, plotly_fig: go.Figure, analysis_result: "AnalysisResult"):
        """
        Wrap a Plotly figure with enhanced functionality.

        Parameters
        ----------
        plotly_fig : go.Figure
            Underlying Plotly figure
        analysis_result : AnalysisResult
            Source analysis result for metadata
        """
        self._fig = plotly_fig
        self._result = analysis_result

    def show(self) -> None:
        """Display figure in browser or notebook."""
        self._fig.show()

    def save_html(
        self,
        filepath: str | Path,
        include_plotlyjs: bool = True,
        auto_open: bool = False
    ) -> None:
        """
        Save as standalone HTML file.

        Parameters
        ----------
        filepath : str or Path
            Output file path
        include_plotlyjs : bool, default True
            Whether to include plotly.js (makes file larger but standalone)
        auto_open : bool, default False
            Whether to open in browser after saving
        """
        filepath = Path(filepath)

        self._fig.write_html(
            str(filepath),
            include_plotlyjs='cdn' if not include_plotlyjs else True,
            auto_open=auto_open
        )

        logger.info(f"✓ Saved interactive chart to: {filepath}")

    def save_image(
        self,
        filepath: str | Path,
        width: int | None = None,
        height: int | None = None,
        scale: float = 2.0
    ) -> None:
        """
        Save as static image (requires kaleido).

        Parameters
        ----------
        filepath : str or Path
            Output file path (.png, .jpg, .svg, .pdf)
        width : int, optional
            Image width in pixels
        height : int, optional
            Image height in pixels
        scale : float, default 2.0
            Scale factor for resolution
        """
        try:
            self._fig.write_image(
                str(filepath),
                width=width,
                height=height,
                scale=scale
            )
            logger.info(f"✓ Saved static image to: {filepath}")
        except Exception as e:
            if 'kaleido' in str(e).lower():
                raise ImportError(
                    "Image export requires kaleido.\n"
                    "Install with: pip install kaleido\n"
                    "Or use .save_html() for interactive HTML export"
                ) from e
            raise

    def add_annotation(self, text: str, x, y, **kwargs) -> ControlChartFigure:
        """Add text annotation to the figure.

        Returns self for method chaining:
        ``fig.add_annotation(...).update_layout(...)``
        """
        self._fig.add_annotation(text=text, x=x, y=y, **kwargs)
        return self

    def update_layout(self, **kwargs) -> ControlChartFigure:
        """Update figure layout (full Plotly API).

        Returns self for method chaining:
        ``fig.update_layout(title='X').update_layout(width=800)``
        """
        self._fig.update_layout(**kwargs)
        return self

    @property
    def figure(self) -> go.Figure:
        """Get underlying Plotly figure."""
        return self._fig

    def __repr__(self):
        return f"ControlChartFigure(charts={list(self._result.charts.keys())})"
