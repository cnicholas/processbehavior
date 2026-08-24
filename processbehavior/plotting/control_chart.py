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


def _translate_image_error(exc: Exception) -> Exception:
    """Turn a static-export failure into an error naming what to actually do.

    Three outcomes, because they need three different fixes:

    - kaleido missing -> ``ImportError`` telling you to install it.
    - kaleido present but no browser -> ``RuntimeError`` about Chrome. This
      used to be reported as "Image export requires kaleido" because the
      handler substring-matched "kaleido" in the message, sending people to
      reinstall a package they already had.
    - anything else -> returned unchanged; not every failure here is about
      the export backend.
    """
    text = str(exc).lower()
    origin = type(exc).__module__ or ''

    if isinstance(exc, ImportError) or 'install kaleido' in text or 'requires kaleido' in text:
        return ImportError(
            'Image export requires kaleido.\n'
            "Install with: pip install 'processbehavior[images]'\n"
            'Or use .save_html() for interactive HTML export'
        )

    if 'chrome' in text or 'chromium' in text or 'browser' in text or origin.startswith('kaleido'):
        return RuntimeError(
            f'Static image export needs a Chrome/Chromium browser, which kaleido could not find.\n'
            f'Install one with: plotly_get_chrome\n'
            f'Or use .save_html() for interactive HTML export.\n'
            f'Underlying error: {exc}'
        )

    return exc


class ControlChartFigure:
    """
    Wrapper around plotly Figure with domain-specific methods.

    This class extends Plotly figures with processbehavior-specific
    functionality while maintaining full access to Plotly's API: anything
    not defined here is delegated to the wrapped figure, so plotly methods
    and attributes work as they would on the figure itself.

    Examples
    --------
    >>> fig = result.plot()
    >>> fig.show()  # Display in browser/notebook
    >>> fig.save_html('report.html')  # Export to HTML
    >>> fig.save_image('chart.png')  # Export to image
    >>> fig.update_layout(title='Custom Title')  # Full Plotly API

    Plotly's own spellings work too, via delegation:

    >>> fig.write_html('report.html')
    >>> fig.update_traces(marker_size=4)
    >>> fig.add_hline(y=10)
    >>> fig.data  # the underlying traces

    The wrapped figure is also available directly as ``fig.figure``.
    """

    def __init__(self, plotly_fig: go.Figure, analysis_result: AnalysisResult):
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

    def save_html(self, filepath: str | Path, include_plotlyjs: bool = True, auto_open: bool = False) -> None:
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
            str(filepath), include_plotlyjs='cdn' if not include_plotlyjs else True, auto_open=auto_open
        )

        logger.info(f'✓ Saved interactive chart to: {filepath}')

    def save_image(
        self, filepath: str | Path, width: int | None = None, height: int | None = None, scale: float = 2.0
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
            self._fig.write_image(str(filepath), width=width, height=height, scale=scale)
            logger.info(f'✓ Saved static image to: {filepath}')
        except Exception as e:
            raise _translate_image_error(e) from e

    # Plotly's own spellings, so muscle memory works. write_image routes through
    # the same error translation as save_image rather than surfacing kaleido's
    # raw failure.
    def write_html(self, *args, **kwargs):
        """Alias for plotly's ``write_html`` (see also :meth:`save_html`)."""
        return self._fig.write_html(*args, **kwargs)

    def write_image(self, *args, **kwargs):
        """Alias for plotly's ``write_image``, with translated export errors."""
        try:
            return self._fig.write_image(*args, **kwargs)
        except Exception as e:
            raise _translate_image_error(e) from e

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

    def __getattr__(self, name: str):
        """Delegate anything we do not define to the wrapped plotly figure.

        Only called when normal lookup fails, so the methods defined above are
        never shadowed. Dunder names are refused outright: letting them fall
        through makes copy/pickle/IPython probing find bound methods of the
        *figure* and behave erratically. ``_fig`` itself is refused for the
        same reason — during unpickling ``__getattr__`` can run before the
        instance dict exists, and delegating would recurse forever.
        """
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        if name in ('_fig', '_result'):
            raise AttributeError(name)
        return getattr(self._fig, name)

    def __dir__(self):
        """Own attributes plus the wrapped figure's, so completion finds both."""
        return sorted(set(super().__dir__()) | set(dir(self._fig)))

    @property
    def figure(self) -> go.Figure:
        """Get underlying Plotly figure."""
        return self._fig

    def __repr__(self):
        return f'ControlChartFigure(charts={list(self._result.charts.keys())})'

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks and nbconvert."""
        return self._fig.to_html(full_html=False, include_plotlyjs='cdn')
