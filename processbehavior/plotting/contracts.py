"""Typed contracts for the plotting pipeline.

Defines the data structures that flow between the coordinator (plotter.py)
and the renderers (renderers.py). Validated at construction to catch
metadata-shape bugs before rendering starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..exceptions import ChartNotAvailableError, ProcessBehaviorError

if TYPE_CHECKING:
    import pandas as pd

    from .themes import ChartTheme

# Valid chart types that renderers accept
VALID_CHART_TYPES = frozenset({'Xbar', 'S', 'XmR', 'R', 'Histogram'})


class PlotError(ProcessBehaviorError):
    """Rendering-specific failure in the plotting pipeline.

    Raised when plotting encounters an issue with chart metadata,
    rendering state, or figure construction that prevents producing
    a valid chart.
    """


@dataclass(frozen=True)
class ChartRenderSpec:
    """Validated metadata contract between analysis outputs and renderers.

    Built from chart_info metadata and validated at construction.
    Prevents silent metadata-shape bugs from reaching the renderer.
    """

    chart_type: str
    value_col: str
    x_col: str | None
    center_key: str | None
    limits_vary: bool
    run_rules_applicable: bool
    lane_boundaries: list[dict[str, Any]] | None
    phased: bool

    def __post_init__(self) -> None:
        if self.chart_type not in VALID_CHART_TYPES:
            raise ChartNotAvailableError(
                f"Unknown chart type: '{self.chart_type}'.\n"
                f"Valid types: {sorted(VALID_CHART_TYPES)}",
                chart=self.chart_type,
                available=sorted(VALID_CHART_TYPES),
            )


@dataclass(frozen=True)
class RenderContext:
    """Everything a renderer needs to draw one chart panel.

    Built once by the coordinator, threaded through the rendering pipeline.
    Immutable to prevent accidental mutation between faceted panels.
    """

    spec: ChartRenderSpec
    data: pd.DataFrame
    stats: dict[str, Any]
    theme: ChartTheme
    result: Any  # AnalysisResult — avoid circular import
    chart_name: str
    highlight_signals: bool
    show_limits: bool
    show_limit_values: bool
    show_zones: bool
    show_rules: bool
    show_stats: bool
    is_faceted: bool
    marker_size: int
    line_width: float


def build_render_spec(
    chart_info: dict[str, Any],
    chart_name: str,
    x_col: str | None,
    center_key: str | None,
) -> ChartRenderSpec:
    """Build a validated ChartRenderSpec from raw chart_info.

    Parameters
    ----------
    chart_info : dict
        Chart info dict with 'data', 'statistics', and 'metadata' keys.
    chart_name : str
        Name of the chart (for error messages).
    x_col : str | None
        Resolved x-axis column (or None for index-based).
    center_key : str | None
        Key for the centerline statistic in stats dict.

    Returns
    -------
    ChartRenderSpec
        Validated render specification.

    Raises
    ------
    PlotError
        If required metadata is missing.
    """
    metadata = chart_info.get('metadata', {})
    stats = chart_info.get('statistics', {})

    if 'metadata' not in chart_info:
        raise PlotError(
            f"Chart '{chart_name}' missing metadata. "
            f"This indicates a bug in chart calculation. "
            f"All charts must have metadata with 'value_col'."
        )

    value_col = metadata.get('value_col')
    if not value_col:
        raise PlotError(
            f"Chart '{chart_name}' metadata missing 'value_col'. "
            f"This indicates a bug in chart calculation."
        )

    chart_type = metadata.get('chart_type', chart_name.split('_')[0])
    limits_vary = stats.get('upl') == 'Varies' or stats.get('lpl') == 'Varies'

    return ChartRenderSpec(
        chart_type=chart_type,
        value_col=value_col,
        x_col=x_col,
        center_key=center_key,
        limits_vary=limits_vary,
        run_rules_applicable=metadata.get('run_rules_applicable', True),
        lane_boundaries=metadata.get('lane_boundaries'),
        phased=metadata.get('phased', False),
    )


def build_render_context(
    spec: ChartRenderSpec,
    chart_info: dict[str, Any],
    chart_name: str,
    theme: ChartTheme,
    result: Any,
    *,
    highlight_signals: bool,
    show_limits: bool,
    show_limit_values: bool,
    show_zones: bool,
    show_rules: bool,
    show_stats: bool,
    is_faceted: bool,
) -> RenderContext:
    """Build a RenderContext from a spec and display options."""
    return RenderContext(
        spec=spec,
        data=chart_info['data'],
        stats=chart_info['statistics'],
        theme=theme,
        result=result,
        chart_name=chart_name,
        highlight_signals=highlight_signals,
        show_limits=show_limits,
        show_limit_values=show_limit_values,
        show_zones=show_zones,
        show_rules=show_rules,
        show_stats=show_stats,
        is_faceted=is_faceted,
        marker_size=theme.facet_marker_size if is_faceted else theme.data_marker_size,
        line_width=theme.facet_line_width if is_faceted else theme.data_line_width,
    )
