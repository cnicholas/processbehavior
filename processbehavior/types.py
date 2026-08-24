"""TypedDict definitions for the chart-info payload shape.

Every entry under ``AnalysisResult.charts`` is structurally a
:class:`ChartPayload` — a dict carrying the chart's data, statistics,
metadata, and (for stratified results) its strata list. Until this
module existed, the shape was an informal ``dict[str, Any]``; producer
sites in :mod:`processbehavior.analysis` and consumer sites in
:mod:`processbehavior.plotting` could disagree about keys silently. The
strata-list bug (commit ``f938fdf``) and the Xbar dead-branch bug
(commit ``1444b63``) were both producer/consumer disagreements about
this contract.

Typing the contract lets mypy catch the next drift at edit time.

Usage
-----
At a producer site::

    from processbehavior.types import ChartPayload, ChartMetadata

    def _calculate_my_chart(self, ...) -> dict[str, ChartPayload]:
        return {
            'Xbar': ChartPayload(
                data=df,
                statistics={'N': 30, 'center': 100.0, 'lpl': 95.0, 'upl': 105.0},
                metadata=ChartMetadata(chart_type='Xbar', value_col='y'),
            )
        }

At a consumer site::

    from processbehavior.types import ChartPayload

    def render(info: ChartPayload) -> Figure:
        df = info['data']               # mypy: pd.DataFrame
        stats = info['statistics']      # mypy: ChartStatistics | nested dict
        metadata = info['metadata']     # mypy: ChartMetadata
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

try:  # Python 3.11+
    from typing import NotRequired
except ImportError:  # Python 3.9-3.10
    from typing_extensions import NotRequired

if TYPE_CHECKING:
    import pandas as pd


class ChartStatistics(TypedDict):
    """Statistics for one chart (unstratified) or one stratum (stratified).

    Every control chart (X, mR, Xbar, S) and the Histogram emit this
    same shape from ``get_statistics()``. Histogram additionally carries
    :class:`HistogramExtras` keys.

    When the control limits and/or subgroup size vary across subgroups
    (e.g. Xbar/S with ``n_mode='actual'`` on unbalanced data),
    ``N``, ``lpl``, and ``upl`` are ``None`` and the optional
    ``limits_vary`` flag is ``True``. ``center`` remains a single scalar.
    """

    N: int | None
    center: float | None
    lpl: float | None
    upl: float | None
    limits_vary: NotRequired[bool]


class HistogramExtras(TypedDict, total=False):
    """Additional keys present in Histogram statistics dicts.

    ``mean`` is an alias for :attr:`ChartStatistics.center`; ``n`` is an
    alias for :attr:`ChartStatistics.N`. ``std`` is the sample standard
    deviation. These keys exist for historical / convenience reasons.
    """

    mean: float | None
    std: float | None
    n: int


class ChartMetadata(TypedDict, total=False):
    """Per-chart metadata keys.

    All keys are optional (``total=False``). Different chart variants
    populate different subsets:

    - ``chart_type`` / ``value_col`` / ``center_col``: every chart.
    - ``stratified`` / ``stratify_col`` / ``stratify_by``: stratified
      charts only.
    - ``lane_boundaries``: stratified X/mR charts with collapsed factors.
    - ``insufficient_strata``: mR/r charts that dropped first-per-stratum.
    - ``phased`` / ``single_point_phases``: charts run with
      ``phased=True``.
    - ``run_rules_applicable``: present on charts where Western Electric
      rules apply (or explicitly don't).
    - ``limits_source`` / ``calibration``: reserved for the standards-
      given charts feature (see issue #79).
    """

    chart_type: str
    value_col: str
    center_col: str
    stratified: bool
    stratify_col: str | None
    stratify_by: list[str]
    lane_boundaries: dict[str, list[float]] | None
    insufficient_strata: list[str] | None
    phased: bool
    single_point_phases: int
    run_rules_applicable: bool
    bins: int  # Histogram only
    focused_stratum: str  # After AnalysisResult.focus()
    stratum_display: str  # Pretty-printed stratum label
    residual_type: str  # 'R1'..'R6' for residual charts
    recentered: bool  # Recentered residual charts
    limits_source: str  # 'data' | 'calibration' (issue #79)
    calibration: dict | None  # Mean/sd payload (issue #79)


# Statistics can be flat (unstratified) or nested-by-stratum (stratified).
ChartStatisticsValue = ChartStatistics | dict[str, ChartStatistics]


class ChartPayload(TypedDict):
    """The dict structure stored in ``AnalysisResult.charts[name]``.

    Required: ``data``, ``statistics``, ``metadata``.
    Optional: ``strata`` (present on stratified charts).
    """

    data: pd.DataFrame
    statistics: ChartStatisticsValue
    metadata: ChartMetadata
    strata: NotRequired[list[str]]


# Type alias for the full charts dict on AnalysisResult.
Charts = dict[str, ChartPayload]


__all__ = [
    'ChartStatistics',
    'HistogramExtras',
    'ChartMetadata',
    'ChartStatisticsValue',
    'ChartPayload',
    'Charts',
]
