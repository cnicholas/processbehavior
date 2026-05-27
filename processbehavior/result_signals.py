"""Signal-detection orchestration for an `AnalysisResult`.

Wraps the `processbehavior.signals` machinery (`SignalDetector`,
`SignalConfig`, `RuleSet`) to produce SignalResults from an analysis
result's charts. Extracted from `AnalysisResult.detect_signals` so the
orchestration is unit-testable without instantiating the full result.

The class method on `AnalysisResult` remains as a thin delegate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .exceptions import ChartNotAvailableError, ProcessBehaviorError

if TYPE_CHECKING:
    from .analysis_result import AnalysisResult
    from .signals.result import SignalResult


# ---------------------------------------------------------------------------
# Public: detect_signals_for_result
# ---------------------------------------------------------------------------


def detect_signals_for_result(
    result: AnalysisResult,
    *,
    chart: str | None = None,
    rules: str | list[str] | Any | None = None,
    config: Any | None = None,
    **kwargs: Any,
) -> SignalResult | dict[str, SignalResult]:
    """Detect rule violations across one or all charts on `result`.

    Parameters mirror `AnalysisResult.detect_signals` exactly. See that
    docstring for the contract.

    Returns one `SignalResult` when `chart` is specified, or a dict
    keyed by chart name otherwise.
    """
    from .signals import SignalDetector

    config = _build_config(rules, config, kwargs)
    detector = SignalDetector()

    if chart:
        chart = result._resolve_chart_name(chart)
        if chart not in result.charts:
            raise ChartNotAvailableError(
                f"Chart '{chart}' not found.\nAvailable: {result.all_charts}",
                chart=chart,
                available=result.all_charts,
            )
        return _detect_for_chart(detector, result, chart, config)

    return {
        name: _detect_for_chart(detector, result, name, config)
        for name in result.charts
    }


# ---------------------------------------------------------------------------
# Public: chart-name → base-type helper
# ---------------------------------------------------------------------------


# Base chart types whose name prefix maps to the same chart_type label.
_KNOWN_CHART_TYPES = ("Xbar", "S", "X", "mR")


def extract_chart_type(chart_name: str) -> str:
    """Base chart type from a chart name.

    Handles stratified chart names like 'X_lane_1' → 'X'. Falls back to
    'Xbar' for anything unrecognised (matches the legacy behaviour).
    """
    for prefix in _KNOWN_CHART_TYPES:
        if chart_name.startswith(prefix):
            return prefix
    return "Xbar"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_config(rules: Any, config: Any | None, kwargs: dict[str, Any]) -> Any:
    """Compose the SignalConfig from the public-API arguments.

    Mirrors the legacy `AnalysisResult.detect_signals` precedence:
    - If `config` is provided, use it as-is (kwargs and rules ignored).
    - Otherwise instantiate a default SignalConfig and apply `rules`
      (str | list | RuleSet) then any matching attributes from kwargs.
    """
    from .signals import RuleSet, SignalConfig

    if config is not None:
        return config

    config = SignalConfig()

    if rules is not None:
        if isinstance(rules, RuleSet):
            config.enabled_rules = rules.get_rules()
        elif isinstance(rules, (str, list)):
            config.enabled_rules = rules

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def _detect_for_chart(
    detector: Any, result: AnalysisResult, chart_name: str, config: Any
) -> SignalResult:
    """Run signal detection on a single chart from `result`.

    Reads `value_col` and `chart_type` from the chart's metadata
    (required) and dispatches to the detector. Raises
    `ProcessBehaviorError` if the chart is missing metadata.
    """
    chart_info = result.charts[chart_name]

    metadata = chart_info.get("metadata")
    if metadata is None:
        raise ProcessBehaviorError(
            f"Chart '{chart_name}' missing metadata. "
            "This indicates a bug in chart calculation."
        )

    value_col = metadata["value_col"]
    chart_type = metadata.get("chart_type") or extract_chart_type(chart_name)

    return detector.detect(
        data=chart_info["data"],
        stats=chart_info["statistics"],
        config=config,
        value_col=value_col,
        chart_name=chart_name,
        chart_type=chart_type,
    )
