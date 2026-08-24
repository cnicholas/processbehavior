"""Stratified-result drill-down: `focus_on(result, stratum)` + `FocusedAnalysisResult`.

Extracted from `AnalysisResult.focus()` and the inline `FocusedAnalysisResult`
subclass so the stratification subsystem lives in one place. The public API
on `AnalysisResult.focus(stratum)` is preserved as a thin delegate to
`focus_on(...)` here.

`FocusedAnalysisResult` is re-exported from `analysis_result` for
backward compat with `from processbehavior.analysis_result import
FocusedAnalysisResult` (any pickled instances continue to resolve too).
"""

from __future__ import annotations

from typing import Any

# Import AnalysisResult at runtime (not just TYPE_CHECKING) because we
# subclass it. analysis_result.py uses lazy imports inside methods to
# avoid the circular path.
from .analysis_result import AnalysisResult
from .data_preparation import encode_rsg
from .exceptions import ProcessBehaviorError, ValidationError

# ---------------------------------------------------------------------------
# Public: focus_on
# ---------------------------------------------------------------------------


def focus_on(result: AnalysisResult, stratum: str) -> AnalysisResult:
    """Return a new `FocusedAnalysisResult` restricted to `stratum`.

    Parameters mirror `AnalysisResult.focus` exactly. See that docstring
    for the contract.

    Raises
    ------
    ValidationError
        If `result` is not stratified, the stratum is missing, or the
        focused data ends up empty (encoding mismatch).
    ProcessBehaviorError
        If the chart lacks both stratify_col metadata and an rsg column.
    """
    if not result.strata:
        raise ValidationError(
            "Cannot focus: this result is not stratified. "
            "Use result.strata to check available subgroups."
        )
    if stratum not in result.strata:
        raise ValidationError(
            f"Stratum '{stratum}' not found. Available strata: {result.strata}"
        )

    focused_charts: dict[str, dict[str, Any]] = {}
    for chart_name, chart_info in result.charts.items():
        if not chart_info.get("strata"):
            # Non-stratified chart in a stratified result — keep verbatim.
            focused_charts[chart_name] = chart_info.copy()
            continue

        focused_charts[chart_name] = _build_focused_chart_info(
            chart_name, chart_info, stratum
        )

    return FocusedAnalysisResult(
        charts=focused_charts, original_result=result, focused_stratum=stratum
    )


# ---------------------------------------------------------------------------
# Internal helpers (per-chart focusing)
# ---------------------------------------------------------------------------


def _build_focused_chart_info(
    chart_name: str, chart_info: dict[str, Any], stratum: str
) -> dict[str, Any]:
    """Filter one chart's data + statistics + metadata to a single stratum."""
    data = chart_info["data"]
    parent_metadata = chart_info.get("metadata", {})

    mask = _stratum_mask(chart_name, data, parent_metadata, stratum)

    # Reset_index so focused row positions are 0..n-1. The renderer falls
    # back to `data.index` for trace x when there's no explicit x_col
    # (integer-position axis); without the reset, that index inherits
    # the unfiltered data's range and the tick positions (which use iloc
    # 0..n-1) land left of the rendered data. Same hazard
    # _get_stratified_charts already handles.
    focused_data = data[mask].copy().reset_index(drop=True)

    if focused_data.empty:
        raise ValidationError(
            f"No data found for stratum '{stratum}' in chart '{chart_name}'. "
            "This may indicate an encoding mismatch between strata keys and rsg values."
        )

    focused_stats = _extract_stratum_statistics(chart_name, chart_info, stratum)
    focused_lb = _extract_stratum_lane_boundaries(parent_metadata, stratum)

    return {
        "data": focused_data,
        "statistics": focused_stats,
        "metadata": {
            **parent_metadata,
            "stratified": False,
            "focused_stratum": stratum,
            "lane_boundaries": focused_lb,
        },
    }


def _stratum_mask(chart_name: str, data, parent_metadata: dict[str, Any], stratum: str):
    """Boolean mask selecting rows of `data` that belong to `stratum`."""
    stratify_col = parent_metadata.get("stratify_col")
    if stratify_col and stratify_col in data.columns:
        return data[stratify_col] == stratum

    # Legacy fallback: locate an rsg-like column.
    rsg_col = next(
        (
            col for col in data.columns
            if col in ("rsg", "RSG") or "rsg" in col.lower()
        ),
        None,
    )
    if rsg_col is None:
        raise ProcessBehaviorError(
            f"Cannot focus stratified chart '{chart_name}': "
            "missing stratification column metadata and no rsg column found. "
            "This indicates a bug in chart construction."
        )
    return data[rsg_col].astype(str) == encode_rsg(stratum)


def _extract_stratum_statistics(
    chart_name: str, chart_info: dict[str, Any], stratum: str
) -> Any:
    """Pull this stratum's stats dict from the chart's nested statistics.

    The dict can be keyed by either the raw stratum label or its rsg-encoded
    form. Raises if the chart is stratified but neither key matches.
    """
    nested = chart_info.get("statistics", {})
    if isinstance(nested, dict):
        if stratum in nested:
            return nested[stratum]
        encoded = encode_rsg(stratum)
        if encoded in nested:
            return nested[encoded]
        if chart_info.get("strata"):
            raise ProcessBehaviorError(
                f"Statistics key mismatch for stratum '{stratum}' in chart "
                f"'{chart_name}'. Available keys: {list(nested.keys())}"
            )
    return nested  # flat stats dict — OK


def _extract_stratum_lane_boundaries(
    parent_metadata: dict[str, Any], stratum: str
) -> Any:
    """Unpack the parent's per-stratum `lane_boundaries` to this stratum's list.

    Parent shape can be:
    - dict {stratum: [boundary_dicts]} — extract this stratum's list
    - list — already flat, pass through
    - None / anything else — return None

    Without this unpack, the plotter silently consumes the dict via
    `next(iter(...))` and uses the first stratum's positions on every
    focused chart. See processbehavior/plotting/x_axis_layout.py.
    """
    raw_lb = parent_metadata.get("lane_boundaries")
    if isinstance(raw_lb, dict):
        focused = raw_lb.get(stratum) or raw_lb.get(encode_rsg(stratum))
        if focused is None and stratum in raw_lb:
            focused = raw_lb[stratum]
        return focused
    if isinstance(raw_lb, list):
        return raw_lb
    return None


# ---------------------------------------------------------------------------
# FocusedAnalysisResult class
# ---------------------------------------------------------------------------


class FocusedAnalysisResult(AnalysisResult):
    """
    Lightweight AnalysisResult for focused (single-stratum) analysis.

    Returned by `AnalysisResult.focus()`. Same public interface as
    `AnalysisResult` but without requiring its own AnalysisDataSet.

    Parameters
    ----------
    charts : dict
        Chart data in standard format
    original_result : AnalysisResult
        The parent result this was focused from
    focused_stratum : str
        The stratum this result is focused on
    """

    def __init__(
        self,
        charts: dict[str, dict[str, Any]],
        original_result: AnalysisResult,
        focused_stratum: str,
    ):
        self.charts = charts
        self._original = original_result
        self._focused_stratum = focused_stratum
        self._ads = original_result._ads
        self._analysis_type = original_result._analysis_type

        parent_dataset = original_result.dataset
        stratum_id = encode_rsg(focused_stratum)

        # Filter by the column the charts were actually stratified on. The
        # rsg column is a different vocabulary whenever `by=` is a subset of
        # the factors — with factors=['machine','op'] and by=['machine'], rsg
        # holds 'M1_A' while the stratum is 'M1', so an rsg comparison matched
        # nothing and the focused result reported zero observations.
        stratify_col = self._parent_stratify_col(original_result)
        if stratify_col and stratify_col in parent_dataset.columns:
            mask = parent_dataset[stratify_col].astype(str) == str(focused_stratum)
            self._dataset = parent_dataset[mask].copy()
            rsg_col = stratify_col
        else:
            rsg_col = next(
                (
                    col for col in parent_dataset.columns
                    if col in ("rsg", "RSG") or "rsg" in col.lower()
                ),
                None,
            )
            if rsg_col:
                mask = parent_dataset[rsg_col].astype(str) == stratum_id
                self._dataset = parent_dataset[mask].copy()
            else:
                self._dataset = parent_dataset.copy()

        self.observed_sds = original_result.observed_sds
        self.analytical_sds = original_result.analytical_sds
        self.observed_sds_info = original_result.observed_sds_info.copy()
        self.analytical_sds_info = original_result.analytical_sds_info.copy()

        # Residuals — filter by rsg if possible, else align by index.
        self._residuals = None
        if original_result._residuals is not None:
            if rsg_col and rsg_col in original_result._residuals.columns:
                mask = original_result._residuals[rsg_col].astype(str) == stratum_id
                self._residuals = original_result._residuals[mask].copy()
            else:
                self._residuals = original_result._residuals.loc[self._dataset.index].copy()

        self._effects = original_result._effects
        self._interactions = original_result._interactions
        self._original_summary = original_result._summary.copy()
        self._summary = self._build_focused_summary()

    @staticmethod
    def _parent_stratify_col(original_result: AnalysisResult) -> str | None:
        """The column the parent's charts were stratified on, if any.

        Same source `focus_on` uses to slice chart rows, so the dataset and the
        charts always agree about what a stratum is.
        """
        for chart_info in original_result.charts.values():
            stratify_col = (chart_info.get("metadata") or {}).get("stratify_col")
            if stratify_col:
                return str(stratify_col)
        return None

    def _build_focused_summary(self) -> dict:
        """Summary for the focused result — n_signals re-counted, strata flag off."""
        n_signals = 0
        for chart_info in self.charts.values():
            data = chart_info.get("data")
            if data is not None and "beyond_limits" in data.columns:
                n_signals += int((data["beyond_limits"] != 0).sum())

        summary = self._original_summary.copy()
        summary.update(
            {
                "n_observations": len(self._dataset),
                "n_charts": len(self.charts),
                "chart_types": list(self.charts.keys()),
                "is_stratified": False,
                "focused_stratum": self._focused_stratum,
                "n_signals_total": n_signals,
            }
        )
        return summary

    @property
    def strata(self) -> list[str]:
        """Focused result has no strata (already single-stratum)."""
        return []

    @property
    def is_stratified(self) -> bool:
        return False

    @property
    def focused_stratum(self) -> str:
        return self._focused_stratum

    def focus(self, stratum: str) -> AnalysisResult:
        """Cannot re-focus a focused result."""
        raise ValidationError(
            f"Cannot focus: this result is already focused on '{self._focused_stratum}'. "
            "Use the original result to focus on a different stratum."
        )

    def __repr__(self) -> str:
        charts_str = ", ".join(self.all_charts)
        return (
            f"FocusedAnalysisResult(\n"
            f"  stratum='{self._focused_stratum}',\n"
            f"  analytical_sds={self.analytical_sds} "
            f"({self.analytical_sds_info['description']}),\n"
            f"  charts=[{charts_str}],\n"
            f"  n_obs={len(self._dataset)},\n"
            f"  has_residuals={self.has_residuals},\n"
            f"  has_effects={self.has_effects}\n"
            f")"
        )
