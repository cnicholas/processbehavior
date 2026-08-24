"""Tabular views of an `AnalysisResult`.

Pure functions that produce DataFrames from a result. Extracted from
`AnalysisResult.chart_table` and `AnalysisResult.get_signals` so the
tabular-formatting logic is independent of the result class and directly
unit-testable (no need to construct a full AnalysisDataSet).

The class methods on `AnalysisResult` remain in place as thin delegates
to the functions here — the public API is unchanged.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pandas as pd

from .data_preparation import encode_rsg_series
from .exceptions import ChartNotAvailableError

if TYPE_CHECKING:
    from .analysis_result import AnalysisResult


# Columns that are never the chart's "value" column — they're metadata,
# join keys, or limits. Used by `_resolve_value_col` when the response
# variable isn't directly identifiable in chart_data.
_NON_VALUE_COLS = frozenset(
    {
        "rsg",
        "center",
        "lpl",
        "upl",
        "beyond_limits",
        "n",
        "N",
        "obs_id",
        "x",
        "pull",
        "time",
        "date",
        "datetime",
        "rsg_key",
        "cell_key",
    }
)

# When the response variable can't be located, prefer one of these
# canonical statistic columns before falling back to "any non-meta column".
_PREFERRED_STATISTIC_COLS = ("xbar", "s", "mr", "r")

# beyond_limits encoding → user-facing signal symbol.
_SIGNAL_SYMBOLS = {-1: "↓", 0: "", 1: "↑"}


# ---------------------------------------------------------------------------
# Public: chart_table
# ---------------------------------------------------------------------------


def build_chart_table(
    result: AnalysisResult,
    *,
    chart: str | None = None,
    include_signal_col: bool = True,
    signal_symbols: bool = True,
) -> pd.DataFrame:
    """Build the summary DataFrame for one chart on `result`.

    Identifies the value column, joins the per-cell `n` from the analysis
    dataset if available, selects logical output columns, and optionally
    converts `beyond_limits` into ↓/↑ symbols.

    Parameters mirror `AnalysisResult.chart_table` exactly. See that
    docstring for the column contract.
    """
    chart = _resolve_chart(result, chart)
    chart_data = result.charts[chart]["data"].copy()

    value_col = _resolve_value_col(result, chart_data)
    chart_data = _join_n_column(result, chart_data)

    output_cols, col_renames = _output_column_plan(
        chart_data, value_col, include_signal_col, chart=chart
    )
    table = chart_data[output_cols].copy().rename(columns=col_renames)

    if include_signal_col and "signal" in table.columns and signal_symbols:
        table["signal"] = table["signal"].map(_SIGNAL_SYMBOLS)

    # 1-based position index so the table's leading column matches the chart's plotted
    # x-axis (which starts at 1) rather than a 0-based row counter.
    table = table.reset_index(drop=True)
    table.index = table.index + 1
    return table


# ---------------------------------------------------------------------------
# Public: signals_table
# ---------------------------------------------------------------------------


def build_signals_table(
    result: AnalysisResult,
    *,
    chart: str | None = None,
) -> pd.DataFrame:
    """Rows from `chart` (or all charts) where `beyond_limits != 0`.

    When `chart` is None, the returned frame has a `chart` column
    identifying which chart each signal row came from.
    """
    if chart:
        data = result.get_chart(chart)
        if "beyond_limits" in data.columns:
            return data[data["beyond_limits"] != 0]
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for name, data, _ in result.iter_charts():
        if "beyond_limits" not in data.columns:
            continue
        signals = data[data["beyond_limits"] != 0].copy()
        signals["chart"] = name
        rows.append(signals)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_chart(result: AnalysisResult, chart: str | None) -> str:
    """Resolve `chart=None` to the first chart and validate existence."""
    resolved = result.all_charts[0] if chart is None else result._resolve_chart_name(chart)
    if resolved not in result.charts:
        raise ChartNotAvailableError(
            f"Chart '{resolved}' not found. Available charts: {result.all_charts}",
            chart=resolved,
            available=result.all_charts,
        )
    return resolved


def _resolve_value_col(result: AnalysisResult, chart_data: pd.DataFrame) -> str | None:
    """Pick the chart's value column.

    Preference order:
    1. The analysis spec's response variable, if present in chart_data.
    2. A known statistic column (`xbar`, `s`, `mr`, `r`).
    3. The first non-metadata column.
    Returns None if nothing fits (degenerate chart).
    """
    if result._ads is not None:
        response_var = result._ads.spec.response_var
        if response_var in chart_data.columns:
            return response_var

    meta_cols = set(_NON_VALUE_COLS)
    if result._ads is not None and result._ads.spec.time_var:
        meta_cols.add(result._ads.spec.time_var)

    value_cols = [c for c in chart_data.columns if c not in meta_cols]
    for preferred in _PREFERRED_STATISTIC_COLS:
        if preferred in value_cols:
            return preferred
    return value_cols[0] if value_cols else None


def _join_n_column(result: AnalysisResult, chart_data: pd.DataFrame) -> pd.DataFrame:
    """Merge per-cell subgroup size `n` from the analysis dataset.

    Returns `chart_data` unchanged when:
    - `n` already in chart_data
    - no analysis dataset available
    - `n` not in analysis dataset
    - no matching kt columns (factor × time grain)

    The join keys are coerced to string on both sides when dtypes differ
    (chart construction sometimes stringifies by-columns while the
    underlying ads keeps them numeric). Merge failures are swallowed —
    a missing `n` column is preferable to a broken table.
    """
    if "n" in chart_data.columns or result._ads is None:
        return chart_data

    ads = result._ads.analysis_dataset
    spec = result._ads.spec
    if "n" not in ads.columns:
        return chart_data

    kt_cols: list[str] = []
    if spec.rsg_var_name and spec.rsg_var_name in ads.columns:
        rsg_col = "rsg" if "rsg" in chart_data.columns else spec.rsg_var_name
        if rsg_col in chart_data.columns:
            kt_cols.append(spec.rsg_var_name)
    if spec.has_time and spec.time_var in ads.columns and spec.time_var in chart_data.columns:
        kt_cols.append(spec.time_var)
    if not kt_cols:
        # Xbar/S identify a subgroup with a single composite "group" column
        # (rsg and time already joined, e.g. 'F1_1_F2_1_3') rather than the
        # separate key columns X/mR carry. Rebuild the same composite on the
        # ads side and join on that — otherwise `n` silently never lands on
        # exactly the charts whose limits vary with it.
        return _join_n_on_composite_group(ads, spec, chart_data)

    n_per_kt = ads.groupby(kt_cols, observed=True)["n"].first().reset_index()

    # Dtype-coerce join keys to string if either side disagrees.
    mismatched = any(chart_data[c].dtype != n_per_kt[c].dtype for c in kt_cols)
    if mismatched:
        for c in kt_cols:
            chart_data[c] = chart_data[c].astype(str)
            n_per_kt[c] = n_per_kt[c].astype(str)

    with contextlib.suppress(ValueError, TypeError):
        chart_data = chart_data.merge(n_per_kt, on=kt_cols, how="left")
    return chart_data


def _join_n_on_composite_group(ads: pd.DataFrame, spec, chart_data: pd.DataFrame) -> pd.DataFrame:
    """Merge `n` onto a chart keyed by a single composite subgroup column.

    Xbar/S label each subgroup with the encoded (rsg, time) cell — the same
    encoding :func:`encode_rsg_series` produces — so the join key is rebuilt
    rather than matched column-by-column. Returns `chart_data` unchanged if
    the composite cannot be formed or does not match.
    """
    if "group" not in chart_data.columns:
        return chart_data

    key_cols = [c for c in (spec.rsg_var_name, spec.time_var) if c and c in ads.columns]
    if len(key_cols) < 2:
        return chart_data

    n_per_cell = ads.groupby(key_cols, observed=True)["n"].first().reset_index()
    n_per_cell["group"] = encode_rsg_series(n_per_cell, key_cols)
    n_per_cell = n_per_cell[["group", "n"]]

    chart_data = chart_data.copy()
    chart_data["group"] = chart_data["group"].astype(str)

    with contextlib.suppress(ValueError, TypeError):
        merged = chart_data.merge(n_per_cell, on="group", how="left")
        # Only adopt the merge if it actually matched; a delimiter collision
        # would otherwise replace a missing column with a column of NaN.
        if merged["n"].notna().any():
            return merged
    return chart_data


def _output_column_plan(
    chart_data: pd.DataFrame,
    value_col: str | None,
    include_signal_col: bool,
    chart: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Decide which columns to keep and how to rename them for display."""
    output_cols: list[str] = []
    renames: dict[str, str] = {}

    # Source-row id ("Obs") — Individuals charts only, where each row maps 1:1 to a
    # single observation, so it traces back to the source data. Xbar/S rows are
    # subgroup aggregates (identified by "subgroup"), so obs_id is not 1:1 there.
    if chart in ("X", "mR") and "obs_id" in chart_data.columns:
        output_cols.append("obs_id")
        renames["obs_id"] = "Obs"

    # The subgroup identifier is "rsg" on X/mR charts but "group" on Xbar/S.
    # Only "rsg" was recognised, so chart_table('Xbar') silently dropped the
    # subgroup column (and "n" with it) that its own docstring promises.
    subgroup_col = next(
        (col for col in ("rsg", "group") if col in chart_data.columns), None
    )
    if subgroup_col:
        output_cols.append(subgroup_col)
        renames[subgroup_col] = "subgroup"

    if "n" in chart_data.columns:
        output_cols.append("n")

    if value_col:
        output_cols.append(value_col)
        renames[value_col] = "value"

    for col in ("center", "lpl", "upl"):
        if col in chart_data.columns:
            output_cols.append(col)

    if include_signal_col and "beyond_limits" in chart_data.columns:
        output_cols.append("beyond_limits")
        renames["beyond_limits"] = "signal"

    return output_cols, renames
