"""Tests for `result_tabular` — `build_chart_table` and `build_signals_table`.

The functions are pure given an AnalysisResult; we use real (small) results
from synthetic data rather than mocking the result class, since the value-col
inference and n-join depend on the ads/spec wiring.
"""

from __future__ import annotations

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic
from processbehavior.exceptions import ChartNotAvailableError
from processbehavior.result_tabular import (
    _NON_VALUE_COLS,
    _PREFERRED_STATISTIC_COLS,
    _SIGNAL_SYMBOLS,
    build_chart_table,
    build_signals_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xbar_s_result():
    """A stratified Xbar/S result on SDS 1 — has rsg column, multiple subgroups."""
    df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return (
        ProcessBehavior(df)
        .formulate(response="y", factors=["factor 1", "factor 2"], time="time")
        .execute()  # default chart for SDS 1 is Xbar/S (cell-keyed by rsg)
    )


@pytest.fixture(scope="module")
def x_mr_result():
    """An X/mR companion result — confirms the same code path works for individuals."""
    df = synthetic.make_sds(2, K1=3, K2=2, T=10, seed=42)
    return (
        ProcessBehavior(df)
        .formulate(response="y", factors=["factor 1", "factor 2"], time="time")
        .execute(chart="X", by=[], companion=True)
    )


# ---------------------------------------------------------------------------
# build_chart_table
# ---------------------------------------------------------------------------


class TestBuildChartTable:
    def test_chart_none_uses_first_chart(self, xbar_s_result):
        table = build_chart_table(xbar_s_result)
        # Should not raise; should return a non-empty DataFrame
        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0

    def test_unknown_chart_raises(self, xbar_s_result):
        with pytest.raises(ChartNotAvailableError):
            build_chart_table(xbar_s_result, chart="DoesNotExist")

    def test_columns_match_documented_contract(self, xbar_s_result):
        """Output has value, center, lpl, upl, (signal). `subgroup` only when
        chart_data has an `rsg` column (chart-type dependent)."""
        table = build_chart_table(xbar_s_result, chart="Xbar")
        assert "value" in table.columns
        assert "center" in table.columns
        assert "lpl" in table.columns
        assert "upl" in table.columns
        # signal column included by default
        assert "signal" in table.columns

    def test_subgroup_column_present_when_rsg_in_chart_data(self, x_mr_result):
        """rsg column → renamed to `subgroup` in the output."""
        # x_mr_result is stratified with rsg present in chart_data
        table = build_chart_table(x_mr_result, chart="X")
        assert "subgroup" in table.columns
        # Original `rsg` column NOT still present under that name
        assert "rsg" not in table.columns

    def test_include_signal_col_false_omits_signal(self, xbar_s_result):
        table = build_chart_table(xbar_s_result, chart="Xbar", include_signal_col=False)
        assert "signal" not in table.columns

    def test_signal_symbols_default(self, xbar_s_result):
        """Default: beyond_limits → ↓/blank/↑."""
        table = build_chart_table(xbar_s_result, chart="Xbar")
        unique_signals = set(table["signal"].dropna().unique())
        # Allowed values are exactly the symbol map values
        assert unique_signals.issubset(set(_SIGNAL_SYMBOLS.values()))

    def test_signal_symbols_false_keeps_numeric(self, xbar_s_result):
        """signal_symbols=False keeps -1/0/1 ints."""
        table = build_chart_table(xbar_s_result, chart="Xbar", signal_symbols=False)
        unique_signals = set(table["signal"].dropna().unique())
        assert unique_signals.issubset({-1, 0, 1})

    def test_value_col_uses_response_var_when_present(self, xbar_s_result):
        """When the chart_data has the response variable column, that's used as value."""
        # For SDS 1 Xbar/S, the data has the response 'y' column directly.
        table = build_chart_table(xbar_s_result, chart="Xbar")
        # The 'y' column should have been renamed to 'value'
        assert "value" in table.columns
        # And 'y' should NOT still be a separate column
        assert "y" not in table.columns

    def test_index_is_reset(self, xbar_s_result):
        """Returned table has a clean RangeIndex starting at 0."""
        table = build_chart_table(xbar_s_result, chart="Xbar")
        assert list(table.index) == list(range(len(table)))

    def test_x_mr_chart_works(self, x_mr_result):
        """Individuals chart also produces a sensible table."""
        table = build_chart_table(x_mr_result, chart="X")
        assert "value" in table.columns
        assert "center" in table.columns
        assert len(table) > 0


# ---------------------------------------------------------------------------
# build_signals_table
# ---------------------------------------------------------------------------


class TestBuildSignalsTable:
    def test_chart_specified_returns_signals_only(self, xbar_s_result):
        signals = build_signals_table(xbar_s_result, chart="Xbar")
        # Either empty (no signals) or all rows have non-zero beyond_limits
        if not signals.empty:
            assert (signals["beyond_limits"] != 0).all()

    def test_chart_none_combines_all_charts(self, xbar_s_result):
        signals = build_signals_table(xbar_s_result, chart=None)
        if not signals.empty:
            # Multi-chart output has a 'chart' identifier column
            assert "chart" in signals.columns
            # Each row is a signal
            assert (signals["beyond_limits"] != 0).all()

    def test_no_beyond_limits_returns_empty_frame(self):
        """A degenerate result without beyond_limits column → empty DataFrame."""

        class _StubResult:
            def get_chart(self, name):
                return pd.DataFrame({"value": [1, 2, 3]})

            def iter_charts(self):
                yield "Xbar", pd.DataFrame({"value": [1, 2, 3]}), {}

        result = _StubResult()
        single = build_signals_table(result, chart="Xbar")  # type: ignore[arg-type]
        all_charts = build_signals_table(result, chart=None)  # type: ignore[arg-type]
        assert single.empty
        assert all_charts.empty


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_signal_symbol_map_is_complete(self):
        """beyond_limits values are {-1, 0, 1} — all must have a symbol."""
        assert set(_SIGNAL_SYMBOLS.keys()) == {-1, 0, 1}

    def test_non_value_cols_includes_typical_metadata(self):
        for col in ("rsg", "center", "lpl", "upl", "beyond_limits", "n", "obs_id"):
            assert col in _NON_VALUE_COLS

    def test_preferred_statistic_cols_order(self):
        """Order matters — xbar before s before mr before r."""
        assert _PREFERRED_STATISTIC_COLS == ("xbar", "s", "mr", "r")
