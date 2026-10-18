"""Tests for `result_tabular` — `build_chart_table` and `build_signals_table`.

The functions are pure given an AnalysisResult; we use real (small) results
from synthetic data rather than mocking the result class, since the value-col
inference and n-join depend on the ads/spec wiring.
"""

from __future__ import annotations

from pathlib import Path

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
    df = synthetic.make_design(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return (
        ProcessBehavior(df)
        .formulate(response="y", factors=["factor 1", "factor 2"], time="time")
        .execute()  # default chart for SDS 1 is Xbar/S (cell-keyed by rsg)
    )


@pytest.fixture(scope="module")
def x_mr_result():
    """An X/mR companion result — confirms the same code path works for individuals."""
    df = synthetic.make_design(2, K1=3, K2=2, T=10, seed=42)
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
        """Output has value, center, lpl, upl, (signal). Xbar/S always add
        `subgroup` and `n` (see TestXbarSSubgroupAndN)."""
        table = build_chart_table(xbar_s_result, chart="Xbar")
        assert "value" in table.columns
        assert "center" in table.columns
        assert "lpl" in table.columns
        assert "upl" in table.columns
        # signal column included by default
        assert "signal" in table.columns

    def test_index_is_one_based(self, xbar_s_result, x_mr_result):
        """The position index starts at 1 (matches the chart's plotted x-axis)."""
        for res, chart in ((xbar_s_result, "Xbar"), (x_mr_result, "X")):
            table = build_chart_table(res, chart=chart)
            assert list(table.index) == list(range(1, len(table) + 1))

    def test_obs_column_on_individuals_only(self, xbar_s_result, x_mr_result):
        """`Obs` (source-row obs_id) is added on X/mR tables, not on Xbar/S."""
        for chart in ("X", "mR"):
            table = build_chart_table(x_mr_result, chart=chart)
            assert "Obs" in table.columns
            assert "obs_id" not in table.columns  # renamed to "Obs"
        assert "Obs" not in build_chart_table(xbar_s_result, chart="Xbar").columns

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

    def test_x_mr_chart_works(self, x_mr_result):
        """Individuals chart also produces a sensible table."""
        table = build_chart_table(x_mr_result, chart="X")
        assert "value" in table.columns
        assert "center" in table.columns
        assert len(table) > 0


REFERENCE_CSV = Path(__file__).resolve().parent.parent / "validation" / "PBTESTDATABASE_T100.csv"
FACTORS = ["FACTOR 1", "FACTOR 2"]
TIME = "PRODUCTION TIME"


@pytest.fixture(scope="module")
def reference_studies():
    """Bishop's reference data: PM SDS 1 (balanced) and PM SDS 3 (cell sizes vary)."""
    df = pd.read_csv(REFERENCE_CSV, na_values=["*"])
    return {
        response: ProcessBehavior(df).formulate(response=response, factors=FACTORS, time=TIME)
        for response in ("PM SDS 1", "PM SDS 3")
    }


def _expected_n(study, by, column, min_n=1):
    """Independent count of `column` per subgroup of `by`, keyed like the table's label."""
    counts = study.dataset.groupby(by, observed=True)[column].count()
    counts = counts[counts >= min_n]
    labels = ["_".join(map(str, k)) if isinstance(k, tuple) else str(k) for k in counts.index]
    return dict(zip(labels, counts.tolist(), strict=True))


class TestXbarSSubgroupAndN:
    """Every Xbar/S table labels its subgroups and gives each one's own n (#143).

    `n` comes from the Xbar/S aggregation, so a subgroup that pools several cells
    reports all of them — a per-cell size, which the table used to re-join, cannot.
    """

    POOLED = [
        ["FACTOR 1", TIME],
        ["FACTOR 2", TIME],
        ["FACTOR 1", "FACTOR 2", TIME],
        ["FACTOR 1", "FACTOR 2"],
        ["FACTOR 1"],
    ]

    @pytest.mark.parametrize("response", ["PM SDS 1", "PM SDS 3"])
    @pytest.mark.parametrize("by", POOLED, ids=lambda b: "+".join(b))
    def test_by_tables_have_subgroup_and_true_n(self, reference_studies, response, by):
        study = reference_studies[response]
        result = study.execute(chart="Xbar", by=by, companion=True)
        for chart, min_n in (("Xbar", 1), ("S", 2)):
            table = build_chart_table(result, chart=chart)
            assert list(table.columns[:2]) == ["subgroup", "n"]
            assert dict(zip(table["subgroup"], table["n"], strict=True)) == _expected_n(study, by, response, min_n)

    @pytest.mark.parametrize("by", POOLED, ids=lambda b: "+".join(b))
    def test_solo_s_matches_companion_s(self, reference_studies, by):
        study = reference_studies["PM SDS 3"]
        companion = build_chart_table(study.execute(chart="Xbar", by=by, companion=True), chart="S")
        solo = build_chart_table(study.execute(chart="S", by=by), chart="S")
        pd.testing.assert_frame_equal(companion, solo)

    @pytest.mark.parametrize("response", ["PM SDS 1", "PM SDS 3"])
    def test_default_and_stratified_tables_give_cell_sizes(self, reference_studies, response):
        study = reference_studies[response]
        cell_n = _expected_n(study, ["rsg", TIME], response)
        default = build_chart_table(study.execute(chart="Xbar"), chart="Xbar")
        assert list(default.columns) == ["subgroup", "n", "value", "center", "lpl", "upl", "signal"]
        assert dict(zip(default["subgroup"], default["n"], strict=True)) == cell_n

        stratified = study.execute(chart="Xbar", by=[TIME])
        table = build_chart_table(stratified, chart="Xbar")
        data = stratified.get_chart("Xbar")
        labels = data["rsg"].astype(str) + "_" + data[TIME].astype(str)
        assert dict(zip(labels, table["n"], strict=True)) == cell_n

    def test_n_is_what_the_limits_rest_on(self, reference_studies):
        """R5's limits rest on R2, undefined at the stream's first value: that cell counts one fewer."""
        study = reference_studies["PM SDS 3"]
        table = build_chart_table(study.execute(chart="Xbar", value="R5"), chart="Xbar")
        assert dict(zip(table["subgroup"], table["n"], strict=True)) == _expected_n(study, ["rsg", TIME], "R2")
        first = table.iloc[0]
        assert first["n"] == _expected_n(study, ["rsg", TIME], "PM SDS 3")[first["subgroup"]] - 1

    def test_n_mode_average_keeps_actual_sizes(self, reference_studies):
        study = reference_studies["PM SDS 3"]
        by = ["FACTOR 1", TIME]
        table = build_chart_table(study.execute(chart="Xbar", by=by, n_mode="average"), chart="Xbar")
        assert dict(zip(table["subgroup"], table["n"], strict=True)) == _expected_n(study, by, "PM SDS 3")

    @pytest.mark.parametrize("by", [None, ["FACTOR 1", TIME], ["FACTOR 1"], [TIME]], ids=str)
    def test_chart_data_carries_n_after_the_label(self, reference_studies, by):
        result = reference_studies["PM SDS 3"].execute(chart="Xbar", by=by, companion=True)
        for chart in ("Xbar", "S"):
            columns = list(result.get_chart(chart).columns)
            assert columns[columns.index("n") - 1] == (result.charts[chart]["metadata"].get("group_col") or TIME)
            assert columns.index("n") < columns.index("center")


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
        for col in ("rsg", "center", "lpl", "upl", "beyond_limits", "n", "phase", "obs_id"):
            assert col in _NON_VALUE_COLS

    def test_preferred_statistic_cols_order(self):
        """Order matters — xbar before s before mr before r."""
        assert _PREFERRED_STATISTIC_COLS == ("xbar", "s", "mr", "r")
