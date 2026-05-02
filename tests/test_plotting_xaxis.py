"""
Tests for x-axis behavior on collapsed and dense charts.

Regression tests for:
- _get_x_column() uniqueness check (categorical compression bug)
- Adaptive tick label rotation for dense charts
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.plotting import Plotter


class TestGetXColumnUniqueness:
    """Test that _get_x_column() returns None for non-unique candidate columns."""

    @pytest.fixture
    def plotter_with_repeating_rsg(self):
        """Create a Plotter whose chart data has repeating rsg values.

        Simulates by=[] XmR on SDS 2 data: rsg repeats across time periods.
        """
        df = pd.DataFrame({
            'value': np.random.default_rng(42).normal(100, 5, 40),
            'factor 1': [1, 1, 2, 2] * 10,
            'factor 2': [1, 2, 1, 2] * 10,
            'time': sorted(list(range(1, 11)) * 4),
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(
            response=pb.cols.value,
            factors=[pb.cols.factor_1, pb.cols.factor_2],
            time=pb.cols.time,
        )
        return study.execute(chart='X', by=[])

    @pytest.fixture
    def plotter_with_unique_rsg(self):
        """Create a Plotter whose chart data has unique rsg values.

        Standard Xbar analysis where each row is a unique subgroup.
        """
        df = pd.DataFrame({
            'value': np.random.default_rng(42).normal(100, 5, 40),
            'factor 1': [1, 1, 2, 2] * 10,
            'time': sorted(list(range(1, 11)) * 4),
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(
            response=pb.cols.value,
            factors=[pb.cols.factor_1],
            time=pb.cols.time,
        )
        return study.execute()

    def test_get_x_column_returns_none_for_repeating_rsg(
        self, plotter_with_repeating_rsg,
    ):
        """When rsg values repeat (by=[] XmR), _get_x_column must return None."""
        plotter = Plotter(plotter_with_repeating_rsg)
        chart_data = next(iter(plotter.charts.values()))['data']
        result = plotter._get_x_column(chart_data)
        assert result is None

    def test_get_x_column_returns_col_for_unique_rsg(
        self, plotter_with_unique_rsg,
    ):
        """When rsg/subgroup values are unique (Xbar), _get_x_column returns it."""
        plotter = Plotter(plotter_with_unique_rsg)
        chart_data = next(iter(plotter.charts.values()))['data']
        result = plotter._get_x_column(chart_data)
        assert result is not None


class TestByEmptyXmrXAxis:
    """End-to-end test: by=[] XmR chart uses integer x positions."""

    def test_by_empty_xmr_uses_integer_xaxis(self):
        """by=[] XmR chart must use sequential integer x-axis, not categorical."""
        df = pd.DataFrame({
            'value': np.random.default_rng(42).normal(100, 5, 40),
            'factor 1': [1, 1, 2, 2] * 10,
            'factor 2': [1, 2, 1, 2] * 10,
            'time': sorted(list(range(1, 11)) * 4),
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(
            response=pb.cols.value,
            factors=[pb.cols.factor_1, pb.cols.factor_2],
            time=pb.cols.time,
        )
        result = study.execute(chart='X', by=[])
        fig = result.plot(chart='X')

        # The x-axis data on the first trace should be numeric (integer index),
        # not categorical strings that repeat.
        trace_x = fig._fig.data[0].x
        if trace_x is not None:
            # Should be numeric / integer-like, not categorical
            assert all(isinstance(v, (int, float, np.integer)) for v in trace_x), (
                f"Expected numeric x values, got types: {set(type(v) for v in trace_x)}"
            )
        # If trace_x is None, plotly uses the DataFrame index (integer) — also correct


class TestAdaptiveTickAngle:
    """Test that tick labels rotate adaptively based on chart density."""

    def _make_result(self, n_time):
        """Helper: create an XmR by=[] result with n_time observations per factor combo."""
        n_combos = 4  # 2 × 2 factor grid
        n_total = n_time * n_combos
        df = pd.DataFrame({
            'value': np.random.default_rng(42).normal(100, 5, n_total),
            'factor 1': [1, 1, 2, 2] * n_time,
            'factor 2': [1, 2, 1, 2] * n_time,
            'time': sorted(list(range(1, n_time + 1)) * n_combos),
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(
            response=pb.cols.value,
            factors=[pb.cols.factor_1, pb.cols.factor_2],
            time=pb.cols.time,
        )
        return study.execute(chart='X', by=[])

    def test_tick_angle_horizontal_for_sparse_charts(self):
        """Charts with <=20 ticks should have tickangle=0 (horizontal)."""
        result = self._make_result(4)  # 4 time × 4 combos = 16 obs → ≤20 ticks
        fig = result.plot(chart='X')

        # Get xaxis tickangle — may be on xaxis or xaxis2 depending on subplot
        layout = fig._fig.layout
        angles = []
        for attr_name in dir(layout):
            if attr_name.startswith('xaxis'):
                ax = getattr(layout, attr_name)
                if hasattr(ax, 'tickangle') and ax.tickangle is not None:
                    angles.append(ax.tickangle)

        # All x-axes should have tickangle=0
        assert all(a == 0 for a in angles), f"Expected tickangle=0, got {angles}"

    def test_tick_angle_rotated_for_dense_long_labels(self):
        """Charts with many time points and long labels should auto-rotate to -45 degrees."""
        # Use long date-like labels so label_footprint exceeds 80
        n_time = 100
        n_combos = 4
        n_total = n_time * n_combos
        long_labels = [f"2024-01-{i:02d}-extra" for i in range(1, n_time + 1)]
        df = pd.DataFrame({
            'value': np.random.default_rng(42).normal(100, 5, n_total),
            'factor 1': [1, 1, 2, 2] * n_time,
            'factor 2': [1, 2, 1, 2] * n_time,
            'time': sorted(long_labels * n_combos),
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(
            response=pb.cols.value,
            factors=[pb.cols.factor_1, pb.cols.factor_2],
            time=pb.cols.time,
        )
        result = study.execute(chart='X', by=[])
        fig = result.plot(chart='X')

        layout = fig._fig.layout
        angles = []
        for attr_name in dir(layout):
            if attr_name.startswith('xaxis'):
                ax = getattr(layout, attr_name)
                if hasattr(ax, 'tickangle') and ax.tickangle is not None:
                    angles.append(ax.tickangle)

        # At least one x-axis should have tickangle=-45
        assert any(a == -45 for a in angles), f"Expected tickangle=-45, got {angles}"

    def test_tick_angle_horizontal_for_dense_short_labels(self):
        """Charts with many time points but short numeric labels stay horizontal."""
        result = self._make_result(100)  # 100 time × 4 combos = 400 obs, labels "1"-"100"
        fig = result.plot(chart='X')

        layout = fig._fig.layout
        angles = []
        for attr_name in dir(layout):
            if attr_name.startswith('xaxis'):
                ax = getattr(layout, attr_name)
                if hasattr(ax, 'tickangle') and ax.tickangle is not None:
                    angles.append(ax.tickangle)

        # Short numeric labels should remain horizontal
        assert all(a == 0 for a in angles), f"Expected tickangle=0, got {angles}"


class TestSingleFactorByAxisLabel:
    """Ensure Xbar/S keep factor labels on x-axis for single-factor by views."""

    @staticmethod
    def _study():
        pb = ProcessBehavior.read_csv("validation/TABVASTESTDATABASE.csv")
        return pb.formulate(response="PM SDS 1", factors=["FACTOR 1", "FACTOR 2"])

    def test_xbar_by_factor_uses_factor_label(self):
        study = self._study()
        result = study.execute(chart="Xbar", by=["FACTOR 1"])
        fig = result.plot(chart="Xbar")
        assert fig._fig.layout.xaxis.title.text == "Factor 1"

    def test_s_by_factor_uses_factor_label(self):
        study = self._study()
        result = study.execute(chart="S", by=["FACTOR 1"])
        fig = result.plot(chart="S")
        assert fig._fig.layout.xaxis.title.text == "Factor 1"


class TestNumericFactorCategoryAxis:
    """Numeric-looking factor levels must produce a categorical x-axis."""

    @staticmethod
    def _study():
        pb = ProcessBehavior.read_csv("validation/TABVASTESTDATABASE.csv")
        return pb.formulate(response="PM SDS 1", factors=["FACTOR 1", "FACTOR 2"])

    def test_xbar_by_numeric_factor_is_categorical(self):
        """Xbar by=['FACTOR 1'] with integer levels → category axis."""
        study = self._study()
        result = study.execute(chart="Xbar", by=["FACTOR 1"])
        fig = result.plot(chart="Xbar")
        assert fig._fig.layout.xaxis.type == "category", (
            f"Expected category axis, got {fig._fig.layout.xaxis.type}"
        )

    def test_s_by_numeric_factor_is_categorical(self):
        """S by=['FACTOR 1'] with integer levels → category axis."""
        study = self._study()
        result = study.execute(chart="S", by=["FACTOR 1"])
        fig = result.plot(chart="S")
        assert fig._fig.layout.xaxis.type == "category", (
            f"Expected category axis, got {fig._fig.layout.xaxis.type}"
        )


# ============================================================================
# Tick label invariants for XmR with lane boundaries (repeated time)
# ============================================================================


class TestTickLabelBlockInvariants:
    """Tick labels must be monotonic within each factor block.

    When XmR charts have lane boundaries (collapsed factors), time values
    repeat across blocks. Tick thinning must select labels per-block to
    avoid garbled cross-block mixing.
    """

    @staticmethod
    def _study():
        from processbehavior.datasets.synthetic import make_sds
        df = make_sds(2, K1=3, K2=2, T=10, seed=42)
        return ProcessBehavior(df).formulate(
            response='y', factors=['factor 1', 'factor 2'], time='time',
        )

    @staticmethod
    def _extract_ticks_and_boundaries(result, chart_name='X'):
        """Extract tick positions, labels, and lane boundary positions."""
        if result.is_stratified:
            focused = result.focus(result.strata[0])
        else:
            focused = result

        fig = focused.plot()
        ax = fig.figure.layout.xaxis
        tickvals = list(ax.tickvals) if ax.tickvals else []
        ticktext = list(ax.ticktext) if ax.ticktext else []

        meta = focused.charts[chart_name].get('metadata', {})
        lb = meta.get('lane_boundaries')
        if isinstance(lb, list):
            boundary_positions = [b['position'] for b in lb]
        elif isinstance(lb, dict):
            first_key = next(iter(lb))
            boundary_positions = [b['position'] for b in lb[first_key]]
        else:
            boundary_positions = []

        data = focused.get_chart(chart_name)
        return tickvals, ticktext, boundary_positions, data

    @pytest.mark.parametrize("by", [
        pytest.param([], id="overall"),
        pytest.param(['factor 1'], id="by_factor1"),
        pytest.param(['factor 2'], id="by_factor2"),
    ])
    def test_tick_labels_monotonic_within_blocks(self, by):
        """Tick labels should increase within each factor block."""
        study = self._study()
        result = study.execute(chart='X', by=by, companion=True)
        tickvals, ticktext, boundaries, data = self._extract_ticks_and_boundaries(result)

        n = len(data)
        edges = sorted({0} | set(boundaries) | {n})

        for i in range(len(edges) - 1):
            start, end = edges[i], edges[i + 1]
            block_labels = [
                int(ticktext[j]) for j, pos in enumerate(tickvals)
                if start <= pos < end
            ]
            if len(block_labels) >= 2:
                for k in range(1, len(block_labels)):
                    assert block_labels[k] > block_labels[k - 1], (
                        f"Block [{start}, {end}): tick labels not monotonic: {block_labels}"
                    )

    @pytest.mark.parametrize("by", [
        pytest.param([], id="overall"),
        pytest.param(['factor 1'], id="by_factor1"),
        pytest.param(['factor 2'], id="by_factor2"),
    ])
    def test_every_block_has_at_least_one_tick(self, by):
        """Every factor block should have at least one tick label."""
        study = self._study()
        result = study.execute(chart='X', by=by, companion=True)
        tickvals, ticktext, boundaries, data = self._extract_ticks_and_boundaries(result)

        n = len(data)
        edges = sorted({0} | set(boundaries) | {n})

        for i in range(len(edges) - 1):
            start, end = edges[i], edges[i + 1]
            block_ticks = [pos for pos in tickvals if start <= pos < end]
            assert len(block_ticks) >= 1, (
                f"Block [{start}, {end}) has no tick labels"
            )

    def test_full_rsg_no_lane_boundaries(self):
        """by=[all factors] has unique time — no lane boundaries, no blocks."""
        study = self._study()
        result = study.execute(chart='X', by=['factor 1', 'factor 2'], companion=True)
        focused = result.focus(result.strata[0])
        meta = focused.charts['X'].get('metadata', {})
        lb = meta.get('lane_boundaries')
        assert not lb, "Full RSG should have no lane boundaries"
