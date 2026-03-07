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
        return study.execute(chart='XmR', by=[])

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
        result = study.execute(chart='XmR', by=[])
        fig = result.plot(chart='XmR')

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
        return study.execute(chart='XmR', by=[])

    def test_tick_angle_horizontal_for_sparse_charts(self):
        """Charts with <=20 ticks should have tickangle=0 (horizontal)."""
        result = self._make_result(4)  # 4 time × 4 combos = 16 obs → ≤20 ticks
        fig = result.plot(chart='XmR')

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
        result = study.execute(chart='XmR', by=[])
        fig = result.plot(chart='XmR')

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
        fig = result.plot(chart='XmR')

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
