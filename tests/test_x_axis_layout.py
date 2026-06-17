"""Tests for the x_axis_layout subsystem.

Three layers:

- **Parse**: `parse_lane_boundaries` shape handling.
- **Boundary types**: `FlatBoundaries.block_edges`, `StratifiedBoundaries.for_stratum`.
- **Layout**: `compute_x_axis_layout` invariants + golden scenarios.
- **Apply**: side effects on a Plotly figure (tickvals, annotations, margin).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from processbehavior.plotting.x_axis_layout import (
    AxisTitle,
    CellBand,
    FlatBoundaries,
    StratifiedBoundaries,
    TickRow,
    XAxisLayout,
    apply_x_axis_layout,
    compute_x_axis_layout,
    parse_lane_boundaries,
)

# ---------------------------------------------------------------------------
# parse_lane_boundaries
# ---------------------------------------------------------------------------


class TestParseLaneBoundaries:
    def test_none_returns_none(self):
        assert parse_lane_boundaries(None) is None

    def test_empty_list_returns_none(self):
        assert parse_lane_boundaries([]) is None

    def test_empty_dict_returns_none(self):
        assert parse_lane_boundaries({}) is None

    def test_flat_list_returns_flat_boundaries(self):
        raw = [{"position": 5}, {"position": 10, "label": "ignored"}]
        result = parse_lane_boundaries(raw)
        assert isinstance(result, FlatBoundaries)
        assert result.positions == (5, 10)

    def test_dict_returns_stratified_boundaries(self):
        raw = {
            "1_1": [{"position": 5}],
            "1_2": [{"position": 7}, {"position": 14}],
        }
        result = parse_lane_boundaries(raw)
        assert isinstance(result, StratifiedBoundaries)
        assert result.per_stratum["1_1"].positions == (5,)
        assert result.per_stratum["1_2"].positions == (7, 14)

    def test_dict_stratum_keys_coerced_to_str(self):
        raw = {1: [{"position": 5}]}
        result = parse_lane_boundaries(raw)
        assert isinstance(result, StratifiedBoundaries)
        assert "1" in result.per_stratum

    def test_unknown_shape_returns_none(self):
        assert parse_lane_boundaries(42) is None
        assert parse_lane_boundaries("string") is None


# ---------------------------------------------------------------------------
# Boundary type behaviour
# ---------------------------------------------------------------------------


class TestFlatBoundaries:
    def test_block_edges_includes_endpoints(self):
        b = FlatBoundaries((5, 10))
        assert b.block_edges(20) == (0, 5, 10, 20)

    def test_block_edges_clips_out_of_range(self):
        b = FlatBoundaries((5, 25, 100))
        assert b.block_edges(20) == (0, 5, 20)

    def test_block_edges_with_no_positions(self):
        b = FlatBoundaries(())
        assert b.block_edges(20) == (0, 20)

    def test_block_edges_drops_position_at_zero_and_n(self):
        # Boundaries AT 0 or n are redundant with the implicit endpoints.
        b = FlatBoundaries((0, 5, 20))
        assert b.block_edges(20) == (0, 5, 20)

    def test_block_edges_sorts_and_dedupes(self):
        b = FlatBoundaries((10, 5, 10, 5))
        assert b.block_edges(20) == (0, 5, 10, 20)


class TestStratifiedBoundaries:
    def test_for_stratum_returns_per_stratum_flat(self):
        s = StratifiedBoundaries(
            {"1_1": FlatBoundaries((5,)), "1_2": FlatBoundaries((7, 14))}
        )
        assert s.for_stratum("1_1") == FlatBoundaries((5,))
        assert s.for_stratum("1_2") == FlatBoundaries((7, 14))

    def test_for_stratum_missing_returns_empty(self):
        s = StratifiedBoundaries({"1_1": FlatBoundaries((5,))})
        assert s.for_stratum("missing") == FlatBoundaries(())

    def test_for_stratum_coerces_arg_to_str(self):
        s = StratifiedBoundaries({"1": FlatBoundaries((5,))})
        assert s.for_stratum(1) == FlatBoundaries((5,))


# ---------------------------------------------------------------------------
# compute_x_axis_layout — invariants (property-style)
# ---------------------------------------------------------------------------


def _make_df(n: int, time_var: str = "t", rsg_values: list[str] | None = None) -> pd.DataFrame:
    """Build a synthetic chart-data slice."""
    cols = {time_var: list(range(n))}
    if rsg_values is not None:
        # Cycle rsg values to length n.
        cols["rsg"] = [rsg_values[i % len(rsg_values)] for i in range(n)]
    return pd.DataFrame(cols)


class TestLayoutInvariants:
    @pytest.mark.parametrize("n", [0, 1, 5, 20, 100, 500])
    @pytest.mark.parametrize(
        "boundary_positions", [(), (10,), (10, 50, 80), (5, 200)]
    )
    def test_all_tick_positions_in_range(self, n, boundary_positions):
        df = _make_df(n)
        boundaries = FlatBoundaries(boundary_positions) if boundary_positions else None
        layout = compute_x_axis_layout(
            data=df,
            time_var="t",
            x_col=None,  # integer-position
            boundaries=boundaries,
            title_text="t",
        )
        if layout.ticks.positions is not None:
            for p in layout.ticks.positions:
                assert 0 <= p < n, f"tick position {p} outside [0, {n})"

    @pytest.mark.parametrize("n_blocks", [2, 3, 5, 10])
    def test_cell_band_labels_match_midpoints(self, n_blocks):
        n = 100
        block_size = n // n_blocks
        positions = tuple(block_size * i for i in range(1, n_blocks))
        df = _make_df(n, rsg_values=[f"cell_{i}" for i in range(n_blocks)])
        layout = compute_x_axis_layout(
            data=df,
            time_var="t",
            x_col=None,
            boundaries=FlatBoundaries(positions),
            title_text="t",
        )
        assert layout.cell_band is not None
        assert len(layout.cell_band.labels) == len(layout.cell_band.midpoints)
        assert len(layout.cell_band.midpoints) == n_blocks

    def test_band_present_only_when_boundaries_and_integer_x(self):
        n = 100
        df = _make_df(n, rsg_values=["c1", "c2"])
        positions = (50,)

        # Integer-position + boundaries → band present
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=FlatBoundaries(positions), title_text="t",
        )
        assert layout.has_cell_band

        # Categorical (x_col set) + boundaries → no band
        df_cat = pd.DataFrame({"t": list(range(n))})
        layout = compute_x_axis_layout(
            data=df_cat, time_var="t", x_col="t",
            boundaries=FlatBoundaries(positions), title_text="t",
        )
        assert not layout.has_cell_band

        # Integer-position, no boundaries → no band
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=None, title_text="t",
        )
        assert not layout.has_cell_band

    def test_bottom_margin_grows_when_band_present(self):
        n = 100
        df = _make_df(n, rsg_values=["c1", "c2"])
        no_band = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=None, title_text="t",
        )
        with_band = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=FlatBoundaries((50,)), title_text="t",
        )
        assert with_band.bottom_margin > no_band.bottom_margin

    def test_title_standoff_set_explicitly_when_band_present(self):
        """Bug B regression: title position is pinned when band present."""
        n = 100
        df = _make_df(n, rsg_values=["c1", "c2"])
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=FlatBoundaries((50,)), title_text="t",
        )
        assert layout.title.standoff is not None
        assert layout.title.standoff > 0

    def test_title_standoff_none_when_no_band(self):
        n = 100
        df = _make_df(n)
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=None, title_text="t",
        )
        assert layout.title.standoff is None


# ---------------------------------------------------------------------------
# compute_x_axis_layout — golden scenarios
# ---------------------------------------------------------------------------


class TestLayoutScenarios:
    def test_empty_data_returns_defaults(self):
        df = pd.DataFrame({"t": []})
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=None, title_text="Time",
        )
        assert layout.ticks.positions is None  # defer to Plotly
        assert not layout.has_cell_band
        assert layout.title.text == "Time"

    def test_missing_time_var_returns_defaults(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=None, title_text="Time",
        )
        assert layout.ticks.positions is None

    def test_small_categorical_returns_no_explicit_ticks(self):
        df = pd.DataFrame({"t": list(range(10))})
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col="t",
            boundaries=None, title_text="Time",
        )
        assert layout.ticks.positions is None  # Plotly handles small categorical

    def test_dense_categorical_returns_thinned_ticks(self):
        df = pd.DataFrame({"t": list(range(100))})
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col="t",
            boundaries=None, title_text="Time",
        )
        assert layout.ticks.is_categorical is True
        assert layout.ticks.positions is not None
        # tickvals are category values (the time values themselves)
        assert all(isinstance(p, int) for p in layout.ticks.positions)

    def test_integer_position_no_blocks_returns_single_tier(self):
        df = pd.DataFrame({"t": [10 * i for i in range(100)]})
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=None, title_text="Time",
        )
        assert layout.ticks.positions is not None
        assert layout.ticks.is_categorical is False
        assert not layout.has_cell_band

    def test_two_tier_with_band(self):
        """Reference scenario: 4 cells × 25 points each, time repeating per cell."""
        n = 100
        df = pd.DataFrame({
            "t": list(range(25)) * 4,  # repeating time
            "rsg": [f"cell_{i // 25}" for i in range(n)],
        })
        boundaries = FlatBoundaries((25, 50, 75))
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=boundaries, title_text="Time",
        )
        assert layout.has_cell_band
        assert layout.ticks.positions is not None
        # 4 ticks per block × 4 blocks = 16 positions
        assert len(layout.ticks.positions) == 16
        # All in range
        assert all(0 <= p < n for p in layout.ticks.positions)
        # 4 cells in band
        assert len(layout.cell_band.midpoints) == 4
        assert layout.cell_band.labels == ("cell_0", "cell_1", "cell_2", "cell_3")
        # Title pinned
        assert layout.title.standoff is not None


# ---------------------------------------------------------------------------
# apply_x_axis_layout — side effects
# ---------------------------------------------------------------------------


class TestApplyLayout:
    def test_apply_writes_ticks_and_title(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1, 2], y=[1, 2, 3]))
        layout = XAxisLayout(
            ticks=TickRow(positions=(0, 1, 2), labels=("a", "b", "c"), angle=0,
                          is_categorical=False),
            cell_band=None,
            title=AxisTitle(text="Time", standoff=None),
            bottom_margin=80,
        )
        apply_x_axis_layout(fig, layout)
        assert tuple(fig.layout.xaxis.tickvals) == (0, 1, 2)
        assert tuple(fig.layout.xaxis.ticktext) == ("a", "b", "c")
        assert fig.layout.xaxis.title.text == "Time"

    def test_apply_writes_band_annotations(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1, 2, 3], y=[1, 2, 3, 4]))
        layout = XAxisLayout(
            ticks=TickRow(positions=(0, 3), labels=("0", "3"), angle=0,
                          is_categorical=False),
            cell_band=CellBand(
                midpoints=(1, 2),
                labels=("c1", "c2"),
                stride=1,
                y_paper=-0.28,
            ),
            title=AxisTitle(text="Time", standoff=8),
            bottom_margin=140,
        )
        apply_x_axis_layout(fig, layout)
        annots = [a for a in fig.layout.annotations if a.yref == "paper"]
        assert len(annots) == 2
        assert fig.layout.xaxis.title.standoff == 8

    def test_apply_reserves_bottom_margin(self):
        fig = go.Figure()
        layout = XAxisLayout(
            ticks=TickRow(None, None, 0, False),
            cell_band=None,
            title=AxisTitle(text="Time", standoff=None),
            bottom_margin=140,
        )
        apply_x_axis_layout(fig, layout)
        assert fig.layout.margin.b == 140

    def test_apply_does_not_shrink_existing_margin(self):
        fig = go.Figure(layout=dict(margin=dict(b=200)))
        layout = XAxisLayout(
            ticks=TickRow(None, None, 0, False),
            cell_band=None,
            title=AxisTitle(text="Time", standoff=None),
            bottom_margin=100,
        )
        apply_x_axis_layout(fig, layout)
        assert fig.layout.margin.b == 200

    def test_apply_band_stride_skips_labels(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(50)), y=list(range(50))))
        layout = XAxisLayout(
            ticks=TickRow(None, None, 0, False),
            cell_band=CellBand(
                midpoints=tuple(range(50)),
                labels=tuple(f"c{i}" for i in range(50)),
                stride=2,
                y_paper=-0.28,
            ),
            title=AxisTitle(text="Time", standoff=8),
            bottom_margin=140,
        )
        apply_x_axis_layout(fig, layout)
        annots = [a for a in fig.layout.annotations if a.yref == "paper"]
        assert len(annots) == 25  # stride=2 → every other label

    def test_apply_show_title_false_suppresses_title(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1, 2], y=[1, 2, 3]))
        layout = XAxisLayout(
            ticks=TickRow(positions=(0, 1, 2), labels=("a", "b", "c"), angle=0,
                          is_categorical=False),
            cell_band=None,
            title=AxisTitle(text="Time", standoff=None),
            bottom_margin=80,
        )
        apply_x_axis_layout(fig, layout, show_title=False)
        assert fig.layout.xaxis.title.text == ""
        # Ticks still applied even when title hidden.
        assert tuple(fig.layout.xaxis.tickvals) == (0, 1, 2)


# ---------------------------------------------------------------------------
# Integration: parse → for_stratum → compute round-trip
# ---------------------------------------------------------------------------


class TestParseRoundTrip:
    def test_stratified_unpacks_to_flat_for_compute(self):
        """`focus()` will follow this exact path: parse raw dict → for_stratum →
        feed to compute_x_axis_layout. This test verifies the contract end-to-end."""
        raw = {
            "1_1": [{"position": 5}, {"position": 10}],
            "1_2": [{"position": 7}],
        }
        parsed = parse_lane_boundaries(raw)
        assert isinstance(parsed, StratifiedBoundaries)

        flat = parsed.for_stratum("1_2")
        assert isinstance(flat, FlatBoundaries)
        assert flat.positions == (7,)

        df = pd.DataFrame({
            "t": list(range(20)),
            "rsg": ["1_2"] * 20,
        })
        layout = compute_x_axis_layout(
            data=df, time_var="t", x_col=None,
            boundaries=flat, title_text="Time",
        )
        # Boundary at position 7 splits into 2 blocks → band rendered
        assert layout.has_cell_band
        assert len(layout.cell_band.midpoints) == 2
