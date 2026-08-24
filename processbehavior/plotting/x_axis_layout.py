"""X-axis tick and cell-band layout for control chart panels.

Owns one cohesive subsystem: given a chart's data and (optional) lane
boundaries, decide where tick labels go, whether a secondary cell-label
band is needed, where the axis title sits, and how much bottom margin
to reserve — all in one place, computed purely, applied to a Plotly
figure as one isolated side effect.

Replaces the prior `_apply_time_tick_labels` and its 5 sibling helpers
on `Plotter`.

Design notes
------------
- `lane_boundaries` metadata can arrive in two shapes (legacy):
  `list[dict]` (flat — single chart's positions) or
  `dict[stratum, list[dict]]` (stratified). Callers must unpack via
  `parse_lane_boundaries(...)` before consumption. `FlatBoundaries`
  and `StratifiedBoundaries` are the typed wrappers; the layout
  function accepts only flat (or None).
- Title and band positions are chosen together so the title always sits
  above the band, by construction. No more silent overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Typed boundary objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatBoundaries:
    """Boundary positions within one chart's data."""

    positions: tuple[int, ...]

    def block_edges(self, n: int) -> tuple[int, ...]:
        """Sorted edges `(0, b1, ..., n)` with positions clipped to `(0, n)`."""
        valid = [p for p in self.positions if 0 < p < n]
        return tuple(sorted({0, *valid, n}))


@dataclass(frozen=True)
class StratifiedBoundaries:
    """Per-stratum boundaries. Caller must unpack via `for_stratum()`."""

    per_stratum: dict[str, FlatBoundaries]

    def for_stratum(self, stratum: str) -> FlatBoundaries:
        return self.per_stratum.get(str(stratum), FlatBoundaries(()))


def parse_lane_boundaries(raw: Any) -> FlatBoundaries | StratifiedBoundaries | None:
    """Normalize legacy raw metadata into a typed boundary object.

    Accepts ``list[dict]`` (flat), ``dict[stratum, list[dict]]`` (stratified),
    or ``None``. Returns the corresponding typed wrapper, or ``None`` for
    missing/empty input.

    Each boundary dict must have at minimum a ``position`` key (int row
    index). Other keys (``label``, etc.) are ignored here — cell labels for
    the band are computed from the chart's `rsg` column at consumption time.
    """
    if not raw:
        return None
    if isinstance(raw, list):
        return FlatBoundaries(tuple(int(b["position"]) for b in raw))
    if isinstance(raw, dict):
        per_stratum = {
            str(k): FlatBoundaries(tuple(int(b["position"]) for b in v))
            for k, v in raw.items()
        }
        return StratifiedBoundaries(per_stratum)
    return None


# ---------------------------------------------------------------------------
# Layout data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickRow:
    """Primary x-axis tick row.

    `positions` and `labels` are parallel. When `positions is None`, the
    caller leaves Plotly's default tick generation in place (e.g. small
    categorical axes).
    """

    positions: tuple[int, ...] | None
    labels: tuple[str, ...] | None
    angle: int  # 0 (horizontal) or -45 (rotated)
    is_categorical: bool  # if True, positions are category values not row indices


@dataclass(frozen=True)
class CellBand:
    """Secondary tier: one label per inter-boundary block.

    `y_paper` is in paper coords for single charts, y-domain coords for
    faceted panels — the apply step picks the right yref accordingly.
    """

    midpoints: tuple[int, ...]
    labels: tuple[str, ...]
    stride: int  # show every Nth label (crowd-thinning for >25 cells)
    y_paper: float  # paper-coord (single) or y-domain (faceted) y position


@dataclass(frozen=True)
class AxisTitle:
    """X-axis title text and explicit standoff.

    `standoff is None` means defer to Plotly's default. An explicit value
    pins the title close to the tick row so the cell band fits below.
    """

    text: str
    standoff: int | None


@dataclass(frozen=True)
class XAxisLayout:
    """Complete x-axis description for one chart panel.

    Invariants (enforced by `compute_x_axis_layout`):

    - When `ticks.positions` is not None, every position is in `[0, n)`.
    - `cell_band` is present only when blocks exist AND x is integer-position.
    - When both `cell_band` and an explicit `title.standoff` are set, the
      band's y position sits below the title's pixel baseline.
    - When `cell_band` is present, `bottom_margin` covers title + band.
    """

    ticks: TickRow
    cell_band: CellBand | None
    title: AxisTitle
    bottom_margin: int  # pixels

    @property
    def has_cell_band(self) -> bool:
        return self.cell_band is not None


# ---------------------------------------------------------------------------
# Internal helpers (pure)
# ---------------------------------------------------------------------------


def _per_block_positions(
    block_edges: tuple[int, ...], per_block_ticks: int
) -> list[int]:
    """Pick `per_block_ticks` evenly-spaced positions within each block.

    Each block contributes first, last, and `per_block_ticks - 2` interior
    indices. Blocks smaller than `per_block_ticks` contribute every position.
    """
    positions: list[int] = []
    for i in range(len(block_edges) - 1):
        start = block_edges[i]
        end = block_edges[i + 1]
        block_n = end - start
        if block_n <= 0:
            continue
        if block_n <= per_block_ticks:
            block_positions = list(range(start, end))
        else:
            step = (block_n - 1) / (per_block_ticks - 1)
            block_positions = [
                int(round(start + step * k)) for k in range(per_block_ticks)
            ]
            block_positions[-1] = end - 1  # snap last to block end
            block_positions = sorted(set(block_positions))
        positions.extend(block_positions)
    return positions


def _thinned_positions(n: int, max_ticks: int) -> list[int]:
    """Evenly-spaced tick positions across `[0, n)` with at most `max_ticks`."""
    if n <= max_ticks:
        return list(range(n))
    step = max(2, (n - 1) // (max_ticks - 1))
    return sorted({0, n - 1} | set(range(0, n, step)))


def _thinned_per_block(
    block_edges: tuple[int, ...], n: int, max_ticks: int
) -> list[int]:
    """Distribute tick budget proportionally across blocks (min 2/block)."""
    n_blocks = len(block_edges) - 1
    if n_blocks <= 0:
        return _thinned_positions(n, max_ticks)
    ticks_per_block = max(2, max_ticks // n_blocks)
    positions: set[int] = set()
    for i in range(n_blocks):
        start = block_edges[i]
        end = block_edges[i + 1]
        block_n = end - start
        if block_n <= 0:
            continue
        if block_n <= ticks_per_block:
            positions.update(range(start, end))
        else:
            step = max(2, (block_n - 1) // (ticks_per_block - 1))
            block_positions = list(range(start, end, step))
            if end - 1 not in block_positions:
                block_positions.append(end - 1)
            positions.update(block_positions)
    return sorted(positions)


def _adaptive_angle(n_ticks: int, max_label_len: int, threshold: int) -> int:
    """0 (horizontal) or -45 (rotated) based on total label footprint."""
    return -45 if (n_ticks * max_label_len) > threshold else 0


def _cell_labels_from_data(
    data: pd.DataFrame, block_edges: tuple[int, ...]
) -> tuple[str, ...]:
    """Read one label per block from the `rsg` column at block start.

    Falls back to `cell N` when the column is missing.
    """
    col = "rsg" if "rsg" in data.columns else None
    labels: list[str] = []
    for i in range(len(block_edges) - 1):
        start = block_edges[i]
        if col is not None and start < len(data):
            labels.append(str(data.iloc[start][col]))
        else:
            labels.append(f"cell {i + 1}")
    return tuple(labels)


def _choose_band_geometry(is_faceted: bool) -> tuple[int, float, int]:
    """Pick `(title_standoff_px, band_y, bottom_margin_px)` for a band layout.

    Title standoff pins the title near the tick row so the band has room
    below it. Band y is in paper coords (single) or y-domain (faceted).
    """
    if is_faceted:
        return (5, -0.30, 160)
    return (8, -0.28, 140)


def _band_stride(n_labels: int, max_visible: int = 25) -> int:
    """Show every Nth label when there are too many blocks."""
    if n_labels <= max_visible:
        return 1
    return max(1, n_labels // max_visible + (1 if n_labels % max_visible else 0))


# ---------------------------------------------------------------------------
# Public: pure layout computation
# ---------------------------------------------------------------------------


def compute_x_axis_layout(
    *,
    data: pd.DataFrame,
    time_var: str | None,
    x_col: str | None,
    boundaries: FlatBoundaries | None,
    title_text: str,
    is_faceted: bool = False,
    max_ticks: int = 20,
    per_block_ticks: int = 4,
) -> XAxisLayout:
    """Decide ticks + (optional) cell band + axis title for one chart panel.

    Pure: no Figure mutation. Returns an `XAxisLayout` describing everything
    the caller needs to apply to the figure.

    Parameters
    ----------
    data : DataFrame
        The chart's data slice (already filtered to the panel/stratum).
    time_var : str | None
        Name of the time column to source tick labels from. If None or
        absent from `data`, ticks fall back to Plotly defaults.
    x_col : str | None
        The column the trace's x-axis uses, if any. `None` indicates an
        integer-position axis (time repeats across the data). When set,
        the axis is categorical and tick positions are category values.
    boundaries : FlatBoundaries | None
        Per-block positions for this chart. Must be flat (unpacked from
        the stratified dict by the caller).
    title_text : str
        X-axis title text. Always rendered; standoff is chosen here.
    is_faceted : bool
        Whether this layout is for a panel inside a faceted figure.
    """
    n = len(data)
    is_categorical = x_col is not None

    # Default title (no explicit standoff) when no cell band is needed.
    default_title = AxisTitle(text=title_text, standoff=None)
    default_margin = 100 if is_faceted else 80

    # Mode 1: time_var missing or empty data → Plotly defaults.
    if not time_var or time_var not in data.columns or n == 0:
        return XAxisLayout(
            ticks=TickRow(None, None, 0, is_categorical),
            cell_band=None,
            title=default_title,
            bottom_margin=default_margin,
        )

    # Mode 2: small categorical → defer to Plotly's tick generation.
    if is_categorical and n <= max_ticks:
        return XAxisLayout(
            ticks=TickRow(None, None, 0, True),
            cell_band=None,
            title=default_title,
            bottom_margin=default_margin,
        )

    block_edges = boundaries.block_edges(n) if boundaries else ()
    has_blocks = len(block_edges) > 2  # need at least 2 blocks for a band

    # Mode 3: integer-position axis with internal block structure →
    # two-tier (per-block ticks + cell band).
    if has_blocks and not is_categorical:
        raw_positions = _per_block_positions(block_edges, per_block_ticks)
        positions = tuple(p for p in raw_positions if 0 <= p < n)
        labels = tuple(data[time_var].iloc[list(positions)].astype(str).tolist())
        max_label_len = max((len(lbl) for lbl in labels), default=1)
        angle = _adaptive_angle(len(positions), max_label_len, threshold=100)

        cell_labels = _cell_labels_from_data(data, block_edges)
        midpoints = tuple(
            (block_edges[i] + block_edges[i + 1]) // 2
            for i in range(len(block_edges) - 1)
        )
        stride = _band_stride(len(midpoints))

        standoff_px, band_y, bottom_margin = _choose_band_geometry(is_faceted)

        return XAxisLayout(
            ticks=TickRow(positions, labels, angle, is_categorical=False),
            cell_band=CellBand(
                midpoints=midpoints,
                labels=cell_labels,
                stride=stride,
                y_paper=band_y,
            ),
            title=AxisTitle(text=title_text, standoff=standoff_px),
            bottom_margin=bottom_margin,
        )

    # Mode 4: thinned single-tier (categorical-large or integer-no-blocks).
    if has_blocks and not is_categorical:
        # (Unreachable in current logic — Mode 3 covered this. Kept for
        # clarity that the per-block thinning path is intentionally fused
        # with the two-tier path; the standalone fallback is below.)
        raw_positions = _thinned_per_block(block_edges, n, max_ticks)
    else:
        raw_positions = _thinned_positions(n, max_ticks)

    positions = tuple(p for p in raw_positions if 0 <= p < n)

    if is_categorical and x_col is not None and x_col in data.columns:
        # Categorical axis: tickvals are the actual category values
        # (not row indices), labels parallel them.
        tick_values = tuple(data[x_col].iloc[list(positions)].tolist())
        labels = tuple(data[time_var].iloc[list(positions)].astype(str).tolist())
        max_label_len = max((len(lbl) for lbl in labels), default=1)
        angle = _adaptive_angle(len(positions), max_label_len, threshold=80)
        # Store category values in `positions` slot for categorical mode.
        return XAxisLayout(
            ticks=TickRow(
                positions=tick_values,  # type: ignore[arg-type]
                labels=labels,
                angle=angle,
                is_categorical=True,
            ),
            cell_band=None,
            title=default_title,
            bottom_margin=default_margin,
        )

    # Integer-position axis, no band.
    labels = tuple(data[time_var].iloc[list(positions)].astype(str).tolist())
    max_label_len = max((len(lbl) for lbl in labels), default=1)
    angle = _adaptive_angle(len(positions), max_label_len, threshold=80)
    return XAxisLayout(
        ticks=TickRow(positions, labels, angle, is_categorical=False),
        cell_band=None,
        title=default_title,
        bottom_margin=default_margin,
    )


# ---------------------------------------------------------------------------
# Public: side-effect application
# ---------------------------------------------------------------------------


def apply_x_axis_layout(
    fig: go.Figure,
    layout: XAxisLayout,
    *,
    row: int | None = None,
    col: int | None = None,
    ncols: int | None = None,
    font_size: int = 12,
    show_title: bool = True,
) -> None:
    """Write a computed layout onto the figure. The only place this module
    touches a Plotly object.

    `show_title=False` suppresses the axis title (used by faceted panels that
    aren't on the bottom row and have no cell band). The cell band still
    renders if present.
    """
    subplot_kwargs: dict = {}
    if row is not None and col is not None:
        subplot_kwargs = {"row": row, "col": col}

    # 1. Ticks + axis title in one call.
    update_kwargs: dict[str, Any] = {"automargin": True}
    if layout.ticks.positions is not None and layout.ticks.labels is not None:
        update_kwargs["tickvals"] = list(layout.ticks.positions)
        update_kwargs["ticktext"] = list(layout.ticks.labels)
        update_kwargs["tickangle"] = layout.ticks.angle

    if show_title:
        title_kwargs: dict[str, Any] = {"text": layout.title.text}
        if layout.title.standoff is not None:
            title_kwargs["standoff"] = layout.title.standoff
        update_kwargs["title"] = title_kwargs
    else:
        update_kwargs["title"] = {"text": ""}

    fig.update_xaxes(**update_kwargs, **subplot_kwargs)

    # 2. Cell-band annotations.
    if layout.cell_band is not None:
        band = layout.cell_band
        if row is not None and col is not None:
            subplot_idx = (row - 1) * (ncols or 1) + col
            xref = "x" if subplot_idx == 1 else f"x{subplot_idx}"
            yref = "y domain" if subplot_idx == 1 else f"y{subplot_idx} domain"
        else:
            xref = "x"
            yref = "paper"

        for idx, (mid, lbl) in enumerate(zip(band.midpoints, band.labels, strict=False)):
            if idx % band.stride != 0:
                continue
            fig.add_annotation(
                x=mid,
                y=band.y_paper,
                xref=xref,
                yref=yref,
                text=f"<b>{lbl}</b>",
                showarrow=False,
                xanchor="center",
                yanchor="top",
                font=dict(size=font_size),
            )

    # 3. Reserve enough bottom margin.
    current = fig.layout.margin
    current_b = current.b if current and current.b else 0
    if current_b < layout.bottom_margin:
        fig.update_layout(margin=dict(b=layout.bottom_margin))
