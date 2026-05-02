"""Visualize the 6 Sampling Design States as 100×100 archetype grids.

Each grid is an abstract schematic showing the N_kt pattern that defines the SDS.
Colors:
  - Dark blue (#1B4F72):  N_kt >= 2  (replicated)
  - Light blue (#85C1E9): N_kt = 1   (singleton)
  - Light grey (#D5D8DC): N_kt = 0   (empty / missing)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROWS, COLS = 100, 100
SEED = 42

# Cell value encoding
EMPTY = 0       # N_kt = 0
SINGLETON = 1   # N_kt = 1
REPLICATED = 2  # N_kt >= 2

# Colors
COLOR_EMPTY = "#D5D8DC"
COLOR_SINGLETON = "#85C1E9"
COLOR_REPLICATED = "#1B4F72"

COLORSCALE = [
    [0.0, COLOR_EMPTY],
    [0.25, COLOR_EMPTY],
    [0.25, COLOR_SINGLETON],
    [0.75, COLOR_SINGLETON],
    [0.75, COLOR_REPLICATED],
    [1.0, COLOR_REPLICATED],
]


def make_sds1():
    """SDS 1: Full replication — all cells N_kt >= 2."""
    return np.full((ROWS, COLS), REPLICATED)


def make_sds2():
    """SDS 2: No replication — all cells N_kt = 1."""
    return np.full((ROWS, COLS), SINGLETON)


def make_sds3(rng):
    """SDS 3: Partial replication — mix of N_kt >= 2 and N_kt = 1, no empties.

    Pattern: Most cells replicated, but certain factor-level bands have
    singletons (mimics real-world under-replication in specific conditions).
    """
    grid = np.full((ROWS, COLS), REPLICATED)

    # Bands of factor levels that got singleton observations at certain times
    # ~30% singleton overall
    singleton_rows = np.concatenate([
        np.arange(15, 25),   # factor levels 15-24
        np.arange(50, 58),   # factor levels 50-57
        np.arange(78, 88),   # factor levels 78-87
    ])
    grid[np.ix_(singleton_rows, np.arange(COLS))] = SINGLETON

    # Some time-period columns also had reduced replication
    singleton_cols = np.concatenate([
        np.arange(5, 12),
        np.arange(40, 48),
        np.arange(85, 92),
    ])
    grid[np.ix_(np.arange(ROWS), singleton_cols)] = SINGLETON

    # But restore some replicated cells in the overlap for texture
    for r in range(0, ROWS, 7):
        for c in singleton_cols:
            if rng.random() < 0.3:
                grid[r, c] = REPLICATED

    return grid


def make_sds4(rng):
    """SDS 4: Incomplete with singletons — has empty, singleton, AND replicated.

    Pattern: A diagonal band of missing cells (production line down during
    certain shifts) plus scattered singletons in transition zones.
    """
    grid = np.full((ROWS, COLS), REPLICATED)

    # Missing diagonal band — production outage swept across factor levels over time
    for col in range(COLS):
        center = int(20 + 60 * (col / COLS))
        width = 8 + int(4 * np.sin(col * 0.1))
        for row in range(max(0, center - width), min(ROWS, center + width)):
            grid[row, col] = EMPTY

    # Transition zone: cells adjacent to empties often got only 1 observation
    empty_mask = grid == EMPTY
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(empty_mask, dr, axis=0), dc, axis=1)
            transition = shifted & (grid == REPLICATED)
            # 60% of transition cells become singletons
            transition_coords = np.argwhere(transition)
            for r, c in transition_coords:
                if rng.random() < 0.6:
                    grid[r, c] = SINGLETON

    # Some additional scattered singletons in the replicated zone
    replicated_coords = np.argwhere(grid == REPLICATED)
    n_scatter = len(replicated_coords) // 20
    scatter_idx = rng.choice(len(replicated_coords), size=n_scatter, replace=False)
    for idx in scatter_idx:
        r, c = replicated_coords[idx]
        grid[r, c] = SINGLETON

    return grid


def make_sds5(rng):
    """SDS 5: Incomplete, no singletons — has empty and replicated only.

    Pattern: Entire factor-level × time-period blocks missing (equipment
    not available for certain conditions), but all collected data is properly
    replicated. Clean gaps, no degraded observations.
    """
    grid = np.full((ROWS, COLS), REPLICATED)

    # Block 1: entire factor levels 60-80 missing during time 0-30
    grid[60:80, 0:30] = EMPTY

    # Block 2: factor levels 10-25 missing during time 55-80
    grid[10:25, 55:80] = EMPTY

    # Block 3: factor levels 85-100 missing during time 35-55
    grid[85:100, 35:55] = EMPTY

    # Scattered missing cells — individual equipment failures
    replicated_coords = np.argwhere(grid == REPLICATED)
    n_scatter = len(replicated_coords) // 15
    scatter_idx = rng.choice(len(replicated_coords), size=n_scatter, replace=False)
    for idx in scatter_idx:
        r, c = replicated_coords[idx]
        grid[r, c] = EMPTY

    return grid


def make_sds6(rng):
    """SDS 6: Incomplete, no replication — has empty and singleton only.

    Pattern: Similar block-missing structure to SDS 5, but all collected
    observations are singletons (e.g., expensive destructive testing where
    you can only test each cell once, and some cells couldn't be tested at all).
    """
    grid = np.full((ROWS, COLS), SINGLETON)

    # Block 1: upper-right corner missing
    grid[0:20, 70:100] = EMPTY

    # Block 2: middle-left block missing
    grid[40:65, 0:25] = EMPTY

    # Block 3: lower band missing
    grid[90:100, 20:70] = EMPTY

    # Scattered missing cells
    singleton_coords = np.argwhere(grid == SINGLETON)
    n_scatter = len(singleton_coords) // 12
    scatter_idx = rng.choice(len(singleton_coords), size=n_scatter, replace=False)
    for idx in scatter_idx:
        r, c = singleton_coords[idx]
        grid[r, c] = EMPTY

    return grid


def build_figure():
    rng = np.random.default_rng(SEED)

    sds_grids = {
        1: make_sds1(),
        2: make_sds2(),
        3: make_sds3(rng),
        4: make_sds4(rng),
        5: make_sds5(rng),
        6: make_sds6(rng),
    }

    sds_titles = {
        1: "SDS 1 — Full Replication",
        2: "SDS 2 — No Replication",
        3: "SDS 3 — Partial Replication",
        4: "SDS 4 — Incomplete + Singletons",
        5: "SDS 5 — Incomplete, No Singletons",
        6: "SDS 6 — Incomplete, No Replication",
    }

    sds_subtitles = {
        1: "All cells N<sub>kt</sub> ≥ 2",
        2: "All cells N<sub>kt</sub> = 1",
        3: "Mix of N<sub>kt</sub> ≥ 2 and N<sub>kt</sub> = 1",
        4: "N<sub>kt</sub> = 0, 1, and ≥ 2",
        5: "N<sub>kt</sub> = 0 and ≥ 2 only",
        6: "N<sub>kt</sub> = 0 and 1 only",
    }

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[sds_titles[i] for i in range(1, 7)],
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    for idx, sds in enumerate(range(1, 7)):
        row = idx // 3 + 1
        col = idx % 3 + 1
        grid = sds_grids[sds]

        # Compute percentages for hover/annotation
        total = ROWS * COLS
        n_empty = int(np.sum(grid == EMPTY))
        n_singleton = int(np.sum(grid == SINGLETON))
        n_replicated = int(np.sum(grid == REPLICATED))

        fig.add_trace(
            go.Heatmap(
                z=grid,
                zmin=0,
                zmax=2,
                colorscale=COLORSCALE,
                showscale=False,
                hovertemplate=(
                    "Row %{y}, Col %{x}<br>"
                    "N<sub>kt</sub>: %{customdata}<extra></extra>"
                ),
                customdata=np.where(
                    grid == EMPTY, "0 (empty)",
                    np.where(grid == SINGLETON, "1 (singleton)", "≥2 (replicated)")
                ),
            ),
            row=row, col=col,
        )

        # Build composition label
        parts = []
        if n_replicated > 0:
            parts.append(f'<span style="color:{COLOR_REPLICATED}">■</span> N≥2: {n_replicated/total:.0%}')
        if n_singleton > 0:
            parts.append(f'<span style="color:{COLOR_SINGLETON}">■</span> N=1: {n_singleton/total:.0%}')
        if n_empty > 0:
            parts.append(f'<span style="color:{COLOR_EMPTY}">■</span> N=0: {n_empty/total:.0%}')

        # Add subtitle annotation below each grid title
        fig.add_annotation(
            text=sds_subtitles[sds] + "<br>" + "  ".join(parts),
            xref=f"x{idx+1 if idx > 0 else ''} domain",
            yref=f"y{idx+1 if idx > 0 else ''} domain",
            x=0.5, y=-0.18,
            showarrow=False,
            font=dict(size=11, color="#555"),
            align="center",
        )

    # Style axes — hide ticks, add thin border
    for i in range(1, 7):
        axis_suffix = "" if i == 1 else str(i)
        fig.update_layout(**{
            f"xaxis{axis_suffix}": dict(
                showticklabels=False, showgrid=False, zeroline=False,
                scaleanchor=f"y{axis_suffix}", scaleratio=1,
                constrain="domain",
            ),
            f"yaxis{axis_suffix}": dict(
                showticklabels=False, showgrid=False, zeroline=False,
                autorange="reversed",
                constrain="domain",
            ),
        })

    fig.update_layout(
        title=dict(
            text=(
                "Sampling Design States — Cell Occupancy Archetypes<br>"
                '<span style="font-size:13px;color:#666">'
                "Each 100×100 grid shows the N<sub>kt</sub> pattern that defines the design state. "
                "Wheeler/Bishop Table 1 classification."
                "</span>"
            ),
            x=0.5,
            font=dict(size=20),
        ),
        height=900,
        width=1300,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=120, b=80, l=40, r=40),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )

    # Style subplot titles
    for annotation in fig.layout.annotations:
        if annotation.text in sds_titles.values():
            annotation.font = dict(size=14, color="#1B4F72", family="Inter, Segoe UI, sans-serif")

    return fig


if __name__ == "__main__":
    fig = build_figure()
    output_path = "sds_design_states.html"
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "scrollZoom": False},
    )
    print(f"Written to {output_path}")
