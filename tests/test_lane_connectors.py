"""On a lane chart the connecting line breaks at every lane boundary (#121).

One subgroup's last point used to be joined to the next subgroup's first point, so every
boundary showed a steep rise or drop that was only the sort order. The line is now its own
trace with a gap at each boundary; the markers trace keeps hover, customdata and legend.
"""

import numpy as np
import pandas as pd

import processbehavior as pb


def _lane_study(n_lanes=3, T=4):
    rng = np.random.default_rng(121)
    rows = []
    for k in range(n_lanes):
        base = 100 * (k + 1)
        for t in range(1, T + 1):
            rows.append({'unit': f'U{k + 1}', 't': t, 'y': base + t + rng.normal(0, 0.1)})
    return pb.formulate(pd.DataFrame(rows), response='y', factors=['unit'], time='t')


def _traces(fig):
    return {t.name: t for t in fig.data}


def test_lane_chart_draws_the_line_with_one_gap_per_boundary():
    fig = _lane_study(n_lanes=3, T=4).execute(chart='X', by=[], companion=True).plot(chart='X')
    traces = _traces(fig)
    line = traces['X (lanes)']
    markers = traces['X']
    assert line.mode == 'lines' and markers.mode == 'markers'
    ys = list(line.y)
    assert ys.count(None) == 2, 'three lanes -> two boundaries -> two gaps'
    assert len([y for y in ys if y is not None]) == 12 and len(markers.y) == 12
    # The gaps sit exactly at the lane starts (positions 4 and 8).
    assert [i for i, y in enumerate(ys) if y is None] == [4, 9]


def test_lines_never_join_two_lanes():
    """No drawn segment spans a boundary: every consecutive non-gap pair is within a lane."""
    fig = _lane_study(n_lanes=4, T=3).execute(chart='X', by=[], companion=True).plot(chart='X')
    line = _traces(fig)['X (lanes)']
    ys = list(line.y)
    # Lanes are at 100, 200, 300, 400 (+ small t); any segment crossing a lane would jump ~100.
    for a, b in zip(ys, ys[1:], strict=False):
        if a is not None and b is not None:
            assert abs(a - b) < 10, (a, b)


def test_markers_keep_hover_and_the_line_has_none():
    fig = _lane_study().execute(chart='X', by=[], companion=True).plot(chart='X')
    traces = _traces(fig)
    assert traces['X'].hovertemplate
    assert traces['X (lanes)'].hoverinfo == 'skip' and traces['X (lanes)'].showlegend is False


def test_single_series_chart_is_unchanged():
    """No lanes -> the original single lines+markers trace, no extra trace."""
    df = pd.DataFrame({'t': range(1, 13), 'y': np.arange(12, dtype=float)})
    fig = pb.formulate(df, response='y', time='t').execute(chart='X', companion=True).plot(chart='X')
    names = [t.name for t in fig.data]
    assert 'X (lanes)' not in names
    assert _traces(fig)['X'].mode == 'lines+markers'


def test_faceted_charts_are_unchanged():
    """by=['unit'] gives one panel per lane: no boundaries inside a panel, so no gaps."""
    fig = _lane_study().execute(chart='X', by=['unit'], companion=True).plot(chart='X', facet=True, ncols=3)
    assert not [t for t in fig.data if t.name and t.name.endswith('(lanes)')]
