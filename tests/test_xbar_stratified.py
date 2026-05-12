"""Tests for stratified Xbar/S charts (by=[time_var] with factors)."""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.plotting.plotter import Plotter


@pytest.fixture
def sds1_study():
    """PM SDS 1 study: 2 factors, time, replication."""
    df = pd.read_csv('validation/PBTESTDATABASE_T100.csv')
    pb = ProcessBehavior(df)
    return pb.formulate(
        response='PM SDS 1',
        factors=['FACTOR 1', 'FACTOR 2'],
        time='PRODUCTION TIME',
    )


class TestXbarStratified:
    """Stratified Xbar charts: by=[time_var] produces per-factor-combo charts."""

    def test_stratified_has_strata_key(self, sds1_study):
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        xbar = result.charts['Xbar']
        assert 'strata' in xbar
        assert xbar['metadata']['stratified'] is True

    def test_strata_count_matches_factor_combos(self, sds1_study):
        """4 levels of F1 × 2 levels of F2 = 8 combinations."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        xbar = result.charts['Xbar']
        assert len(xbar['strata']) == 8

    def test_reference_values_combo_1_2(self, sds1_study):
        """Validate against Tom's reference: combo 1_2."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        stats = result.charts['Xbar']['statistics']['1_2']
        assert stats['center'] == pytest.approx(238.615, abs=0.01)
        assert stats['lpl'] == pytest.approx(237.467, abs=0.01)
        assert stats['upl'] == pytest.approx(239.763, abs=0.01)

    def test_time_on_x_axis(self, sds1_study):
        """Each stratum has time points on x-axis."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        xbar = result.charts['Xbar']
        data = xbar['data']
        assert 'PRODUCTION TIME' in data.columns
        # Each stratum should have 100 time points
        for stratum in xbar['strata']:
            stratum_data = data[data['rsg'] == stratum]
            assert len(stratum_data) == 100

    def test_metadata_fields(self, sds1_study):
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        meta = result.charts['Xbar']['metadata']
        assert meta['chart_type'] == 'Xbar'
        assert meta['stratified'] is True
        assert meta['stratify_col'] == 'rsg'
        assert meta['stratify_by'] == ['rsg']
        assert meta['value_col'] == 'xbar'


class TestSChartStratified:
    """Companion S chart must also be stratified."""

    def test_companion_s_stratified(self, sds1_study):
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True
        )
        s_chart = result.charts['S']
        assert 'strata' in s_chart
        assert s_chart['metadata']['stratified'] is True

    def test_companion_s_matching_strata(self, sds1_study):
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True
        )
        xbar_strata = result.charts['Xbar']['strata']
        s_strata = result.charts['S']['strata']
        assert xbar_strata == s_strata

    def test_s_reference_values_combo_1_2(self, sds1_study):
        """Validate S chart statistics for combo 1_2."""
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True
        )
        stats = result.charts['S']['statistics']['1_2']
        assert stats['center'] == pytest.approx(0.80, abs=0.01)
        assert stats['lpl'] == pytest.approx(0.0, abs=0.01)
        assert stats['upl'] == pytest.approx(1.68, abs=0.01)


class TestDefaultBehaviorUnchanged:
    """Existing by=None and by=[factor] paths must not regress."""

    def test_by_none_no_strata(self, sds1_study):
        """Default by=None: Kt-level single chart, no stratification."""
        result = sds1_study.execute(chart='Xbar')
        xbar = result.charts['Xbar']
        assert 'strata' not in xbar
        assert xbar['metadata'].get('stratified') is None

    def test_by_factor_no_strata(self, sds1_study):
        """by=[FACTOR 1]: factor levels on x-axis, no stratification."""
        result = sds1_study.execute(chart='Xbar', by=['FACTOR 1'])
        xbar = result.charts['Xbar']
        assert 'strata' not in xbar
        assert xbar['metadata'].get('stratified') is None


class TestResidualNotStratified:
    """Residuals have factor effects removed — stratification is meaningless."""

    def test_r4_by_time_not_stratified(self, sds1_study):
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], value='R4'
        )
        xbar = result.charts['Xbar']
        assert 'strata' not in xbar
        assert xbar['metadata'].get('stratified') is None

    def test_r5_by_time_not_stratified(self, sds1_study):
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], value='R5'
        )
        xbar = result.charts['Xbar']
        assert 'strata' not in xbar
        assert xbar['metadata'].get('stratified') is None


class TestSharedYAxis:
    """Stratified charts must share y-axis range within each chart type."""

    def test_stratified_xbar_subplots_share_yaxis(self, sds1_study):
        """All Xbar facets should have identical y-axis range."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar').figure
        ranges = [
            fig.layout[k].range
            for k in fig.layout
            if k.startswith('yaxis') and fig.layout[k].range
        ]
        assert len(ranges) > 1
        assert all(r == ranges[0] for r in ranges), \
            "All Xbar subplots should share y-axis range"

    def test_stratified_xbar_autorange_disabled(self, sds1_study):
        """autorange must be False when shared range is set."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar').figure
        for k in fig.layout:
            if k.startswith('yaxis') and fig.layout[k].range:
                assert fig.layout[k].autorange is False

    def test_companion_xbar_s_share_within_type(self, sds1_study):
        """Companion: Xbar facets share one range, S facets share another."""
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True,
        )
        plotter = Plotter(result)
        fig = plotter.plot().figure
        all_ranges = []
        for k in sorted(fig.layout):
            if k.startswith('yaxis') and fig.layout[k].range:
                all_ranges.append(tuple(fig.layout[k].range))
        unique_ranges = set(all_ranges)
        assert len(unique_ranges) == 2, \
            "Xbar and S should have different y-axis ranges"
        for r in unique_ranges:
            count = all_ranges.count(r)
            assert count > 1, f"Range {r} should appear in multiple subplots"

    def test_companion_autorange_disabled(self, sds1_study):
        """All companion subplots must have autorange=False."""
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True,
        )
        plotter = Plotter(result)
        fig = plotter.plot().figure
        for k in fig.layout:
            if k.startswith('yaxis') and fig.layout[k].range:
                assert fig.layout[k].autorange is False

    def test_companion_subplot_minimum_height(self, sds1_study):
        """Each subplot in companion chart must get reasonable vertical space."""
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True,
        )
        plotter = Plotter(result)
        fig = plotter.plot().figure
        # Collect all y-axis domains
        domains = []
        for k in sorted(fig.layout):
            if k.startswith('yaxis'):
                domain = fig.layout[k].domain
                if domain:
                    domains.append(domain)
        assert len(domains) > 0, "Should have y-axis domains"
        for domain in domains:
            height = domain[1] - domain[0]
            assert height >= 0.03, (
                f"Subplot domain height {height:.4f} is too small; "
                f"domain={domain}"
            )

    def test_companion_all_axes_have_range(self, sds1_study):
        """Every yaxis in companion figure must have an explicit range."""
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True,
        )
        plotter = Plotter(result)
        fig = plotter.plot().figure
        yaxis_keys = [k for k in fig.layout if k.startswith('yaxis')]
        assert len(yaxis_keys) > 0
        for k in yaxis_keys:
            assert fig.layout[k].range is not None, (
                f"{k} has no explicit range set"
            )

    def test_companion_ranges_match_standalone(self, sds1_study):
        """Per-type companion ranges should match standalone chart ranges."""
        companion_result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True,
        )
        standalone_result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'],
        )
        # Get standalone Xbar range
        standalone_fig = Plotter(standalone_result).plot(chart='Xbar').figure
        standalone_ranges = [
            standalone_fig.layout[k].range
            for k in standalone_fig.layout
            if k.startswith('yaxis') and standalone_fig.layout[k].range
        ]
        assert len(standalone_ranges) > 0
        standalone_xbar_range = standalone_ranges[0]

        # Get companion Xbar ranges
        companion_fig = Plotter(companion_result).plot().figure
        companion_ranges = []
        for k in sorted(companion_fig.layout):
            if k.startswith('yaxis') and companion_fig.layout[k].range:
                companion_ranges.append(tuple(companion_fig.layout[k].range))
        # Xbar range should appear in the companion
        assert tuple(standalone_xbar_range) in set(companion_ranges), (
            f"Standalone Xbar range {standalone_xbar_range} not found in "
            f"companion ranges {set(companion_ranges)}"
        )

    def test_non_shared_yaxis_keeps_autorange(self, sds1_study):
        """shared_yaxis=False should NOT set autorange=False."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar', shared_yaxis=False).figure
        for k in fig.layout:
            if k.startswith('yaxis'):
                assert fig.layout[k].autorange is not False


class TestLimitSummaryAnnotation:
    """Limit values should appear as a summary above each subplot."""

    def test_limit_summary_annotation_present(self, sds1_study):
        """Summary annotation text contains UPL, CL, LPL with pipes."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar').figure
        annotations = [a for a in fig.layout.annotations
                       if a.text and 'UPL' in a.text and '|' in a.text]
        assert len(annotations) > 0, "Should have limit summary annotations"
        for ann in annotations:
            assert 'CL' in ann.text
            assert 'LPL' in ann.text
            assert ann.xanchor == 'right'
            assert ann.yanchor == 'top'

    def test_no_right_side_limit_annotations(self, sds1_study):
        """No annotations with xanchor='left' at x=1 (old per-line labels)."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar').figure
        right_side = [a for a in fig.layout.annotations
                      if getattr(a, 'xanchor', None) == 'left'
                      and getattr(a, 'x', None) == 1]
        assert len(right_side) == 0, \
            f"Found {len(right_side)} old right-side limit annotations"

    def test_limit_summary_hidden_when_values_off(self, sds1_study):
        """show_limit_values=False suppresses the summary annotation."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar', show_limit_values=False).figure
        annotations = [a for a in fig.layout.annotations
                       if a.text and 'UPL' in a.text and '|' in a.text]
        assert len(annotations) == 0

    def test_limit_summary_hidden_when_limits_off(self, sds1_study):
        """show_limits=False suppresses lines AND summary."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar', show_limits=False).figure
        annotations = [a for a in fig.layout.annotations
                       if a.text and 'UPL' in a.text and '|' in a.text]
        assert len(annotations) == 0


class TestFacetedAxisLabels:
    """Y-axis title only on leftmost column, x-axis title only on bottom row."""

    def test_faceted_ylabel_only_leftmost_column(self, sds1_study):
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar', ncols=2).figure
        # Check yaxis titles: only odd-indexed subplots (col 1) should have title
        n_charts = len([k for k in fig.layout if k.startswith('yaxis')])
        for idx in range(n_charts):
            c = idx % 2 + 1
            axis_key = 'yaxis' if idx == 0 else f'yaxis{idx + 1}'
            title = fig.layout[axis_key].title
            if c == 1:
                assert title and title.text, \
                    f"{axis_key} (col 1) should have y-axis title"
            else:
                assert not (title and title.text), \
                    f"{axis_key} (col {c}) should NOT have y-axis title"

    def test_faceted_xlabel_only_bottom_row(self, sds1_study):
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        fig = plotter.plot(chart='Xbar', ncols=2).figure
        n_charts = 8  # 8 strata
        ncols = 2
        nrows = 4
        for idx in range(n_charts):
            r = idx // ncols + 1
            axis_key = 'xaxis' if idx == 0 else f'xaxis{idx + 1}'
            title = fig.layout[axis_key].title
            is_bottom = (r == nrows) or (idx + ncols >= n_charts)
            if is_bottom:
                assert title and title.text, \
                    f"{axis_key} (row {r}) should have x-axis title"
            else:
                assert not (title and title.text), \
                    f"{axis_key} (row {r}) should NOT have x-axis title"


class TestCompanionPairedLayout:
    """Stratified companion charts should pair Xbar/S per stratum."""

    def test_companion_paired_layout_ordering(self, sds1_study):
        """Keys should alternate Xbar/S per stratum."""
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True,
        )
        plotter = Plotter(result)
        plotter.plot()
        # Access the resolved charts via internal path
        plotter._resolve_charts('Xbar', False)
        # Merge companion
        companion = plotter._resolve_charts(None, False)
        reordered, forced_ncols = plotter._reorder_companion_pairs(companion)
        assert forced_ncols == 2
        keys = list(reordered.keys())
        # Should alternate: Xbar_X, S_X, Xbar_Y, S_Y, ...
        for i in range(0, len(keys), 2):
            assert keys[i].startswith('Xbar'), \
                f"Even index {i} should be Xbar, got {keys[i]}"
            if i + 1 < len(keys):
                assert keys[i + 1].startswith('S'), \
                    f"Odd index {i+1} should be S, got {keys[i+1]}"

    def test_non_companion_stratified_unchanged(self, sds1_study):
        """Xbar-only stratified (no companion) should not be reordered."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        plotter = Plotter(result)
        charts = plotter._resolve_charts('Xbar', False)
        reordered, forced_ncols = plotter._reorder_companion_pairs(charts)
        assert forced_ncols is None
        assert list(reordered.keys()) == list(charts.keys())

    def test_simple_companion_unchanged(self, sds1_study):
        """Non-stratified companion (Xbar+S, 2 charts) is not reordered."""
        result = sds1_study.execute(chart='Xbar', companion=True)
        plotter = Plotter(result)
        charts = plotter._resolve_charts(None, False)
        reordered, forced_ncols = plotter._reorder_companion_pairs(charts)
        assert forced_ncols is None
