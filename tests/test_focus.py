"""Tests for AnalysisResult.focus() — stratified chart drill-down.

Covers single-factor, multi-factor, and Histogram stratification,
plus error cases. Validates the strata/focus/rsg invariant:
every stratified chart has an rsg column, and focus() produces
single-stratum data with flat statistics.
"""

import pytest

from processbehavior import ProcessBehavior, ValidationError
from processbehavior.data_preparation import encode_rsg
from processbehavior.datasets import synthetic

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope='module')
def sds1_study_single_factor():
    """SDS1 study with single factor — ideal for single-factor focus tests."""
    df = synthetic.make_sds(1, K1=3, K2=1, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1'])


@pytest.fixture(scope='module')
def sds1_study_two_factor():
    """SDS1 study with two factors — for multi-factor focus tests."""
    df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return ProcessBehavior(df).formulate(response='y', time='time', factors=['factor 1', 'factor 2'])


@pytest.fixture(scope='module')
def non_stratified_xmr_result(sds1_study_single_factor):
    """Non-stratified XmR result (by=[]) — for non-stratified error tests."""
    return sds1_study_single_factor.execute(chart='X', by=[])


@pytest.fixture(scope='module')
def single_factor_xmr_result(sds1_study_single_factor):
    """Stratified XmR result with single factor."""
    return sds1_study_single_factor.execute(chart='X', by=['factor 1'])


@pytest.fixture(scope='module')
def two_factor_xmr_result(sds1_study_two_factor):
    """Stratified XmR result with two factors."""
    return sds1_study_two_factor.execute(chart='X', by=['factor 1', 'factor 2'])


@pytest.fixture(scope='module')
def single_factor_histogram_result(sds1_study_single_factor):
    """Stratified Histogram result with single factor."""
    return sds1_study_single_factor.execute(chart='Histogram', by=['factor 1'])


# =============================================================================
# Single-factor focus
# =============================================================================


class TestFocusSingleFactor:
    """focus() with single-factor stratification (by=['factor 1'])."""

    def test_focus_returns_filtered_data(self, single_factor_xmr_result):
        """Focused rsg column has exactly one unique value matching the stratum."""
        stratum = single_factor_xmr_result.strata[0]
        focused = single_factor_xmr_result.focus(stratum)
        data = focused.get_chart('X')
        assert 'rsg' in data.columns
        assert data['rsg'].nunique() == 1
        assert data['rsg'].iloc[0] == encode_rsg(stratum)

    def test_focus_row_count_matches_original(self, single_factor_xmr_result):
        """Focused row count == original filtered by rsg."""
        stratum = single_factor_xmr_result.strata[0]
        original = single_factor_xmr_result.get_chart('X')
        focused = single_factor_xmr_result.focus(stratum).get_chart('X')
        expected = len(original[original['rsg'] == encode_rsg(stratum)])
        assert len(focused) == expected

    def test_focus_statistics_are_flat(self, single_factor_xmr_result):
        """Focused stats are a flat dict, not nested."""
        stratum = single_factor_xmr_result.strata[0]
        stats = single_factor_xmr_result.focus(stratum).get_statistics('X')
        assert not any(isinstance(v, dict) for v in stats.values())

    def test_focus_metadata_not_stratified(self, single_factor_xmr_result):
        """Focused metadata has stratified=False."""
        stratum = single_factor_xmr_result.strata[0]
        focused = single_factor_xmr_result.focus(stratum)
        metadata = focused.charts['X']['metadata']
        assert metadata.get('stratified') is False

    def test_every_stratum_roundtrip(self, single_factor_xmr_result):
        """Every value from result.strata works with focus()."""
        for s in single_factor_xmr_result.strata:
            focused = single_factor_xmr_result.focus(s)
            assert len(focused.get_chart('X')) > 0


# =============================================================================
# Multi-factor focus
# =============================================================================


class TestFocusMultiFactor:
    """focus() with multi-factor stratification (by=['factor 1', 'factor 2'])."""

    def test_focus_with_multi_factor_stratum(self, two_factor_xmr_result):
        """Multi-factor strata are strings (encode_rsg format); focus accepts them."""
        for s in two_factor_xmr_result.strata:
            assert isinstance(s, str)  # multi-factor -> encoded string
        focused = two_factor_xmr_result.focus(two_factor_xmr_result.strata[0])
        assert len(focused.get_chart('X')) > 0

    def test_focus_every_stratum_roundtrip(self, two_factor_xmr_result):
        """Every stratum in multi-factor result works with focus()."""
        for s in two_factor_xmr_result.strata:
            focused = two_factor_xmr_result.focus(s)
            data = focused.get_chart('X')
            assert len(data) > 0
            assert data['rsg'].nunique() == 1


# =============================================================================
# Histogram focus
# =============================================================================


class TestFocusHistogram:
    """focus() on stratified Histogram."""

    def test_histogram_has_rsg_column(self, single_factor_histogram_result):
        """Stratified Histogram output includes rsg column."""
        data = single_factor_histogram_result.get_chart('Histogram')
        assert 'rsg' in data.columns

    def test_focus_histogram_single_factor(self, single_factor_histogram_result):
        """focus on Histogram by=['factor 1'] returns filtered data."""
        stratum = single_factor_histogram_result.strata[0]
        focused = single_factor_histogram_result.focus(stratum)
        data = focused.get_chart('Histogram')
        assert len(data) > 0

    def test_focus_histogram_no_data_leak(self, single_factor_histogram_result):
        """Focused Histogram data has exactly one stratum in rsg column."""
        stratum = single_factor_histogram_result.strata[0]
        focused = single_factor_histogram_result.focus(stratum)
        data = focused.get_chart('Histogram')
        assert data['rsg'].nunique() == 1
        assert data['rsg'].iloc[0] == encode_rsg(stratum)


# =============================================================================
# Error cases
# =============================================================================


class TestFocusErrors:
    """focus() error cases."""

    def test_focus_non_stratified_raises(self, non_stratified_xmr_result):
        """focus on non-stratified result (by=[]) raises ValidationError."""
        with pytest.raises(ValidationError, match='not stratified'):
            non_stratified_xmr_result.focus('anything')

    def test_focus_invalid_stratum_raises(self, single_factor_xmr_result):
        """focus with nonexistent stratum raises ValidationError."""
        with pytest.raises(ValidationError, match='not found'):
            single_factor_xmr_result.focus('nonexistent_stratum')

    def test_focus_encoding_equivalence(self, single_factor_xmr_result):
        """encode_rsg(raw_stratum) matches rsg values in chart data for every stratum."""
        data = single_factor_xmr_result.get_chart('X')
        for s in single_factor_xmr_result.strata:
            encoded = encode_rsg(s)
            assert encoded in data['rsg'].values, f'encode_rsg({s!r}) = {encoded!r} not found in rsg column'


# =============================================================================
# Stratified chart rsg invariant
# =============================================================================


class TestStratifiedChartsHaveRsg:
    """Every stratified chart type includes an rsg column — invariant test."""

    def test_all_stratified_charts_have_rsg(self, sds1_study_single_factor):
        """For a stratified run, every chart with strata has rsg column."""
        result = sds1_study_single_factor.execute(chart='X', by=['factor 1'], companion=True)
        for chart_name, chart_info in result.charts.items():
            if chart_info.get('strata'):
                data = chart_info['data']
                assert 'rsg' in data.columns, f"Stratified chart '{chart_name}' missing rsg column"

    def test_histogram_stratified_has_rsg(self, single_factor_histogram_result):
        """Stratified Histogram has rsg column."""
        chart_info = single_factor_histogram_result.charts['Histogram']
        assert chart_info.get('strata')
        assert 'rsg' in chart_info['data'].columns

    def test_focus_every_stratified_chart_no_leak(self, sds1_study_single_factor):
        """For each stratified chart, focus produces single-stratum data."""
        result = sds1_study_single_factor.execute(chart='X', by=['factor 1'], companion=True)
        stratum = result.strata[0]
        focused = result.focus(stratum)
        for chart_name, chart_info in focused.charts.items():
            data = chart_info['data']
            if 'rsg' in data.columns:
                assert data['rsg'].nunique() == 1, f"Chart '{chart_name}' has data from multiple strata after focus"


class TestFocusLaneBoundaries:
    """focus() must unpack the parent's stratified lane_boundaries dict into
    the focused stratum's flat list. Without this, the plotter consumed the
    full dict and silently fell back to the first stratum's positions —
    producing wrong x-axis tick labels on every non-first focused stratum."""

    def test_focused_metadata_carries_flat_lane_boundaries(self, sds1_study_single_factor):
        """Focused metadata's `lane_boundaries` is a flat list (or None), never a dict."""
        result = sds1_study_single_factor.execute(chart='X', by=['factor 1'], companion=True)
        for stratum in result.strata:
            focused = result.focus(stratum)
            for chart_name, chart_info in focused.charts.items():
                lb = chart_info.get('metadata', {}).get('lane_boundaries')
                assert lb is None or isinstance(lb, list), (
                    f"Stratum {stratum!r} chart {chart_name!r}: lane_boundaries "
                    f"should be flat list or None, got {type(lb).__name__}: {lb!r}"
                )

    def test_focused_plot_for_each_stratum_renders(self, sds1_study_single_factor):
        """Each stratum's focused chart plots without raising. Symptom of
        Bug A: the plotter's defensive guard would reject the dict shape with
        PlotError before any tick labels could be computed."""
        result = sds1_study_single_factor.execute(chart='X', by=['factor 1'], companion=True)
        for stratum in result.strata:
            focused = result.focus(stratum)
            for cname in ('X', 'mR'):
                if cname not in focused.charts:
                    continue
                fig = focused.plot(chart=cname)
                assert fig is not None

    def test_focused_chart_tick_positions_in_range(self, sds1_study_two_factor):
        """Tick positions on a focused stratum's chart are inside the FOCUSED
        data's row range. Was the symptom of Bug A: the plotter was using the
        first stratum's boundaries against the focused stratum's data, which
        could produce tick positions beyond the focused stratum's row count."""
        result = sds1_study_two_factor.execute(chart='X', by=['factor 1', 'factor 2'], companion=True)
        stratum = result.strata[-1]  # non-first stratum
        focused = result.focus(stratum)
        fig = focused.plot(chart='X')
        ax = fig.figure.layout.xaxis
        if ax.tickvals is None:
            return  # small categorical case — Plotly handles natively
        n = len(focused.charts['X']['data'])
        for pos in ax.tickvals:
            if isinstance(pos, int):
                assert 0 <= pos < n, (
                    f"Tick position {pos} outside focused stratum's [0, {n}) "
                    f"— likely Bug A regression."
                )
