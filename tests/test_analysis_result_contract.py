"""
Contract tests for AnalysisResult public API surface.

Covers: chart_table, get_statistics, iter_charts, list_strata,
FocusedAnalysisResult, get_residual, get_signals, convenience accessors.
"""

import pandas as pd
import pytest

from processbehavior import (
    ChartNotAvailableError,
    ProcessBehavior,
    ValidationError,
)
from processbehavior.datasets.synthetic import make_sds

# ============================================================================
# Module-scoped fixtures (expensive — computed once)
# ============================================================================

@pytest.fixture(scope='module')
def sds1_xbar_result():
    """SDS 1 with replication — Xbar/S charts available."""
    df = make_sds(1, K1=3, K2=2, T=4, n_min=3, n_max=3, seed=42)
    study = ProcessBehavior(df).formulate(
        response='y', time='time', factors=['factor 1', 'factor 2'],
    )
    return study.execute(chart='Xbar')


@pytest.fixture(scope='module')
def sds1_stratified_xmr_result():
    """SDS 1 stratified XmR — strata available."""
    df = make_sds(1, K1=3, K2=2, T=4, n_min=3, n_max=3, seed=42)
    study = ProcessBehavior(df).formulate(
        response='y', time='time', factors=['factor 1', 'factor 2'],
    )
    return study.execute(chart='XmR', by=['factor 1'])


@pytest.fixture(scope='module')
def sds4_xmr_result():
    """SDS 4-like — single factor, single level, no residual decomposition."""
    import numpy as np
    df = pd.DataFrame({
        'y': np.random.default_rng(42).normal(50, 5, 30),
        'time': range(1, 31),
        'group': ['A'] * 30,
    })
    study = ProcessBehavior(df).formulate(
        response='y', time='time', factors=['group'],
    )
    return study.execute(chart='XmR', by=['group'])


# ============================================================================
# chart_table()
# ============================================================================

class TestChartTable:
    """Contract tests for chart_table() summary table generation."""

    def test_default_uses_first_chart(self, sds1_xbar_result):
        """chart=None should use first chart."""
        table = sds1_xbar_result.chart_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0

    def test_xbar_columns(self, sds1_xbar_result):
        """Xbar chart table should have subgroup, n, value, center, limits, signal."""
        table = sds1_xbar_result.chart_table('Xbar')
        expected_cols = {'value', 'center', 'lpl', 'upl'}
        assert expected_cols.issubset(set(table.columns))

    def test_signal_symbols_default(self, sds1_xbar_result):
        """Default signal_symbols=True should use arrow symbols."""
        table = sds1_xbar_result.chart_table('Xbar')
        if 'signal' in table.columns:
            unique_signals = set(table['signal'].dropna().unique())
            # Should contain only arrow symbols or empty string
            assert unique_signals.issubset({'↑', '↓', ''})

    def test_signal_symbols_false(self, sds1_xbar_result):
        """signal_symbols=False should use numeric values."""
        table = sds1_xbar_result.chart_table('Xbar', signal_symbols=False)
        if 'signal' in table.columns:
            # Should be numeric -1/0/1
            assert table['signal'].dtype in ('int64', 'float64', 'Int64')

    def test_no_signal_col(self, sds1_xbar_result):
        """include_signal_col=False should omit signal column."""
        table = sds1_xbar_result.chart_table('Xbar', include_signal_col=False)
        assert 'signal' not in table.columns

    def test_invalid_chart_raises(self, sds1_xbar_result):
        """Nonexistent chart should raise ChartNotAvailableError."""
        with pytest.raises(ChartNotAvailableError):
            sds1_xbar_result.chart_table('Nonexistent')

    def test_xmr_no_subgroup_col(self, sds4_xmr_result):
        """SDS 4 (no factors) chart_table should work without subgroup column."""
        table = sds4_xmr_result.chart_table('XmR')
        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0
        assert 'value' in table.columns

    def test_n_joined_from_ads(self, sds1_xbar_result):
        """n column should be present and contain positive integers."""
        table = sds1_xbar_result.chart_table('Xbar')
        if 'n' in table.columns:
            assert (table['n'] > 0).all()


# ============================================================================
# get_statistics()
# ============================================================================

class TestGetStatistics:
    """Contract tests for get_statistics() method."""

    def test_nonstratified_returns_flat_dict(self, sds1_xbar_result):
        """Non-stratified result should return flat dict with center/lpl/upl."""
        stats = sds1_xbar_result.get_statistics('Xbar')
        assert isinstance(stats, dict)
        assert 'center' in stats
        assert 'lpl' in stats
        assert 'upl' in stats

    def test_stratified_returns_nested_dict(self, sds1_stratified_xmr_result):
        """Stratified result should return dict keyed by stratum."""
        stats = sds1_stratified_xmr_result.get_statistics('XmR')
        assert isinstance(stats, dict)
        # Should be nested — values are dicts, not scalars
        strata = sds1_stratified_xmr_result.strata
        if strata:
            # At least one stratum key should be present
            first_value = next(iter(stats.values()))
            assert isinstance(first_value, dict), \
                f"Expected nested dict for stratified stats, got {type(first_value)}"

    def test_returns_copy(self, sds1_xbar_result):
        """Returned dict should be a copy — mutations don't affect original."""
        stats1 = sds1_xbar_result.get_statistics('Xbar')
        stats1['center'] = -999
        stats2 = sds1_xbar_result.get_statistics('Xbar')
        assert stats2['center'] != -999


# ============================================================================
# iter_charts()
# ============================================================================

class TestIterCharts:
    """Contract tests for iter_charts() iterator."""

    def test_yields_all_charts(self, sds1_xbar_result):
        """Should yield entries for every chart in all_charts."""
        names = [name for name, _, _ in sds1_xbar_result.iter_charts()]
        assert names == sds1_xbar_result.all_charts

    def test_tuple_structure(self, sds1_xbar_result):
        """Each yield should be (str, DataFrame, dict)."""
        for name, data, stats in sds1_xbar_result.iter_charts():
            assert isinstance(name, str)
            assert isinstance(data, pd.DataFrame)
            assert isinstance(stats, dict)

    def test_data_content_matches_get_chart(self, sds1_xbar_result):
        """Yielded data should match get_chart() content."""
        for name, data, _ in sds1_xbar_result.iter_charts():
            chart_copy = sds1_xbar_result.get_chart(name)
            pd.testing.assert_frame_equal(data, chart_copy)


# ============================================================================
# list_strata()
# ============================================================================

class TestListStrata:
    """Contract tests for list_strata() method."""

    def test_equals_strata_property(self, sds1_stratified_xmr_result):
        """list_strata() should return same value as strata property."""
        assert sds1_stratified_xmr_result.list_strata() == \
               sds1_stratified_xmr_result.strata

    def test_empty_for_nonstratified(self, sds1_xbar_result):
        """Non-stratified result should return empty list."""
        assert sds1_xbar_result.list_strata() == []

    def test_nonempty_for_stratified(self, sds1_stratified_xmr_result):
        """Stratified result should return non-empty list."""
        strata = sds1_stratified_xmr_result.list_strata()
        assert len(strata) > 0


# ============================================================================
# FocusedAnalysisResult contract
# ============================================================================

class TestFocusedAnalysisResult:
    """Contract tests for FocusedAnalysisResult behavior."""

    def test_focus_raises_validation_error(self, sds1_stratified_xmr_result):
        """Calling focus() on already-focused result should raise ValidationError."""
        stratum = sds1_stratified_xmr_result.strata[0]
        focused = sds1_stratified_xmr_result.focus(stratum)
        with pytest.raises(ValidationError, match="already focused"):
            focused.focus('anything')

    def test_strata_is_empty(self, sds1_stratified_xmr_result):
        """Focused result should have empty strata."""
        stratum = sds1_stratified_xmr_result.strata[0]
        focused = sds1_stratified_xmr_result.focus(stratum)
        assert focused.strata == []

    def test_is_stratified_false(self, sds1_stratified_xmr_result):
        """Focused result should not be stratified."""
        stratum = sds1_stratified_xmr_result.strata[0]
        focused = sds1_stratified_xmr_result.focus(stratum)
        assert focused.is_stratified is False

    def test_focused_stratum_property(self, sds1_stratified_xmr_result):
        """focused_stratum should return the stratum name."""
        stratum = sds1_stratified_xmr_result.strata[0]
        focused = sds1_stratified_xmr_result.focus(stratum)
        assert focused.focused_stratum == stratum


# ============================================================================
# Convenience accessors and edge cases
# ============================================================================

class TestConvenienceAccessors:
    """Contract tests for properties and convenience methods."""

    def test_get_residual_valid(self, sds1_xbar_result):
        """SDS 1 result should have R2 residual."""
        r2 = sds1_xbar_result.get_residual('R2')
        assert isinstance(r2, pd.Series)
        assert len(r2) > 0

    def test_get_residual_invalid_returns_empty(self, sds1_xbar_result):
        """Invalid residual type should return empty Series."""
        r99 = sds1_xbar_result.get_residual('R99')
        assert isinstance(r99, pd.Series)
        assert len(r99) == 0

    def test_get_signals_returns_dataframe(self, sds1_xbar_result):
        """get_signals() should return a DataFrame."""
        signals = sds1_xbar_result.get_signals('Xbar')
        assert isinstance(signals, pd.DataFrame)

    def test_get_signals_all_charts(self, sds1_xbar_result):
        """get_signals(None) should check all charts."""
        signals = sds1_xbar_result.get_signals()
        assert isinstance(signals, pd.DataFrame)

    def test_has_residuals_true_for_sds1(self, sds1_xbar_result):
        """SDS 1 with factors should have residuals."""
        assert sds1_xbar_result.has_residuals is True

    def test_has_effects_true_for_sds1(self, sds1_xbar_result):
        """SDS 1 with factors should have effects."""
        assert sds1_xbar_result.has_effects is True

    def test_has_interactions_is_bool(self, sds1_xbar_result):
        """has_interactions should return a boolean."""
        assert isinstance(sds1_xbar_result.has_interactions, bool)

    def test_all_charts_matches_chart_keys(self, sds1_xbar_result):
        """all_charts should match charts dict keys."""
        assert sds1_xbar_result.all_charts == list(sds1_xbar_result.charts.keys())

    def test_summary_returns_dict(self, sds1_xbar_result):
        """summary should return a dict with key metadata."""
        s = sds1_xbar_result.summary
        assert isinstance(s, dict)
        assert 'analytical_sds' in s

    def test_summary_returns_copy(self, sds1_xbar_result):
        """summary should return a copy."""
        s1 = sds1_xbar_result.summary
        s1['analytical_sds'] = -999
        s2 = sds1_xbar_result.summary
        assert s2['analytical_sds'] != -999
