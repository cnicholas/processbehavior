"""
Tripwire tests for plotting module hardening.

P0 tests verify:
- P0-1: Unexpected exceptions in run-rules visualization propagate (not swallowed)
- P0-2: PDC length mismatch in interaction chart raises ValueError
- P0-4: Missing limit column logs debug and returns fig unchanged

P1 tests verify:
- #14: Compact rounding cascade produces 3 distinct tiers
- #17: ncols=0 in stats box doesn't ZeroDivisionError
- #18: Length mismatch in build_stepped_coordinates raises ValueError
- #16: Zone boundaries return None with debug log for missing stats
- #20: MultiIndex obs_id in run_rules doesn't crash
- #35: Rule name without underscore doesn't IndexError
- #37: HTML escaping in generate_report
"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from processbehavior.plotting.effects_charts import create_time_interaction_chart
from processbehavior.plotting.limits import add_stepped_limit_line, build_stepped_coordinates
from processbehavior.plotting.run_rules_viz import add_run_rules_visualization
from processbehavior.plotting.stats_box import add_stats_box, format_stat_value
from processbehavior.plotting.themes import get_theme
from processbehavior.plotting.zones import calculate_zone_boundaries


@pytest.fixture
def theme():
    return get_theme('processbehavior')


class TestRunRulesUnexpectedErrorPropagates:
    """P0-1: RuntimeError from detect_signals must not be swallowed."""

    def test_unexpected_error_propagates(self, theme, caplog):
        fig = go.Figure()
        data = pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        stats = {'center': 2.0, 'upl': 4.0, 'lpl': 0.0}

        mock_result = MagicMock()
        mock_result.detect_signals.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            with caplog.at_level(logging.ERROR):
                add_run_rules_visualization(
                    fig, data, stats, 'X', 'value', 'x',
                    theme, result=mock_result
                )

        assert "failed unexpectedly" in caplog.text

    def test_known_skip_returns_fig(self, theme, caplog):
        """AttributeError/KeyError should be caught and return fig."""
        fig = go.Figure()
        data = pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        stats = {'center': 2.0, 'upl': 4.0, 'lpl': 0.0}

        mock_result = MagicMock()
        mock_result.detect_signals.side_effect = AttributeError("no signals attr")

        with caplog.at_level(logging.INFO):
            returned = add_run_rules_visualization(
                fig, data, stats, 'X', 'value', 'x',
                theme, result=mock_result
            )

        assert returned is fig
        assert "Skipping run-rule annotations" in caplog.text


class TestInteractionChartPDCMismatchRaises:
    """P0-2: PDC length != data length must raise, not fall back."""

    def test_pdc_mismatch_raises(self, theme):
        # 10-row dataset
        df = pd.DataFrame({
            'factor1': ['A', 'B'] * 5,
            'time': list(range(5)) * 2,
            'value': range(10),
        })
        # 5-element PDC (mismatched)
        pdc = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

        interactions = {'factor_time': pdc}
        effects = {}

        with pytest.raises(ValueError, match=r"len\(pdc\)=5 != len\(df\)=10"):
            create_time_interaction_chart(
                interactions=interactions,
                effects=effects,
                factors=['factor1'],
                time_var='time',
                dataset=df,
                theme=theme,
            )


class TestSteppedLimitMissingColumnLogsDebug:
    """P0-4: Missing limit_col should log debug and return fig unchanged."""

    def test_missing_column_logs_debug_and_returns_fig(self, theme, caplog):
        fig = go.Figure()
        data = pd.DataFrame({'x': [1, 2, 3], 'value': [10, 20, 30]})

        with caplog.at_level(logging.DEBUG):
            returned = add_stepped_limit_line(
                fig, data, 'x', 'nonexistent_col',
                'red', 'dash', 1.5, 'UPL', theme
            )

        assert returned is fig
        assert len(fig.data) == 0
        assert "nonexistent_col" in caplog.text
        assert "not found" in caplog.text


# =========================================================================
# P1 #14: Compact rounding cascade
# =========================================================================


class TestFormatStatValueCompactTiers:
    """#14: All 3 compact tiers produce distinct formatting."""

    def test_compact_large(self):
        assert format_stat_value(152.3456, compact=True) == '152'

    def test_compact_medium(self):
        assert format_stat_value(52.3456, compact=True) == '52.3'

    def test_compact_small(self):
        assert format_stat_value(5.12345, compact=True) == '5.12'

    def test_compact_tiers_are_distinct(self):
        """Each tier must produce a different number of decimal places."""
        large = format_stat_value(152.3456, compact=True)
        medium = format_stat_value(52.3456, compact=True)
        small = format_stat_value(5.12345, compact=True)
        # Count decimal places
        assert '.' not in large  # 0 decimals
        assert len(medium.split('.')[1]) == 1  # 1 decimal
        assert len(small.split('.')[1]) == 2  # 2 decimals


# =========================================================================
# P1 #17: Stats box guard for zero grid dimensions
# =========================================================================


class TestStatsBoxNcolsZeroNoCrash:
    """#17: ncols=0 logs debug and doesn't ZeroDivisionError."""

    def test_ncols_zero_skips(self, theme, caplog):
        fig = go.Figure()
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        data = pd.DataFrame({'val': range(10)})

        with caplog.at_level(logging.DEBUG):
            add_stats_box(fig, stats, data, theme, row=1, col=1, nrows=1, ncols=0)

        assert len(fig.layout.annotations) == 0
        assert "Invalid grid dimensions" in caplog.text

    def test_nrows_zero_skips(self, theme, caplog):
        fig = go.Figure()
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        data = pd.DataFrame({'val': range(10)})

        with caplog.at_level(logging.DEBUG):
            add_stats_box(fig, stats, data, theme, row=1, col=1, nrows=0, ncols=2)

        assert len(fig.layout.annotations) == 0

    def test_none_ncols_skips(self, theme, caplog):
        fig = go.Figure()
        stats = {'center': 50.0, 'upl': 65.0, 'lpl': 35.0}
        data = pd.DataFrame({'val': range(10)})

        with caplog.at_level(logging.DEBUG):
            add_stats_box(fig, stats, data, theme, row=1, col=1, nrows=1, ncols=None)

        assert len(fig.layout.annotations) == 0


# =========================================================================
# P1 #18: Stepped coordinates length mismatch
# =========================================================================


class TestSteppedCoordinatesLengthMismatch:
    """#18: Unequal lengths raise ValueError."""

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            build_stepped_coordinates([1, 2, 3], [10.0, 20.0])

    def test_equal_length_ok(self):
        x, y = build_stepped_coordinates([1, 2], [10.0, 20.0])
        assert len(x) == 3
        assert len(y) == 3


# =========================================================================
# P1 #16: Zone boundaries missing stats logs debug
# =========================================================================


class TestZoneBoundariesMissingLogsDebug:
    """#16: Returns None + debug log present."""

    def test_missing_stats_logs_debug(self, theme, caplog):
        with caplog.at_level(logging.DEBUG):
            result = calculate_zone_boundaries({}, theme)

        assert result is None
        assert "missing center/upl/lpl" in caplog.text

    def test_varying_limits_logs_debug(self, theme, caplog):
        with caplog.at_level(logging.DEBUG):
            result = calculate_zone_boundaries(
                {'center': 50.0, 'upl': 'Varies', 'lpl': 35.0}, theme
            )

        assert result is None
        assert "limits vary" in caplog.text


# =========================================================================
# P1 #20: MultiIndex safety in run_rules
# =========================================================================


class TestRunRulesMultiIndexSkip:
    """#20: MultiIndex data.loc doesn't crash, logs debug."""

    def test_multiindex_obs_id_skipped(self, theme, caplog):
        """When data.loc[obs_id] returns a DataFrame (MultiIndex), skip gracefully."""
        # Create data with a duplicate index to simulate MultiIndex expansion
        data = pd.DataFrame(
            {'value': [1.0, 2.0, 3.0], 'x': [0, 1, 2]},
            index=[0, 0, 1]  # duplicate index 0
        )
        stats = {'center': 2.0, 'upl': 4.0, 'lpl': 0.0}

        # Mock result that returns violations referencing obs_id=0
        mock_result = MagicMock()
        mock_signal = MagicMock()
        mock_signal.has_signals = True
        violations = pd.DataFrame({
            'obs_id': pd.Categorical([0]),
            'rule_name': ['rule_2'],
        })
        mock_signal.violations = violations
        mock_result.detect_signals.return_value = mock_signal

        with caplog.at_level(logging.DEBUG):
            returned = add_run_rules_visualization(
                fig=go.Figure(),
                data=data,
                stats=stats,
                chart_name='X',
                value_col='value',
                x_col='x',
                theme=theme,
                result=mock_result,
            )

        assert returned is not None
        assert "MultiIndex expanded to DataFrame" in caplog.text


# =========================================================================
# P2 #35: Rule name without underscore
# =========================================================================


class TestRuleNumsNoUnderscore:
    """#35: Rule name without '_' doesn't IndexError."""

    def test_rule_without_underscore(self, theme):
        """A rule name like 'custom' (no underscore) should not crash."""
        data = pd.DataFrame(
            {'value': [1.0, 2.0, 3.0], 'x': [0, 1, 2]},
            index=[0, 1, 2]
        )
        stats = {'center': 2.0, 'upl': 4.0, 'lpl': 0.0}

        mock_result = MagicMock()
        mock_signal = MagicMock()
        mock_signal.has_signals = True
        violations = pd.DataFrame({
            'obs_id': pd.Categorical([1]),
            'rule_name': ['custom'],  # no underscore
        })
        mock_signal.violations = violations
        mock_result.detect_signals.return_value = mock_signal

        # Should not raise IndexError
        returned = add_run_rules_visualization(
            fig=go.Figure(),
            data=data,
            stats=stats,
            chart_name='X',
            value_col='value',
            x_col='x',
            theme=theme,
            result=mock_result,
        )
        assert returned is not None


# =========================================================================
# P2 #37: HTML escaping in generate_report
# =========================================================================


class TestGenerateReportEscapesHtml:
    """#37: <script> in response_var is escaped in output."""

    def test_xss_in_response_var_escaped(self, theme, tmp_path):
        """User-controlled strings must be HTML-escaped in the report."""
        from processbehavior.plotting.plotter import Plotter

        mock_result = MagicMock()
        mock_result.summary = {
            'sds': 1,
            'sds_description': 'test',
            'response_var': '<script>alert("xss")</script>',
            'n_observations': 100,
            'chart_types': ['XmR'],
            'n_signals_total': 0,
            'has_residuals': False,
            'has_effects': False,
            'is_stratified': False,
        }
        mock_result.charts = {'XmR': {'data': pd.DataFrame({'value': [1, 2, 3]}), 'statistics': {}}}
        mock_result.has_residuals = False
        mock_result.has_effects = False

        plotter = Plotter(mock_result)
        filepath = tmp_path / "report.html"
        plotter.generate_report(
            str(filepath),
            include_charts=False,
            include_residuals=False,
            include_effects=False,
            include_summary=True,
        )

        content = filepath.read_text()
        # The raw <script> tag must NOT appear unescaped
        assert '<script>alert' not in content
        # The escaped version should be present
        assert '&lt;script&gt;' in content

    def test_xss_in_title_escaped(self, theme, tmp_path):
        """Report title must be HTML-escaped."""
        from processbehavior.plotting.plotter import Plotter

        mock_result = MagicMock()
        mock_result.summary = {
            'sds': 1,
            'sds_description': 'test',
            'response_var': 'Y',
            'n_observations': 10,
            'chart_types': ['XmR'],
            'n_signals_total': 0,
            'has_residuals': False,
            'has_effects': False,
            'is_stratified': False,
        }
        mock_result.charts = {'XmR': {'data': pd.DataFrame({'value': [1]}), 'statistics': {}}}
        mock_result.has_residuals = False
        mock_result.has_effects = False

        plotter = Plotter(mock_result)
        filepath = tmp_path / "report.html"
        plotter.generate_report(
            str(filepath),
            include_charts=False,
            include_residuals=False,
            include_effects=False,
            include_summary=False,
            title='<img src=x onerror=alert(1)>',
        )

        content = filepath.read_text()
        assert '<img src=x' not in content
        assert '&lt;img src=x' in content


# =========================================================================
# X-axis label: XmR gets "Observation", Xbar gets "Subgroup"
# =========================================================================


class TestXAxisLabel:
    """XmR charts should label x-axis 'Observation', not 'Subgroup'."""

    def _make_plotter(self, summary_overrides=None):
        from processbehavior.plotting.plotter import Plotter

        summary = {
            'sds': 1,
            'sds_description': 'test',
            'response_var': 'Y',
            'n_observations': 10,
            'chart_types': ['XmR'],
            'n_signals_total': 0,
            'has_residuals': False,
            'has_effects': False,
            'is_stratified': False,
        }
        if summary_overrides:
            summary.update(summary_overrides)

        mock_result = MagicMock()
        mock_result.summary = summary
        mock_result.charts = {'XmR': {'data': pd.DataFrame({'value': [1]}), 'statistics': {}}}
        mock_result.has_residuals = False
        mock_result.has_effects = False
        return Plotter(mock_result)

    def test_xmr_xaxis_label_not_subgroup(self):
        """XmR chart with 'rsg' column gets 'Observation', not 'Subgroup'."""
        plotter = self._make_plotter()
        label = plotter._get_xaxis_label('rsg', 'XmR')
        assert label == 'Observation'

    def test_xbar_xaxis_label_is_subgroup(self):
        """Xbar chart with 'rsg' column still gets 'Subgroup'."""
        plotter = self._make_plotter()
        label = plotter._get_xaxis_label('rsg', 'Xbar')
        assert label == 'Subgroup'

    def test_xmr_xaxis_label_uses_time_var(self):
        """XmR chart with time variable gets the time variable name."""
        plotter = self._make_plotter({'time_var': 'pull_date'})
        label = plotter._get_xaxis_label('rsg', 'XmR')
        assert label == 'Pull Date'

    def test_xmr_residuals_xaxis_label(self):
        """XmR_residuals chart with 'rsg' gets 'Observation'."""
        plotter = self._make_plotter()
        label = plotter._get_xaxis_label('rsg', 'XmR_residuals')
        assert label == 'Observation'

    def test_r_chart_xaxis_label(self):
        """R chart with 'rsg' gets 'Observation'."""
        plotter = self._make_plotter()
        label = plotter._get_xaxis_label('rsg', 'R')
        assert label == 'Observation'

    def test_s_chart_xaxis_label_is_subgroup(self):
        """S chart with 'rsg' gets 'Subgroup'."""
        plotter = self._make_plotter()
        label = plotter._get_xaxis_label('rsg', 'S')
        assert label == 'Subgroup'
