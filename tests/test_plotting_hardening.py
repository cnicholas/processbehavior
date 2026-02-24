"""
Tripwire tests for P0 plotting module hardening.

These tests verify that:
- P0-1: Unexpected exceptions in run-rules visualization propagate (not swallowed)
- P0-2: PDC length mismatch in interaction chart raises ValueError
- P0-4: Missing limit column logs debug and returns fig unchanged
"""

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from processbehavior.plotting.effects_charts import create_time_interaction_chart
from processbehavior.plotting.limits import add_stepped_limit_line
from processbehavior.plotting.run_rules_viz import add_run_rules_visualization
from processbehavior.plotting.themes import get_theme


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
