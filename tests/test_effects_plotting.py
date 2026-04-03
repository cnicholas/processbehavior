"""
Tests for effects and interactions visualization.

Tests the new chart types: 'Effects', 'TimeInteraction', 'FactorInteraction'
that can be accessed via result.plot(chart='...')
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from processbehavior import ProcessBehavior
from processbehavior.exceptions import ChartNotAvailableError, ValidationError
from processbehavior.plotting import ControlChartFigure
from processbehavior.plotting.effects_charts import (
    create_factor_effects_chart,
    create_factor_interaction_chart,
    create_main_effects_chart,
    create_time_effects_chart,
    create_time_interaction_chart,
)
from processbehavior.plotting.themes import get_theme

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def result_with_effects():
    """Create analysis result with effects (SDS 1 - full replication)."""
    np.random.seed(42)

    # 2 factors × 5 time points × 3 replicates
    data = []
    for f1 in ['A', 'B']:
        for f2 in [1, 2]:
            for t in range(1, 6):
                for _ in range(3):
                    # Add effects for testing
                    f1_effect = 2 if f1 == 'A' else -2
                    f2_effect = 1 if f2 == 1 else -1
                    time_interaction = 0.5 * t if f1 == 'A' else -0.5 * t
                    value = 100 + f1_effect + f2_effect + time_interaction + np.random.normal(0, 2)
                    data.append({'value': value, 'factor1': f1, 'factor2': f2, 'time': t})

    df = pd.DataFrame(data)
    pdf = ProcessBehavior(df)
    study = pdf.formulate(
        response=pdf.cols.value,
        factors=[pdf.cols.factor1, pdf.cols.factor2],
        time=pdf.cols.time
    )
    return study.execute()


@pytest.fixture
def result_single_factor():
    """Create analysis result with single factor."""
    np.random.seed(42)

    data = []
    for f1 in ['A', 'B', 'C']:
        for t in range(1, 6):
            for _ in range(3):
                f1_effect = {'A': 2, 'B': 0, 'C': -2}[f1]
                value = 100 + f1_effect + np.random.normal(0, 2)
                data.append({'value': value, 'factor1': f1, 'time': t})

    df = pd.DataFrame(data)
    pdf = ProcessBehavior(df)
    study = pdf.formulate(
        response=pdf.cols.value,
        factors=[pdf.cols.factor1],
        time=pdf.cols.time
    )
    return study.execute()


@pytest.fixture
def result_no_time():
    """Create analysis result with factors but no time variable.

    This produces a result with factor effects but no time interaction.
    """
    np.random.seed(42)
    data = []
    for f1 in ['A', 'B']:
        for _ in range(15):
            value = 100 + (2 if f1 == 'A' else -2) + np.random.normal(0, 2)
            data.append({'value': value, 'factor1': f1})

    df = pd.DataFrame(data)
    pdf = ProcessBehavior(df)
    study = pdf.formulate(
        response=pdf.cols.value,
        factors=[pdf.cols.factor1]
    )
    return study.execute(chart='XmR', by=['factor1'])


@pytest.fixture
def theme():
    """Get default theme for testing."""
    return get_theme('processbehavior')


# ============================================================================
# Test: Effect Key Names (Phase 1 verification)
# ============================================================================

class TestEffectKeyNames:
    """Verify effect keys use new naming convention."""

    def test_time_effect_key(self, result_with_effects):
        """Time effects should use 'time' key, not 'pt_me'."""
        effects = result_with_effects.effects
        assert 'time' in effects
        assert 'pt_me' not in effects

    def test_factor_time_interaction_key(self, result_with_effects):
        """Factor × time interaction should use 'factor_time' key."""
        interactions = result_with_effects.interactions
        assert 'factor_time' in interactions
        assert 'pdc_by_pt' not in interactions

    def test_factor_factor_interaction_key(self, result_with_effects):
        """Factor × factor interaction should use 'factor_factor' key."""
        interactions = result_with_effects.interactions
        assert 'factor_factor' in interactions
        assert 'factor_interaction' not in result_with_effects.effects


# ============================================================================
# Test: create_main_effects_chart
# ============================================================================

class TestCreateMainEffectsChart:
    """Test main effects bar chart creation."""

    def test_creates_figure(self, result_with_effects, theme):
        """Should create a valid Plotly figure."""
        effects = result_with_effects.effects
        fig = create_main_effects_chart(effects, theme)

        assert isinstance(fig, go.Figure)

    def test_horizontal_bar_orientation(self, result_with_effects, theme):
        """Should create horizontal bars."""
        effects = result_with_effects.effects
        fig = create_main_effects_chart(effects, theme)

        # Check trace is horizontal bar
        assert len(fig.data) > 0
        assert fig.data[0].orientation == 'h'

    def test_includes_all_factors(self, result_with_effects, theme):
        """Should include effects for all factors."""
        effects = result_with_effects.effects
        fig = create_main_effects_chart(effects, theme)

        # Should have labels for factor1 and factor2 levels
        labels = list(fig.data[0].y)
        factor1_labels = [lbl for lbl in labels if lbl.startswith('factor1:')]
        factor2_labels = [lbl for lbl in labels if lbl.startswith('factor2:')]

        assert len(factor1_labels) == 2  # A, B
        assert len(factor2_labels) == 2  # 1, 2

    def test_custom_dimensions(self, result_with_effects, theme):
        """Should respect custom width/height."""
        effects = result_with_effects.effects
        fig = create_main_effects_chart(effects, theme, width=800, height=600)

        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_raises_if_no_effects(self, theme):
        """Should raise error if no effects data found."""
        empty_effects = {}
        with pytest.raises(ChartNotAvailableError, match="No main effects found"):
            create_main_effects_chart(empty_effects, theme)


# ============================================================================
# Test: create_factor_effects_chart
# ============================================================================

class TestCreateFactorEffectsChart:
    """Test factor effects bar chart creation (MainEffects)."""

    def test_creates_figure(self, result_with_effects, theme):
        """Should create a valid Plotly figure."""
        effects = result_with_effects.effects
        fig = create_factor_effects_chart(effects, theme)

        assert isinstance(fig, go.Figure)

    def test_vertical_bar_orientation(self, result_with_effects, theme):
        """Should create vertical bars (no orientation specified)."""
        effects = result_with_effects.effects
        fig = create_factor_effects_chart(effects, theme)

        # Check trace is vertical bar (no orientation = vertical)
        assert len(fig.data) > 0
        assert fig.data[0].orientation is None  # Vertical bars have no orientation

    def test_excludes_time_effects(self, result_with_effects, theme):
        """Should only show factor effects, not time effects."""
        effects = result_with_effects.effects
        fig = create_factor_effects_chart(effects, theme)

        # Should have labels only for factors, not time
        labels = list(fig.data[0].x)
        time_labels = [lbl for lbl in labels if lbl.startswith('Time:')]

        assert len(time_labels) == 0

    def test_includes_all_factors(self, result_with_effects, theme):
        """Should include effects for all factors."""
        effects = result_with_effects.effects
        fig = create_factor_effects_chart(effects, theme)

        labels = list(fig.data[0].x)
        factor1_labels = [lbl for lbl in labels if lbl.startswith('factor1:')]
        factor2_labels = [lbl for lbl in labels if lbl.startswith('factor2:')]

        assert len(factor1_labels) == 2  # A, B
        assert len(factor2_labels) == 2  # 1, 2

    def test_custom_dimensions(self, result_with_effects, theme):
        """Should respect custom width/height."""
        effects = result_with_effects.effects
        fig = create_factor_effects_chart(effects, theme, width=800, height=600)

        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_raises_if_no_factor_effects(self, theme, result_with_effects):
        """Should raise error if no factor effects data found."""
        # Create effects dict with only time effects
        time_only_effects = {'time': result_with_effects.effects.get('time')}
        with pytest.raises(ChartNotAvailableError, match="No factor main effects found"):
            create_factor_effects_chart(time_only_effects, theme)


# ============================================================================
# Test: create_time_effects_chart
# ============================================================================

class TestCreateTimeEffectsChart:
    """Test time effects bar chart creation (TimeEffects)."""

    def test_creates_figure(self, result_with_effects, theme):
        """Should create a valid Plotly figure."""
        effects = result_with_effects.effects
        fig = create_time_effects_chart(effects, theme)

        assert isinstance(fig, go.Figure)

    def test_horizontal_bar_orientation(self, result_with_effects, theme):
        """Should create horizontal bars."""
        effects = result_with_effects.effects
        fig = create_time_effects_chart(effects, theme)

        # Check trace is horizontal bar
        assert len(fig.data) > 0
        assert fig.data[0].orientation == 'h'

    def test_only_time_labels(self, result_with_effects, theme):
        """Should only show time labels, not factor labels."""
        effects = result_with_effects.effects
        fig = create_time_effects_chart(effects, theme)

        labels = list(fig.data[0].y)

        # All labels should be time labels
        for lbl in labels:
            assert lbl.startswith('Time:')

        # Should have labels for all 5 time points
        assert len(labels) == 5

    def test_auto_height_calculation(self, result_with_effects, theme):
        """Should auto-calculate height based on number of bars."""
        effects = result_with_effects.effects
        fig = create_time_effects_chart(effects, theme)

        # 5 time points should give height = max(400, 25*5 + 100) = 400
        assert fig.layout.height == 400

    def test_custom_dimensions(self, result_with_effects, theme):
        """Should respect custom width/height."""
        effects = result_with_effects.effects
        fig = create_time_effects_chart(effects, theme, width=800, height=600)

        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_raises_if_no_time_effects(self, theme, result_no_time):
        """Should raise error if no time effects data found."""
        effects = result_no_time.effects
        with pytest.raises(ChartNotAvailableError, match="No time effects found"):
            create_time_effects_chart(effects, theme)


# ============================================================================
# Test: create_time_interaction_chart
# ============================================================================

class TestCreateTimeInteractionChart:
    """Test factor × time interaction line chart creation."""

    def test_creates_figure(self, result_with_effects, theme):
        """Should create a valid Plotly figure."""
        fig = create_time_interaction_chart(
            interactions=result_with_effects.interactions,
            effects=result_with_effects.effects,
            factors=['factor1', 'factor2'],
            time_var='time',
            dataset=result_with_effects.dataset,
            theme=theme
        )

        assert isinstance(fig, go.Figure)

    def test_line_per_factor_level(self, result_single_factor, theme):
        """Should create one line per factor level."""
        fig = create_time_interaction_chart(
            interactions=result_single_factor.interactions,
            effects=result_single_factor.effects,
            factors=['factor1'],
            time_var='time',
            dataset=result_single_factor.dataset,
            theme=theme
        )

        # Should have 3 traces (A, B, C)
        assert len(fig.data) == 3

    def test_raises_if_no_factor_time(self, theme, result_no_time):
        """Should raise error if factor_time not available."""
        with pytest.raises(ChartNotAvailableError, match="Factor × time interaction not available"):
            create_time_interaction_chart(
                interactions={},  # Empty interactions
                effects={},
                factors=['factor1'],
                time_var='time',
                dataset=result_no_time.dataset,
                theme=theme
            )


# ============================================================================
# Test: create_factor_interaction_chart
# ============================================================================

class TestCreateFactorInteractionChart:
    """Test factor × factor interaction chart creation."""

    def test_creates_line_chart(self, result_with_effects, theme):
        """Should create a line chart with scatter traces."""
        fig = create_factor_interaction_chart(
            interactions=result_with_effects.interactions,
            factors=['factor1', 'factor2'],
            theme=theme
        )

        assert isinstance(fig, go.Figure)
        # Check it's a scatter (line) trace
        assert isinstance(fig.data[0], go.Scatter)
        # Should have one trace per factor2 level (2 levels: 1, 2)
        assert len(fig.data) == 2

    def test_raises_if_no_factor_factor(self, theme):
        """Should raise error if factor_factor not available."""
        with pytest.raises(ChartNotAvailableError, match="Factor × factor interaction not available"):
            create_factor_interaction_chart(
                interactions={},
                factors=['factor1', 'factor2'],
                theme=theme
            )

    def test_raises_if_less_than_two_factors(self, theme, result_with_effects):
        """Should raise error if less than 2 factors."""
        with pytest.raises(ValidationError, match="at least 2 factors"):
            create_factor_interaction_chart(
                interactions=result_with_effects.interactions,
                factors=['factor1'],  # Only 1 factor
                theme=theme
            )


# ============================================================================
# Test: result.plot() Integration
# ============================================================================

class TestPlotEffectsIntegration:
    """Test plot(chart='...') integration with AnalysisResult."""

    def test_plot_effects(self, result_with_effects):
        """Should create Effects chart via result.plot()."""
        fig = result_with_effects.plot(chart='Effects')

        assert isinstance(fig, ControlChartFigure)
        assert isinstance(fig.figure, go.Figure)

    def test_plot_main_effects(self, result_with_effects):
        """Should create MainEffects chart via result.plot()."""
        fig = result_with_effects.plot(chart='MainEffects')

        assert isinstance(fig, ControlChartFigure)
        assert isinstance(fig.figure, go.Figure)
        # Should have vertical bars (x-axis labels, y-axis values)
        assert fig.figure.data[0].orientation is None

    def test_plot_time_effects(self, result_with_effects):
        """Should create TimeEffects chart via result.plot()."""
        fig = result_with_effects.plot(chart='TimeEffects')

        assert isinstance(fig, ControlChartFigure)
        assert isinstance(fig.figure, go.Figure)
        # Should have horizontal bars
        assert fig.figure.data[0].orientation == 'h'

    def test_plot_time_interaction(self, result_with_effects):
        """Should create TimeInteraction chart via result.plot()."""
        fig = result_with_effects.plot(chart='TimeInteraction')

        assert isinstance(fig, ControlChartFigure)
        assert isinstance(fig.figure, go.Figure)

    def test_plot_factor_interaction(self, result_with_effects):
        """Should create FactorInteraction chart via result.plot()."""
        fig = result_with_effects.plot(chart='FactorInteraction')

        assert isinstance(fig, ControlChartFigure)
        assert isinstance(fig.figure, go.Figure)

    def test_effects_chart_with_custom_title(self, result_with_effects):
        """Should apply custom title."""
        fig = result_with_effects.plot(chart='Effects', title='My Effects Chart')

        assert fig.figure.layout.title.text == 'My Effects Chart'

    def test_effects_chart_with_custom_width(self, result_with_effects):
        """Should apply custom width."""
        fig = result_with_effects.plot(chart='Effects', width=1200)

        assert fig.figure.layout.width == 1200

    def test_main_effects_with_custom_title(self, result_with_effects):
        """Should apply custom title to MainEffects chart."""
        fig = result_with_effects.plot(chart='MainEffects', title='Factor Effects')

        assert fig.figure.layout.title.text == 'Factor Effects'

    def test_time_effects_with_custom_title(self, result_with_effects):
        """Should apply custom title to TimeEffects chart."""
        fig = result_with_effects.plot(chart='TimeEffects', title='Time Effects')

        assert fig.figure.layout.title.text == 'Time Effects'


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestEffectsPlottingErrors:
    """Test error handling for effects plotting."""

    def test_time_interaction_requires_time(self, result_no_time):
        """Should raise error when time interaction not available."""
        with pytest.raises(ValidationError, match="Time interaction not available"):
            result_no_time.plot(chart='TimeInteraction')

    def test_factor_interaction_requires_two_factors(self, result_single_factor):
        """Should raise error when factor interaction not available."""
        with pytest.raises(ValidationError, match="Factor interaction not available"):
            result_single_factor.plot(chart='FactorInteraction')

    def test_time_effects_requires_time_variable(self, result_no_time):
        """Should raise error when time effects not available."""
        with pytest.raises(ValidationError, match="Time effects not available"):
            result_no_time.plot(chart='TimeEffects')

    def test_main_effects_requires_effects(self, result_no_time):
        """MainEffects should raise error when effects not computed."""
        # result_no_time has factors but has_effects=False (XmR chart with by param)
        with pytest.raises(ValidationError, match="Effects not available"):
            result_no_time.plot(chart='MainEffects')


# ============================================================================
# Test: Different SDS Types
# ============================================================================

class TestEffectsPottingDifferentSDS:
    """Test effects plotting with different SDS types."""

    def test_sds1_full_replication(self, result_with_effects):
        """SDS 1 should have all effects available."""
        assert result_with_effects.sds == 1
        assert result_with_effects.has_effects

        # All chart types should work
        result_with_effects.plot(chart='Effects')
        result_with_effects.plot(chart='TimeInteraction')
        result_with_effects.plot(chart='FactorInteraction')

    def test_sds2_no_replication(self):
        """SDS 2 (no replication) should still have effects."""
        np.random.seed(42)

        # SDS 2: one observation per cell - need XmR chart with by parameter
        data = []
        for f1 in ['A', 'B']:
            for f2 in [1, 2]:
                for t in range(1, 6):
                    # Only one observation per cell
                    value = 100 + np.random.normal(0, 2)
                    data.append({'value': value, 'factor1': f1, 'factor2': f2, 'time': t})

        df = pd.DataFrame(data)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.value,
            factors=[pdf.cols.factor1, pdf.cols.factor2],
            time=pdf.cols.time
        )
        result = study.execute(chart='XmR', by=['factor1', 'factor2'])

        assert result.analytical_sds == 2
        assert result.has_effects

        # Effects should still be plottable
        result.plot(chart='Effects')


# ============================================================================
# Test: Data Access
# ============================================================================

class TestEffectsDataAccess:
    """Test accessing effects data directly."""

    def test_access_factor_effects(self, result_with_effects):
        """Should access factor effects by name."""
        effects = result_with_effects.effects

        # Access by factor name
        factor1_effects = effects['factor1']
        assert isinstance(factor1_effects, pd.DataFrame)
        assert 'factor1' in factor1_effects.columns
        assert 'Main_Effect' in factor1_effects.columns

    def test_access_time_effects(self, result_with_effects):
        """Should access time effects."""
        effects = result_with_effects.effects

        time_effects = effects['time']
        assert isinstance(time_effects, pd.DataFrame)
        assert 'time' in time_effects.columns
        assert 'PT_ME' in time_effects.columns

    def test_access_factor_time_interaction(self, result_with_effects):
        """Should access factor × time interaction."""
        interactions = result_with_effects.interactions

        factor_time = interactions['factor_time']
        assert isinstance(factor_time, pd.Series)

    def test_access_factor_factor_interaction(self, result_with_effects):
        """Should access factor × factor interaction."""
        interactions = result_with_effects.interactions

        factor_factor = interactions['factor_factor']
        assert isinstance(factor_factor, pd.DataFrame)
        assert 'Rx' in factor_factor.columns
