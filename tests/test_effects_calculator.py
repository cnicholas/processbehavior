"""
Unit tests for EffectsCalculator class.

Tests cover:
- Factor main effects calculation
- Time main effects calculation
- Main effect scores
- Interaction cell means
- PDC calculation (SDS 1, 2, 3)
- Factor interaction effects
- Integration scenarios
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior.formulation_spec import FormulationSpec
from processbehavior.effects_calculator import (
    EffectsCalculator,
    calculate_factor_interaction_effects,
    calculate_factor_interaction_scores,
    calculate_factor_main_effects,
    calculate_interaction_cell_means,
    calculate_main_effect_scores,
    calculate_pdc_by_time_sds2,
    calculate_time_main_effects,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def calc():
    """Create EffectsCalculator instance."""
    return EffectsCalculator()


@pytest.fixture
def simple_df_with_residuals():
    """Simple DataFrame with VAS residuals."""
    return pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'rsg': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],  # Added for backward compatibility
        'pull': [1, 2, 1, 2, 1, 2, 1, 2],
        'weight': [10.2, 10.4, 10.1, 10.3, 9.8, 10.0, 9.9, 9.7],
        'R1': [0.1, 0.3, 0.0, 0.2, -0.2, 0.0, -0.1, -0.3],
        'R2': [0.05, 0.05, -0.05, -0.05, 0.03, 0.03, -0.03, -0.03],
        'R3': [0.1, -0.1, 0.1, -0.1, 0.05, -0.05, 0.05, -0.05],
        'R4': [0.02, 0.02, -0.02, -0.02, 0.01, 0.01, -0.01, -0.01],
        'R5': [0.2, 0.2, 0.2, 0.2, -0.1, -0.1, -0.1, -0.1],
        'Ybar': [10.1] * 8,
        'Ybar_k': [10.25, 10.25, 10.25, 10.25, 9.85, 9.85, 9.85, 9.85],
        'Ybar_t': [10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1],
        'Ybar_kt': [10.15, 10.35, 10.15, 10.35, 9.85, 9.95, 9.85, 9.95]
    })


@pytest.fixture
def multi_factor_df():
    """DataFrame with multiple factors and residuals."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'head': [1, 1, 2, 2, 1, 1, 2, 2],
        'pull': [1, 2, 1, 2, 1, 2, 1, 2],
        'R1': [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
        'R2': [0.05, 0.05, 0.1, 0.1, -0.05, -0.05, -0.1, -0.1],
        'R3': [0.02, -0.02, 0.03, -0.03, 0.01, -0.01, 0.015, -0.015],
        'R5': [0.3, 0.3, 0.4, 0.4, -0.2, -0.2, -0.3, -0.3]
    })
    # Add composite rsg column (for backward compatibility)
    df['rsg'] = df['lane'] + '_' + df['head'].astype(str)
    return df


@pytest.fixture
def spec_single_factor():
    """Single factor specification."""
    return FormulationSpec(
        response_var='weight',
        rsg_vars=('lane',),
        time_var='pull',
    )


@pytest.fixture
def spec_multi_factor():
    """Multi-factor specification."""
    return FormulationSpec(
        response_var='weight',
        rsg_vars=('lane', 'head'),
        time_var='pull',
    )


@pytest.fixture
def spec_no_grouping():
    """Specification without grouping variables (IMR)."""
    return FormulationSpec(
        response_var='weight',
        time_var='pull',
    )


# ============================================================================
# Test: calculate_factor_main_effects (Pure Function)
# ============================================================================

def test_calculate_factor_main_effects_basic():
    """Should calculate mean of R5 per factor level."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'R5': [0.2, 0.3, -0.2, -0.3]
    })

    result = calculate_factor_main_effects(df, 'lane')

    assert 'Main_Effect' in result.columns
    assert result.shape[0] == 2  # Two factor levels

    # Check values
    a_effect = result[result['lane'] == 'A']['Main_Effect'].iloc[0]
    b_effect = result[result['lane'] == 'B']['Main_Effect'].iloc[0]

    assert abs(a_effect - 0.25) < 0.01
    assert abs(b_effect - (-0.25)) < 0.01


def test_calculate_factor_main_effects_raises_if_no_r5():
    """Should raise helpful error if R5 missing."""
    df = pd.DataFrame({
        'lane': ['A', 'B'],
        'weight': [10.0, 9.0]
    })

    with pytest.raises(ValueError, match="R5 column missing"):
        calculate_factor_main_effects(df, 'lane')


def test_calculate_factor_main_effects_raises_if_factor_missing():
    """Should raise helpful error if factor not found."""
    df = pd.DataFrame({
        'lane': ['A', 'B'],
        'R5': [0.1, -0.1]
    })

    with pytest.raises(ValueError, match="Factor 'missing' not found"):
        calculate_factor_main_effects(df, 'missing')


def test_calculate_factor_main_effects_multiple_levels():
    """Should handle factors with many levels."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
        'R5': [0.1, 0.2, -0.1, -0.2, 0.3, 0.4, -0.3, -0.4]
    })

    result = calculate_factor_main_effects(df, 'lane')

    assert result.shape[0] == 4  # Four levels
    assert set(result['lane']) == {'A', 'B', 'C', 'D'}


# ============================================================================
# Test: calculate_time_main_effects (Pure Function)
# ============================================================================

def test_calculate_time_main_effects_basic():
    """Should calculate mean of R1 per time point."""
    df = pd.DataFrame({
        'pull': [1, 1, 2, 2],
        'R1': [0.1, 0.2, -0.1, -0.2]
    })

    result = calculate_time_main_effects(df, 'pull')

    assert 'PT_ME' in result.columns  # Column is called PT_ME
    assert result.shape[0] == 2  # Two time points

    # Check values
    t1_effect = result[result['pull'] == 1]['PT_ME'].iloc[0]
    t2_effect = result[result['pull'] == 2]['PT_ME'].iloc[0]

    assert abs(t1_effect - 0.15) < 0.01
    assert abs(t2_effect - (-0.15)) < 0.01


def test_calculate_time_main_effects_raises_if_no_r1():
    """Should raise helpful error if R1 missing."""
    df = pd.DataFrame({
        'pull': [1, 2],
        'weight': [10.0, 9.0]
    })

    with pytest.raises(ValueError, match="R1 column missing"):
        calculate_time_main_effects(df, 'pull')


# ============================================================================
# Test: calculate_main_effect_scores (Pure Function)
# ============================================================================

def test_calculate_main_effect_scores_basic():
    """Should add main effect to R2 for each row."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'R2': [0.1, 0.2, -0.1, -0.2]
    })
    main_effects = pd.DataFrame({
        'lane': ['A', 'B'],
        'Main_Effect': [0.5, -0.5]
    })

    result = calculate_main_effect_scores(df, 'lane', main_effects)

    assert 'lane_MEs' in result.columns  # Column is called lane_MEs

    # Check values: MEs = R2 + Main_Effect
    assert abs(result.loc[0, 'lane_MEs'] - 0.6) < 0.01  # 0.1 + 0.5
    assert abs(result.loc[1, 'lane_MEs'] - 0.7) < 0.01  # 0.2 + 0.5
    assert abs(result.loc[2, 'lane_MEs'] - (-0.6)) < 0.01  # -0.1 + -0.5
    assert abs(result.loc[3, 'lane_MEs'] - (-0.7)) < 0.01  # -0.2 + -0.5


def test_calculate_main_effect_scores_raises_if_no_r2():
    """Should raise helpful error if R2 missing."""
    df = pd.DataFrame({'lane': ['A', 'B']})
    main_effects = pd.DataFrame({'lane': ['A', 'B'], 'Main_Effect': [0.1, -0.1]})

    with pytest.raises(ValueError, match="R2 column missing"):
        calculate_main_effect_scores(df, 'lane', main_effects)


# ============================================================================
# Test: calculate_interaction_cell_means (Pure Function)
# ============================================================================

def test_calculate_interaction_cell_means_basic():
    """Should calculate mean of R3 per cell (factor × time)."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'pull': [1, 2, 1, 2, 1, 2, 1, 2],
        'R3': [0.1, 0.2, 0.15, 0.25, -0.1, -0.2, -0.15, -0.25]
    })

    result = calculate_interaction_cell_means(df, ['lane'], 'pull')

    # Returns a Series (broadcast to all rows)
    assert isinstance(result, pd.Series)
    assert len(result) == 8  # Same length as input

    # Check cell values (mean of R3 per cell, broadcast to all rows in that cell)
    assert abs(result.iloc[0] - 0.125) < 0.01  # A×1: mean of [0.1, 0.15]
    assert abs(result.iloc[1] - 0.225) < 0.01  # A×2: mean of [0.2, 0.25]


def test_calculate_interaction_cell_means_raises_if_no_r3():
    """Should raise helpful error if R3 missing."""
    df = pd.DataFrame({
        'lane': ['A', 'B'],
        'pull': [1, 2]
    })

    with pytest.raises(ValueError, match="R3 column missing"):
        calculate_interaction_cell_means(df, ['lane'], 'pull')


# ============================================================================
# Test: calculate_pdc_by_time_sds2 (Pure Function)
# ============================================================================

def test_calculate_pdc_by_time_sds2_basic():
    """Should calculate PDC for SDS 2: Ybar_kt - Ybar_k - Ybar_t + Ybar."""
    df = pd.DataFrame({
        'pull': [1, 2, 1, 2],
        'lane': ['A', 'A', 'B', 'B']
    })
    ybar_kt = pd.Series([10.2, 10.4, 9.8, 10.0])
    ybar_k = pd.Series([10.3, 10.3, 9.9, 9.9])
    ybar_t = pd.Series([10.0, 10.2, 10.0, 10.2])
    ybar = 10.1

    result = calculate_pdc_by_time_sds2(df, ybar_kt, ybar_k, ybar_t, ybar)

    # Returns a Series (PDC per row)
    assert isinstance(result, pd.Series)
    assert len(result) == 4

    # PDC = Ybar_kt - Ybar_k - Ybar_t + Ybar
    # For row 0: 10.2 - 10.3 - 10.0 + 10.1 = 0.0
    assert abs(result.iloc[0] - 0.0) < 0.01


# ============================================================================
# Test: calculate_factor_interaction_effects (Pure Function)
# ============================================================================

def test_calculate_factor_interaction_effects_two_factors():
    """Should calculate interaction effects between two factors."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'head': [1, 1, 2, 2, 1, 1, 2, 2],
        'R5': [0.3, 0.3, 0.4, 0.4, -0.2, -0.2, -0.3, -0.3]
    })

    # Pre-calculate main effects
    lane_me = calculate_factor_main_effects(df, 'lane')
    head_me = calculate_factor_main_effects(df, 'head')
    effects = {'lane': lane_me, 'head': head_me}

    result = calculate_factor_interaction_effects(df, ['lane', 'head'], effects)

    assert 'Rx' in result.columns  # Column is called Rx
    # Should have 4 combinations: A×1, A×2, B×1, B×2
    assert result.shape[0] == 4


def test_calculate_factor_interaction_effects_returns_empty_if_insufficient_factors():
    """Should return empty DataFrame if fewer than 2 factors."""
    df = pd.DataFrame({'lane': ['A', 'B'], 'R5': [0.1, -0.1]})
    effects = {}

    result = calculate_factor_interaction_effects(df, ['lane'], effects)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ============================================================================
# Test: calculate_factor_interaction_scores (Pure Function)
# ============================================================================

def test_calculate_factor_interaction_scores_basic():
    """Should merge interaction effects back to data."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'head': [1, 2, 1, 2],
        'R2': [0.05, 0.05, -0.05, -0.05]
    })
    interaction_effects = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'head': [1, 2, 1, 2],
        'Rx': [0.1, 0.2, -0.1, -0.2]
    })

    result = calculate_factor_interaction_scores(df, ['lane', 'head'], interaction_effects)

    assert 'factor_interaction_effects' in result.columns
    assert result.shape[0] == 4
    # FIE = R2 + Rx
    assert abs(result['factor_interaction_effects'].iloc[0] - 0.15) < 0.01  # 0.05 + 0.1


# ============================================================================
# Test: EffectsCalculator.calculate_all_effects
# ============================================================================

def test_calculate_all_effects_single_factor(calc, simple_df_with_residuals, spec_single_factor):
    """Should calculate all effects for single factor."""
    result = calc.calculate_all_effects(simple_df_with_residuals, spec_single_factor)

    # Should have lane main effects
    assert 'lane' in result
    assert result['lane'].shape[0] == 2  # Two levels

    # Should have time main effects
    assert 'time' in result
    assert result['time'].shape[0] == 2  # Two time points

    # Should have main effect scores
    assert 'lane_MEs' in result


def test_calculate_all_effects_multi_factor(calc, multi_factor_df, spec_multi_factor):
    """Should calculate all effects for multiple factors."""
    result = calc.calculate_all_effects(multi_factor_df, spec_multi_factor)

    # Should have both factor main effects
    assert 'lane' in result
    assert 'head' in result

    # Should have main effect scores for each factor
    assert 'lane_MEs' in result
    assert 'head_MEs' in result

    # Should have factor interaction effects (2+ factors)
    # Note: factor_interaction is now in interactions dict, not effects
    assert 'factor_interaction_effects' in result


def test_calculate_all_effects_no_grouping(calc, spec_no_grouping):
    """Should return empty dict when no grouping variables."""
    df = pd.DataFrame({
        'pull': [1, 2, 3],
        'weight': [10.1, 10.2, 10.3],
        'R1': [0.0, 0.1, 0.2],
        'R2': [0.0, 0.0, 0.0],
        'R5': [0.0, 0.0, 0.0]
    })

    result = calc.calculate_all_effects(df, spec_no_grouping)

    assert result == {}


def test_calculate_all_effects_raises_if_missing_residuals(calc, spec_single_factor):
    """Should raise helpful error if residuals missing."""
    df = pd.DataFrame({
        'lane': ['A', 'B'],
        'weight': [10.0, 9.0]
    })

    with pytest.raises(ValueError, match="missing residuals"):
        calc.calculate_all_effects(df, spec_single_factor)


# ============================================================================
# Test: EffectsCalculator.calculate_interactions
# ============================================================================

def test_calculate_interactions_sds1(calc, simple_df_with_residuals, spec_single_factor):
    """Should calculate PDC using cell means for SDS 1."""
    result = calc.calculate_interactions(simple_df_with_residuals, spec_single_factor, sds=1)

    assert 'factor_time' in result
    assert isinstance(result['factor_time'], pd.Series)  # Returns Series


def test_calculate_interactions_sds2(calc, simple_df_with_residuals, spec_single_factor):
    """Should calculate PDC using direct formula for SDS 2."""
    result = calc.calculate_interactions(simple_df_with_residuals, spec_single_factor, sds=2)

    assert 'factor_time' in result
    assert isinstance(result['factor_time'], pd.Series)  # Returns Series


def test_calculate_interactions_sds3(calc, simple_df_with_residuals, spec_single_factor):
    """Should calculate PDC using cell means for SDS 3."""
    result = calc.calculate_interactions(simple_df_with_residuals, spec_single_factor, sds=3)

    assert 'factor_time' in result
    assert isinstance(result['factor_time'], pd.Series)  # Returns Series


def test_calculate_interactions_no_grouping(calc, spec_no_grouping):
    """Should return empty dict when no grouping."""
    df = pd.DataFrame({
        'pull': [1, 2],
        'R3': [0.1, -0.1]
    })

    result = calc.calculate_interactions(df, spec_no_grouping, sds=1)

    assert result == {}


def test_calculate_interactions_missing_r3(calc, spec_single_factor, caplog):
    """Should warn and return empty dict if R3 missing."""
    df = pd.DataFrame({
        'lane': ['A', 'B'],
        'pull': [1, 2],
        'R1': [0.1, -0.1],
        'R2': [0.1, -0.1],
        'R5': [0.1, -0.1]
    })

    with caplog.at_level('WARNING'):
        result = calc.calculate_interactions(df, spec_single_factor, sds=1)

    assert result == {}
    assert 'R3 not found' in caplog.text


# ============================================================================
# Test: Integration Scenarios
# ============================================================================

def test_full_effects_pipeline(calc, simple_df_with_residuals, spec_single_factor):
    """Test complete effects calculation pipeline."""
    # Step 1: Calculate all effects
    effects = calc.calculate_all_effects(simple_df_with_residuals, spec_single_factor)

    # Step 2: Calculate interactions (pass effects for factor_factor calculation)
    interactions = calc.calculate_interactions(
        simple_df_with_residuals, spec_single_factor, sds=1, effects=effects
    )

    # Verify all expected outputs
    assert 'lane' in effects
    assert 'time' in effects
    assert 'lane_MEs' in effects
    assert 'factor_time' in interactions

    # Verify data structure
    assert isinstance(effects['lane'], pd.DataFrame)
    assert isinstance(effects['time'], pd.DataFrame)
    assert isinstance(interactions['factor_time'], pd.Series)  # Returns Series


def test_multi_factor_effects_pipeline(calc, multi_factor_df, spec_multi_factor):
    """Test effects with multiple factors and interactions."""
    effects = calc.calculate_all_effects(multi_factor_df, spec_multi_factor)

    # Should have both factors
    assert 'lane' in effects
    assert 'head' in effects

    # Should have factor_interaction_effects scores in effects
    assert 'factor_interaction_effects' in effects

    # Calculate interactions to get factor_factor
    interactions = calc.calculate_interactions(
        multi_factor_df, spec_multi_factor, sds=1, effects=effects
    )

    # Should have factor × factor interaction in interactions dict
    assert 'factor_factor' in interactions

    # Verify interaction has correct structure
    fi = interactions['factor_factor']
    assert 'lane' in fi.columns
    assert 'head' in fi.columns
    assert 'Rx' in fi.columns  # Column is called Rx


# ============================================================================
# Test: Edge Cases
# ============================================================================

def test_effects_with_missing_values_in_factors():
    """Should handle NaN values gracefully."""
    df = pd.DataFrame({
        'lane': ['A', 'A', np.nan, 'B'],
        'R5': [0.1, 0.2, 0.3, -0.1]
    })

    # Should drop NaN rows during groupby
    result = calculate_factor_main_effects(df, 'lane')

    # Should only have A and B (NaN dropped)
    assert result.shape[0] == 2
    assert set(result['lane']) == {'A', 'B'}


def test_effects_with_single_observation_per_level():
    """Should work with single observation per factor level."""
    df = pd.DataFrame({
        'lane': ['A', 'B', 'C'],
        'R5': [0.1, -0.1, 0.2]
    })

    result = calculate_factor_main_effects(df, 'lane')

    assert result.shape[0] == 3
    # Main effect = R5 when n=1 per level
    assert result[result['lane'] == 'A']['Main_Effect'].iloc[0] == 0.1


def test_effects_preserves_data_types():
    """Should preserve numeric data types."""
    df = pd.DataFrame({
        'lane': ['A', 'A', 'B', 'B'],
        'R5': np.array([0.1, 0.2, -0.1, -0.2], dtype=np.float64)
    })

    result = calculate_factor_main_effects(df, 'lane')

    assert pd.api.types.is_numeric_dtype(result['Main_Effect'])
