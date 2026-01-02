"""
Edge case tests for processbehavior robustness.

These tests verify the library handles unusual, minimal, and edge-case data
gracefully without crashing and produces reasonable output.
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessBehavior

# ============================================================================
# TestMissingData: NaN handling
# ============================================================================

class TestMissingData:
    """Tests for handling missing/NaN data."""

    def test_all_nan_response(self):
        """All NaN in response column should fail gracefully."""
        df = pd.DataFrame({
            'Value': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        # Should either raise ValidationError or handle gracefully
        # Current behavior: check what happens
        study = pb.formulate(response='Value', time='Time')
        # If we get here, verify result is reasonable
        assert study.sds >= 0

    def test_mostly_nan_response(self):
        """Response with >50% NaN should still produce results if enough valid."""
        df = pd.DataFrame({
            'Value': [1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0, np.nan, 9.0, 10.0],
            'Time': list(range(1, 11))
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(response='Value', time='Time')

        # Should handle gracefully
        assert study is not None
        assert study.sds >= 0

    def test_nan_in_factors(self):
        """NaN values in grouping factors."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': ['A', 'A', np.nan, 'B', 'B', np.nan],
            'Time': [1, 2, 1, 1, 2, 2]
        })
        pb = ProcessBehavior(df)

        # Should handle NaN in factors
        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        assert study is not None

    def test_nan_in_time(self):
        """NaN values in time variable."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Time': [1, 2, np.nan, 4, 5]
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(response='Value', time='Time')
        assert study is not None

    def test_sparse_nan_pattern(self):
        """Scattered NaN values throughout data."""
        np.random.seed(42)
        values = np.random.normal(50, 5, 100)
        # Insert NaN at random positions (10%)
        nan_indices = np.random.choice(100, 10, replace=False)
        values[nan_indices] = np.nan

        df = pd.DataFrame({
            'Value': values,
            'Time': list(range(1, 101))
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(response='Value', time='Time')

        assert study is not None
        result = study.execute()
        assert result is not None


# ============================================================================
# TestDegenerateCases: Minimal/edge data structures
# ============================================================================

class TestDegenerateCases:
    """Tests for minimal/degenerate data structures."""

    def test_single_observation(self):
        """Single row dataset should not crash."""
        df = pd.DataFrame({'Value': [100.0]})
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value')
        # SDS 4: single condition over time (implicit via obs_id)
        assert study.sds == 4

    def test_two_observations(self):
        """Two observations - minimum for moving range calculation."""
        df = pd.DataFrame({
            'Value': [100.0, 102.0],
            'Time': [1, 2]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        assert study is not None
        result = study.execute()
        assert result is not None

    def test_three_observations(self):
        """Three observations - minimal time series."""
        df = pd.DataFrame({
            'Value': [100.0, 102.0, 99.0],
            'Time': [1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        # With only 3 observations, may be SDS 0 or 4 depending on detection
        assert study.sds >= 0
        result = study.execute()
        assert result is not None

    def test_single_factor_level(self):
        """Only one level in grouping factor."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Factor': ['A', 'A', 'A', 'A', 'A'],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        # Single factor level should still work
        assert study is not None
        assert study.sds >= 0

    def test_single_time_point(self):
        """Only one time point with multiple factors - requires replication for Xbar."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0],
            'Factor': ['A', 'B', 'C'],
            'Time': [1, 1, 1]
        })
        pb = ProcessBehavior(df)

        # formulate() succeeds (chart-agnostic)
        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        assert study is not None

        # execute(chart='Xbar') fails because can't calculate within-group variance
        # when all subgroups have n=1
        with pytest.raises(ValueError, match="Subgroup size must be >= 2"):
            study.execute(chart='Xbar')

    def test_constant_response(self):
        """All response values identical (zero variance)."""
        df = pd.DataFrame({
            'Value': [50.0, 50.0, 50.0, 50.0, 50.0],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        # Should work - limits will be at center line
        assert result is not None
        chart_df = result.get_chart('Imr')
        assert chart_df is not None
        # Center should equal all values (get first row's center)
        assert chart_df['center'].iloc[0] == 50.0

    def test_near_constant_response(self):
        """Very small variance - numerical stability check."""
        df = pd.DataFrame({
            'Value': [50.0, 50.0 + 1e-10, 50.0 - 1e-10, 50.0 + 1e-10, 50.0],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        # Should not produce NaN or Inf
        chart_df = result.get_chart('Imr')
        assert not chart_df['center'].isna().any()
        assert not np.isinf(chart_df['upl']).any()
        assert not np.isinf(chart_df['lpl']).any()

    def test_two_factor_levels_minimal_obs(self):
        """Two factor levels with 2 observations each."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0],
            'Factor': ['A', 'A', 'B', 'B'],
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'])
        assert study is not None
        result = study.execute()
        assert result is not None


# ============================================================================
# TestExtremeValues: Unusual numeric ranges
# ============================================================================

class TestExtremeValues:
    """Tests for extreme/unusual values."""

    def test_very_large_values(self):
        """Values in 1e15 range."""
        df = pd.DataFrame({
            'Value': [1e15, 1.01e15, 0.99e15, 1.02e15, 0.98e15],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        chart_df = result.get_chart('Imr')
        assert not chart_df['center'].isna().any()
        assert not np.isinf(chart_df['upl']).any()

    def test_very_small_values(self):
        """Values in 1e-15 range."""
        df = pd.DataFrame({
            'Value': [1e-15, 1.01e-15, 0.99e-15, 1.02e-15, 0.98e-15],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        chart_df = result.get_chart('Imr')
        assert not chart_df['center'].isna().any()
        assert not np.isinf(chart_df['upl']).any()

    def test_extreme_outlier(self):
        """Single observation 1000x larger than others."""
        df = pd.DataFrame({
            'Value': [100.0, 101.0, 99.0, 102.0, 100000.0, 100.0, 101.0],
            'Time': list(range(1, 8))
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        # Should complete without crash
        assert result is not None
        chart = result.get_chart('Imr')
        assert chart is not None

    def test_negative_values(self):
        """All negative response values."""
        df = pd.DataFrame({
            'Value': [-100.0, -102.0, -99.0, -101.0, -100.0],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        chart_df = result.get_chart('Imr')
        # Center should be negative
        assert chart_df['center'].iloc[0] < 0

    def test_mixed_sign_values(self):
        """Mix of positive and negative values."""
        df = pd.DataFrame({
            'Value': [-10.0, 5.0, -3.0, 8.0, -1.0, 4.0],
            'Time': list(range(1, 7))
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        chart_df = result.get_chart('Imr')
        assert not chart_df['center'].isna().any()

    def test_values_crossing_zero(self):
        """Values transitioning from negative to positive."""
        df = pd.DataFrame({
            'Value': [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            'Time': list(range(1, 7))
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        assert result is not None

    @pytest.mark.parametrize('scale', [1e-12, 1e-6, 1, 1e6, 1e12])
    def test_various_scales(self, scale):
        """Test across different orders of magnitude."""
        base_values = np.array([100.0, 102.0, 99.0, 101.0, 100.0])
        df = pd.DataFrame({
            'Value': base_values * scale,
            'Time': list(range(1, 6))
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        chart_df = result.get_chart('Imr')
        assert not chart_df['center'].isna().any()
        assert not np.isinf(chart_df['upl']).any()


# ============================================================================
# TestUnbalancedData: Non-uniform designs
# ============================================================================

class TestUnbalancedData:
    """Tests for unbalanced designs."""

    def test_highly_unbalanced_factors(self):
        """One factor level with many obs, another with few."""
        # Factor A: 50 observations, Factor B: 3 observations
        values_a = list(np.random.normal(50, 2, 50))
        values_b = list(np.random.normal(52, 2, 3))

        df = pd.DataFrame({
            'Value': values_a + values_b,
            'Factor': ['A'] * 50 + ['B'] * 3
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'])
        result = study.execute()

        assert result is not None

    def test_missing_cells_factorial(self):
        """Factor x time combinations with no observations."""
        # Create data with some factor/time cells missing
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': ['A', 'A', 'B', 'B', 'A', 'B'],
            'Time': [1, 2, 1, 3, 3, 2]  # A missing at T3 cell, B missing at T2 cell
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        assert study is not None

    def test_sparse_replication(self):
        """Some cells have many replicates, others have 1."""
        np.random.seed(42)
        data = []
        for factor in ['A', 'B']:
            for time in [1, 2, 3]:
                # Vary replication: some cells n=1, others n=10
                n = 10 if (factor == 'A' and time == 1) else 1
                for _ in range(n):
                    data.append({
                        'Value': np.random.normal(50, 2),
                        'Factor': factor,
                        'Time': time
                    })

        df = pd.DataFrame(data)
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        # Use factor-level aggregation since most cells have n=1
        result = study.execute(chart='Xbar', by=['Factor'])

        assert result is not None

    def test_single_replicate_per_cell(self):
        """n=1 in every factor×time cell but replication at factor level."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        # Per Wheeler/Bishop, SDS is based on N_kt (factor × time cells):
        # Each (Factor, Time) cell has exactly n=1, so this is SDS 2
        # Note: Analysis subgrouping uses factor-only (A:3, B:3) for chart selection
        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        assert study.sds == 2  # Wheeler/Bishop: all N_kt = 1 → SDS 2
        assert 'Xbar' in study.valid_charts  # Still valid due to factor-level subgrouping

    def test_unbalanced_time_points(self):
        """Different number of observations per time point."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'Time': [1, 1, 1, 2, 2, 3, 3, 3]  # T1:3, T2:2, T3:3
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        assert result is not None


# ============================================================================
# TestDataTypes: Various input data types
# ============================================================================

class TestDataTypes:
    """Tests for various data types and formats."""

    def test_integer_response(self):
        """Integer response variable (should convert to float)."""
        df = pd.DataFrame({
            'Value': [100, 102, 99, 101, 100],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        assert result is not None

    def test_string_factors(self):
        """String factor levels."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': ['Alpha', 'Alpha', 'Beta', 'Beta', 'Gamma', 'Gamma']
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'])
        assert study is not None

    def test_numeric_factors(self):
        """Numeric factor levels (should work as categorical)."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': [1, 1, 2, 2, 3, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'])
        assert study is not None

    def test_categorical_dtype(self):
        """pandas Categorical dtype for factors."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': pd.Categorical(['A', 'A', 'B', 'B', 'C', 'C'])
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'])
        assert study is not None

    def test_datetime_time_variable(self):
        """Datetime as time variable."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Time': pd.date_range('2024-01-01', periods=5)
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        assert study is not None


# ============================================================================
# TestChartGeneration: Edge cases in chart output
# ============================================================================

class TestChartGeneration:
    """Tests for chart generation with edge case data."""

    def test_chart_with_constant_data(self):
        """Generate chart when all values are identical."""
        df = pd.DataFrame({
            'Value': [100.0] * 20,
            'Time': list(range(1, 21))
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        # Should be able to get chart
        chart_df = result.get_chart('Imr')
        assert chart_df is not None
        # UPL and LPL should equal center (no variation)
        assert (chart_df['upl'] == chart_df['center']).all()
        assert (chart_df['lpl'] == chart_df['center']).all()

    def test_chart_data_columns_present(self):
        """Verify chart data has expected columns."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0],
            'Time': [1, 2, 3, 4, 5]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', time='Time')
        result = study.execute()

        # get_chart returns a DataFrame with chart columns
        chart_df = result.get_chart('Imr')
        assert isinstance(chart_df, pd.DataFrame)
        assert 'center' in chart_df.columns
        assert 'upl' in chart_df.columns  # upper process limit
        assert 'lpl' in chart_df.columns  # lower process limit

    def test_residual_charts_with_minimal_data(self):
        """Residual charts with minimal but valid data."""
        # Create minimal SDS 1 data
        np.random.seed(42)
        data = []
        for factor in ['A', 'B']:
            for time in [1, 2]:
                for _ in range(2):  # n=2 per cell
                    data.append({
                        'Value': np.random.normal(50, 2),
                        'Factor': factor,
                        'Time': time
                    })

        df = pd.DataFrame(data)
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')

        # Should be SDS 1
        assert study.sds == 1

        # Should have residual charts available
        assert len(study.residual_charts) > 0

        # Execute a residual chart using new syntax
        if 'R2_S' in study.residual_charts:
            result = study.execute(chart='S', value='R2')
            assert result is not None
