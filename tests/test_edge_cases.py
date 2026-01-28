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


# ============================================================================
# TestDegenerateCases: Minimal/edge data structures
# ============================================================================

class TestDegenerateCases:
    """Tests for minimal/degenerate data structures."""

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
        # when all subgroups have n=1 (filtered out, leaving no valid groups)
        with pytest.raises(ValueError, match="All subgroups have 1 or less observations"):
            study.execute(chart='Xbar')

    def test_constant_response(self):
        """All response values identical (zero variance)."""
        df = pd.DataFrame({
            'Value': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

        # Should work - limits will be at center line
        assert result is not None
        chart_df = result.get_chart('Imr')
        assert chart_df is not None
        # Center should equal all values (get first row's center)
        assert chart_df['center'].iloc[0] == 50.0

    def test_near_constant_response(self):
        """Very small variance - numerical stability check."""
        df = pd.DataFrame({
            'Value': [50.0, 50.0 + 1e-10, 50.0 - 1e-10, 50.0 + 1e-10, 50.0, 50.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

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
            'Value': [1e15, 1.01e15, 0.99e15, 1.02e15, 0.98e15, 1e15],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

        chart_df = result.get_chart('Imr')
        assert not chart_df['center'].isna().any()
        assert not np.isinf(chart_df['upl']).any()

    def test_very_small_values(self):
        """Values in 1e-15 range."""
        df = pd.DataFrame({
            'Value': [1e-15, 1.01e-15, 0.99e-15, 1.02e-15, 0.98e-15, 1e-15],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

        chart_df = result.get_chart('Imr')
        assert not chart_df['center'].isna().any()
        assert not np.isinf(chart_df['upl']).any()

    def test_extreme_outlier(self):
        """Single observation 1000x larger than others."""
        df = pd.DataFrame({
            'Value': [100.0, 101.0, 99.0, 102.0, 100000.0, 100.0, 101.0, 100.0],
            'Factor': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 4, 1, 2, 3, 4]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

        # Should complete without crash
        assert result is not None
        chart = result.get_chart('Imr')
        assert chart is not None

    def test_negative_values(self):
        """All negative response values."""
        df = pd.DataFrame({
            'Value': [-100.0, -102.0, -99.0, -101.0, -100.0, -99.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

        chart_df = result.get_chart('Imr')
        # Center should be negative
        assert chart_df['center'].iloc[0] < 0

    def test_mixed_sign_values(self):
        """Mix of positive and negative values."""
        df = pd.DataFrame({
            'Value': [-10.0, 5.0, -3.0, 8.0, -1.0, 4.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

        chart_df = result.get_chart('Imr')
        assert not chart_df['center'].isna().any()

    def test_values_crossing_zero(self):
        """Values transitioning from negative to positive."""
        df = pd.DataFrame({
            'Value': [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

        assert result is not None

    @pytest.mark.parametrize('scale', [1e-12, 1e-6, 1, 1e6, 1e12])
    def test_various_scales(self, scale):
        """Test across different orders of magnitude."""
        base_values = np.array([100.0, 102.0, 99.0, 101.0, 100.0, 99.0])
        df = pd.DataFrame({
            'Value': base_values * scale,
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

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
            'Factor': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'Time': [1, 1, 2, 2, 1, 1, 2, 2]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
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
            'Value': [100, 102, 99, 101, 100, 99],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

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
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': list(pd.date_range('2024-01-01', periods=3)) * 2
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
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
            'Factor': ['A'] * 10 + ['B'] * 10,
            'Time': list(range(1, 11)) * 2
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

        # Should be able to get chart
        chart_df = result.get_chart('Imr')
        assert chart_df is not None
        # UPL and LPL should equal center (no variation)
        assert (chart_df['upl'] == chart_df['center']).all()
        assert (chart_df['lpl'] == chart_df['center']).all()

    def test_chart_data_columns_present(self):
        """Verify chart data has expected columns."""
        df = pd.DataFrame({
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3]
        })
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='Imr', by=['Factor'])

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


# ============================================================================
# TestPartialReplication: SDS 3 (mixed n=1 and n>=2 cells)
# ============================================================================

class TestPartialReplication:
    """Tests for SDS 3 partial replication handling."""

    def test_xbar_with_sds3_partial_replication(self):
        """Xbar chart works with SDS 3 data (mixed n=1 and n>=2 cells)."""
        from processbehavior.datasets.synthetic import make_sds

        # SDS 3: partial replication (50% cells have n>=2)
        df = make_sds(3, K1=3, K2=2, T=8, p_replicated=0.5, n_when_replicated=3, seed=42)

        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')

        # Verify we have SDS 3
        assert study.sds == 3

        # Should NOT raise ValueError - n=1 groups are filtered
        result = study.execute(chart='Xbar')

        # Verify we got valid results
        assert 'Xbar' in result.charts
        xbar_data = result.get_chart('Xbar')
        assert len(xbar_data) > 0

    def test_s_chart_with_sds3_partial_replication(self):
        """S chart also works with SDS 3 data (filters n=1 groups)."""
        from processbehavior.datasets.synthetic import make_sds

        df = make_sds(3, K1=3, K2=2, T=8, p_replicated=0.5, n_when_replicated=3, seed=42)

        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')

        # S chart should also work
        result = study.execute(chart='S')

        assert 'S' in result.charts
        s_data = result.get_chart('S')
        assert len(s_data) > 0
