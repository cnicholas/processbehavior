"""
Edge case tests for processbehavior robustness.

These tests verify the library handles unusual, minimal, and edge-case data
gracefully without crashing and produces reasonable output.
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.exceptions import ValidationError

# ============================================================================
# TestMissingData: NaN handling
# ============================================================================


class TestMissingData:
    """Tests for handling missing/NaN data."""

    def test_nan_in_factors(self):
        """NaN values in grouping factors."""
        df = pd.DataFrame(
            {
                'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'Factor': ['A', 'A', np.nan, 'B', 'B', np.nan],
                'Time': [1, 2, 1, 1, 2, 2],
            }
        )
        pb = ProcessBehavior(df)

        # Should handle NaN in factors
        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        assert study is not None


# ============================================================================
# TestDegenerateCases: Minimal/edge data structures
# ============================================================================


def _degenerate_single_factor_level():
    return pd.DataFrame(
        {'Value': [1.0, 2.0, 3.0, 4.0, 5.0], 'Factor': ['A', 'A', 'A', 'A', 'A'], 'Time': [1, 2, 3, 4, 5]}
    )


def _degenerate_constant_response():
    return pd.DataFrame(
        {
            'Value': [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3],
        }
    )


def _degenerate_near_constant_response():
    return pd.DataFrame(
        {
            'Value': [50.0, 50.0 + 1e-10, 50.0 - 1e-10, 50.0 + 1e-10, 50.0, 50.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3],
        }
    )


class TestDegenerateCases:
    """Tests for minimal/degenerate data structures."""

    @pytest.mark.parametrize(
        'make_df, check',
        [
            pytest.param(
                _degenerate_single_factor_level,
                lambda study, _: study.observed_design_state.sds >= 0,
                id='single_factor_level',
            ),
            pytest.param(
                _degenerate_constant_response,
                lambda _, result: result.get_chart('X')['center'].iloc[0] == 50.0,
                id='constant_response',
            ),
            pytest.param(
                _degenerate_near_constant_response,
                lambda _, result: (
                    not result.get_chart('X')['center'].isna().any()
                    and not np.isinf(result.get_chart('X')['upl']).any()
                    and not np.isinf(result.get_chart('X')['lpl']).any()
                ),
                id='near_constant_response',
            ),
        ],
    )
    def test_degenerate_formulate_execute(self, make_df, check):
        """Degenerate data structures formulate and execute without error."""
        df = make_df()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        assert study is not None
        result = study.execute(chart='X', by=['Factor'])
        assert result is not None
        assert check(study, result)

    def test_single_time_point(self):
        """Only one time point with multiple factors - requires replication for Xbar."""
        df = pd.DataFrame({'Value': [1.0, 2.0, 3.0], 'Factor': ['A', 'B', 'C'], 'Time': [1, 1, 1]})
        pb = ProcessBehavior(df)

        # formulate() succeeds (chart-agnostic)
        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        assert study is not None

        # execute(chart='Xbar') fails because can't calculate within-group variance
        # when all subgroups have n=1 (filtered out, leaving no valid groups)
        with pytest.raises(ValidationError, match='No subgroups with n > 1 found'):
            study.execute(chart='Xbar')

    def test_two_factor_levels_minimal_obs(self):
        """Two factor levels with 2 observations each."""
        df = pd.DataFrame(
            {
                'Value': [1.0, 2.0, 3.0, 4.0],
                'Factor': ['A', 'A', 'B', 'B'],
            }
        )
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'])
        assert study is not None
        result = study.execute()
        assert result is not None


# ============================================================================
# TestExtremeValues: Unusual numeric ranges
# ============================================================================


def _extreme_very_large():
    return pd.DataFrame(
        {
            'Value': [1e15, 1.01e15, 0.99e15, 1.02e15, 0.98e15, 1e15],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3],
        }
    )


def _extreme_very_small():
    return pd.DataFrame(
        {
            'Value': [1e-15, 1.01e-15, 0.99e-15, 1.02e-15, 0.98e-15, 1e-15],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3],
        }
    )


def _extreme_outlier():
    return pd.DataFrame(
        {
            'Value': [100.0, 101.0, 99.0, 102.0, 100000.0, 100.0, 101.0, 100.0],
            'Factor': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )


def _extreme_negative():
    return pd.DataFrame(
        {
            'Value': [-100.0, -102.0, -99.0, -101.0, -100.0, -99.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3],
        }
    )


def _extreme_mixed_sign():
    return pd.DataFrame(
        {
            'Value': [-10.0, 5.0, -3.0, 8.0, -1.0, 4.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3],
        }
    )


def _extreme_crossing_zero():
    return pd.DataFrame(
        {
            'Value': [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3],
        }
    )


class TestExtremeValues:
    """Tests for extreme/unusual values."""

    @pytest.mark.parametrize(
        'make_df, check',
        [
            pytest.param(
                _extreme_very_large,
                lambda c: not c['center'].isna().any() and not np.isinf(c['upl']).any(),
                id='very_large_values',
            ),
            pytest.param(
                _extreme_very_small,
                lambda c: not c['center'].isna().any() and not np.isinf(c['upl']).any(),
                id='very_small_values',
            ),
            pytest.param(
                _extreme_outlier,
                lambda c: c is not None,
                id='extreme_outlier',
            ),
            pytest.param(
                _extreme_negative,
                lambda c: c['center'].iloc[0] < 0,
                id='negative_values',
            ),
            pytest.param(
                _extreme_mixed_sign,
                lambda c: not c['center'].isna().any(),
                id='mixed_sign_values',
            ),
            pytest.param(
                _extreme_crossing_zero,
                lambda c: c is not None,
                id='values_crossing_zero',
            ),
        ],
    )
    def test_extreme_values(self, make_df, check):
        """Extreme numeric values formulate, execute, and produce valid charts."""
        df = make_df()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='X', by=['Factor'])
        assert result is not None
        chart_df = result.get_chart('X')
        assert chart_df is not None
        assert check(chart_df)

    @pytest.mark.parametrize('scale', [1e-12, 1e-6, 1, 1e6, 1e12])
    def test_various_scales(self, scale):
        """Test across different orders of magnitude."""
        base_values = np.array([100.0, 102.0, 99.0, 101.0, 100.0, 99.0])
        df = pd.DataFrame(
            {'Value': base_values * scale, 'Factor': ['A', 'A', 'A', 'B', 'B', 'B'], 'Time': [1, 2, 3, 1, 2, 3]}
        )
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='X', by=['Factor'])

        chart_df = result.get_chart('X')
        assert not chart_df['center'].isna().any()
        assert not np.isinf(chart_df['upl']).any()


# ============================================================================
# TestUnbalancedData: Non-uniform designs
# ============================================================================


def _unbalanced_highly_unbalanced():
    np.random.seed(42)
    values_a = list(np.random.normal(50, 2, 50))
    values_b = list(np.random.normal(52, 2, 3))
    return pd.DataFrame({'Value': values_a + values_b, 'Factor': ['A'] * 50 + ['B'] * 3})


def _unbalanced_missing_cells():
    return pd.DataFrame(
        {'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'Factor': ['A', 'A', 'B', 'B', 'A', 'B'], 'Time': [1, 2, 1, 3, 3, 2]}
    )


def _unbalanced_sparse_replication():
    np.random.seed(42)
    data = []
    for factor in ['A', 'B']:
        for time in [1, 2, 3]:
            n = 10 if (factor == 'A' and time == 1) else 1
            for _ in range(n):
                data.append({'Value': np.random.normal(50, 2), 'Factor': factor, 'Time': time})
    return pd.DataFrame(data)


def _unbalanced_single_replicate():
    return pd.DataFrame(
        {'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'Factor': ['A', 'A', 'A', 'B', 'B', 'B'], 'Time': [1, 2, 3, 1, 2, 3]}
    )


def _unbalanced_time_points():
    return pd.DataFrame(
        {
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'Factor': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
            'Time': [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )


class TestUnbalancedData:
    """Tests for unbalanced designs."""

    @pytest.mark.parametrize(
        'make_df, formulate_kw, execute_kw, check',
        [
            pytest.param(
                _unbalanced_highly_unbalanced,
                dict(response='Value', factors=['Factor']),
                dict(),
                None,
                id='highly_unbalanced_factors',
            ),
            pytest.param(
                _unbalanced_missing_cells,
                dict(response='Value', factors=['Factor'], time='Time'),
                None,  # formulate-only
                None,
                id='missing_cells_factorial',
            ),
            pytest.param(
                _unbalanced_sparse_replication,
                dict(response='Value', factors=['Factor'], time='Time'),
                dict(chart='Xbar', by=['Factor']),
                None,
                id='sparse_replication',
            ),
            pytest.param(
                _unbalanced_single_replicate,
                dict(response='Value', factors=['Factor'], time='Time'),
                None,  # formulate-only, with custom check
                lambda study: (study.observed_design_state.sds == 2 and 'Xbar' in study.valid_charts),
                id='single_replicate_per_cell',
            ),
            pytest.param(
                _unbalanced_time_points,
                dict(response='Value', factors=['Factor'], time='Time'),
                dict(),
                None,
                id='unbalanced_time_points',
            ),
        ],
    )
    def test_unbalanced_data(self, make_df, formulate_kw, execute_kw, check):
        """Unbalanced designs formulate and execute without error."""
        df = make_df()
        pb = ProcessBehavior(df)
        study = pb.formulate(**formulate_kw)
        assert study is not None

        if check is not None:
            assert check(study)

        if execute_kw is not None:
            result = study.execute(**execute_kw)
            assert result is not None


# ============================================================================
# TestDataTypes: Various input data types
# ============================================================================


def _dtype_integer_response():
    return pd.DataFrame(
        {'Value': [100, 102, 99, 101, 100, 99], 'Factor': ['A', 'A', 'A', 'B', 'B', 'B'], 'Time': [1, 2, 3, 1, 2, 3]}
    )


def _dtype_string_factors():
    return pd.DataFrame(
        {'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'Factor': ['Alpha', 'Alpha', 'Beta', 'Beta', 'Gamma', 'Gamma']}
    )


def _dtype_numeric_factors():
    return pd.DataFrame({'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'Factor': [1, 1, 2, 2, 3, 3]})


def _dtype_categorical():
    return pd.DataFrame(
        {'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'Factor': pd.Categorical(['A', 'A', 'B', 'B', 'C', 'C'])}
    )


def _dtype_datetime_time():
    return pd.DataFrame(
        {
            'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
            'Time': list(pd.date_range('2024-01-01', periods=3)) * 2,
        }
    )


class TestDataTypes:
    """Tests for various data types and formats."""

    @pytest.mark.parametrize(
        'make_df, formulate_kw, execute_kw',
        [
            pytest.param(
                _dtype_integer_response,
                dict(response='Value', factors=['Factor'], time='Time'),
                dict(chart='X', by=['Factor']),
                id='integer_response',
            ),
            pytest.param(
                _dtype_string_factors,
                dict(response='Value', factors=['Factor']),
                None,
                id='string_factors',
            ),
            pytest.param(
                _dtype_numeric_factors,
                dict(response='Value', factors=['Factor']),
                None,
                id='numeric_factors',
            ),
            pytest.param(
                _dtype_categorical,
                dict(response='Value', factors=['Factor']),
                None,
                id='categorical_dtype',
            ),
            pytest.param(
                _dtype_datetime_time,
                dict(response='Value', factors=['Factor'], time='Time'),
                None,
                id='datetime_time_variable',
            ),
        ],
    )
    def test_data_types(self, make_df, formulate_kw, execute_kw):
        """Various data types formulate (and execute) without error."""
        df = make_df()
        pb = ProcessBehavior(df)
        study = pb.formulate(**formulate_kw)
        assert study is not None

        if execute_kw is not None:
            result = study.execute(**execute_kw)
            assert result is not None


# ============================================================================
# TestChartGeneration: Edge cases in chart output
# ============================================================================


class TestChartGeneration:
    """Tests for chart generation with edge case data."""

    def test_chart_with_constant_data(self):
        """Generate chart when all values are identical."""
        df = pd.DataFrame({'Value': [100.0] * 20, 'Factor': ['A'] * 10 + ['B'] * 10, 'Time': list(range(1, 11)) * 2})
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='X', by=['Factor'])

        # Should be able to get chart
        chart_df = result.get_chart('X')
        assert chart_df is not None
        # UPL and LPL should equal center (no variation)
        assert (chart_df['upl'] == chart_df['center']).all()
        assert (chart_df['lpl'] == chart_df['center']).all()

    def test_chart_data_columns_present(self):
        """Verify chart data has expected columns."""
        df = pd.DataFrame(
            {
                'Value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'Factor': ['A', 'A', 'A', 'B', 'B', 'B'],
                'Time': [1, 2, 3, 1, 2, 3],
            }
        )
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')
        result = study.execute(chart='X', by=['Factor'])

        # get_chart returns a DataFrame with chart columns
        chart_df = result.get_chart('X')
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
                    data.append({'Value': np.random.normal(50, 2), 'Factor': factor, 'Time': time})

        df = pd.DataFrame(data)
        pb = ProcessBehavior(df)

        study = pb.formulate(response='Value', factors=['Factor'], time='Time')

        # Should be SDS 1
        assert study.observed_design_state.sds == 1

        # Should have residual charts available
        assert len(study.residual_charts) > 0

        # Execute a residual chart using new syntax
        if ('S', 'R2') in study.residual_charts:
            result = study.execute(chart='S', value='R2')
            assert result is not None


# ============================================================================
# TestPartialReplication: SDS 3 (mixed n=1 and n>=2 cells)
# ============================================================================


class TestPartialReplication:
    """Tests for SDS 3 partial replication handling."""

    def test_xbar_with_sds3_partial_replication(self):
        """Xbar chart works with SDS 3 data (mixed n=1 and n>=2 cells)."""
        from processbehavior.datasets.synthetic import make_design

        # SDS 3: partial replication (50% cells have n>=2)
        df = make_design(3, K1=3, K2=2, T=8, p_replicated=0.5, n_when_replicated=3, seed=42)

        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')

        # Verify we have SDS 3
        assert study.observed_design_state.sds == 3

        # Should NOT raise ValueError - n=1 groups are filtered
        result = study.execute(chart='Xbar')

        # Verify we got valid results
        assert 'Xbar' in result.charts
        xbar_data = result.get_chart('Xbar')
        assert len(xbar_data) > 0

    def test_s_chart_with_sds3_partial_replication(self):
        """S chart also works with SDS 3 data (filters n=1 groups)."""
        from processbehavior.datasets.synthetic import make_design

        df = make_design(3, K1=3, K2=2, T=8, p_replicated=0.5, n_when_replicated=3, seed=42)

        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')

        # S chart should also work
        result = study.execute(chart='S')

        assert 'S' in result.charts
        s_data = result.get_chart('S')
        assert len(s_data) > 0
