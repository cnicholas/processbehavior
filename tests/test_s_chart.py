"""
Unit tests for S chart calculations with and without subgroups.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add the processbehavior package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from processbehavior import engine
from processbehavior.charts.s import calculate_statistics_S
from processbehavior.data_prep import prepare_dataset
from processbehavior.spec import AnalysisSpecification


class TestSChart(unittest.TestCase):
    """Test S chart calculations with and without subgroups."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic data for testing
        np.random.seed(42)  # For reproducible results

        # Data with subgroups - varying subgroup sizes
        self.df_grouped = pd.DataFrame(
            {
                'batch': np.repeat(['A', 'B', 'C', 'D'], [5, 4, 6, 5]),  # Different subgroup sizes
                'measurement': np.concatenate(
                    [
                        np.random.normal(100, 2, 5),  # Batch A
                        np.random.normal(105, 3, 4),  # Batch B
                        np.random.normal(98, 1.5, 6),  # Batch C
                        np.random.normal(102, 2.5, 5),  # Batch D
                    ]
                ),
            }
        )

        # Data with equal subgroup sizes
        self.df_equal_groups = pd.DataFrame(
            {
                'group': np.repeat(['X', 'Y', 'Z'], 5),
                'value': np.concatenate(
                    [
                        np.random.normal(50, 4, 5),  # Group X
                        np.random.normal(55, 3, 5),  # Group Y
                        np.random.normal(48, 5, 5),  # Group Z
                    ]
                ),
            }
        )

    def test_s_chart_with_subgroups_using_engine(self):
        """Test S chart with subgroups using engine.perform_analysis."""
        spec = {
            'analysis_type': 'S',
            'response_var': 'measurement',
            'rsg_vars': ['batch'],
            'round_to': 3,
        }

        # Perform analysis
        result = engine.perform_analysis(df=self.df_grouped, specification=spec)

        # Verify result structure - S chart returns a DataFrame directly
        self.assertIsInstance(result, pd.DataFrame)

        # Check expected columns
        expected_columns = ['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_columns:
            self.assertIn(col, result.columns, f'Missing column: {col}')

        # Verify data types
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in result['s']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in result['S']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in result['lcl']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in result['ucl']))
        self.assertTrue(
            all(isinstance(x, (bool, np.bool_, np.integer, int)) for x in result['beyond_limits'])
        )

        # Verify mathematical relationships
        for _, row in result.iterrows():
            # Standard deviation should be positive
            self.assertGreaterEqual(row['s'], 0)
            self.assertGreater(row['S'], 0)  # Average standard deviation should be positive

            # UCL should be greater than LCL
            self.assertGreaterEqual(row['ucl'], row['lcl'])

    def test_s_chart_direct_function_call(self):
        """Test S chart calculation directly using calculate_statistics_S."""
        spec = AnalysisSpecification.from_dict(
            analysis_type='S',
            analysis_specification={
                'analysis_type': 'S',
                'response_var': 'measurement',
                'rsg_vars': ['batch'],
                'round_to': 3,
            },
        )

        # Prepare dataset
        prepared_df = prepare_dataset(df=self.df_grouped, analysis_specification=spec)

        # Calculate S statistics
        result = calculate_statistics_S(df=prepared_df, analysis_specification=spec)

        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)

        # Check expected columns
        expected_columns = ['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_columns:
            self.assertIn(col, result.columns, f'Missing column: {col}')

        # Verify no NaN values in critical columns
        self.assertFalse(result['s'].isna().any())
        self.assertFalse(result['S'].isna().any())
        self.assertFalse(result['lcl'].isna().any())
        self.assertFalse(result['ucl'].isna().any())

        # Each subgroup should have s >= 0
        self.assertTrue(all(result['s'] >= 0))

    def test_s_chart_with_equal_subgroup_sizes(self):
        """Test S chart with equal subgroup sizes (should produce consistent limits)."""
        spec = {
            'analysis_type': 'S',
            'response_var': 'value',
            'rsg_vars': ['group'],
            'round_to': 3,
        }

        result = engine.perform_analysis(df=self.df_equal_groups, specification=spec)

        # Verify structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)  # Should have 3 groups

        # For equal subgroup sizes, control limits should be the same for all groups
        ucl_values = result['ucl'].unique()
        lcl_values = result['lcl'].unique()

        # Should have consistent control limits (allowing for small numerical differences)
        if len(ucl_values) > 1:
            ucl_range = max(ucl_values) - min(ucl_values)
            self.assertLess(
                ucl_range, 0.001, 'UCL values should be nearly identical for equal subgroup sizes'
            )

        if len(lcl_values) > 1:
            lcl_range = max(lcl_values) - min(lcl_values)
            self.assertLess(
                lcl_range, 0.001, 'LCL values should be nearly identical for equal subgroup sizes'
            )

    def test_s_chart_with_real_data_fillweight(self):
        """Test S chart with real FILLWEIGHTDATA_800.csv data."""
        f_path = (
            '/Users/cnicholas/Documents/projects/processbehavior/'
            'processbehavior/datasets/data/FILLWEIGHTDATA_800.csv'
        )

        if not os.path.exists(f_path):
            self.skipTest('FILLWEIGHTDATA_800.csv not found')

        df = pd.read_csv(f_path)

        spec = {
            'analysis_type': 'S',
            'response_var': 'fill_weight',
            'rsg_vars': ['pull', 'lane'],
            'round_to': 2,
        }

        # Perform analysis
        result = engine.perform_analysis(df=df, specification=spec)

        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

        # Verify no NaN values in critical columns
        self.assertFalse(result['s'].isna().any())
        self.assertFalse(result['S'].isna().any())
        self.assertFalse(result['lcl'].isna().any())
        self.assertFalse(result['ucl'].isna().any())

        # All standard deviations should be non-negative
        self.assertTrue(all(result['s'] >= 0))
        self.assertGreater(result['S'].iloc[0], 0)

    def test_s_chart_removes_single_observation_subgroups(self):
        """Test that S chart properly removes subgroups with single observations."""
        # Create data where some subgroups have only 1 observation
        df_mixed = pd.DataFrame(
            {
                'group': ['A', 'A', 'A', 'B', 'C', 'C', 'C', 'C'],  # B has only 1 observation
                'value': [10, 12, 11, 50, 20, 22, 21, 19],
            }
        )

        spec = {
            'analysis_type': 'S',
            'response_var': 'value',
            'rsg_vars': ['group'],
            'round_to': 3,
        }

        result = engine.perform_analysis(df=df_mixed, specification=spec)

        # Should only have groups A and C (B should be removed)
        self.assertEqual(len(result), 2)
        group_names = result['rsg'].tolist()
        self.assertIn('A', group_names)
        self.assertIn('C', group_names)
        self.assertNotIn('B', group_names)  # Single observation group should be removed

    def test_s_chart_mathematical_properties(self):
        """Test mathematical properties of S chart calculations."""
        # Create controlled test data with known standard deviations
        test_data = pd.DataFrame(
            {
                'group': ['G1'] * 5 + ['G2'] * 5,
                'value': [10, 20, 30, 40, 50, 100, 110, 120, 130, 140],  # Different spreads
            }
        )

        spec = AnalysisSpecification.from_dict(
            analysis_type='S',
            analysis_specification={
                'analysis_type': 'S',
                'response_var': 'value',
                'rsg_vars': ['group'],
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=test_data, analysis_specification=spec)
        result = calculate_statistics_S(df=prepared_df, analysis_specification=spec)

        # Verify mathematical relationships
        for _, row in result.iterrows():
            # Standard deviation should be non-negative
            self.assertGreaterEqual(row['s'], 0)

            # UCL should be greater than or equal to LCL
            self.assertGreaterEqual(row['ucl'], row['lcl'])

            # LCL should be non-negative (can't have negative standard deviation)
            self.assertGreaterEqual(row['lcl'], 0)

            # Center line (S) should be positive
            self.assertGreater(row['S'], 0)

    def test_s_chart_with_sds_data(self):
        """Test S chart with SDS_1_DATA.csv if available."""
        f_path = (
            '/Users/cnicholas/Documents/projects/processbehavior/'
            'processbehavior/datasets/data/SDS_1_DATA.csv'
        )

        if not os.path.exists(f_path):
            self.skipTest('SDS_1_DATA.csv not found')

        df = pd.read_csv(f_path)

        spec = {
            'analysis_type': 'S',
            'response_var': 'Y',
            'rsg_vars': ['FACTOR_1', 'FACTOR_2'],
            'time_var': 'TIME',
            'round_to': 3,
        }

        # Perform analysis
        result = engine.perform_analysis(df=df, specification=spec)

        # Verify result structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

        # Check that we have valid results
        self.assertTrue(all(result['S'] > 0))
        self.assertTrue(all(result['s'] >= 0))
        self.assertTrue(all(result['ucl'] >= result['lcl']))

    def test_s_chart_precision_and_rounding(self):
        """Test that S chart respects the rounding specification."""
        test_data = pd.DataFrame(
            {
                'group': ['A'] * 5 + ['B'] * 5,
                'value': [
                    1.123456,
                    2.234567,
                    3.345678,
                    4.456789,
                    5.567890,
                    6.678901,
                    7.789012,
                    8.890123,
                    9.901234,
                    10.012345,
                ],
            }
        )

        # Test with different rounding specifications
        for round_to in [1, 2, 3]:
            spec = {
                'analysis_type': 'S',
                'response_var': 'value',
                'rsg_vars': ['group'],
                'round_to': round_to,
            }

            result = engine.perform_analysis(df=test_data, specification=spec)

            # Check that all values are rounded to the specified precision
            for col in ['s', 'S', 'lcl', 'ucl']:
                for value in result[col]:
                    # Convert to string to check decimal places
                    str_value = f'{value:.{round_to + 2}f}'  # Add buffer for checking
                    decimal_part = str_value.split('.')[-1]
                    # Count significant digits after decimal (ignoring trailing zeros)
                    significant_decimals = len(decimal_part.rstrip('0'))
                    self.assertLessEqual(
                        significant_decimals,
                        round_to,
                        f'Value {value} in column {col} not rounded to {round_to} places',
                    )


if __name__ == '__main__':
    unittest.main()
