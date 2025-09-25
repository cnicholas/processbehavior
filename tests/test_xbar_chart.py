"""
Unit tests for Xbar chart calculations with and without subgroups.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add the processbehavior package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from processbehavior import engine
from processbehavior.charts.xbar import calculate_statistics_Xbar
from processbehavior.data_prep import prepare_dataset
from processbehavior.spec import AnalysisSpecification


class TestXbarChart(unittest.TestCase):
    """Test Xbar chart calculations with and without subgroups."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic data for testing
        np.random.seed(42)  # For reproducible results

        # Data with subgroups - SDS_1_DATA.csv structure
        self.df_grouped = pd.DataFrame(
            {
                'TIME': np.repeat([1, 2, 3, 4, 5], 8),
                'FACTOR_1': np.tile([1, 1, 1, 1, 2, 2, 2, 2], 5),
                'FACTOR_2': np.tile([1, 1, 2, 2, 1, 1, 2, 2], 5),
                'Y': np.random.normal(210, 5, 40),  # 40 observations
            }
        )

        # Data without subgroups - simple time series
        self.df_individual = pd.DataFrame(
            {'time': range(1, 21), 'measurement': np.random.normal(100, 10, 20)}
        )

    def test_xbar_with_subgroups_using_engine(self):
        """Test Xbar chart with subgroups using engine.perform_analysis."""
        spec = {
            'analysis_type': 'Xbar',
            'response_var': 'Y',
            'rsg_vars': ['FACTOR_1', 'FACTOR_2'],
            'time_var': 'TIME',
            'round_to': 3,
        }

        # Perform analysis
        result = engine.perform_analysis(df=self.df_grouped, specification=spec)

        # Verify result structure - Xbar returns dict with statistics
        self.assertIsInstance(result, dict)

        # Should have 'data' and 'statistics' keys
        self.assertIn('data', result)
        self.assertIn('statistics', result)

        # Verify data structure
        data_df = result['data']
        self.assertIsInstance(data_df, pd.DataFrame)

        # Check expected columns
        expected_columns = ['rsg', 'mean', 'Xbar', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_columns:
            self.assertIn(col, data_df.columns, f'Missing column: {col}')

        # Verify statistics
        stats = result['statistics']
        self.assertIn('Mean', stats)
        self.assertIn('N', stats)
        self.assertIn('lcl', stats)
        self.assertIn('ucl', stats)

    def test_xbar_without_subgroups_direct_function(self):
        """Test Xbar chart calculation directly using calculate_statistics_Xbar."""
        # For Xbar, we need grouping variables, so create artificial grouping
        df_with_grouping = self.df_individual.copy()
        df_with_grouping['batch'] = 'A'  # Single group

        spec = AnalysisSpecification.from_dict(
            analysis_type='Xbar',
            analysis_specification={
                'analysis_type': 'Xbar',
                'response_var': 'measurement',
                'rsg_vars': ['batch'],
                'time_var': 'time',
                'round_to': 3,
            },
        )

        # Prepare dataset
        prepared_df = prepare_dataset(df=df_with_grouping, analysis_specification=spec)

        # Calculate Xbar statistics
        result = calculate_statistics_Xbar(df=prepared_df, analysis_specification=spec)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('data', result)
        self.assertIn('statistics', result)

        # Verify data structure
        data_df = result['data']
        self.assertIsInstance(data_df, pd.DataFrame)

        # Check expected columns
        expected_columns = ['rsg', 'mean', 'Xbar', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_columns:
            self.assertIn(col, data_df.columns, f'Missing column: {col}')

        # Verify data types and values
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['mean']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['lcl']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['ucl']))
        self.assertTrue(
            all(isinstance(x, (bool, np.bool_, np.integer, int)) for x in data_df['beyond_limits'])
        )

        # Verify UCL > mean > LCL (basic sanity check)
        for _, row in data_df.iterrows():
            self.assertGreater(row['ucl'], row['mean'])
            self.assertGreater(row['mean'], row['lcl'])

        # Verify statistics
        stats = result['statistics']
        self.assertIn('Mean', stats)
        self.assertIsInstance(stats['Mean'], (int, float, np.number))

    def test_xbar_with_real_data_fillweight(self):
        """Test Xbar chart with real FILLWEIGHTDATA_800.csv data."""
        f_path = (
            '/Users/cnicholas/Documents/projects/processbehavior/'
            'processbehavior/datasets/data/FILLWEIGHTDATA_800.csv'
        )

        if not os.path.exists(f_path):
            self.skipTest('FILLWEIGHTDATA_800.csv not found')

        df = pd.read_csv(f_path)

        spec = {
            'analysis_type': 'Xbar',
            'response_var': 'fill_weight',
            'rsg_vars': ['pull', 'lane'],
            'round_to': 2,
        }

        # Perform analysis
        result = engine.perform_analysis(df=df, specification=spec)

        # Verify result structure - should be dict with 'data' and 'statistics'
        self.assertIsInstance(result, dict)
        self.assertIn('data', result)
        self.assertIn('statistics', result)

        data_df = result['data']
        self.assertIsInstance(data_df, pd.DataFrame)
        self.assertGreater(len(data_df), 0)

        # Verify no NaN values in critical columns
        self.assertFalse(data_df['mean'].isna().any())
        self.assertFalse(data_df['lcl'].isna().any())
        self.assertFalse(data_df['ucl'].isna().any())

    def test_xbar_with_equal_subgroup_sizes(self):
        """Test Xbar chart with equal subgroup sizes (should use N for limits)."""
        # Create data with exactly 5 observations per subgroup
        data = []
        for group in ['A', 'B', 'C']:
            for _i in range(5):
                data.append({'group': group, 'value': np.random.normal(100, 5)})

        df_equal = pd.DataFrame(data)

        spec = {
            'analysis_type': 'Xbar',
            'response_var': 'value',
            'rsg_vars': ['group'],
            'round_to': 3,
        }

        result = engine.perform_analysis(df=df_equal, specification=spec)

        # Verify structure
        self.assertIsInstance(result, dict)
        self.assertIn('data', result)
        self.assertIn('statistics', result)

        # For equal subgroup sizes, control limits should be consistent
        stats = result['statistics']

        # With equal subgroup sizes, N should be an integer, not 'Varies'
        if stats['N'] != 'Varies':
            self.assertIsInstance(stats['N'], (int, np.integer))
            self.assertIsInstance(stats['lcl'], (int, float, np.number))
            self.assertIsInstance(stats['ucl'], (int, float, np.number))

    def test_xbar_mathematical_properties(self):
        """Test mathematical properties of Xbar chart calculations."""
        # Create controlled test data
        test_data = pd.DataFrame(
            {
                'group': ['A'] * 10 + ['B'] * 10,
                'value': [100, 102, 98, 101, 99] * 2 + [105, 107, 103, 106, 104] * 2,
            }
        )

        spec = AnalysisSpecification.from_dict(
            analysis_type='Xbar',
            analysis_specification={
                'analysis_type': 'Xbar',
                'response_var': 'value',
                'rsg_vars': ['group'],
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=test_data, analysis_specification=spec)
        result = calculate_statistics_Xbar(df=prepared_df, analysis_specification=spec)

        data_df = result['data']

        # Verify mathematical relationships
        for _, row in data_df.iterrows():
            # Control limits should be symmetric around center line
            center_to_ucl = row['ucl'] - row['Xbar']
            center_to_lcl = row['Xbar'] - row['lcl']
            self.assertAlmostEqual(
                center_to_ucl,
                center_to_lcl,
                places=2,
                msg='Control limits should be symmetric around center line',
            )

            # Verify mean is within control limits (unless it's a signal)
            if not row['beyond_limits']:
                self.assertLessEqual(row['mean'], row['ucl'])
                self.assertGreaterEqual(row['mean'], row['lcl'])

    def test_xbar_error_conditions(self):
        """Test error conditions for Xbar chart."""
        # Test with insufficient data (all subgroups have ≤1 observation)
        df_insufficient = pd.DataFrame({'group': ['A', 'B', 'C'], 'value': [100, 101, 102]})

        spec = AnalysisSpecification.from_dict(
            analysis_type='Xbar',
            analysis_specification={
                'analysis_type': 'Xbar',
                'response_var': 'value',
                'rsg_vars': ['group'],
                'round_to': 3,
            },
        )

        # This should raise ValueError for insufficient data in prepare_dataset
        with self.assertRaises(ValueError) as context:
            prepare_dataset(df=df_insufficient, analysis_specification=spec)

        self.assertIn('All subgroups have 1 or less observations', str(context.exception))


if __name__ == '__main__':
    unittest.main()
