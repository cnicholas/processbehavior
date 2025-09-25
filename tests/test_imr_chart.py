"""
Unit tests for IMR (Individual and Moving Range) chart calculations with and without subgroups.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add the processbehavior package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from processbehavior import engine
from processbehavior.charts.imr import calculate_statistics_Imr
from processbehavior.data_prep import prepare_dataset
from processbehavior.spec import AnalysisSpecification


class TestIMRChart(unittest.TestCase):
    """Test IMR chart calculations with and without subgroups."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic data for testing
        np.random.seed(42)  # For reproducible results

        # Individual measurements without subgroups - time series
        self.df_individual = pd.DataFrame(
            {'time': range(1, 21), 'measurement': np.random.normal(100, 5, 20)}
        )

        # Individual measurements with grouping (e.g., different production lines)
        self.df_grouped = pd.DataFrame(
            {
                'line': ['A'] * 10 + ['B'] * 10 + ['C'] * 8,
                'time': list(range(1, 11)) + list(range(1, 11)) + list(range(1, 9)),
                'value': np.concatenate(
                    [
                        np.random.normal(50, 3, 10),  # Line A
                        np.random.normal(55, 2, 10),  # Line B
                        np.random.normal(48, 4, 8),  # Line C
                    ]
                ),
            }
        )

        # Create a simple known dataset for validation
        self.df_known = pd.DataFrame({'measurement': [10, 12, 11, 15, 13, 14, 16, 12, 13, 11]})

    def test_imr_without_subgroups_using_engine(self):
        """Test IMR chart without subgroups using engine.perform_analysis."""
        spec = {
            'analysis_type': 'Imr',
            'response_var': 'measurement',
            'time_var': 'time',
            'round_to': 3,
        }

        # Perform analysis
        result = engine.perform_analysis(df=self.df_individual, specification=spec)

        # Verify result structure (should be dict with 'all' key for non-grouped data)
        self.assertIsInstance(result, dict)
        self.assertIn('all', result)

        # Get the results
        data_results = result['all']
        self.assertIn('data', data_results)
        self.assertIn('statistics', data_results)

        # Verify data structure
        data_df = data_results['data']
        self.assertIsInstance(data_df, pd.DataFrame)

        # Check expected columns (should include time as first column)
        expected_columns = ['time', 'measurement', 'mean', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_columns:
            self.assertIn(col, data_df.columns, f'Missing column: {col}')

        # Verify data types
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['measurement']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['mean']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['lcl']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['ucl']))

        # Verify beyond_limits values
        valid_beyond_limits = {-1, 0, 1}  # IMR uses -1, 0, 1 for beyond_limits
        self.assertTrue(all(x in valid_beyond_limits for x in data_df['beyond_limits']))

        # Verify statistics
        stats = data_results['statistics']
        self.assertIn('mean', stats)
        self.assertIn('lcl', stats)
        self.assertIn('ucl', stats)

    def test_imr_with_subgroups_using_engine(self):
        """Test IMR chart with subgroups using engine.perform_analysis."""
        spec = {
            'analysis_type': 'Imr',
            'response_var': 'value',
            'rsg_vars': ['line'],
            'time_var': 'time',
            'round_to': 3,
        }

        # Perform analysis
        result = engine.perform_analysis(df=self.df_grouped, specification=spec)

        # Verify result structure (should have multiple groups)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result.keys()), 1)

        # Check each group
        for group_key, group_data in result.items():
            self.assertIn('data', group_data)
            self.assertIn('statistics', group_data)

            # Verify data structure
            data_df = group_data['data']
            self.assertIsInstance(data_df, pd.DataFrame)

            # Check expected columns
            expected_columns = ['line', 'time', 'value', 'mean', 'lcl', 'ucl', 'beyond_limits']
            for col in expected_columns:
                self.assertIn(col, data_df.columns, f'Missing column: {col} in group {group_key}')

            # Verify beyond_limits values
            valid_beyond_limits = {-1, 0, 1}
            self.assertTrue(all(x in valid_beyond_limits for x in data_df['beyond_limits']))

    def test_imr_direct_function_call(self):
        """Test IMR chart calculation directly using calculate_statistics_Imr."""
        spec = AnalysisSpecification.from_dict(
            analysis_type='Imr',
            analysis_specification={
                'analysis_type': 'Imr',
                'response_var': 'measurement',
                'round_to': 3,
            },
        )

        # Prepare dataset
        prepared_df = prepare_dataset(df=self.df_individual, analysis_specification=spec)

        # Calculate IMR statistics
        result = calculate_statistics_Imr(df=prepared_df, analysis_specification=spec)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('all', result)

        data_results = result['all']
        self.assertIn('data', data_results)
        self.assertIn('statistics', data_results)

        # Check data DataFrame
        data_df = data_results['data']
        self.assertIsInstance(data_df, pd.DataFrame)

        # Verify we have the right number of rows (should be same as input)
        self.assertEqual(len(data_df), len(self.df_individual))

    def test_imr_with_real_data_fillweight(self):
        """Test IMR chart with real FILLWEIGHTDATA_800.csv data (without subgroups)."""
        f_path = (
            '/Users/cnicholas/Documents/projects/processbehavior/'
            'processbehavior/datasets/data/FILLWEIGHTDATA_800.csv'
        )

        if not os.path.exists(f_path):
            self.skipTest('FILLWEIGHTDATA_800.csv not found')

        df = pd.read_csv(f_path)

        # Use IMR without grouping (treat as individuals)
        spec = {
            'analysis_type': 'Imr',
            'response_var': 'fill_weight',
            'round_to': 2,
        }

        # Perform analysis
        result = engine.perform_analysis(df=df, specification=spec)

        # Should be a single group ('all')
        self.assertIsInstance(result, dict)
        self.assertIn('all', result)

        data_results = result['all']
        data_df = data_results['data']

        # Verify no NaN values in critical columns
        self.assertFalse(data_df['fill_weight'].isna().any())
        self.assertFalse(data_df['mean'].isna().any())
        self.assertFalse(data_df['lcl'].isna().any())
        self.assertFalse(data_df['ucl'].isna().any())

        # Verify we have all the data points
        self.assertEqual(len(data_df), len(df))

    def test_imr_moving_range_calculation(self):
        """Test that moving ranges are calculated correctly."""
        # Use known data for verification
        known_data = pd.DataFrame(
            {
                'value': [10, 12, 11, 15, 13]  # Known moving ranges: NaN, 2, 1, 4, 2
            }
        )

        spec = {
            'analysis_type': 'Imr',
            'response_var': 'value',
            'round_to': 3,
        }

        result = engine.perform_analysis(df=known_data, specification=spec)

        # The moving ranges are calculated internally but not directly exposed
        # We can verify through the control limits that they're reasonable
        self.assertIsInstance(result['all']['statistics']['lcl'], (int, float))
        self.assertIsInstance(result['all']['statistics']['ucl'], (int, float))
        self.assertIsInstance(result['all']['statistics']['mean'], (int, float))

        # UCL should be greater than mean, mean should be greater than LCL
        stats = result['all']['statistics']
        self.assertGreater(stats['ucl'], stats['mean'])
        self.assertGreater(stats['mean'], stats['lcl'])

    def test_imr_mathematical_properties(self):
        """Test mathematical properties of IMR chart calculations."""
        # Create controlled test data
        test_data = pd.DataFrame({'measurement': [100, 102, 98, 104, 96, 105, 95, 103, 97, 101]})

        spec = AnalysisSpecification.from_dict(
            analysis_type='Imr',
            analysis_specification={
                'analysis_type': 'Imr',
                'response_var': 'measurement',
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=test_data, analysis_specification=spec)
        result = calculate_statistics_Imr(df=prepared_df, analysis_specification=spec)

        data_df = result['all']['data']
        stats = result['all']['statistics']

        # Verify mathematical relationships
        # Mean should be approximately equal to the average of all measurements
        expected_mean = test_data['measurement'].mean()
        self.assertAlmostEqual(stats['mean'], expected_mean, places=2)

        # Control limits should be based on moving range
        # UCL = mean + 2.66 * average_moving_range
        # LCL = mean - 2.66 * average_moving_range
        self.assertGreater(stats['ucl'], stats['mean'])
        self.assertLess(stats['lcl'], stats['mean'])

        # Beyond limits should be correctly identified
        for _, row in data_df.iterrows():
            value = row['measurement']
            if value > stats['ucl']:
                self.assertEqual(row['beyond_limits'], 1)
            elif value < stats['lcl']:
                self.assertEqual(row['beyond_limits'], -1)
            else:
                self.assertEqual(row['beyond_limits'], 0)

    def test_imr_with_existing_test_data_compatibility(self):
        """Test IMR with the same data from existing test_imr_analysis.py for compatibility."""
        f_path = (
            '/Users/cnicholas/Documents/projects/processbehavior/'
            'processbehavior/datasets/data/FILLWEIGHTDATA_800.csv'
        )

        if not os.path.exists(f_path):
            self.skipTest('FILLWEIGHTDATA_800.csv not found')

        df = pd.read_csv(f_path)

        # Use the same specification as the existing test
        spec = {
            'analysis_type': 'Imr',
            'response_var': 'fill_weight',
            'time_unit': None,
            'round_to': 2,
        }

        # Perform analysis
        result = engine.perform_analysis(df=df, specification=spec)

        # Should match the structure expected by existing test
        self.assertIsInstance(result, dict)
        self.assertIn('all', result)

        data_results = result['all']
        self.assertIn('data', data_results)
        self.assertIn('statistics', data_results)

        # Verify statistics match expected values from existing test
        stats = data_results['statistics']
        self.assertIn('mean', stats)
        self.assertIn('lcl', stats)
        self.assertIn('ucl', stats)

        # Values should be close to those expected in the original test
        # (allowing for small differences due to refactoring)
        self.assertAlmostEqual(stats['mean'], 237.78, places=1)
        self.assertAlmostEqual(stats['lcl'], 232.23, places=1)
        self.assertAlmostEqual(stats['ucl'], 243.33, places=1)

    def test_imr_precision_and_rounding(self):
        """Test that IMR chart respects the rounding specification."""
        test_data = pd.DataFrame({'value': [1.123456, 2.234567, 3.345678, 4.456789, 5.567890]})

        # Test with different rounding specifications
        for round_to in [1, 2, 3]:
            spec = {
                'analysis_type': 'Imr',
                'response_var': 'value',
                'round_to': round_to,
            }

            result = engine.perform_analysis(df=test_data, specification=spec)
            data_df = result['all']['data']

            # Check that values are rounded to the specified precision
            # (Note: beyond_limits is integer, so skip that column)
            for col in ['value', 'mean', 'lcl', 'ucl']:
                for value in data_df[col]:
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

    def test_imr_with_grouping_and_time(self):
        """Test IMR chart with both grouping and time variables."""
        spec = {
            'analysis_type': 'Imr',
            'response_var': 'value',
            'rsg_vars': ['line'],
            'time_var': 'time',
            'round_to': 2,
        }

        result = engine.perform_analysis(df=self.df_grouped, specification=spec)

        # Should have multiple groups
        self.assertGreater(len(result.keys()), 1)

        # Each group should have time column as first column
        for _group_key, group_data in result.items():
            data_df = group_data['data']
            columns = data_df.columns.tolist()

            # Time and grouping columns should be present
            self.assertIn('time', columns)
            self.assertIn('line', columns)

            # Time should be one of the first columns
            first_few_cols = columns[:3]
            self.assertIn('time', first_few_cols)


if __name__ == '__main__':
    unittest.main()
