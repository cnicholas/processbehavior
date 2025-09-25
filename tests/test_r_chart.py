"""
Unit tests for R (Range) chart calculations with and without subgroups.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add the processbehavior package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from processbehavior import engine
from processbehavior.charts.r import calculate_statistics_R
from processbehavior.data_prep import prepare_dataset
from processbehavior.spec import AnalysisSpecification


class TestRChart(unittest.TestCase):
    """Test R chart calculations with and without subgroups."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic data for testing
        np.random.seed(42)  # For reproducible results

        # Individual measurements without subgroups - time series for moving ranges
        self.df_individual = pd.DataFrame(
            {'time': range(1, 21), 'measurement': np.random.normal(100, 5, 20)}
        )

        # Individual measurements with grouping (different production lines)
        self.df_grouped = pd.DataFrame(
            {
                'line': ['A'] * 12 + ['B'] * 12 + ['C'] * 10,
                'time': list(range(1, 13)) + list(range(1, 13)) + list(range(1, 11)),
                'value': np.concatenate(
                    [
                        np.random.normal(50, 3, 12),  # Line A
                        np.random.normal(55, 2, 12),  # Line B
                        np.random.normal(48, 4, 10),  # Line C
                    ]
                ),
            }
        )

        # Create a simple known dataset for validation of moving range calculation
        self.df_known = pd.DataFrame(
            {
                'measurement': [10, 12, 11, 15, 13, 14, 16, 12, 13, 11]
                # Expected moving ranges: |12-10|=2, |11-12|=1, |15-11|=4, |13-15|=2, etc.
            }
        )

    def test_r_chart_without_subgroups_using_engine(self):
        """Test R chart without subgroups using engine.perform_analysis."""
        spec = {
            'analysis_type': 'R',
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
        expected_columns = ['time', 'mr', 'mR', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_columns:
            self.assertIn(col, data_df.columns, f'Missing column: {col}')

        # Verify data types
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['mr']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['mR']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['lcl']))
        self.assertTrue(all(isinstance(x, (int, float, np.number)) for x in data_df['ucl']))

        # Verify beyond_limits values
        valid_beyond_limits = {-1, 0, 1}  # R uses -1, 0, 1 for beyond_limits
        self.assertTrue(all(x in valid_beyond_limits for x in data_df['beyond_limits']))

        # R chart should have fewer rows than input (first row has no moving range)
        self.assertLess(len(data_df), len(self.df_individual))

        # Verify statistics
        stats = data_results['statistics']
        self.assertIn('mR', stats)
        self.assertIn('lcl', stats)
        self.assertIn('ucl', stats)

    def test_r_chart_with_subgroups_using_engine(self):
        """Test R chart with subgroups using engine.perform_analysis."""
        spec = {
            'analysis_type': 'R',
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
            expected_columns = ['line', 'time', 'mr', 'mR', 'lcl', 'ucl', 'beyond_limits']
            for col in expected_columns:
                self.assertIn(col, data_df.columns, f'Missing column: {col} in group {group_key}')

            # Verify beyond_limits values
            valid_beyond_limits = {-1, 0, 1}
            self.assertTrue(all(x in valid_beyond_limits for x in data_df['beyond_limits']))

            # All moving ranges should be non-negative
            self.assertTrue(all(data_df['mr'] >= 0))

    def test_r_chart_direct_function_call(self):
        """Test R chart calculation directly using calculate_statistics_R."""
        spec = AnalysisSpecification.from_dict(
            analysis_type='R',
            analysis_specification={
                'analysis_type': 'R',
                'response_var': 'measurement',
                'round_to': 3,
            },
        )

        # Prepare dataset
        prepared_df = prepare_dataset(df=self.df_individual, analysis_specification=spec)

        # Calculate R statistics
        result = calculate_statistics_R(df=prepared_df, analysis_specification=spec)

        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('all', result)

        data_results = result['all']
        self.assertIn('data', data_results)
        self.assertIn('statistics', data_results)

        # Check data DataFrame
        data_df = data_results['data']
        self.assertIsInstance(data_df, pd.DataFrame)

        # Should have fewer rows than input (NAs dropped, first row has no moving range)
        self.assertLess(len(data_df), len(self.df_individual))

    def test_r_chart_moving_range_properties(self):
        """Test properties of moving range calculations."""
        # Use known data for verification
        test_data = pd.DataFrame(
            {
                'value': [10, 12, 11, 15, 13]  # Expected moving ranges: NaN, 2, 1, 4, 2
            }
        )

        spec = {
            'analysis_type': 'R',
            'response_var': 'value',
            'round_to': 3,
        }

        result = engine.perform_analysis(df=test_data, specification=spec)
        data_df = result['all']['data']

        # All moving ranges should be non-negative
        self.assertTrue(all(data_df['mr'] >= 0))

        # Should have 4 moving ranges (original 5 values - 1 for first NaN)
        self.assertEqual(len(data_df), 4)

        # Average moving range (mR) should be positive
        self.assertGreater(data_df['mR'].iloc[0], 0)

        # UCL should be greater than average moving range
        stats = result['all']['statistics']
        self.assertGreater(stats['ucl'], stats['mR'])

    def test_r_chart_with_real_data_fillweight(self):
        """Test R chart with real FILLWEIGHTDATA_800.csv data."""
        f_path = (
            '/Users/cnicholas/Documents/projects/processbehavior/'
            'processbehavior/datasets/data/FILLWEIGHTDATA_800.csv'
        )

        if not os.path.exists(f_path):
            self.skipTest('FILLWEIGHTDATA_800.csv not found')

        df = pd.read_csv(f_path)

        # Use R chart without grouping (treat as individuals for moving ranges)
        spec = {
            'analysis_type': 'R',
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

        # Verify no NaN values in critical columns (after NAs are dropped)
        self.assertFalse(data_df['mr'].isna().any())
        self.assertFalse(data_df['mR'].isna().any())
        self.assertFalse(data_df['lcl'].isna().any())
        self.assertFalse(data_df['ucl'].isna().any())

        # Should have one less row than original (first moving range is NaN and dropped)
        self.assertEqual(len(data_df), len(df) - 1)

        # All moving ranges should be non-negative
        self.assertTrue(all(data_df['mr'] >= 0))

    def test_r_chart_mathematical_properties(self):
        """Test mathematical properties of R chart calculations."""
        # Create controlled test data
        test_data = pd.DataFrame({'measurement': [100, 102, 98, 104, 96, 105, 95, 103, 97, 101]})

        spec = AnalysisSpecification.from_dict(
            analysis_type='R',
            analysis_specification={
                'analysis_type': 'R',
                'response_var': 'measurement',
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=test_data, analysis_specification=spec)
        result = calculate_statistics_R(df=prepared_df, analysis_specification=spec)

        data_df = result['all']['data']
        stats = result['all']['statistics']

        # Verify mathematical relationships
        # All moving ranges should be non-negative
        self.assertTrue(all(data_df['mr'] >= 0))

        # Average moving range should equal the mean of all moving ranges
        calculated_average = data_df['mr'].mean()
        self.assertAlmostEqual(stats['mR'], calculated_average, places=2)

        # LCL for R chart is typically 0 (ranges can't be negative)
        self.assertGreaterEqual(stats['lcl'], 0)

        # UCL should be greater than average moving range
        self.assertGreater(stats['ucl'], stats['mR'])

        # Beyond limits should be correctly identified
        for _, row in data_df.iterrows():
            mr_value = row['mr']
            if mr_value > stats['ucl']:
                self.assertEqual(row['beyond_limits'], 1)
            elif mr_value < stats['lcl']:
                self.assertEqual(row['beyond_limits'], -1)
            else:
                self.assertEqual(row['beyond_limits'], 0)

    def test_r_chart_with_grouping_different_lengths(self):
        """Test R chart with groups of different lengths."""
        # Create data with uneven group sizes
        uneven_data = pd.DataFrame(
            {
                'group': ['A'] * 8 + ['B'] * 5 + ['C'] * 12,
                'measure': np.concatenate(
                    [
                        np.random.normal(10, 1, 8),  # Group A: 8 observations
                        np.random.normal(15, 2, 5),  # Group B: 5 observations
                        np.random.normal(20, 1.5, 12),  # Group C: 12 observations
                    ]
                ),
            }
        )

        spec = {
            'analysis_type': 'R',
            'response_var': 'measure',
            'rsg_vars': ['group'],
            'round_to': 3,
        }

        result = engine.perform_analysis(df=uneven_data, specification=spec)

        # Should have 3 groups
        self.assertEqual(len(result), 3)

        # Each group should have different numbers of moving ranges
        group_lengths = {}
        for group_key, group_data in result.items():
            data_df = group_data['data']
            group_lengths[group_key] = len(data_df)

        # Group A should have 7 moving ranges (8 - 1)
        # Group B should have 4 moving ranges (5 - 1)
        # Group C should have 11 moving ranges (12 - 1)
        expected_lengths = {'A': 7, 'B': 4, 'C': 11}

        for group, expected_length in expected_lengths.items():
            self.assertEqual(
                group_lengths[group],
                expected_length,
                f'Group {group} should have {expected_length} moving ranges',
            )

    def test_r_chart_precision_and_rounding(self):
        """Test that R chart respects the rounding specification."""
        test_data = pd.DataFrame({'value': [1.123456, 2.234567, 3.345678, 4.456789, 5.567890]})

        # Test with different rounding specifications
        for round_to in [1, 2, 3]:
            spec = {
                'analysis_type': 'R',
                'response_var': 'value',
                'round_to': round_to,
            }

            result = engine.perform_analysis(df=test_data, specification=spec)
            data_df = result['all']['data']

            # Check that values are rounded to the specified precision
            # (Note: beyond_limits is integer, so skip that column)
            for col in ['mr', 'mR', 'lcl', 'ucl']:
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

    def test_r_chart_lcl_properties(self):
        """Test properties of Lower Control Limit for R charts."""
        test_data = pd.DataFrame({'measurement': [100, 102, 98, 104, 96, 105, 95, 103, 97, 101]})

        spec = {
            'analysis_type': 'R',
            'response_var': 'measurement',
            'round_to': 3,
        }

        result = engine.perform_analysis(df=test_data, specification=spec)
        stats = result['all']['statistics']

        # LCL for R chart should be non-negative (ranges can't be negative)
        self.assertGreaterEqual(stats['lcl'], 0)

        # For moving ranges with small sample sizes, LCL is often 0
        # This is mathematically correct behavior

    def test_r_chart_with_constant_values(self):
        """Test R chart behavior with constant values (zero variation)."""
        constant_data = pd.DataFrame(
            {
                'value': [10.0] * 10  # All same values
            }
        )

        spec = {
            'analysis_type': 'R',
            'response_var': 'value',
            'round_to': 3,
        }

        result = engine.perform_analysis(df=constant_data, specification=spec)
        data_df = result['all']['data']

        # All moving ranges should be zero
        self.assertTrue(all(data_df['mr'] == 0))

        # Average moving range should also be zero
        self.assertEqual(data_df['mR'].iloc[0], 0)

        # All beyond_limits should be 0 (no points out of control)
        self.assertTrue(all(data_df['beyond_limits'] == 0))


if __name__ == '__main__':
    unittest.main()
