"""
Unit tests for XbarS (combined Xbar and S) chart calculations.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

# Add the processbehavior package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from processbehavior.charts.xbar_s import calculate_statistics_XbarS
from processbehavior.data_prep import prepare_dataset
from processbehavior.spec import AnalysisSpecification


class TestXbarSChart(unittest.TestCase):
    """Test XbarS combined chart calculations."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic data for testing
        np.random.seed(42)  # For reproducible results

        # Data with subgroups suitable for XbarS analysis
        self.df_grouped = pd.DataFrame(
            {
                'batch': np.repeat(['A', 'B', 'C', 'D'], [6, 6, 5, 7]),  # Different subgroup sizes
                'measurement': np.concatenate(
                    [
                        np.random.normal(100, 2, 6),  # Batch A: 6 observations
                        np.random.normal(105, 3, 6),  # Batch B: 6 observations
                        np.random.normal(98, 1.5, 5),  # Batch C: 5 observations
                        np.random.normal(102, 2.5, 7),  # Batch D: 7 observations
                    ]
                ),
            }
        )

        # Data with equal subgroup sizes for XbarS
        self.df_equal_groups = pd.DataFrame(
            {
                'group': np.repeat(['X', 'Y', 'Z'], 5),
                'value': np.concatenate(
                    [
                        np.random.normal(50, 4, 5),  # Group X: 5 observations
                        np.random.normal(55, 3, 5),  # Group Y: 5 observations
                        np.random.normal(48, 5, 5),  # Group Z: 5 observations
                    ]
                ),
            }
        )

        # Larger dataset for more comprehensive testing
        self.df_large = pd.DataFrame(
            {
                'line': np.repeat(['L1', 'L2', 'L3'], [10, 10, 10]),
                'shift': np.tile(['Day', 'Night'], 15),
                'quality': np.concatenate(
                    [
                        np.random.normal(75, 2, 10),  # Line 1
                        np.random.normal(77, 1.5, 10),  # Line 2
                        np.random.normal(73, 3, 10),  # Line 3
                    ]
                ),
            }
        )

    def test_xbar_s_with_subgroups_using_engine(self):
        """Test XbarS chart with subgroups using engine.perform_analysis."""
        # Note: XbarS is not directly exposed in engine.perform_analysis
        # But we can test it through direct function calls
        spec = AnalysisSpecification.from_dict(
            analysis_type='XbarS',
            analysis_specification={
                'analysis_type': 'XbarS',
                'response_var': 'measurement',
                'rsg_vars': ['batch'],
                'round_to': 3,
            },
        )

        # Prepare dataset
        prepared_df = prepare_dataset(df=self.df_grouped, analysis_specification=spec)

        # Calculate XbarS statistics
        result = calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

        # Verify result structure (should contain both Xbar and Sbar)
        self.assertIsInstance(result, dict)
        self.assertIn('Xbar', result)
        self.assertIn('Sbar', result)

        # Verify Xbar component
        xbar_result = result['Xbar']
        self.assertIn('data', xbar_result)
        self.assertIn('statistics', xbar_result)

        xbar_data = xbar_result['data']
        self.assertIsInstance(xbar_data, pd.DataFrame)

        # Check expected Xbar columns
        expected_xbar_columns = ['rsg', 'mean', 'Xbar', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_xbar_columns:
            self.assertIn(col, xbar_data.columns, f'Missing Xbar column: {col}')

        # Verify Sbar component
        sbar_result = result['Sbar']
        self.assertIsInstance(sbar_result, pd.DataFrame)

        # Check expected Sbar columns
        expected_sbar_columns = ['rsg', 's', 'S', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_sbar_columns:
            self.assertIn(col, sbar_result.columns, f'Missing Sbar column: {col}')

    def test_xbar_s_composition_consistency(self):
        """Test that XbarS composition gives consistent results with individual calculations."""
        spec = AnalysisSpecification.from_dict(
            analysis_type='XbarS',
            analysis_specification={
                'analysis_type': 'XbarS',
                'response_var': 'measurement',
                'rsg_vars': ['batch'],
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=self.df_grouped, analysis_specification=spec)

        # Get XbarS results
        xbar_s_result = calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

        # Import individual functions for comparison
        from processbehavior.charts.s import calculate_statistics_S
        from processbehavior.charts.xbar import calculate_statistics_Xbar

        # Calculate individual results
        xbar_spec = AnalysisSpecification.from_dict(
            analysis_type='Xbar',
            analysis_specification={
                'analysis_type': 'Xbar',
                'response_var': 'measurement',
                'rsg_vars': ['batch'],
                'round_to': 3,
            },
        )

        s_spec = AnalysisSpecification.from_dict(
            analysis_type='S',
            analysis_specification={
                'analysis_type': 'S',
                'response_var': 'measurement',
                'rsg_vars': ['batch'],
                'round_to': 3,
            },
        )

        prepared_xbar_df = prepare_dataset(df=self.df_grouped, analysis_specification=xbar_spec)
        prepared_s_df = prepare_dataset(df=self.df_grouped, analysis_specification=s_spec)

        individual_xbar = calculate_statistics_Xbar(
            df=prepared_xbar_df, analysis_specification=xbar_spec
        )
        individual_s = calculate_statistics_S(df=prepared_s_df, analysis_specification=s_spec)

        # Compare Xbar results
        xbar_from_combined = xbar_s_result['Xbar']['data']
        xbar_individual_data = individual_xbar['data']

        # Should have same number of rows
        self.assertEqual(len(xbar_from_combined), len(xbar_individual_data))

        # Key statistical values should match (within rounding tolerance)
        for col in ['mean', 'Xbar', 'lcl', 'ucl']:
            combined_values = xbar_from_combined[col].values
            individual_values = xbar_individual_data[col].values
            np.testing.assert_array_almost_equal(
                combined_values,
                individual_values,
                decimal=2,
                err_msg=f"Xbar {col} values don't match",
            )

        # Compare S results
        sbar_from_combined = xbar_s_result['Sbar']

        # Should have same number of rows
        self.assertEqual(len(sbar_from_combined), len(individual_s))

        # Key statistical values should match (within rounding tolerance)
        for col in ['s', 'S', 'lcl', 'ucl']:
            combined_values = sbar_from_combined[col].values
            individual_values = individual_s[col].values
            np.testing.assert_array_almost_equal(
                combined_values, individual_values, decimal=2, err_msg=f"S {col} values don't match"
            )

    def test_xbar_s_with_equal_subgroup_sizes(self):
        """Test XbarS chart with equal subgroup sizes."""
        spec = AnalysisSpecification.from_dict(
            analysis_type='XbarS',
            analysis_specification={
                'analysis_type': 'XbarS',
                'response_var': 'value',
                'rsg_vars': ['group'],
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=self.df_equal_groups, analysis_specification=spec)
        result = calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

        # Verify both components exist
        self.assertIn('Xbar', result)
        self.assertIn('Sbar', result)

        # For equal subgroup sizes, control limits should be consistent
        xbar_data = result['Xbar']['data']
        sbar_data = result['Sbar']

        # Each component should have same number of groups
        self.assertEqual(len(xbar_data), 3)  # 3 groups: X, Y, Z
        self.assertEqual(len(sbar_data), 3)

        # With equal subgroup sizes, Xbar control limits should be the same
        xbar_stats = result['Xbar']['statistics']
        if xbar_stats['N'] != 'Varies':
            # All UCL and LCL values should be the same
            ucl_values = xbar_data['ucl'].unique()
            lcl_values = xbar_data['lcl'].unique()
            self.assertEqual(len(ucl_values), 1, 'UCL should be same for equal subgroup sizes')
            self.assertEqual(len(lcl_values), 1, 'LCL should be same for equal subgroup sizes')

    def test_xbar_s_with_real_data_sds(self):
        """Test XbarS chart with real SDS_1_DATA.csv if available."""
        f_path = (
            '/Users/cnicholas/Documents/projects/processbehavior/'
            'processbehavior/datasets/data/SDS_1_DATA.csv'
        )

        if not os.path.exists(f_path):
            self.skipTest('SDS_1_DATA.csv not found')

        df = pd.read_csv(f_path)

        spec = AnalysisSpecification.from_dict(
            analysis_type='XbarS',
            analysis_specification={
                'analysis_type': 'XbarS',
                'response_var': 'Y',
                'rsg_vars': ['FACTOR_1', 'FACTOR_2'],
                'time_var': 'TIME',
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=df, analysis_specification=spec)
        result = calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

        # Verify result structure
        self.assertIn('Xbar', result)
        self.assertIn('Sbar', result)

        # Verify we have valid results
        xbar_data = result['Xbar']['data']
        sbar_data = result['Sbar']

        self.assertGreater(len(xbar_data), 0)
        self.assertGreater(len(sbar_data), 0)

        # Check that we have reasonable values
        self.assertTrue(all(xbar_data['Xbar'] > 0))  # Assuming positive measurements
        self.assertTrue(all(sbar_data['S'] > 0))  # Standard deviations should be positive
        self.assertTrue(all(sbar_data['s'] >= 0))  # Individual standard deviations non-negative

    def test_xbar_s_mathematical_properties(self):
        """Test mathematical properties of XbarS chart calculations."""
        # Create controlled test data
        test_data = pd.DataFrame(
            {
                'group': ['G1'] * 5 + ['G2'] * 5 + ['G3'] * 5,
                'value': [
                    10,
                    12,
                    8,
                    14,
                    11,  # Group G1
                    20,
                    22,
                    18,
                    24,
                    21,  # Group G2
                    30,
                    32,
                    28,
                    34,
                    31,
                ],  # Group G3
            }
        )

        spec = AnalysisSpecification.from_dict(
            analysis_type='XbarS',
            analysis_specification={
                'analysis_type': 'XbarS',
                'response_var': 'value',
                'rsg_vars': ['group'],
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=test_data, analysis_specification=spec)
        result = calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

        # Verify mathematical relationships for Xbar
        xbar_data = result['Xbar']['data']
        for _, row in xbar_data.iterrows():
            # Control limits should be symmetric around center line
            center_to_ucl = row['ucl'] - row['Xbar']
            center_to_lcl = row['Xbar'] - row['lcl']
            self.assertAlmostEqual(
                center_to_ucl,
                center_to_lcl,
                places=2,
                msg='Xbar control limits should be symmetric',
            )

            # UCL should be greater than mean, mean should be greater than LCL
            self.assertGreater(row['ucl'], row['mean'])
            self.assertGreater(row['mean'], row['lcl'])

        # Verify mathematical relationships for S
        sbar_data = result['Sbar']
        for _, row in sbar_data.iterrows():
            # Standard deviation should be non-negative
            self.assertGreaterEqual(row['s'], 0)
            self.assertGreater(row['S'], 0)  # Average should be positive

            # UCL should be greater than or equal to LCL
            self.assertGreaterEqual(row['ucl'], row['lcl'])

            # LCL should be non-negative (can't have negative standard deviation)
            self.assertGreaterEqual(row['lcl'], 0)

    def test_xbar_s_with_insufficient_data(self):
        """Test XbarS behavior with insufficient data (single observations per subgroup)."""
        insufficient_data = pd.DataFrame(
            {
                'group': ['A', 'B', 'C'],  # Each group has only 1 observation
                'value': [10, 20, 30],
            }
        )

        spec = AnalysisSpecification.from_dict(
            analysis_type='XbarS',
            analysis_specification={
                'analysis_type': 'XbarS',
                'response_var': 'value',
                'rsg_vars': ['group'],
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=insufficient_data, analysis_specification=spec)

        # XbarS should handle this gracefully - Xbar should fail, S should return empty
        with self.assertRaises(ValueError) as context:
            calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

        # Should get error from Xbar component about insufficient subgroup size
        self.assertIn('All subgroups have 1 or less observations', str(context.exception))

    def test_xbar_s_precision_and_rounding(self):
        """Test that XbarS chart respects the rounding specification."""
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
            spec = AnalysisSpecification.from_dict(
                analysis_type='XbarS',
                analysis_specification={
                    'analysis_type': 'XbarS',
                    'response_var': 'value',
                    'rsg_vars': ['group'],
                    'round_to': round_to,
                },
            )

            prepared_df = prepare_dataset(df=test_data, analysis_specification=spec)
            result = calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

            # Check Xbar rounding
            xbar_data = result['Xbar']['data']
            for col in ['mean', 'Xbar', 'lcl', 'ucl']:
                for value in xbar_data[col]:
                    str_value = f'{value:.{round_to + 2}f}'
                    decimal_part = str_value.split('.')[-1]
                    significant_decimals = len(decimal_part.rstrip('0'))
                    self.assertLessEqual(
                        significant_decimals,
                        round_to,
                        f'Xbar {col} value {value} not rounded to {round_to} places',
                    )

            # Check S rounding
            sbar_data = result['Sbar']
            for col in ['s', 'S', 'lcl', 'ucl']:
                for value in sbar_data[col]:
                    str_value = f'{value:.{round_to + 2}f}'
                    decimal_part = str_value.split('.')[-1]
                    significant_decimals = len(decimal_part.rstrip('0'))
                    self.assertLessEqual(
                        significant_decimals,
                        round_to,
                        f'S {col} value {value} not rounded to {round_to} places',
                    )

    def test_xbar_s_debug_output(self):
        """Test that XbarS prints the expected debug message."""
        spec = AnalysisSpecification.from_dict(
            analysis_type='XbarS',
            analysis_specification={
                'analysis_type': 'XbarS',
                'response_var': 'value',
                'rsg_vars': ['group'],
                'round_to': 3,
            },
        )

        prepared_df = prepare_dataset(df=self.df_equal_groups, analysis_specification=spec)

        # Capture print output
        import contextlib
        import io

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

        output = f.getvalue()

        # Should contain the composition message
        self.assertIn('In calculate statistics XbarS (using composition)', output)

    def test_xbar_s_with_multiple_grouping_variables(self):
        """Test XbarS with multiple grouping variables."""
        spec = AnalysisSpecification.from_dict(
            analysis_type='XbarS',
            analysis_specification={
                'analysis_type': 'XbarS',
                'response_var': 'quality',
                'rsg_vars': ['line', 'shift'],
                'round_to': 2,
            },
        )

        prepared_df = prepare_dataset(df=self.df_large, analysis_specification=spec)
        result = calculate_statistics_XbarS(df=prepared_df, analysis_specification=spec)

        # Verify both components exist
        self.assertIn('Xbar', result)
        self.assertIn('Sbar', result)

        # Should have multiple groups (line × shift combinations)
        xbar_data = result['Xbar']['data']
        sbar_data = result['Sbar']

        self.assertGreater(len(xbar_data), 3)  # More than just line count
        self.assertGreater(len(sbar_data), 3)

        # Each result should have the composite grouping variable
        expected_rsg_values = ['L1_Day', 'L1_Night', 'L2_Day', 'L2_Night', 'L3_Day', 'L3_Night']
        xbar_rsg_values = set(xbar_data['rsg'].tolist())
        sbar_rsg_values = set(sbar_data['rsg'].tolist())

        # Should have some of the expected combinations (might not have all if some are filtered)
        self.assertTrue(len(xbar_rsg_values.intersection(expected_rsg_values)) > 0)
        self.assertTrue(len(sbar_rsg_values.intersection(expected_rsg_values)) > 0)


if __name__ == '__main__':
    unittest.main()
