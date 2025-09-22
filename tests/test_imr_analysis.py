"""
Test for IMR analysis functionality using FILLWEIGHTDATA_800.csv dataset.
"""

import os
import sys
import unittest

import pandas as pd

# Add the processbehavior package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'processbehavior'))

from processbehavior import analysis_dataset as ad


class TestIMRAnalysis(unittest.TestCase):
    """Test IMR analysis without grouping variables."""

    def test_IMR_w_o_grouping_var_FW800(self):
        """Test IMR analysis without grouping variable using FILLWEIGHTDATA_800.csv dataset."""

        spec = {
            'analysis_type': 'Imr',
            'response_var': 'fill_weight',
            'time_unit': None,
            'round_to': 2,
        }

        f_path = 'processbehavior/datasets/data/FILLWEIGHTDATA_800.csv'
        df = pd.read_csv(f_path)
        print(f'\nDataset columns: {df.columns.tolist()}')
        print(f'Dataset shape: {df.shape}')
        print(f'Missing values in fill_weight: {df["fill_weight"].isna().sum()}')

        # Create analysis specification
        a_spec = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=spec)
        print(f'Analysis specification created: {a_spec.analysis_type}')

        # Perform analysis
        result = ad.perform_analysis(df=df, specification=spec)

        # Verify result structure
        self.assertEqual(type(result), type({}))
        _keys = result.keys()
        print("Verifying dictionary returned has the key: 'all'...")
        self.assertEqual(list(result)[0], 'all')
        print("Verifying 'all' dictionary key references a dictionary...")
        self.assertEqual(type(result.get('all')), type({}))

        out = result.get('all')
        print(f'Result keys: {list(out.keys())}')

        # Verify data and statistics structure
        self.assertIn('data', out)
        self.assertIn('statistics', out)

        # Verify statistics values (Match results from R (qcc))
        print('Verify lcl, ucl, and mean...(Match results from R (qcc))...')
        statistics = out['statistics']
        print(f'Calculated statistics: {statistics}')

        self.assertEqual(statistics['mean'], 237.78)
        self.assertEqual(statistics['lcl'], 232.23)
        self.assertEqual(statistics['ucl'], 243.33)

        # Verify data structure
        data_df = out['data']
        self.assertIsInstance(data_df, pd.DataFrame)

        # Verify required columns exist
        expected_columns = ['x', 'fill_weight', 'mean', 'lcl', 'ucl', 'beyond_limits']
        for col in expected_columns:
            self.assertIn(col, data_df.columns)

        print('Test completed successfully!')

    def test_IMR_with_grouping_var_FW800(self):
        """Test IMR analysis with grouping variable using FILLWEIGHTDATA_800.csv dataset."""

        spec = {
            'analysis_type': 'Imr',
            'response_var': 'fill_weight',
            'rsg_vars': ['pull', 'lane'],  # Use pull and lane as grouping variables
            'time_unit': None,
            'round_to': 2,
        }

        f_path = 'processbehavior/datasets/data/FILLWEIGHTDATA_800.csv'
        df = pd.read_csv(f_path)
        print(f'\nDataset columns: {df.columns.tolist()}')
        print(f'Dataset shape: {df.shape}')

        # Perform analysis
        result = ad.perform_analysis(df=df, specification=spec)

        # Verify result structure
        self.assertEqual(type(result), type({}))
        print(f'Groups found: {list(result.keys())}')

        # Should have multiple groups (pull_lane combinations)
        self.assertGreater(len(result.keys()), 1)

        # Check first group structure
        first_group_key = list(result.keys())[0]
        first_group = result[first_group_key]

        # Verify data and statistics structure
        self.assertIn('data', first_group)
        self.assertIn('statistics', first_group)

        # Verify data structure
        data_df = first_group['data']
        self.assertIsInstance(data_df, pd.DataFrame)

        print('Grouped IMR test completed successfully!')


if __name__ == '__main__':
    unittest.main()
