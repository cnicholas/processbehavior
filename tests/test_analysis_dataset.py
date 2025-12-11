"""
Integration tests for Analysis and AnalysisDataSet classes.

These tests validate:
1. Correct control chart calculations (Xbar, S, IMR, R)
2. Proper handling of grouping variables
3. Correct limit calculations
4. Data type handling (datetime columns, etc.)
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior import Analysis
from processbehavior import analysis_dataset as ad
from processbehavior.analysis import gather_analysis_statistics, package_analysis
from processbehavior.data_preparation import DataPreparation
from processbehavior.sds_detector import SamplingDesignDetector
from processbehavior.spc_constants import c4


def detect_sds_for_test(df: pd.DataFrame, spec: dict) -> int:
    """
    Helper to detect SDS for tests that need to create AnalysisDataSet or Analysis directly.

    Returns only the SDS integer, not the (sds, min_cell_size) tuple.
    """
    from processbehavior.analysis_specification import AnalysisSpecification
    config = AnalysisSpecification(spec)
    prep = DataPreparation()
    prep.validate_columns(df, config)
    prepared_df = prep.prepare_dataset(df, config)
    detector = SamplingDesignDetector()
    sds, _min_cell_size = detector.detect_sds(prepared_df, config)
    return sds


# ========================
# Fixtures
# ========================

@pytest.fixture
def analysis_types():
    return ['Xbar', 'S', 'Imr', 'R']


@pytest.fixture
def df():
    """Basic test data with integer time variable."""
    data = {
        'a': ['a', 'a', 'a', 'b', 'b', 'b', 'c'],
        'b': ['c', 'c', 'c', 'd', 'd', 'd', 'e'],
        'c': [1.5, 2.0, 3.5, 5.0, 8.0, 10.0, 1.0],
        'd': [1, 2, 3, 1, 2, 3, 1],
        'a1': [1, 1, 1, 1, 1, 1, 1],
        'a2': [2, 2, 2, 2, 2, 2, 2],
    }
    return pd.DataFrame(data=data)


@pytest.fixture
def df_differing_Ns():
    """Data with varying group sizes."""
    data = {
        'a': ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'c'],
        'b': ['c', 'c', 'c', 'd', 'd', 'd', 'd', 'e'],
        'c': [1.5, 2.0, 3.5, 5.0, 8.0, 10.0, 1.0, 1.0],
        'd': [1, 2, 3, 1, 2, 3, 1, 1],
        'a1': [1, 1, 1, 1, 1, 1, 1, 1],
        'a2': [2, 2, 2, 2, 2, 2, 2, 2],
    }
    return pd.DataFrame(data=data)


@pytest.fixture
def df_dt():
    """Data with datetime column."""
    data = {
        'a': ['a', 'a', 'a', 'b', 'b', 'b', 'd'],
        'b': ['c', 'c', 'c', 'd', 'd', 'd', 'e'],
        'c': [1.5, 2, 3.5, 40, 55, 60, 1],
        'd': pd.to_datetime([
            "2022-01-01", "2022-01-02", "2022-01-03",
            "2022-01-01", "2022-01-02", "2022-01-03", pd.NA
        ]),
        'a1': [1, 1, 1, 1, 1, 1, 1],
        'a2': [2, 2, 2, 2, 2, 2, 2],
        'd2': ["4/1/2000", "2/1/2000", "3/1/2000", "5/1/2000", "2/1/2000", "1/1/2000", pd.NA]
    }
    return pd.DataFrame(data=data)


# ========================
# Xbar-S Chart Tests
# ========================

class TestXbarSAnalysis:
    """Tests for Xbar and S chart calculations."""

    def test_xbar_s_basic(self, df):
        """Test basic Xbar-S analysis with expected statistics."""
        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['a', 'b'],
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'time_var': 'd',
            'round_to': 2
        }

        sds = detect_sds_for_test(df, spec)
        result = Analysis(df, spec, sds=sds).calculate()

        # Should return both Xbar and Sbar charts
        assert len(result) == 2
        assert 'Xbar' in result
        assert 'Sbar' in result

        # Check Xbar statistics
        xbar_stats = result['Xbar']['statistics']
        assert xbar_stats['center'] == 5.0
        assert xbar_stats['lpl'] == 1.52
        assert xbar_stats['upl'] == 8.48

        # Check Sbar statistics
        sbar_stats = result['Sbar']['statistics']
        assert sbar_stats['center'] == 1.78
        assert sbar_stats['lpl'] == 0
        assert sbar_stats['upl'] == 4.57

    def test_xbar_s_differing_ns(self, df_differing_Ns):
        """Test Xbar-S with varying group sizes (limits vary by subgroup)."""
        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['a', 'b'],
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'time_var': 'd',
            'round_to': 2
        }

        sds = detect_sds_for_test(df_differing_Ns, spec)
        result = Analysis(df_differing_Ns, spec, sds=sds).calculate()

        # Center should be calculated
        assert result['Xbar']['statistics']['center'] == 4.17
        # Limits should vary when group sizes differ
        assert result['Xbar']['statistics']['lpl'] == 'Varies'
        assert result['Xbar']['statistics']['upl'] == 'Varies'

        assert result['Sbar']['statistics']['center'] == 2.48
        assert result['Sbar']['statistics']['lpl'] == 'Varies'
        assert result['Sbar']['statistics']['upl'] == 'Varies'

    def test_xbar_zero_center(self, df):
        """Test Xbar analysis with zero-centered option."""
        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['a', 'b'],
            'time_var': 'd',
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'zero-center': True
        }

        sds = detect_sds_for_test(df, spec)
        result = Analysis(df, spec, sds=sds).calculate()

        assert result['Xbar']['statistics']['center'] == 0


# ========================
# IMR Chart Tests
# ========================

class TestImrAnalysis:
    """Tests for Individual Moving Range (IMR) chart calculations."""

    def test_imr_with_grouping(self, df):
        """Test stratified IMR analysis with grouping."""
        spec = {
            'analysis_type': 'Imr',
            'rsg_vars': ['a', 'b'],
            'time_var': 'd',
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'round_to': 2
        }

        sds = detect_sds_for_test(df, spec)
        result = Analysis(df=df, specification=spec, sds=sds).calculate()

        # Should return dict-like with multiple groups
        assert hasattr(result, 'keys') and hasattr(result, 'values')
        # Third group only had 1 obs so should be dropped
        assert len(result) == 2

        keys = list(result.keys())
        assert 'a_c' in keys
        assert 'b_d' in keys

        # Check centers
        assert result['a_c']['statistics']['center'] == 2.33
        assert result['b_d']['statistics']['center'] == 7.67

        # Check limits
        assert result['a_c']['statistics']['lpl'] == -0.33
        assert result['b_d']['statistics']['lpl'] == 1.02
        assert result['a_c']['statistics']['upl'] == 4.99
        assert result['b_d']['statistics']['upl'] == 14.32

        # Check sample sizes
        assert result['a_c']['statistics']['n'] == 3
        assert result['b_d']['statistics']['n'] == 3

    def test_imr_without_grouping(self, df):
        """Test IMR analysis without grouping variable."""
        spec = {
            'analysis_type': 'Imr',
            'response_var': 'c',
            'round_to': 2
        }

        ad.AnalysisSpecification(spec)
        sds = detect_sds_for_test(df, spec)
        result = Analysis(df, spec, sds=sds).calculate()

        assert hasattr(result, "keys") and hasattr(result, "values")
        assert 'all' in result

    def test_imr_with_fillweight_data(self):
        """Test IMR without grouping on FillWeight800 dataset."""
        f_path = "processbehavior/datasets/data/FILLWEIGHTDATA_800.csv"
        df = pd.read_csv(f_path)

        spec = {
            'analysis_type': 'Imr',
            'response_var': 'fill_weight',
            'round_to': 2
        }

        ad.AnalysisSpecification(spec)
        sds = detect_sds_for_test(df, spec)
        result = Analysis(df, spec, sds=sds).calculate()

        assert hasattr(result, "keys") and hasattr(result, "values")
        assert list(result)[0] == 'all'
        assert isinstance(result.get("all"), dict)

        out = result.get("all")
        # Verify against R (qcc) results
        assert out['statistics']['center'] == 237.78
        assert out['statistics']['lpl'] == 232.23
        assert out['statistics']['upl'] == 243.33


# ========================
# R Chart Tests
# ========================

class TestRChartAnalysis:
    """Tests for Moving Range (R) chart calculations."""

    def test_r_with_grouping(self, df):
        """Test R chart with grouping variable."""
        spec = {
            'analysis_type': 'R',
            'rsg_vars': ['a', 'b'],
            'time_var': 'd',
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'round_to': 2
        }
        ad.AnalysisSpecification(spec)

        sds = detect_sds_for_test(df, spec)
        result = Analysis(df, spec, sds=sds).calculate()

        assert hasattr(result, "keys") and hasattr(result, "values")
        # Third group had 1 obs and should be dropped
        assert len(result) == 2

        # Check centers
        assert result['a_c']['statistics']['center'] == 1
        assert result['b_d']['statistics']['center'] == 2.5

        # Check limits
        assert result['a_c']['statistics']['lpl'] == 0
        assert result['b_d']['statistics']['lpl'] == 0
        assert result['a_c']['statistics']['upl'] == 3.27
        assert result['b_d']['statistics']['upl'] == 8.17

        # Check data row counts (first row drops for MR calculation)
        assert len(result['a_c']['data']) == 2
        assert len(result['b_d']['data']) == 2

    def test_r_without_grouping(self, df):
        """Test R chart without grouping variable."""
        spec = {
            'analysis_type': 'R',
            'response_var': 'c',
            'rsg_var_name': 'rsg'
        }

        sds = detect_sds_for_test(df, spec)
        result = Analysis(df, spec, sds=sds).calculate()

        assert hasattr(result, "keys") and hasattr(result, "values")
        assert result['all']['statistics']['center'] == 2.917
        assert result['all']['statistics']['n'] == 6
        assert result['all']['statistics']['lpl'] == 0
        assert result['all']['statistics']['upl'] == 9.532

    def test_r_with_fillweight_data(self):
        """Test R chart on FillWeight800 dataset with stratification."""
        f_path = "processbehavior/datasets/data/FILLWEIGHTDATA_800.csv"
        df = pd.read_csv(f_path)

        spec = {
            'analysis_type': 'R',
            'rsg_vars': ['lane', 'phase'],
            'response_var': 'fill_weight',
            'rsg_var_name': 'rsg',
            'time_var': 'pull'
        }

        sds = detect_sds_for_test(df, spec)
        result = Analysis(df, spec, sds=sds).calculate()

        assert hasattr(result, "keys") and hasattr(result, "values")
        assert len(result) == 8  # 8 lane-phase combinations

        # Check all result dfs have no nulls
        for key in result:
            assert not result[key]['data'].isnull().values.any()


# ========================
# DateTime Handling Tests
# ========================

class TestDateTimeHandling:
    """Tests for datetime column handling in analysis."""

    def test_datetime_column_preserved(self, df_dt, analysis_types):
        """Test that datetime columns are preserved in output."""
        spec = {
            'analysis_type': None,
            'rsg_vars': ['a', 'b'],
            'time_var': 'd',
            'response_var': 'c',
            'rsg_var_name': 'rsg'
        }

        has_time = 'd'
        for analysis in analysis_types:
            spec['analysis_type'] = analysis
            sds = detect_sds_for_test(df_dt, spec)
            result = Analysis(df_dt, spec, sds=sds).calculate()

            if analysis in ['Imr', 'R']:
                out = result['a_c']['data']
                assert out.columns.tolist()[0] == has_time

    def test_string_date_converted_and_sorted(self, df_dt):
        """Test that string date columns are converted and sorted chronologically."""
        spec = {
            'analysis_type': 'Imr',
            'rsg_vars': ['a', 'b'],
            'time_var': 'd2',
            'response_var': 'c',
            'rsg_var_name': 'rsg'
        }

        sds = detect_sds_for_test(df_dt, spec)
        result = Analysis(df_dt, spec, sds=sds).calculate()

        # String dates should be converted to datetime
        o_type = result['a_c']['data']['d2'].dtype
        assert pd.api.types.is_datetime64_any_dtype(o_type), f"Expected datetime type, got {o_type}"

        # Should be chronologically ordered (2000-02-01 is first)
        dt_val = result['a_c']['data'].iloc[0, 0]
        expected = pd.Timestamp('2000-02-01')
        assert dt_val == expected


# ========================
# AnalysisDataSet Tests
# ========================

class TestAnalysisDataSet:
    """Tests for AnalysisDataSet functionality."""

    def test_no_grouping_without_time(self, df):
        """Test AnalysisDataSet with no grouping and no time variable."""
        spec = {
            'analysis_type': 'Imr',
            'response_var': 'c',
            'round_to': 2
        }

        sds = detect_sds_for_test(df, spec)
        a_spec = ad.AnalysisSpecification(spec)
        dataset = ad.AnalysisDataSet(df=df, analysis_specification=a_spec, sds=sds)

        assert dataset.sampling_design_state == 0
        assert dataset.analysis_dataset.columns.tolist() == ['c', 'obs_id', 'rsg_key', 'cell_key']

    def test_no_grouping_with_time(self, df):
        """Test AnalysisDataSet with time variable but no grouping."""
        spec = {
            'analysis_type': 'Imr',
            'time_var': 'd',
            'response_var': 'c',
            'round_to': 2
        }

        sds = detect_sds_for_test(df, spec)
        a_spec = ad.AnalysisSpecification(spec)
        dataset = ad.AnalysisDataSet(df=df, analysis_specification=a_spec, sds=sds)

        assert dataset.sampling_design_state == 0
        assert dataset.analysis_dataset.columns.tolist() == ['d', 'c', 'obs_id', 'rsg_key', 'cell_key']


# ========================
# Utility Function Tests
# ========================

class TestUtilityFunctions:
    """Tests for analysis utility functions."""

    def test_package_statistics(self):
        """Test packaging analysis output with statistics."""
        analysis_output = {'a': "dataframe_a", 'b': "dataframe_b"}
        statistics = {'a': "statistics_a", 'b': "statistics_b"}

        out = package_analysis(
            analysis_output=analysis_output,
            summary_statistics_output=statistics
        )

        assert isinstance(out.get("a"), dict)
        assert out.get("a").get('statistics') == "statistics_a"
        assert isinstance(out.get("b"), dict)
        assert out.get("b").get('statistics') == "statistics_b"

    def test_gather_statistics_with_grouping(self):
        """Test gathering statistics with grouping variable."""
        data = {
            'rsg': ['a_c', 'a_c', 'a_c', 'b_d', 'b_d', 'b_d', 'b_d'],
            'stat1': [1.5, 1.5, 1.5, 5.0, 5.0, 5.0, 5.0],
            'stat2': [2.5, 2.5, 2.5, 6.0, 6.0, 6.0, 6.0],
            'stat3': [1, 1, 1, 2, 2, 2, 2],
            'response': [1.1, 1.2, 1.5, 2.1, 2.2, 2.3, 2.4]
        }
        df = pd.DataFrame(data=data)

        out = gather_analysis_statistics(df, ['stat1', 'stat2', 'stat3'], grouping_var='rsg')

        assert len(out) == 2  # Two groups
        assert len(out['a_c']) == 4  # 3 stats + n

    def test_gather_statistics_without_grouping(self):
        """Test gathering statistics without grouping variable."""
        data = {
            'rsg': ['a_c', 'a_c', 'a_c', 'b_d', 'b_d', 'b_d', 'b_d'],
            'stat1': [1.5, 1.5, 1.5, 5.0, 5.0, 5.0, 5.0],
            'stat2': [2.5, 2.5, 2.5, 6.0, 6.0, 6.0, 6.0],
            'stat3': [1, 1, 1, 2, 2, 2, 2],
            'response': [1.1, 1.2, 1.5, 2.1, 2.2, 2.3, 2.4]
        }
        df = pd.DataFrame(data=data)

        out = gather_analysis_statistics(df, ['stat1', 'stat2', 'stat3'])

        assert len(out) == 1  # Single 'all' group
        assert len(out['all']) == 4  # 3 stats + n

    def test_c4_limit_calculation(self):
        """Test c4 constant and limit calculation."""
        mean = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        sd = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        N = pd.Series([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

        result = pd.DataFrame({'mean': mean, 'sd': sd, 'N': N})
        result['c4'] = result['N'].apply(c4)
        result['Wd'] = result['sd'] / result['c4']
        result['lpl'] = result['mean'] + (-1 * ((3 * result['Wd']) / np.sqrt(result['N'])))
        result['upl'] = result['mean'] + ((3 * result['Wd']) / np.sqrt(result['N']))

        # Verify c4 is calculated for each row
        assert len(result['c4']) == 10
        # Verify limits are symmetric around mean
        assert (result['upl'] - result['mean']).equals(result['mean'] - result['lpl'])
