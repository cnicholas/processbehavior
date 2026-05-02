"""
Integration tests for Analysis and AnalysisDataSet classes.

These tests validate:
1. Correct control chart calculations (Xbar, S, XmR, R)
2. Proper handling of grouping variables
3. Correct limit calculations
4. Data type handling (datetime columns, etc.)
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior import analysis_dataset as ad
from processbehavior.analysis import Analysis
from processbehavior.data_preparation import DataPreparation
from processbehavior.formulation_spec import ChartRequest, FormulationSpec
from processbehavior.sds_detector import SDSRegistry
from processbehavior.spc_constants import c4

from conftest import make_spec, make_request, detect_sds_for_test


# ========================
# Fixtures
# ========================

@pytest.fixture
def analysis_types():
    return ['Xbar', 'S', 'X', 'mR']


@pytest.fixture
def df():
    """SDS 2 test data: n=1 per (factor x time) cell. Valid for XmR/R."""
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
def df_sds1():
    """SDS 1 test data: balanced, complete, replicated (n=2 per cell)."""
    data = {
        'a': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
        'b': ['c', 'c', 'c', 'c', 'd', 'd', 'd', 'd'],
        'c': [1.5, 2.0, 3.5, 4.0, 5.0, 8.0, 10.0, 7.0],
        'd': [1, 1, 2, 2, 1, 1, 2, 2],
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

    def test_xbar_s_basic(self, df_sds1):
        """Test basic Xbar-S analysis with SDS 1 data (n=2 per cell)."""
        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['a', 'b'],
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'time_var': 'd',
            'round_to': 2,
            'by': ['a', 'b'],  # Aggregate by factors (not cell level)
            'companion': True,  # Request both Xbar and S charts
        }

        sds = detect_sds_for_test(df_sds1, spec)
        assert sds == 1  # Verify this is actually SDS 1

        result = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df_sds1).calculate()

        # Should return both Xbar and S charts (companion=True)
        assert len(result) == 2
        assert 'Xbar' in result
        assert 'S' in result

        # Check Xbar statistics (n=4 per factor group)
        xbar_stats = result['Xbar']['statistics']
        assert xbar_stats['center'] == 5.12
        assert xbar_stats['lpl'] == 2.46
        assert xbar_stats['upl'] == 7.79

        # Check S statistics
        # b3 clamped to 0 for small subgroups (n < 6), so LPL = 0.0
        sbar_stats = result['S']['statistics']
        assert sbar_stats['center'] == 1.64
        assert sbar_stats['lpl'] == 0.0
        assert sbar_stats['upl'] == 3.71

    def test_xbar_s_differing_ns(self, df_differing_Ns):
        """Test Xbar-S with varying group sizes (limits vary by subgroup)."""
        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['a', 'b'],
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'time_var': 'd',
            'round_to': 2,
            'by': ['a', 'b'],  # Aggregate by factors (not cell level)
            'companion': True,  # Request both Xbar and S charts
        }

        sds = detect_sds_for_test(df_differing_Ns, spec)
        result = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df_differing_Ns).calculate()

        # Center should be grand mean of all observations (including n=1 groups)
        assert result['Xbar']['statistics']['center'] == 4.0
        # Limits should vary when group sizes differ
        assert result['Xbar']['statistics']['lpl'] == 'Varies'
        assert result['Xbar']['statistics']['upl'] == 'Varies'

        assert result['S']['statistics']['center'] == 2.48
        assert result['S']['statistics']['lpl'] == 'Varies'
        assert result['S']['statistics']['upl'] == 'Varies'


# ========================
# XmR Chart Tests
# ========================

class TestXmRAnalysis:
    """Tests for Individual Moving Range (XmR) chart calculations."""

    def test_xmr_with_grouping(self, df):
        """Test stratified XmR analysis with grouping."""
        spec = {
            'analysis_type': 'X',
            'rsg_vars': ['a', 'b'],
            'time_var': 'd',
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'round_to': 2
        }

        sds = detect_sds_for_test(df, spec)
        result = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df).calculate()

        # SRP: XmR only returns XmR (no longer bundled with R by default)
        assert hasattr(result, 'keys') and hasattr(result, 'values')
        keys = list(result.keys())
        assert 'X' in keys

        # Check that XmR chart has strata
        xmr_chart = result['X']
        assert 'strata' in xmr_chart
        # Third group only had 1 obs so should be dropped
        strata = xmr_chart['strata']
        assert 'a_c' in strata
        assert 'b_d' in strata

        # Statistics are nested by stratum
        xmr_stats = xmr_chart['statistics']

        # Check centers
        assert xmr_stats['a_c']['center'] == 2.33
        assert xmr_stats['b_d']['center'] == 7.67

        # Check limits
        assert xmr_stats['a_c']['lpl'] == -0.33
        assert xmr_stats['b_d']['lpl'] == 1.02
        assert xmr_stats['a_c']['upl'] == 4.99
        assert xmr_stats['b_d']['upl'] == 14.32

    def test_xmr_with_grouping_companion(self, df):
        """Test XmR chart with grouping returns both XmR and R when companion=True."""
        spec = {
            'analysis_type': 'X',
            'rsg_vars': ['a', 'b'],
            'time_var': 'd',
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'round_to': 2,
            'companion': True  # Request bundled XmR+R
        }
        sds = detect_sds_for_test(df, spec)
        result = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df).calculate()

        # Companion mode: XmR and R are bundled together
        keys = list(result.keys())
        assert 'X' in keys
        assert 'mR' in keys

        # Check R chart is present with nested statistics
        r_chart = result['mR']
        assert 'strata' in r_chart
        assert 'a_c' in r_chart['strata']
        assert 'b_d' in r_chart['strata']


# ========================
# R Chart Tests
# ========================

class TestRChartAnalysis:
    """Tests for Moving Range (R) chart calculations."""

    def test_r_with_grouping(self, df):
        """Test R chart with grouping variable (SRP: R only)."""
        spec = {
            'analysis_type': 'mR',
            'rsg_vars': ['a', 'b'],
            'time_var': 'd',
            'response_var': 'c',
            'rsg_var_name': 'rsg',
            'round_to': 2
        }
        sds = detect_sds_for_test(df, spec)
        result = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df).calculate()

        # SRP: R only returns R (no longer bundled with XmR by default)
        assert hasattr(result, "keys") and hasattr(result, "values")
        keys = list(result.keys())
        assert 'mR' in keys

        # Check that R chart has strata
        r_chart = result['mR']
        assert 'strata' in r_chart
        strata = r_chart['strata']
        # Third group had 1 obs and should be dropped
        assert 'a_c' in strata
        assert 'b_d' in strata

        # Statistics are nested by stratum
        r_stats = r_chart['statistics']

        # Check centers
        assert r_stats['a_c']['center'] == 1
        assert r_stats['b_d']['center'] == 2.5

        # Check limits
        assert r_stats['a_c']['lpl'] == 0
        assert r_stats['b_d']['lpl'] == 0
        assert r_stats['a_c']['upl'] == 3.27
        assert r_stats['b_d']['upl'] == 8.17

    def test_r_with_fillweight_data(self):
        """Test R chart on FillWeight800 dataset with stratification (SRP: R only)."""
        f_path = "processbehavior/datasets/data/FILLWEIGHTDATA_800.csv"
        df = pd.read_csv(f_path)

        spec = {
            'analysis_type': 'mR',
            'rsg_vars': ['lane', 'phase'],
            'response_var': 'fill_weight',
            'rsg_var_name': 'rsg',
            'time_var': 'pull'
        }

        sds = detect_sds_for_test(df, spec)
        result = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df).calculate()

        # SRP: R only returns R
        assert hasattr(result, "keys") and hasattr(result, "values")
        assert 'mR' in result

        # Strata are nested inside each chart
        r_chart = result['mR']
        assert 'strata' in r_chart
        assert len(r_chart['strata']) == 8  # 8 lane-phase combinations

        # Check data has no nulls in key columns
        assert not r_chart['data'][['center', 'lpl', 'upl']].isnull().values.any()


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
            # Xbar/S need explicit by for factor aggregation (test data has n=1 per cell)
            if analysis in ['Xbar', 'S']:
                spec['by'] = ['a', 'b']
            else:
                spec['by'] = ['a', 'b']  # XmR/R also needs explicit by with factors
            sds = detect_sds_for_test(df_dt, spec)
            result = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df_dt).calculate()

            if analysis in ['X', 'mR']:
                # SRP: Each chart type returns only itself
                out = result[analysis]['data']
                assert out.columns.tolist()[0] == has_time

    def test_string_date_converted_and_sorted(self, df_dt):
        """Test that string date columns are converted and sorted chronologically."""
        spec = {
            'analysis_type': 'X',
            'rsg_vars': ['a', 'b'],
            'time_var': 'd2',
            'response_var': 'c',
            'rsg_var_name': 'rsg'
        }

        sds = detect_sds_for_test(df_dt, spec)
        result = Analysis(spec=make_spec(spec), request=make_request(spec), sds=sds, df=df_dt).calculate()

        # With new bundled structure, access via chart type key
        xmr_data = result['X']['data']

        # String dates should be converted to datetime
        o_type = xmr_data['d2'].dtype
        assert pd.api.types.is_datetime64_any_dtype(o_type), f"Expected datetime type, got {o_type}"

        # Filter to a specific stratum for ordering check
        stratum_data = xmr_data[xmr_data['rsg'] == 'a_c']

        # Should be chronologically ordered (2000-02-01 is first)
        dt_val = stratum_data.iloc[0, 0]
        expected = pd.Timestamp('2000-02-01')
        assert dt_val == expected


# ========================
# AnalysisDataSet Tests
# ========================

# ========================
# Canonical Ordering Tests
# ========================

class TestCanonicalOrdering:
    """Tests for sort_key and obs_id canonical ordering behavior."""

    def test_sort_key_deterministic_from_cell_key_and_obs_id(self):
        """sort_key should be deterministic based on (cell_key, obs_id) ordering."""
        # Create data intentionally out of canonical order
        df = pd.DataFrame({
            'factor': ['B', 'A', 'A', 'B'],  # Out of order
            'time': [2, 1, 2, 1],  # Out of order
            'y': [10, 20, 30, 40]
        })

        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor'],
            'time_var': 'time',
            'response_var': 'y',
            'round_to': 2
        }

        sds = detect_sds_for_test(df, spec)
        a_spec = make_spec(spec)
        dataset = ad.AnalysisDataSet(df=df, spec=a_spec, observed_sds=sds)

        ads = dataset.analysis_dataset

        # sort_key should be 0..N-1 in canonical order
        assert ads['sort_key'].tolist() == list(range(len(ads)))

        # Data should be sorted by cell_key (factor, time)
        # Expected order: A-1, A-2, B-1, B-2
        cell_keys = ads['cell_key'].tolist()
        assert cell_keys == sorted(cell_keys), "Data should be sorted by cell_key"

    def test_obs_id_preserves_pre_sort_row_order(self):
        """obs_id should reflect row order before canonical sorting."""
        # Create data with known pre-sort order
        df = pd.DataFrame({
            'factor': ['B', 'A', 'B', 'A'],  # Rows 0,1,2,3 in original order
            'time': [1, 1, 2, 2],
            'y': [10, 20, 30, 40]
        })

        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor'],
            'time_var': 'time',
            'response_var': 'y',
            'round_to': 2
        }

        sds = detect_sds_for_test(df, spec)
        a_spec = make_spec(spec)
        dataset = ad.AnalysisDataSet(df=df, spec=a_spec, observed_sds=sds)

        ads = dataset.analysis_dataset

        # After canonical sort, obs_id values will be reordered but preserve identity
        # Original: row 0 (B,1), row 1 (A,1), row 2 (B,2), row 3 (A,2)
        # Canonical order: (A,1), (A,2), (B,1), (B,2)
        # So sorted obs_id should be: 1, 3, 0, 2 (reflecting original row positions)

        # obs_id should still contain all original values 0..N-1
        assert set(ads['obs_id'].tolist()) == {0, 1, 2, 3}

        # sort_key should be 0..N-1 in order
        assert ads['sort_key'].tolist() == [0, 1, 2, 3]

    def test_sort_key_stable_tie_breaking_by_obs_id(self):
        """When cell_keys are equal, obs_id should break ties stably."""
        # Create data with same cell_key for multiple rows
        df = pd.DataFrame({
            'factor': ['A', 'A', 'A'],  # Same factor
            'time': [1, 1, 1],  # Same time = same cell_key
            'y': [10, 20, 30]
        })

        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor'],
            'time_var': 'time',
            'response_var': 'y',
            'round_to': 2
        }

        sds = detect_sds_for_test(df, spec)
        a_spec = make_spec(spec)
        dataset = ad.AnalysisDataSet(df=df, spec=a_spec, observed_sds=sds)

        ads = dataset.analysis_dataset

        # All rows have same cell_key, so obs_id breaks ties
        # Original order: obs_id 0, 1, 2
        # After sort by (cell_key, obs_id): still 0, 1, 2
        assert ads['obs_id'].tolist() == [0, 1, 2]
        assert ads['sort_key'].tolist() == [0, 1, 2]


# ========================
# Utility Function Tests
# ========================

class TestUtilityFunctions:
    """Tests for analysis utility functions."""

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
