import logging

import pytest

from processbehavior import analysis_dataset as ad
from processbehavior.analysis_specification import DataPrepConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DataPrepConfig Tests (Base Class - No analysis_type)
# =============================================================================

def test_data_prep_config_basic():
    """Test DataPrepConfig works without analysis_type."""
    spec = {
        'response_var': 'Height',
        'rsg_vars': ['Operator'],
        'time_var': 'Time'
    }
    config = DataPrepConfig(spec)

    assert config.response_var == 'Height'
    assert config.rsg_vars == ['Operator']
    assert config.time_var == 'Time'
    assert config.has_grouping
    assert config.has_time
    assert config.requires_sort


def test_data_prep_config_requires_response_var():
    """Test DataPrepConfig raises error without response_var."""
    spec_no_response = {'rsg_vars': ['a', 'b'], 'time_var': 'd'}

    with pytest.raises(ValueError, match='response variable is required'):
        DataPrepConfig(spec_no_response)


def test_data_prep_config_has_grouping():
    """Test has_grouping property."""
    spec_with_grouping = {'response_var': 'y', 'rsg_vars': ['a', 'b']}
    config = DataPrepConfig(spec_with_grouping)
    assert config.has_grouping

    spec_no_grouping = {'response_var': 'y'}
    config = DataPrepConfig(spec_no_grouping)
    assert not config.has_grouping


def test_data_prep_config_has_time():
    """Test has_time property."""
    spec_with_time = {'response_var': 'y', 'time_var': 't'}
    config = DataPrepConfig(spec_with_time)
    assert config.has_time

    spec_no_time = {'response_var': 'y'}
    config = DataPrepConfig(spec_no_time)
    assert not config.has_time


def test_data_prep_config_sort_cols():
    """Test sort_cols are built correctly."""
    # Both grouping and time
    spec = {'response_var': 'y', 'rsg_vars': ['a'], 'time_var': 't'}
    config = DataPrepConfig(spec)
    assert config.sort_cols == ['rsg', 't']
    assert config.requires_sort

    # Only time
    spec = {'response_var': 'y', 'time_var': 't'}
    config = DataPrepConfig(spec)
    assert config.sort_cols == ['t']
    assert config.requires_sort

    # No time
    spec = {'response_var': 'y', 'rsg_vars': ['a']}
    config = DataPrepConfig(spec)
    assert config.sort_cols == []
    assert not config.requires_sort


def test_data_prep_config_rsg_delim():
    """Test rsg_var_delim configuration."""
    # Default delimiter
    spec = {'response_var': 'y', 'rsg_vars': ['a', 'b']}
    config = DataPrepConfig(spec)
    assert config.rsg_var_delim == '_'

    # Custom delimiter
    spec = {'response_var': 'y', 'rsg_vars': ['a', 'b'], 'rsg_var_delim': '|'}
    config = DataPrepConfig(spec)
    assert config.rsg_var_delim == '|'


# =============================================================================
# AnalysisSpecification Tests (Extended Class - With analysis_type)
# =============================================================================

def test_analysis_specification_valid():
    """Test AnalysisSpecification validation with new unified constructor."""
    # Handles missing key and None value for valid key - get() resolves to None in both cases
    spec_with_no_response = {
        'analysis_type': 'Imr',
        'rsg_vars': ['a', 'b'],
        'time_var': 'd',
        'rsg_var_name': 'rsg',
        'response_var': None
    }

    with pytest.raises(ValueError):
        ad.AnalysisSpecification(spec_with_no_response)

    # TODO: Make loop to test all analysis types
    spec_with_no_rsg_xbar = {
        'analysis_type': 'Xbar',
        'response_var': 'c',
        'time_var': 'd',
        'rsg_var_name': 'rsg'
    }
    with pytest.raises(ValueError):
        ad.AnalysisSpecification(spec_with_no_rsg_xbar)

    spec_with_no_rsg_s = {
        'analysis_type': 'S',
        'response_var': 'c',
        'time_var': 'd',
        'rsg_var_name': 'rsg'
    }
    with pytest.raises(ValueError):
        ad.AnalysisSpecification(spec_with_no_rsg_s)

    # expect time and rsg to be first two columns of output cols

    # Check configuration
def test_analysis_specification_Imr_no_time_var():
    spec = {
        'analysis_type': 'Imr',
        'rsg_vars': ['a', 'b'],
        'response_var': 'c',
        'rsg_var_name': 'rsg'
    }
    logger.info(f'Key is: {spec.get("time_var")}')
    asImr = ad.AnalysisSpecification(spec)
    logger.debug(f'{spec}')
    logger.info(f'\nAnalysis Type is: {asImr.analysis_type}')

    # rsg_vars provided
    assert asImr.has_grouping

    # group if there is an rsg
    assert not asImr.has_time  # no time_var

    # rsg is present sort by it only add time if it is provided
    assert asImr.sort_cols == []

    # expect x and rsg to be first two columns of output cols
    assert asImr.analysis_output_cols == ['x', 'rsg', spec['response_var'], 'mean', 'lpl', 'upl', 'beyond_limits']


def test_analysis_specification_Imr_w_time_var():
    spec = {
        'analysis_type': 'Imr',
        'rsg_vars': ['a', 'b'],
        'time_var': 'd',
        'response_var': 'c',
        'rsg_var_name': 'rsg'
    }
    # [specs['time_var'], 'rsg', specs['response_var'], 'mean', 'lpl', 'upl', 'beyond_limits']
    asImr = ad.AnalysisSpecification(spec)
    logger.debug(f'{spec}')
    logger.info(f'\nAnalysis Type is: {asImr.analysis_type}')
    assert asImr.has_grouping
    assert asImr.has_time
    assert asImr.sort_cols == [spec['rsg_var_name'], spec['time_var']]
    assert asImr.requires_sort

    # expect time_var and rsg to be first two columns of output cols
    assert asImr.analysis_output_cols == [
        spec['time_var'], spec['rsg_var_name'], spec['response_var'],
        'mean', 'lpl', 'upl', 'beyond_limits'
    ]
        
    # def test_analysis_specification_test_time_unit(self):
    #     no_time_invalid_spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 
    #             'response_var': 'c', 'rsg_var_name': 'rsg','time_unit':'Month'}
        
    #     with self.assertRaises(ValueError):
    #         ad.AnalysisSpecification(
    #             analysis_type='Imr',
    #             analysis_specification=no_time_invalid_spec
    #         )

    #     invalid_time_unit_spec = {
    #         'analysis_type': 'Imr',
    #         'rsg_vars': ['a', 'b'],
    #         'time_var':'d',
    #         'response_var': 'c',
    #         'rsg_var_name': 'rsg',
    #         'time_unit':'month'
    #     }

    #     with self.assertRaises(ValueError):
    #         ad.AnalysisSpecification(
    #             analysis_type='Imr',
    #             analysis_specification=invalid_time_unit_spec
    #         )
        
        
    #     valid_spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 
    #             'response_var': 'c','rsg_var_name': 'rsg'}
                
    #     print(f'Testing:data prep output cols with response, and rsg for spec: \n')
    #     print(f'{valid_spec}')
    #     result = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=valid_spec)
        
    #     #test data prep output cols   
    #     expected = ['rsg','n', 'c']
    #     self.assertEqual(result.data_prep_output_cols, expected)
    #     print(f'Requires sort: {result.requires_sort}')

def test_analysis_specification_sort_required():
    time_w_group_spec = {
        'analysis_type': 'Imr',
        'rsg_vars': ['a', 'b'],
        'time_var': 'd',
        'response_var': 'c',
        'rsg_var_name': 'rsg',
        'time_unit': 'Month'
    }
    sortable = True

    logger.info(
        '\nTesting: Sort required when time_var and rsg_var specified '
        'and that sortcols from match....'
    )
    aspec = ad.AnalysisSpecification(time_w_group_spec)
    assert aspec.requires_sort == sortable
    assert aspec.sort_cols == ['rsg', 'd']

    time_no_group_spec = {
        'analysis_type': 'Imr',
        'time_var': 'd',
        'response_var': 'c'
    }
    logger.info('\nTesting: Sort required when only time_var specified and that sortcols from match...')
    aspec = ad.AnalysisSpecification(time_no_group_spec)
    assert aspec.requires_sort == sortable
    assert aspec.sort_cols == ['d']


def test_analysis_specification_rsg_delim():
    no_delim_spec = {
        'analysis_type': 'Imr',
        'rsg_vars': ['a', 'b'],
        'time_var': 'd',
        'response_var': 'c',
        'rsg_var_name': 'rsg',
        'time_unit': 'Month'
    }
    aspec = ad.AnalysisSpecification(no_delim_spec)
    logger.info('\nTesting rsg_var_delim is set properly to default: "_" specified in spec:...')
    assert aspec.rsg_var_delim == "_"

    logger.info(f'\nTesting n in data prep output cols for grouped data: {aspec.data_prep_output_cols}')
    actual = "n" in aspec.data_prep_output_cols
    assert actual

    delim_spec = {
        'analysis_type': 'Imr',
        'rsg_vars': ['a', 'b'],
        'time_var': 'd',
        'response_var': 'c',
        'rsg_var_name': 'rsg',
        'rsg_var_delim': '|',
        'time_unit': 'Month'
    }
    aspec = ad.AnalysisSpecification(delim_spec)

    logger.info('\nTesting rsg_var_delim is set properly to: "|" specified in spec:...')
    assert aspec.rsg_var_delim == "|"
