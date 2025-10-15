import logging
import pytest

import analysis_dataset as ad

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_analysis_specification_valid():

    #Handles missing key and None value for valid key - get() resolves to None in both cases
    spec_with_no_response = {'rsg_vars': ['a', 'b'], 'time_var': 'd',
            'rsg_var_name': 'rsg','response_var': None}


    with pytest.raises(ValueError):
        ad.AnalysisSpecification(analysis_type='Imr',analysis_specification=spec_with_no_response)
    #TODO: Make loop to test all analyis types
    spec_with_no_rsg = {'response_var': 'c', 'time_var': 'd', 'rsg_var_name': 'rsg'}
    with pytest.raises(ValueError):
        ad.AnalysisSpecification(analysis_type='Xbar',analysis_specification=spec_with_no_rsg)

    with pytest.raises(ValueError):
        ad.AnalysisSpecification(analysis_type='S',analysis_specification=spec_with_no_rsg)

        #expect time and rsg to be first two columns of output cols

    #Check configuration
def test_analysis_specification_Imr_no_time_var():
    spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'],'response_var': 'c', 'rsg_var_name': 'rsg'}
    logger.info(f'Key is: {spec.get("time_var")}')
    asImr = ad.AnalysisSpecification(analysis_type='Imr',analysis_specification=spec)
    logger.debug(f'{spec}')
    logger.info(f'\nAnalysis Type is: {asImr.analysis_type}')

    #rsg_vars provided
    assert asImr.has_grouping == True

    #group if there is an rsg
    assert asImr.has_time == False #no time_var

    #rsg is present sort by it only add time if it is provided
    assert asImr.sort_cols == []

    #expect x and rsg to be first two columns of output cols
    assert asImr.analysis_output_cols == ['x', 'rsg', spec['response_var'], 'mean', 'lcl', 'ucl', 'beyond_limits']

def test_analysis_specification_Imr_w_time_var():

    spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'],'time_var':'d', 'response_var': 'c', 'rsg_var_name': 'rsg'}
    #[specs['time_var'], 'rsg', specs['response_var'], 'mean', 'lcl', 'ucl', 'beyond_limits']
    asImr = ad.AnalysisSpecification(analysis_type='Imr',analysis_specification=spec)
    logger.debug(f'{spec}')
    logger.info(f'\nAnalysis Type is: {asImr.analysis_type}')
    assert asImr.has_grouping == True
    assert asImr.has_time == True
    assert asImr.sort_cols == [spec['rsg_var_name'], spec['time_var']]
    assert asImr.requires_sort == True

    #expect time_var and rsg to be first two columns of output cols
    assert asImr.analysis_output_cols == [spec['time_var'], spec['rsg_var_name'], spec['response_var'], 'mean', 'lcl', 'ucl', 'beyond_limits']
        
    # def test_analysis_specification_test_time_unit(self):
    #     no_time_invalid_spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 
    #             'response_var': 'c', 'rsg_var_name': 'rsg','time_unit':'Month'}
        
    #     with self.assertRaises(ValueError):
    #         ad.AnalysisSpecification(analysis_type='Imr',analysis_specification=no_time_invalid_spec)

    #     invalid_time_unit_spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var':'d', 
    #             'response_var': 'c', 'rsg_var_name': 'rsg','time_unit':'month'}

    #     with self.assertRaises(ValueError):
    #         ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=invalid_time_unit_spec)
        
        
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
    time_w_group_spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var':'d',
            'response_var': 'c', 'rsg_var_name': 'rsg','time_unit':'Month'}
    sortable = True

    logger.info('\nTesting: Sort required when time_var and rsg_var specified and that sortcols from match....')
    aspec = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=time_w_group_spec)
    assert aspec.requires_sort == sortable
    assert aspec.sort_cols == ['rsg','d']

    time_no_group_spec = {'analysis_type': 'Imr', 'time_var':'d',
            'response_var': 'c'}
    logger.info('\nTesting: Sort required when only time_var specified and that sortcols from match...')
    aspec = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=time_no_group_spec)
    assert aspec.requires_sort == sortable
    assert aspec.sort_cols == ['d']

def test_analysis_specification_rsg_delim():
    no_delim_spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var':'d',
            'response_var': 'c', 'rsg_var_name': 'rsg', 'time_unit':'Month'}
    aspec = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=no_delim_spec)
    logger.info('\nTesting rsg_var_delim is set properly to default: "_" specified in spec:...')
    assert aspec.rsg_var_delim == "_"

    logger.info(f'\nTesting n in data prep output cols for grouped data: {aspec.data_prep_output_cols}')
    actual = "n" in aspec.data_prep_output_cols
    assert actual == True

    delim_spec = {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var':'d',
            'response_var': 'c', 'rsg_var_name': 'rsg', 'rsg_var_delim': '|', 'time_unit':'Month'}
    aspec = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=delim_spec)

    logger.info('\nTesting rsg_var_delim is set properly to: "|" specified in spec:...')
    assert aspec.rsg_var_delim == "|"



def test_analysis_specification_zero_center():
    zero_center_spec_false= {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var':'d',
            'response_var': 'c', 'rsg_var_name': 'rsg','time_unit':'Month'}

    logger.info('\nTesting: Value of zero-center is False when not set in spec....')
    aspec = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=zero_center_spec_false)
    assert aspec.zero_center == False

    zero_center_spec_true= {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var':'d',
            'response_var': 'c', 'rsg_var_name': 'rsg','time_unit':'Month','zero-center':True}

    logger.info('\nTesting: Value of zero-center is True when set in spec....')
    aspec = ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=zero_center_spec_true)
    assert aspec.zero_center == True

    zero_center_spec_invalid= {'analysis_type': 'Imr', 'rsg_vars': ['a', 'b'], 'time_var':'d',
            'response_var': 'c', 'rsg_var_name': 'rsg','time_unit':'Month','zero-center':1234}

    logger.info('\nTesting: ValueError raised when zero-center is set to none boolean value in spec....')
    with pytest.raises(ValueError):
        ad.AnalysisSpecification(analysis_type='Imr', analysis_specification=zero_center_spec_invalid)
        
       