import pytest

from processbehavior.formulation_spec import ChartRequest, FormulationSpec

# =============================================================================
# FormulationSpec Tests
# =============================================================================

def test_formulation_spec_basic():
    """Test FormulationSpec with all primary fields."""
    spec = FormulationSpec(
        response_var='Height',
        rsg_vars=('Operator',),
        time_var='Time'
    )

    assert spec.response_var == 'Height'
    assert spec.rsg_vars == ('Operator',)
    assert spec.time_var == 'Time'
    assert spec.has_grouping
    assert spec.has_time
    assert spec.requires_sort


def test_formulation_spec_requires_response_var():
    """Test FormulationSpec raises error when response_var is None."""
    with pytest.raises(ValueError, match='response variable is required'):
        FormulationSpec(response_var=None)


def test_formulation_spec_has_grouping():
    """Test has_grouping property."""
    spec_with_grouping = FormulationSpec(
        response_var='y',
        rsg_vars=('a', 'b')
    )
    assert spec_with_grouping.has_grouping

    spec_no_grouping = FormulationSpec(response_var='y')
    assert not spec_no_grouping.has_grouping


def test_formulation_spec_has_time():
    """Test has_time property."""
    spec_with_time = FormulationSpec(response_var='y', time_var='t')
    assert spec_with_time.has_time

    spec_no_time = FormulationSpec(response_var='y')
    assert not spec_no_time.has_time


def test_formulation_spec_requires_sort():
    """Test requires_sort is True when time_var is present, False otherwise."""
    # Both grouping and time -> requires sort
    spec = FormulationSpec(response_var='y', rsg_vars=('a',), time_var='t')
    assert spec.requires_sort

    # Only time -> requires sort
    spec = FormulationSpec(response_var='y', time_var='t')
    assert spec.requires_sort

    # Only grouping, no time -> no sort required
    spec = FormulationSpec(response_var='y', rsg_vars=('a',))
    assert not spec.requires_sort

    # Neither grouping nor time -> no sort required
    spec = FormulationSpec(response_var='y')
    assert not spec.requires_sort


def test_formulation_spec_rsg_delim_default():
    """Test rsg_var_delim defaults to underscore."""
    spec = FormulationSpec(response_var='y', rsg_vars=('a', 'b'))
    assert spec.rsg_var_delim == '_'


def test_formulation_spec_rsg_delim_custom():
    """Test rsg_var_delim accepts custom delimiter."""
    spec = FormulationSpec(
        response_var='y',
        rsg_vars=('a', 'b'),
        rsg_var_delim='|'
    )
    assert spec.rsg_var_delim == '|'


def test_formulation_spec_defaults():
    """Test default values for all optional fields."""
    spec = FormulationSpec(response_var='y')

    assert spec.rsg_vars is None
    assert spec.time_var is None
    assert spec.round_to == 3
    assert spec.rsg_var_name == 'rsg'
    assert spec.rsg_var_delim == '_'
    assert spec.unit_of_analysis is None


def test_formulation_spec_round_to():
    """Test custom round_to value."""
    spec = FormulationSpec(response_var='y', round_to=5)
    assert spec.round_to == 5


def test_formulation_spec_unit_of_analysis():
    """Test unit_of_analysis field."""
    spec = FormulationSpec(response_var='y', unit_of_analysis='wafer')
    assert spec.unit_of_analysis == 'wafer'


def test_formulation_spec_rsg_vars_list():
    """Test rsg_vars_list returns a list copy, or empty list if None."""
    spec_with = FormulationSpec(response_var='y', rsg_vars=('a', 'b'))
    assert spec_with.rsg_vars_list == ['a', 'b']
    assert isinstance(spec_with.rsg_vars_list, list)
    # Each call returns a fresh list (safe to mutate)
    assert spec_with.rsg_vars_list is not spec_with.rsg_vars_list

    spec_none = FormulationSpec(response_var='y')
    assert spec_none.rsg_vars_list == []


def test_formulation_spec_is_frozen():
    """Test that FormulationSpec is immutable."""
    spec = FormulationSpec(response_var='y')
    with pytest.raises(AttributeError):
        spec.response_var = 'z'


def test_formulation_spec_rsg_vars_is_tuple():
    """Test that rsg_vars is stored as a tuple, not a list."""
    spec = FormulationSpec(response_var='y', rsg_vars=('a', 'b'))
    assert isinstance(spec.rsg_vars, tuple)


def test_formulation_spec_multi_rsg_vars():
    """Test FormulationSpec with multiple rational subgrouping variables."""
    spec = FormulationSpec(
        response_var='c',
        rsg_vars=('a', 'b'),
        time_var='d',
        rsg_var_name='rsg'
    )
    assert spec.has_grouping
    assert spec.has_time
    assert spec.requires_sort
    assert spec.rsg_vars == ('a', 'b')


def test_formulation_spec_grouping_no_time():
    """Test FormulationSpec with grouping but no time variable."""
    spec = FormulationSpec(
        response_var='c',
        rsg_vars=('a', 'b'),
        rsg_var_name='rsg'
    )
    assert spec.has_grouping
    assert not spec.has_time
    assert not spec.requires_sort


# =============================================================================
# ChartRequest Tests
# =============================================================================

def test_chart_request_basic():
    """Test ChartRequest with only the required chart field."""
    req = ChartRequest(chart='X')
    assert req.chart == 'X'
    assert req.by is None
    assert req.value_col is None
    assert req.residual is None
    assert req.residual_chart_type is None
    assert req.recentered is False
    assert req.companion is False
    assert req.bins == 10


def test_chart_request_xbar():
    """Test ChartRequest for Xbar chart type."""
    req = ChartRequest(chart='Xbar', by=('Operator',))
    assert req.chart == 'Xbar'
    assert req.by == ('Operator',)


def test_chart_request_s_chart():
    """Test ChartRequest for S chart type."""
    req = ChartRequest(chart='S', by=('Operator',))
    assert req.chart == 'S'
    assert req.by == ('Operator',)


def test_chart_request_companion():
    """Test ChartRequest with companion=True for Xbar+S or X+mR."""
    req = ChartRequest(chart='Xbar', companion=True)
    assert req.companion is True


def test_chart_request_residual():
    """Test ChartRequest for residual charting."""
    req = ChartRequest(
        chart='X',
        residual='R2',
        residual_chart_type='X',
        value_col='R2'
    )
    assert req.residual == 'R2'
    assert req.residual_chart_type == 'X'
    assert req.value_col == 'R2'


def test_chart_request_recentered():
    """Test ChartRequest with recentered residuals."""
    req = ChartRequest(chart='X', residual='R2', recentered=True)
    assert req.recentered is True


def test_chart_request_histogram():
    """Test ChartRequest for Histogram with custom bins."""
    req = ChartRequest(chart='Histogram', bins=20)
    assert req.chart == 'Histogram'
    assert req.bins == 20


def test_chart_request_is_frozen():
    """Test that ChartRequest is immutable."""
    req = ChartRequest(chart='X')
    with pytest.raises(AttributeError):
        req.chart = 'Xbar'


def test_chart_request_value_col():
    """Test ChartRequest with explicit value_col."""
    req = ChartRequest(chart='X', value_col='Height')
    assert req.value_col == 'Height'


def test_chart_request_by_tuple():
    """Test that by field is stored as a tuple."""
    req = ChartRequest(chart='X', by=('a', 'b'))
    assert isinstance(req.by, tuple)
    assert req.by == ('a', 'b')
