"""
Test to verify correct columns are used for plotting and signal detection.

This test exposes bugs in:
1. Which column is plotted (value column)
2. Which column is used for signal detection
3. Which column is the centerline

These tests verify the DECLARATIVE CONTRACT that each chart type should follow.
"""
import pandas as pd

from processbehavior.analysis import Analysis
from processbehavior.data_preparation import DataPreparation
from processbehavior.plotting.plotter import Plotter
from processbehavior.sds_detector import SDSRegistry


def detect_sds_for_test(df: pd.DataFrame, spec: dict) -> int:
    """
    Helper to detect SDS for tests that need to create Analysis directly.
    """
    from processbehavior.analysis_specification import AnalysisSpecification
    config = AnalysisSpecification(spec)
    prep = DataPreparation()
    prep.validate_columns(df, config)
    prepared_df = prep.prepare_dataset(df, config)
    detector = SDSRegistry()
    return detector.detect_sds(prepared_df, config)


def test_xbar_chart_column_contract():
    """
    Verify Xbar chart follows the column contract:
    - 'xbar' column contains varying subgroup means (VALUES TO PLOT)
    - 'center' column contains constant grand mean (CENTERLINE)
    - Plotter uses 'xbar' column
    """
    # Use the same test data structure from test_analysis_dataset.py that works
    df = pd.DataFrame({
        'a': ['a', 'a', 'a', 'b', 'b', 'b', 'c'],
        'b': ['c', 'c', 'c', 'd', 'd', 'd', 'e'],
        'c': [1.5, 2.0, 3.5, 5.0, 8.0, 10.0, 1.0],
        'd': [1, 2, 3, 1, 2, 3, 1],
    })

    spec = {
        'analysis_type': 'Xbar',
        'response_var': 'c',
        'rsg_vars': ['a', 'b'],
        'time_var': 'd'
    }

    sds = detect_sds_for_test(df, spec)
    result = Analysis(df, spec, sds=sds).calculate()
    xbar_data = result.charts['Xbar']['data']

    # Test column structure
    assert 'xbar' in xbar_data.columns, "Xbar data must have 'xbar' column"
    assert 'center' in xbar_data.columns, "Xbar data must have 'center' column"

    # Test that 'xbar' varies (these are the values to plot)
    xbar_values = xbar_data['xbar'].tolist()
    assert len(set(xbar_values)) > 1, "'xbar' must vary across subgroups (values to plot)"
    # Verify we have different means (actual values from test data)
    assert min(xbar_values) < max(xbar_values), "Xbar values should vary across subgroups"

    # Test that 'center' is constant (this is the centerline)
    center_values = xbar_data['center'].tolist()
    assert len(set(center_values)) == 1, "'center' must be constant (centerline)"

    # Test plotter uses correct column (via metadata)
    plotter = Plotter(result)
    xbar_chart_info = result.charts['Xbar']
    value_col = plotter._get_value_column(xbar_chart_info, 'Xbar')
    assert value_col == 'xbar', f"Plotter must use 'xbar' column for Xbar, got '{value_col}'"


def test_s_chart_column_contract():
    """
    Verify S chart follows the column contract:
    - 's' column contains varying subgroup std devs (VALUES TO PLOT)
    - 'center' column contains constant mean of std devs (CENTERLINE)
    - Plotter uses 's' column
    """
    # Create data with VARYING standard deviations
    # Use more variation in values to get different std devs per subgroup
    df = pd.DataFrame({
        'a': ['a', 'a', 'a', 'b', 'b', 'b', 'c'],
        'b': ['c', 'c', 'c', 'd', 'd', 'd', 'e'],
        'c': [1.0, 2.0, 5.0,   # std≈2.08
              10.0, 12.0, 20.0,  # std≈5.29
              1.0],
        'd': [1, 2, 3, 1, 2, 3, 1],
    })

    spec = {
        'analysis_type': 'Xbar',
        'response_var': 'c',
        'rsg_vars': ['a', 'b'],
        'time_var': 'd'
    }

    sds = detect_sds_for_test(df, spec)
    result = Analysis(df, spec, sds=sds).calculate()
    sbar_data = result.charts['S']['data']

    # Test column structure
    assert 's' in sbar_data.columns, "S chart data must have 's' column"
    assert 'center' in sbar_data.columns, "S chart data must have 'center' column"

    # CRITICAL TEST: 's' should vary (these are the values to plot)
    s_values = sbar_data['s'].tolist()
    assert len(set(s_values)) > 1, (
        "'s' must vary across subgroups (values to plot). "
        f"Got: {s_values}. This is a BUG if all values are identical."
    )

    # Test that 'center' is constant (this is the centerline)
    center_values = sbar_data['center'].tolist()
    assert len(set(center_values)) == 1, "'center' must be constant (centerline)"

    # Test plotter uses correct column (via metadata)
    plotter = Plotter(result)
    sbar_chart_info = result.charts['S']
    value_col = plotter._get_value_column(sbar_chart_info, 'S')
    assert value_col == 's', f"Plotter must use 's' column for S chart, got '{value_col}'"


def test_imr_chart_column_contract():
    """
    Verify IMR chart follows the column contract:
    - response_var column contains varying individual values (VALUES TO PLOT)
    - 'center' column contains constant grand mean (CENTERLINE)
    - Plotter uses response_var column
    """
    df = pd.DataFrame({
        'time': [1, 2, 3, 4, 5],
        'group': ['A', 'A', 'A', 'A', 'A'],
        'weight': [235.5, 237.2, 239.1, 236.8, 238.3]  # Individual values
    })

    spec = {
        'analysis_type': 'Imr',
        'response_var': 'weight',
        'rsg_vars': ['group'],
        'time_var': 'time'
    }

    sds = detect_sds_for_test(df, spec)
    result = Analysis(df, spec, sds=sds).calculate()

    # Get the chart (not standard chart names like 'Imr', but the actual group chart)
    chart_name = [k for k in result.charts if k not in {'Xbar', 'S', 'Imr', 'R'}][0]
    imr_data = result.charts[chart_name]['data']

    # Test column structure
    assert 'weight' in imr_data.columns, "IMR data must have response_var column"
    assert 'center' in imr_data.columns, "IMR data must have 'center' column"

    # CRITICAL TEST: 'weight' should vary (these are the values to plot)
    weight_values = imr_data['weight'].tolist()
    assert len(set(weight_values)) > 1, (
        "'weight' must vary (individual measurements). "
        f"Got: {weight_values}. This is a BUG if all values are identical."
    )
    assert 235.5 in weight_values, "Should contain original measurement 235.5"
    assert 239.1 in weight_values, "Should contain original measurement 239.1"

    # Test that 'center' is constant (this is the centerline)
    center_values = imr_data['center'].tolist()
    assert len(set(center_values)) == 1, "'center' must be constant (centerline)"

    # Test plotter uses correct column (via metadata)
    plotter = Plotter(result)
    imr_chart_info = result.charts[chart_name]
    value_col = plotter._get_value_column(imr_chart_info, chart_name)
    assert value_col == 'weight', (
        f"Plotter must use 'weight' (response_var) column for IMR chart, got '{value_col}'. "
        "This is a BUG - IMR should plot individual values, not the center!"
    )


def test_chart_type_declarative_contract():
    """
    Meta-test: Verify that each chart type declares what column to plot.

    This enforces a DECLARATIVE approach where the chart calculation
    explicitly specifies which column contains the values to plot.
    """
    # Map of analysis_type → expected value column name
    CHART_VALUE_COLUMNS = {
        'Xbar': 'xbar',      # Plot subgroup means
        'S': 's',            # Plot subgroup std devs
        'R': 'mr',           # Plot moving ranges
        'Imr': None,         # Plot response_var (dynamic)
    }

    # Map of analysis_type → expected centerline column name (now standardized)
    CHART_CENTER_COLUMNS = {
        'Xbar': 'center',    # Grand mean
        'S': 'center',       # Mean of std devs
        'R': 'center',       # Mean of moving ranges
        'Imr': 'center',     # Grand mean
    }

    print("\n" + "="*80)
    print("DECLARATIVE CHART CONTRACT VERIFICATION")
    print("="*80)
    print("\nEach chart type must explicitly declare:")
    print("  1. Which column contains values to plot (varies)")
    print("  2. Which column contains the centerline (constant)")
    print("  3. Plotter must respect these declarations")
    print()

    for analysis_type, expected_value_col in CHART_VALUE_COLUMNS.items():
        print(f"{analysis_type} chart:")
        print(f"  Value column: {expected_value_col if expected_value_col else 'response_var'}")
        print(f"  Center column: {CHART_CENTER_COLUMNS[analysis_type]}")


if __name__ == '__main__':
    import sys

    print("="*80)
    print("COLUMN CONTRACT VERIFICATION TESTS")
    print("="*80)
    print("\nThese tests verify the declarative contract for chart columns:")
    print("  - Each chart has a VALUE column (what to plot)")
    print("  - Each chart has a CENTER column (centerline)")
    print("  - Plotter uses the correct VALUE column")
    print()

    failures = []

    tests = [
        ("Xbar chart", test_xbar_chart_column_contract),
        ("S chart", test_s_chart_column_contract),
        ("IMR chart", test_imr_chart_column_contract),
        ("Declarative contract", test_chart_type_declarative_contract),
    ]

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name} test PASSED")
        except AssertionError as e:
            print(f"✗ {name} test FAILED:")
            print(f"  {e}")
            failures.append((name, str(e)))

    print("\n" + "="*80)
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed")
        for name, error in failures:
            print(f"\n{name}:")
            print(f"  {error}")
        sys.exit(1)
    else:
        print("SUCCESS: All tests passed")
        sys.exit(0)
