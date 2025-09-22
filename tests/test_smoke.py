import pandas as pd

from processbehavior import AnalysisSpec, analyze, load_demo


def test_analyze_smoke():
    df = load_demo()
    spec = AnalysisSpec(response_var='y', time_var='t', grouping=['line'])
    result = analyze(df, spec)
    assert 'charts' in result and 'meta' in result
    assert 'Xbar' in result['charts'] and 'Imr' in result['charts']
    xbar = result['charts']['Xbar']['data']
    assert isinstance(xbar, pd.DataFrame)
    assert xbar.loc[0, 'count'] == len(df)
