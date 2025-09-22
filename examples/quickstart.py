from processbehavior import AnalysisSpec, analyze, load_demo

df = load_demo()
spec = AnalysisSpec(response_var='y', time_var='t', grouping=['line'])
res = analyze(df, spec)
print(res['charts'].keys())
