#!/usr/bin/env python
"""
Demonstration of the new AnalysisResult unified access pattern.

This script shows how the AnalysisResult class provides easy access to:
- Chart data and statistics
- VAS residuals (R1-R5)
- Main effects and interactions
- Comprehensive summary metadata
- Stratified individuals charts

All in one place!
"""

import numpy as np
import pandas as pd

from processbehavior.analysis import Analysis

np.random.seed(42)


def example1_unified_access():
    """Example 1: Unified access to all analysis outputs"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Unified Access Pattern")
    print("="*80)

    # Create sample data
    df = pd.DataFrame({
        'Height': np.random.normal(50, 3, 60),
        'Operator': ['Alice', 'Bob'] * 30,
        'Machine': ['M1', 'M2'] * 30,
        'Time': list(range(1, 21)) * 3
    })

    spec = {
        'analysis_type': 'Xbar',
        'response_var': 'Height',
        'time_var': 'Time',
        'rsg_vars': ['Operator', 'Machine']
    }

    analysis = Analysis(df, spec)
    result = analysis.calculate()

    print(f"\nResult type: {type(result)}")
    print(f"\n{result}")  # __str__ gives nice summary

    # Access charts
    print("\n📊 Charts Available:")
    for chart_name in result.all_charts:
        print(f"  - {chart_name}")

    # Get specific chart
    xbar_data = result.get_chart('Xbar')
    print(f"\nXbar chart has {len(xbar_data)} points")

    # Get statistics
    xbar_stats = result.get_statistics('Xbar')
    print(f"Xbar Mean: {xbar_stats.get('Mean')}")

    # Access residuals if available
    if result.has_residuals:
        print(f"\n📈 Residuals available: {list(result.residuals.columns)}")
        print(f"R1 mean: {result.residuals['R1'].mean():.3f}")
    else:
        print("\n📈 No residuals calculated (SDS doesn't support VAS)")

    # Access effects if available
    if result.has_effects:
        print("\n✨ Effects calculated:")
        print(f"  Factor effects: {len(result.effects.get('k_effects', []))}")
        print(f"  Time effects: {len(result.effects.get('t_effects', []))}")
    else:
        print("\n✨ No effects calculated (SDS doesn't support effects)")

    # Access summary
    print("\n📋 Summary:")
    print(f"  SDS: {result.summary['sds']} - {result.summary['sds_description']}")
    print(f"  Observations: {result.summary['n_observations']}")
    print(f"  Signals: {result.summary['n_signals_total']}")


def example2_stratified_xmr():
    """Example 2: Stratified XmR charts (killer feature!)"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Stratified Individuals Charts")
    print("="*80)

    # Create data with 3 operators
    n = 30
    df = pd.DataFrame({
        'Measurement': np.concatenate([
            np.random.normal(100, 2, n),  # Alice
            np.random.normal(105, 3, n),  # Bob
            np.random.normal(98, 2, n)    # Charlie
        ]),
        'Operator': ['Alice'] * n + ['Bob'] * n + ['Charlie'] * n,
        'Time': list(range(1, n + 1)) * 3
    })

    spec = {
        'analysis_type': 'XmR',
        'response_var': 'Measurement',
        'time_var': 'Time',
        'rsg_vars': ['Operator']
    }

    analysis = Analysis(df, spec)
    result = analysis.calculate()

    print(f"\nStratified analysis: {result.summary['is_stratified']}")
    print(f"Number of charts: {len(result)}")

    print("\n📊 Individual charts per operator:")
    for chart_name, data, stats in result.iter_charts():
        print(f"\n  {chart_name}:")
        print(f"    Points: {len(data)}")
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    LCL: {stats['lcl']:.2f}")
        print(f"    UCL: {stats['ucl']:.2f}")

    # Detect which operators have signals
    print("\n⚠️  Signal Detection:")
    signals = result.get_signals()
    if len(signals) > 0:
        for _, row in signals.iterrows():
            print(f"  {row.get('chart', 'Unknown')}: Point at beyond limits")
    else:
        print("  All operators in control!")


def example3_with_residuals():
    """Example 3: Full VAS analysis with residuals"""
    print("\n" + "="*80)
    print("EXAMPLE 3: VAS Residuals and Effects")
    print("="*80)

    # Create SDS1 data (full replication)
    factors = []
    time_points = []
    measurements = []

    for f1 in ['Low', 'High']:
        for f2 in ['A', 'B']:
            for t in range(1, 6):
                # 3 replicates per cell
                for _ in range(3):
                    factors.append(f"{f1}_{f2}")
                    time_points.append(t)
                    measurements.append(np.random.normal(100, 5))

    df = pd.DataFrame({
        'Y': measurements,
        'Factor': factors,
        'Time': time_points
    })

    spec = {
        'analysis_type': 'Xbar',
        'response_var': 'Y',
        'time_var': 'Time',
        'rsg_vars': ['Factor']
    }

    analysis = Analysis(df, spec)
    result = analysis.calculate()

    print(f"\n{result}")

    if result.has_residuals:
        print("\n📊 VAS Residuals (R1-R5):")
        for residual in ['R1', 'R2', 'R3', 'R4', 'R5']:
            r = result.get_residual(residual)
            if r is not None:
                print(f"  {residual}: mean={r.mean():.3f}, sd={r.std():.3f}")

    if result.has_effects:
        print("\n✨ Main Effects:")
        k_effects = result.effects.get('k_effects')
        if k_effects is not None:
            print(f"  Factor effects range: {k_effects.min():.3f} to {k_effects.max():.3f}")

    if result.has_interactions:
        print("\n🔗 Interactions calculated:")
        for key in result.interactions:
            print(f"  - {key}")


def example4_backward_compatibility():
    """Example 4: Backward compatibility - works like a dict!"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Backward Compatibility (Dict-like Access)")
    print("="*80)

    df = pd.DataFrame({
        'Value': np.random.normal(100, 5, 20),
        'Time': range(1, 21)
    })

    spec = {
        'analysis_type': 'XmR',
        'response_var': 'Value',
        'time_var': 'Time'
    }

    analysis = Analysis(df, spec)
    result = analysis.calculate()

    print("\nOld dict-like access still works:")

    # Dict-like iteration
    print(f"  len(result) = {len(result)}")
    print(f"  'all' in result = {'all' in result}")

    # Dict-like access
    chart = result['all']
    print(f"  result['all'] works: {type(chart)}")

    # Dict methods
    print(f"  result.keys() = {list(result.keys())}")
    print(f"  result.get('all') works: {result.get('all') is not None}")

    # Iteration
    for name, chart_info in result.items():
        print(f"  Iterating: {name} has {len(chart_info['data'])} rows")

    print("\nNew convenient access also works:")
    print(f"  result.get_chart('all') = {len(result.get_chart('all'))} rows")
    print(f"  result.all_charts = {result.all_charts}")
    print(f"  result.has_residuals = {result.has_residuals}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("AnalysisResult Unified Access Demonstration")
    print("="*80)

    example1_unified_access()
    example2_stratified_xmr()
    example3_with_residuals()
    example4_backward_compatibility()

    print("\n" + "="*80)
    print("Key Takeaways")
    print("="*80)
    print("""
1. Everything in one place:
   - Charts:        result.charts or result.get_chart('name')
   - Residuals:     result.residuals
   - Effects:       result.effects
   - Interactions:  result.interactions
   - Summary:       result.summary

2. Stratified XmR charts:
   - Separate XmR chart per group with group-specific limits
   - Access via result.iter_charts() or result['group_name']

3. Backward compatible:
   - Works like a dict: result['chart'], len(result), 'chart' in result
   - All old code still works!

4. Discoverable:
   - result.has_residuals, result.has_effects, result.has_interactions
   - result.all_charts shows what's available
   - print(result) gives comprehensive summary

🎉 Unified, consistent, pythonic access to all analysis outputs!
    """)


if __name__ == '__main__':
    main()
