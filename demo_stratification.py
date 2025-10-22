"""
Stratification Feature Demo

This script demonstrates the powerful stratification capability that enables
drill-down analysis by creating separate control charts for each subgroup.

This is a market-differentiating feature - no other SPC software makes
stratified analysis this easy!

Usage:
    python demo_stratification.py
"""

import pandas as pd
import numpy as np
from processbehavior import ProcessDataFrame


def demo_1_basic_stratification():
    """Demo 1: Basic stratified IMR for drill-down analysis"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Stratification - Drill into Subgroups")
    print("="*70)
    print("\nUse Case: You have data grouped by Operator, but want to see")
    print("separate IMR charts for each Operator to identify differences.\n")

    # Create data with 3 operators, each with different mean
    np.random.seed(42)
    data = []

    for operator in ['Alice', 'Bob', 'Carol']:
        offset = ord(operator[0]) - ord('A')  # Alice=0, Bob=1, Carol=2
        for i in range(25):
            data.append({
                'Operator': operator,
                'Measurement': np.random.normal(100 + offset * 3, 2),
                'Time': i + 1
            })

    df = pd.DataFrame(data)
    pdata = ProcessDataFrame(df)

    print(f"Data: {len(df)} measurements from {df['Operator'].nunique()} operators")
    print(f"Operators: {', '.join(df['Operator'].unique())}\n")

    # Run stratified analysis
    analysis = pdata.analyze(
        response_var=pdata.columns.Measurement,
        time_var=pdata.columns.Time,
        grouping_vars=[pdata.columns.Operator],
        chart_type='Imr',
        stratify=True  # 🎯 KEY FEATURE: Creates separate charts per operator!
    )

    result = analysis.calculate()

    print(f"\n✅ Stratified Analysis Complete!")
    print(f"   Created {result.summary['n_charts']} separate IMR charts")
    print(f"   Strata: {', '.join(result.list_strata())}\n")

    # Show chart summary for each stratum
    print("Chart Summaries:")
    print("-" * 70)
    for stratum in result.list_strata():
        chart = result.get_stratified_chart(stratum)
        signals = (chart['beyond_limits'] != 0).sum()
        print(f"{stratum}:")
        print(f"  • Observations: {len(chart)}")
        print(f"  • Mean: {chart['mean'].iloc[0]:.2f}")
        print(f"  • UCL: {chart['ucl'].iloc[0]:.2f}, LCL: {chart['lcl'].iloc[0]:.2f}")
        print(f"  • Signals: {signals}")
        print()


def demo_2_multi_factor_stratification():
    """Demo 2: Stratify by multiple factors (e.g., Operator × Machine)"""
    print("\n" + "="*70)
    print("DEMO 2: Multi-Factor Stratification")
    print("="*70)
    print("\nUse Case: Stratify by combination of factors")
    print("(e.g., Operator × Machine combinations)\n")

    # Create data with 2 operators × 2 machines
    np.random.seed(42)
    data = []

    for operator in ['Day', 'Night']:
        for machine in ['M1', 'M2']:
            for i in range(15):
                offset = (ord(operator[0]) - ord('D')) * 2 + int(machine[1]) - 1
                data.append({
                    'Shift': operator,
                    'Machine': machine,
                    'Yield': np.random.normal(95 + offset, 3),
                    'Batch': i + 1
                })

    df = pd.DataFrame(data)
    pdata = ProcessDataFrame(df)

    print(f"Data: {len(df)} measurements")
    print(f"Shifts: {', '.join(df['Shift'].unique())}")
    print(f"Machines: {', '.join(df['Machine'].unique())}")
    print(f"Expected strata: {df['Shift'].nunique() * df['Machine'].nunique()}\n")

    # Stratify by BOTH factors
    analysis = pdata.analyze(
        response_var=pdata.columns.Yield,
        time_var=pdata.columns.Batch,
        grouping_vars=[pdata.columns.Shift, pdata.columns.Machine],
        chart_type='Imr',
        stratify=True  # Stratifies by Shift × Machine combination
    )

    result = analysis.calculate()

    print(f"\n✅ Multi-Factor Stratification Complete!")
    print(f"   Created {result.summary['n_charts']} charts (one per Shift×Machine)")
    print(f"\nStrata:")
    for stratum in result.list_strata():
        print(f"  • {stratum}")


def demo_3_selective_stratification():
    """Demo 3: Stratify by specific variable only"""
    print("\n" + "="*70)
    print("DEMO 3: Selective Stratification")
    print("="*70)
    print("\nUse Case: You have multiple grouping variables but only want")
    print("to stratify by one of them.\n")

    # Create data with Operator and Machine
    np.random.seed(42)
    data = []

    for operator in ['A', 'B']:
        for machine in ['M1', 'M2', 'M3']:
            for i in range(10):
                data.append({
                    'Operator': operator,
                    'Machine': machine,
                    'Output': np.random.normal(100, 5),
                    'Hour': i + 1
                })

    df = pd.DataFrame(data)
    pdata = ProcessDataFrame(df)

    print(f"Data: {len(df)} measurements")
    print(f"Grouping by: Operator ({df['Operator'].nunique()}) and Machine ({df['Machine'].nunique()})")
    print("Stratify by: Machine only (creates 3 charts, not 6)\n")

    # Stratify by specific variable
    analysis = pdata.analyze(
        response_var=pdata.columns.Output,
        time_var=pdata.columns.Hour,
        grouping_vars=[pdata.columns.Operator, pdata.columns.Machine],
        chart_type='Imr',
        stratify='Machine'  # Only stratify by Machine, not Operator
    )

    result = analysis.calculate()

    print(f"\n✅ Selective Stratification Complete!")
    print(f"   Created {result.summary['n_charts']} charts (one per Machine)")
    print(f"   Strata: {', '.join(result.list_strata())}")


def demo_4_accessing_stratified_results():
    """Demo 4: Different ways to access stratified results"""
    print("\n" + "="*70)
    print("DEMO 4: Accessing Stratified Results")
    print("="*70)
    print("\nMultiple ways to work with stratified charts:\n")

    # Create simple data
    np.random.seed(42)
    data = []
    for dept in ['QC', 'Production', 'R&D']:
        for i in range(15):
            data.append({
                'Department': dept,
                'Defects': max(0, int(np.random.normal(5 + ord(dept[0]) % 3, 2))),
                'Week': i + 1
            })

    df = pd.DataFrame(data)
    pdata = ProcessDataFrame(df)

    analysis = pdata.analyze(
        response_var=pdata.columns.Defects,
        time_var=pdata.columns.Week,
        grouping_vars=[pdata.columns.Department],
        chart_type='Imr',
        stratify=True
    )

    result = analysis.calculate()

    # Method 1: List all strata
    print("Method 1: List all strata")
    print(f"  Strata: {result.list_strata()}\n")

    # Method 2: Get all stratified charts at once
    print("Method 2: Get all stratified charts")
    strat_charts = result.get_stratified_charts()
    print(f"  Found {len(strat_charts)} stratified charts\n")

    # Method 3: Get specific chart by stratum name
    print("Method 3: Get specific chart by stratum")
    qc_chart = result.get_stratified_chart('QC')
    print(f"  QC Department chart: {len(qc_chart)} observations\n")

    # Method 4: Iterate over all charts
    print("Method 4: Iterate over all stratified charts")
    for stratum in result.list_strata():
        chart = result.get_stratified_chart(stratum)
        print(f"  {stratum}: mean={chart['mean'].iloc[0]:.2f}, n={len(chart)}")


def demo_5_comparison_to_combined():
    """Demo 5: Compare stratified vs combined analysis"""
    print("\n" + "="*70)
    print("DEMO 5: Stratified vs Combined Analysis")
    print("="*70)
    print("\nShowing the difference between combined and stratified approaches.\n")

    # Create data where operators have different means (assignable cause!)
    np.random.seed(42)
    data = []

    for operator in ['Novice', 'Expert']:
        skill_offset = 0 if operator == 'Novice' else 10  # Expert is more accurate
        for i in range(20):
            data.append({
                'Operator': operator,
                'Accuracy': np.random.normal(90 + skill_offset, 3),
                'Trial': i + 1
            })

    df = pd.DataFrame(data)
    pdata = ProcessDataFrame(df)

    # Combined analysis (hides operator effect)
    print("Scenario A: Combined Analysis (ignores operator)")
    analysis_combined = pdata.analyze(
        response_var=pdata.columns.Accuracy,
        time_var=pdata.columns.Trial,
        chart_type='Imr',
        stratify=False
    )
    result_combined = analysis_combined.calculate()

    combined_chart = result_combined.get_chart('A')
    print(f"  Overall mean: {combined_chart['mean'].iloc[0]:.2f}")
    print(f"  Overall UCL: {combined_chart['ucl'].iloc[0]:.2f}")
    print(f"  Signals: {(combined_chart['beyond_limits'] != 0).sum()}")
    print("  Problem: Combines two different processes (Novice + Expert)")
    print()

    # Stratified analysis (reveals operator effect)
    print("Scenario B: Stratified Analysis (separate charts per operator)")
    analysis_strat = pdata.analyze(
        response_var=pdata.columns.Accuracy,
        time_var=pdata.columns.Trial,
        grouping_vars=[pdata.columns.Operator],
        chart_type='Imr',
        stratify=True
    )
    result_strat = analysis_strat.calculate()

    for stratum in result_strat.list_strata():
        chart = result_strat.get_stratified_chart(stratum)
        signals = (chart['beyond_limits'] != 0).sum()
        print(f"  {stratum}:")
        print(f"    Mean: {chart['mean'].iloc[0]:.2f}")
        print(f"    UCL: {chart['ucl'].iloc[0]:.2f}")
        print(f"    Signals: {signals}")

    print("\n  Insight: Each operator is in control within their own capability!")
    print("  The variation is BETWEEN operators, not WITHIN.")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("STRATIFICATION FEATURE DEMONSTRATION")
    print("Market-Differentiating Drill-Down Analysis")
    print("="*70)

    # Run all demos
    demo_1_basic_stratification()
    demo_2_multi_factor_stratification()
    demo_3_selective_stratification()
    demo_4_accessing_stratified_results()
    demo_5_comparison_to_combined()

    print("\n" + "="*70)
    print("WHY THIS IS A KILLER FEATURE")
    print("="*70)
    print("\n✨ Key Benefits:")
    print("   1. Drill-down analysis - see each subgroup separately")
    print("   2. Identify assignable causes - operator/machine/shift effects")
    print("   3. Compare subgroups - who's in control, who's not")
    print("   4. Frictionless API - just add stratify=True")
    print("   5. Flexible - stratify by one or multiple factors")
    print("\n🏆 Market Differentiation:")
    print("   • Minitab: Requires manual filtering and separate analyses")
    print("   • JMP: Can stratify but lacks Python integration")
    print("   • Python libraries: Don't even have the concept")
    print("   • processbehavior: ONE parameter (stratify=True) does it all!")
    print("\n" + "="*70 + "\n")
