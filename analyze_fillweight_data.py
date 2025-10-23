"""
Full Analysis of Tom Bishop's Fill Weight Dataset

This is Tom's favorite real production dataset - fill weight measurements
from a production line with 4 lanes and 2 phases over 100 time points.

This demonstrates the complete workflow:
1. Load real data
2. Process with ProcessDataFrame
3. Run stratified analysis
4. Export to Excel for review
"""

import pandas as pd

from processbehavior import ProcessDataFrame

print("=" * 80)
print("FILL WEIGHT DATA ANALYSIS")
print("Tom Bishop's Favorite Real Production Dataset")
print("=" * 80)

# Load the data
print("\n1. Loading data...")
df = pd.read_csv('processbehavior/datasets/data/FILLWEIGHTDATA_800.csv')

print(f"   Loaded {len(df)} observations")
print(f"   Columns: {list(df.columns)}")
print(f"   Pulls (time): {df['pull'].min()} to {df['pull'].max()}")
print(f"   Lanes: {sorted(df['lane'].unique())}")
print(f"   Phases: {sorted(df['phase'].unique())}")
print(f"   Missing values: {df['fill_weight'].isna().sum()}")

# Create ProcessDataFrame
print("\n2. Creating ProcessDataFrame...")
pdata = ProcessDataFrame(df)

print(f"   Auto-detected columns: {dir(pdata.columns)}")

# Show data summary
print("\n3. Data Summary:")
print(f"   Mean fill weight: {df['fill_weight'].mean():.2f}")
print(f"   Std dev: {df['fill_weight'].std():.2f}")
print(f"   Min: {df['fill_weight'].min():.2f}, Max: {df['fill_weight'].max():.2f}")

# Analyze with stratification by lane and phase
print("\n" + "=" * 80)
print("RUNNING STRATIFIED ANALYSIS")
print("=" * 80)

print("\n4. Configuring analysis:")
print("   Response: fill_weight")
print("   Time: pull")
print("   Grouping: lane, phase")
print("   Stratify: True (separate charts per lane×phase combination)")

analysis = pdata.analyze(
    response_var=pdata.columns.fill_weight,
    time_var=pdata.columns.pull,
    grouping_vars=[pdata.columns.lane, pdata.columns.phase],
    stratify=True  # Create separate chart for each lane×phase combination
)

print("\n5. Running calculation...")
result = analysis.calculate()

print("\n✅ Analysis Complete!")
print(f"   Sampling Design State: {result.summary['sds']}")
print(f"   Analysis type: {result.summary.get('analysis_type', 'N/A')}")
print(f"   Is stratified: {result.summary['is_stratified']}")
print(f"   Number of charts: {result.summary.get('n_charts', len(result.all_charts))}")

# Show stratified results
print("\n" + "=" * 80)
print("STRATIFIED RESULTS")
print("=" * 80)

print(f"\n6. Strata detected: {len(result.list_strata())}")
for stratum in result.list_strata():
    chart = result.get_stratified_chart(stratum)
    signals = (chart['beyond_limits'] != 0).sum()
    mean_val = chart['mean'].iloc[0]
    ucl_val = chart['ucl'].iloc[0]
    lcl_val = chart['lcl'].iloc[0]

    print(f"\n   {stratum}:")
    print(f"      Observations: {len(chart)}")
    print(f"      Mean: {mean_val:.2f}")
    print(f"      UCL: {ucl_val:.2f}, LCL: {lcl_val:.2f}")
    print(f"      Signals (beyond limits): {signals}")

# Export to Excel
print("\n" + "=" * 80)
print("EXPORTING TO EXCEL")
print("=" * 80)

excel_file = 'fillweight_analysis_results.xlsx'
print(f"\n7. Exporting to {excel_file}...")

result.to_excel(
    excel_file,
    include_summary=True,
    include_charts=True,
    include_residuals=True,
    include_effects=True,
    include_interactions=True,
    include_full_dataset=True,
    format_cells=True
)

print(f"\n✅ Excel file created: {excel_file}")
print("\nExcel file contains:")
print("   • Summary tab: SDS info, analysis configuration, signal counts")
print("   • Chart tab: Combined stratified chart (all lane×phase combinations)")
print("   • Residuals tab: R1-R5 VAS residuals (if available)")
print("   • Effects tab: Main effects analysis (if calculated)")
print("   • Interactions tab: Interaction effects (if calculated)")
print("   • Dataset tab: Complete dataset with all calculations")

# Also run combined (non-stratified) for comparison
print("\n" + "=" * 80)
print("RUNNING COMBINED ANALYSIS (for comparison)")
print("=" * 80)

print("\n8. Running non-stratified analysis...")
analysis_combined = pdata.analyze(
    response_var=pdata.columns.fill_weight,
    time_var=pdata.columns.pull,
    grouping_vars=[pdata.columns.lane, pdata.columns.phase],
    stratify=False  # Combined analysis
)

result_combined = analysis_combined.calculate()

print("\n✅ Combined Analysis Complete!")
all_charts = result_combined.all_charts
for chart_name in all_charts:
    chart = result_combined.get_chart(chart_name)
    if chart is not None and len(chart) > 0:
        signals = (chart['beyond_limits'] != 0).sum()
        print(f"\n   {chart_name} Chart:")
        print(f"      Observations: {len(chart)}")
        # Different charts have different value columns
        if 'mean' in chart.columns:
            center_val = chart['mean'].iloc[0]
            print(f"      Mean: {center_val:.2f}")
        elif 'sbar' in chart.columns:
            center_val = chart['sbar'].iloc[0]
            print(f"      Sbar (average range): {center_val:.2f}")
        print(f"      Signals: {signals}")

# Export combined analysis
excel_file_combined = 'fillweight_analysis_combined.xlsx'
print(f"\n9. Exporting combined analysis to {excel_file_combined}...")
result_combined.to_excel(excel_file_combined, format_cells=True)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
Results exported to Excel files:
1. {excel_file} - Stratified analysis (separate charts per lane×phase)
2. {excel_file_combined} - Combined analysis (all data together)

Key Insights:
- This is Tom Bishop's favorite dataset for demonstrating SPC methodology
- Stratified analysis reveals differences between lane×phase combinations
- Combined analysis shows overall process performance
- Compare the two approaches to see the value of stratification

Next Steps:
1. Open the Excel files to review the charts and statistics
2. Look for patterns in the stratified charts (which lanes/phases are in control?)
3. Review the VAS residuals to understand variance decomposition
4. Check for interaction effects between lane and phase
""")

print("\n" + "=" * 80)
