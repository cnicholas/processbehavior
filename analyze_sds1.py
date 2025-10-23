"""
SDS 1 Data Analysis
Analyzing fully replicated experimental data with two factors
"""

import pandas as pd

from processbehavior import ProcessDataFrame

print("=" * 80)
print("SDS 1 DATA ANALYSIS")
print("Fully Replicated Factorial Design")
print("=" * 80)

# Load the data
print("\n1. Loading data...")
df = pd.read_csv('processbehavior/datasets/data/SDS_1_DATA.csv')

print(f"   Loaded {len(df)} observations")
print(f"   Columns: {list(df.columns)}")
print(f"   Time points: {df['TIME'].min()} to {df['TIME'].max()}")
print(f"   FACTOR_1 levels: {sorted(df['FACTOR_1'].unique())}")
print(f"   FACTOR_2 levels: {sorted(df['FACTOR_2'].unique())}")

# Create ProcessDataFrame
print("\n2. Creating ProcessDataFrame...")
pdata = ProcessDataFrame(df)

# Show data summary
print("\n3. Data Summary:")
print(f"   Mean Y: {df['Y'].mean():.2f}")
print(f"   Std dev: {df['Y'].std():.2f}")
print(f"   Min: {df['Y'].min():.2f}, Max: {df['Y'].max():.2f}")

# Check cell structure
print("\n4. Verifying replication structure...")
cell_counts = df.groupby(['TIME', 'FACTOR_1', 'FACTOR_2']).size()
print(f"   Number of cells: {len(cell_counts)}")
print(f"   Replicates per cell: {cell_counts.min()} to {cell_counts.max()}")
print(f"   All cells have n≥2: {(cell_counts >= 2).all()}")

# Analyze with rational subgroups defined by FACTOR_1 and FACTOR_2
print("\n" + "=" * 80)
print("RUNNING ANALYSIS")
print("=" * 80)

print("\n5. Configuring analysis:")
print("   Response: Y")
print("   Time: TIME")
print("   Rational subgroups: FACTOR_1, FACTOR_2")

analysis = pdata.analyze(
    response_var='Y',
    time_var='TIME',
    grouping_vars=['FACTOR_1', 'FACTOR_2']
)

print("\n6. Running calculation...")
result = analysis.calculate()

print("\n✅ Analysis Complete!")
print(f"   Sampling Design State: {result.summary['sds']}")
print(f"   Analysis type: {result.summary.get('analysis_type', 'N/A')}")
print(f"   Chart types: {result.all_charts}")

# Show chart details
print("\n" + "=" * 80)
print("CHART RESULTS")
print("=" * 80)

for chart_name in result.all_charts:
    chart = result.get_chart(chart_name)
    if chart is not None and len(chart) > 0:
        signals = (chart['beyond_limits'] != 0).sum()
        print(f"\n   {chart_name} Chart:")
        print(f"      Observations: {len(chart)}")

        # Show statistics based on chart type
        if 'mean' in chart.columns:
            center_val = chart['mean'].iloc[0]
            ucl = chart['ucl'].iloc[0]
            lcl = chart['lcl'].iloc[0]
            print(f"      Center line: {center_val:.2f}")
            print(f"      UCL: {ucl:.2f}, LCL: {lcl:.2f}")
        elif 'sbar' in chart.columns:
            center_val = chart['sbar'].iloc[0]
            ucl = chart['ucl'].iloc[0]
            print(f"      Sbar: {center_val:.2f}")
            print(f"      UCL: {ucl:.2f}")

        print(f"      Signals (beyond limits): {signals}")

# Export to Excel
print("\n" + "=" * 80)
print("EXPORTING TO EXCEL")
print("=" * 80)

excel_file = 'sds1_analysis_results.xlsx'
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
print("   • Chart tabs: Xbar and S charts with control limits")
print("   • Residuals tab: VAS residual decomposition (R1-R5)")
print("   • Effects tab: Main effects for FACTOR_1 and FACTOR_2")
print("   • Interactions tab: FACTOR_1 × FACTOR_2 interaction effects")
print("   • Dataset tab: Complete dataset with all calculations")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
Key Insights:
- SDS 1 detected: Fully replicated factorial design
- All cells have multiple observations (n≥2)
- Appropriate for Xbar & S chart analysis
- Variance decomposition available via VAS residuals
- Main effects and interactions can be analyzed

Next Steps:
1. Open {excel_file} to review charts and statistics
2. Check Xbar chart for process mean stability
3. Check S chart for process variation stability
4. Review VAS residuals (R1-R5) for variance sources
5. Analyze main effects and interactions
""")

print("=" * 80)
