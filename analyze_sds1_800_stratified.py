"""
SDS 1 Data Analysis - 800 Observations - STRATIFIED
Analyzing fully replicated experimental data with separate charts per stratum
"""

import pandas as pd

from processbehavior import ProcessDataFrame

print("=" * 80)
print("SDS 1 DATA ANALYSIS - 800 OBSERVATIONS - STRATIFIED")
print("Fully Replicated Factorial Design with Stratification")
print("=" * 80)

# Load the data
print("\n1. Loading data...")
df = pd.read_csv('processbehavior/datasets/data/SDS_1_DATA.csv')

# Clean up any trailing empty rows
df = df.dropna(how='all')

print(f"   Loaded {len(df)} observations")
print(f"   Columns: {list(df.columns)}")
print(f"   Time points: {df['TIME'].min()} to {df['TIME'].max()}")
print(f"   FACTOR 1 levels: {sorted(df['FACTOR 1'].unique())}")
print(f"   FACTOR 2 levels: {sorted(df['FACTOR 2'].unique())}")

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
cell_counts = df.groupby(['TIME', 'FACTOR 1', 'FACTOR 2']).size()
print(f"   Number of cells: {len(cell_counts)}")
print(f"   Replicates per cell: {cell_counts.min()} to {cell_counts.max()}")
print(f"   All cells have n≥2: {(cell_counts >= 2).all()}")

# Analyze with STRATIFICATION
print("\n" + "=" * 80)
print("RUNNING STRATIFIED ANALYSIS")
print("=" * 80)

print("\n5. Configuring analysis:")
print("   Response: Y")
print("   Time: TIME")
print("   Rational subgroups: FACTOR 1, FACTOR 2")
print("   Stratify: TRUE (separate charts per FACTOR 1 × FACTOR 2 combination)")

analysis = pdata.analyze(
    response_var='Y',
    time_var='TIME',
    grouping_vars=['FACTOR 1', 'FACTOR 2'],
    stratify=True  # Create separate charts for each factor combination
)

print("\n6. Running calculation...")
result = analysis.calculate()

print("\n✅ Analysis Complete!")
print(f"   Sampling Design State: {result.summary['sds']}")
print(f"   Analysis type: {result.summary.get('analysis_type', 'N/A')}")
print(f"   Is stratified: {result.summary['is_stratified']}")
print(f"   Number of strata: {len(result.list_strata()) if result.summary['is_stratified'] else 'N/A'}")

# Show stratified results
if result.summary['is_stratified']:
    print("\n" + "=" * 80)
    print("STRATIFIED RESULTS")
    print("=" * 80)

    strata = result.list_strata()
    print(f"\n7. Strata detected: {len(strata)}")

    for stratum in strata:
        chart = result.get_stratified_chart(stratum)
        if chart is not None and len(chart) > 0:
            signals = (chart['beyond_limits'] != 0).sum()

            # Get statistics based on what's available
            if 'mean' in chart.columns:
                center_val = chart['mean'].iloc[0]
                ucl = chart['ucl'].iloc[0]
                lcl = chart['lcl'].iloc[0]
                chart_type = 'Xbar'
            elif 's' in chart.columns:
                center_val = chart['sbar'].iloc[0]
                ucl = chart['ucl'].iloc[0]
                lcl = chart.get('lcl', [None]).iloc[0]
                chart_type = 'S'
            else:
                center_val = None
                chart_type = 'Unknown'

            print(f"\n   {stratum}:")
            print(f"      Observations: {len(chart)}")
            if center_val is not None:
                print(f"      Center: {center_val:.3f}")
                print(f"      UCL: {ucl:.3f}", end='')
                if lcl is not None:
                    print(f", LCL: {lcl:.3f}")
                else:
                    print()
            print(f"      Signals (beyond limits): {signals}")

# Export to Excel
print("\n" + "=" * 80)
print("EXPORTING TO EXCEL")
print("=" * 80)

excel_file = 'sds1_800_stratified_results.xlsx'
print(f"\n8. Exporting to {excel_file}...")

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
print("   • Chart tabs: Separate Xbar and S charts for EACH stratum")
print("   • Residuals tab: VAS residual decomposition (R1-R5)")
print("   • Effects tab: Main effects for FACTOR 1 and FACTOR 2")
print("   • Interactions tab: FACTOR 1 × FACTOR 2 interaction effects")
print("   • Dataset tab: Complete dataset with all calculations")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"""
Key Insights:
- Dataset: 800 observations across 50 time points
- SDS 1 detected: Fully replicated factorial design
- Stratified analysis: Separate charts per FACTOR 1 × FACTOR 2 combination
- Each stratum has its own control limits
- Compare between strata to identify factor effects

Stratified vs Combined:
- Stratified: Shows how EACH factor combination behaves over time
- Combined: Shows OVERALL process performance (already done)
- Use stratified when factors create genuinely different processes

Next Steps:
1. Open {excel_file} to review stratified charts
2. Compare control limits across strata
3. Identify which factor combinations are in/out of control
4. Look for patterns: Are all FACTOR 1=3 strata low? All FACTOR 2=2 high?
5. Use this to optimize factor settings
""")

print("=" * 80)
