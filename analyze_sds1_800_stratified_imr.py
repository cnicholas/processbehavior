"""
SDS 1 Data Analysis - 800 Observations - STRATIFIED IMR
Separate IMR chart for each FACTOR 1 × FACTOR 2 combination
Shows time-series behavior WITHIN each factor combination
"""

import pandas as pd

from processbehavior import ProcessDataFrame

print("=" * 80)
print("SDS 1 DATA ANALYSIS - STRATIFIED IMR")
print("Separate time-series chart per FACTOR 1 × FACTOR 2 combination")
print("=" * 80)

# Load the data
print("\n1. Loading data...")
df = pd.read_csv('processbehavior/datasets/data/SDS_1_DATA.csv')
df = df.dropna(how='all')

print(f"   Loaded {len(df)} observations")
print(f"   FACTOR 1 levels: {sorted(df['FACTOR 1'].unique())}")
print(f"   FACTOR 2 levels: {sorted(df['FACTOR 2'].unique())}")
print(f"   Time points: {df['TIME'].min()} to {df['TIME'].max()}")

# Create ProcessDataFrame
print("\n2. Creating ProcessDataFrame...")
pdata = ProcessDataFrame(df)

# Analyze with STRATIFICATION by factors, NO grouping
print("\n" + "=" * 80)
print("RUNNING STRATIFIED IMR ANALYSIS")
print("=" * 80)

print("\n3. Configuring analysis:")
print("   Response: Y")
print("   Time: TIME")
print("   Grouping vars: FACTOR 1, FACTOR 2 (defines strata)")
print("   Chart type: IMR (forced - individual measurements)")
print("   Stratify: True")
print("   Result: Separate IMR chart per FACTOR 1 × FACTOR 2 combination")

analysis = pdata.analyze(
    response_var='Y',
    time_var='TIME',
    grouping_vars=['FACTOR 1', 'FACTOR 2'],  # Define factor grouping
    chart_type='Imr',                         # Force IMR instead of Xbar
    stratify=True                             # Create separate chart per stratum
)

print("\n4. Running calculation...")
result = analysis.calculate()

print("\n✅ Analysis Complete!")
print(f"   Sampling Design State: {result.summary['sds']}")
print(f"   Is stratified: {result.summary['is_stratified']}")
print(f"   Number of strata: {len(result.list_strata()) if result.summary['is_stratified'] else 'N/A'}")

# Show stratified results summary
if result.summary['is_stratified']:
    print("\n" + "=" * 80)
    print("STRATIFIED IMR CHARTS SUMMARY")
    print("=" * 80)

    strata = result.list_strata()
    print(f"\nCreated {len(strata)} separate IMR charts (one per factor combination)")

    for i, stratum in enumerate(strata, 1):
        chart = result.get_stratified_chart(stratum)
        if chart is not None:
            signals = (chart['beyond_limits'] != 0).sum() if 'beyond_limits' in chart.columns else 0
            print(f"   {i}. {stratum}: {len(chart)} time points, {signals} signals")

# Export to Excel
print("\n" + "=" * 80)
print("EXPORTING TO EXCEL")
print("=" * 80)

excel_file = 'sds1_800_stratified_imr_results.xlsx'
print(f"\n5. Exporting to {excel_file}...")

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
print("\nEach stratum shows:")
print("   • Time-series of individual measurements (not subgroup means)")
print("   • How that specific factor combination behaves over time")
print("   • Which time points have signals for that combination")

print("\n" + "=" * 80)
print("USE CASE")
print("=" * 80)
print("""
This stratified IMR approach answers:
  • Does FACTOR 1=1, FACTOR 2=1 stay stable over time?
  • Which factor combinations drift or show special causes?
  • Are certain combinations more variable than others?
  • What time periods are problematic for each combination?

Compare this to combined analysis which only shows:
  • Overall differences BETWEEN factor combinations
  • Not time-series behavior WITHIN each combination
""")

print("=" * 80)
