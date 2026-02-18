"""
Demo: Excel Export Functionality

This example demonstrates how to export analysis results to Excel workbooks
with organized, professional formatting.

The to_excel() method creates multi-sheet workbooks with:
- Summary tab: Analysis metadata and SDS information
- Chart tabs: One tab per chart with data and control limits
- Residuals: VAS residuals (R1-R5) if calculated
- Effects: Main effects if available
- Interactions: Interaction terms if available
- Full Dataset: Complete analysis data (optional)
"""


from processbehavior import ProcessBehavior
from processbehavior import analysis_dataset as ad
from processbehavior.datasets import make_sds1, make_sds2, make_sds3

print("=" * 70)
print("EXCEL EXPORT DEMO")
print("=" * 70)

# =============================================================================
# Example 1: Basic Export (Xbar/S Chart)
# =============================================================================
print("\n" + "=" * 70)
print("Example 1: Basic Xbar/S Chart Export")
print("=" * 70)

df1 = make_sds1(K=3, T=8, n_min=2, n_max=4, seed=42)

spec1 = {
    'analysis_type': 'Xbar',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y'
}

analysis1 = ad.Analysis(df1, spec1)
result1 = analysis1.calculate()

# Export with defaults (all data except full dataset)
result1.to_excel('example1_xbar_chart.xlsx')

print("✓ Exported to: example1_xbar_chart.xlsx")
print(f"  - Tabs created: {result1.summary['n_charts'] + 1}")  # +1 for Summary
print(f"  - SDS: {result1.sds} ({result1.summary['sds_description']})")
print(f"  - Charts: {', '.join(result1.all_charts)}")
print(f"  - Has residuals: {result1.has_residuals}")
print(f"  - Has effects: {result1.has_effects}")

# =============================================================================
# Example 2: Stratified XmR Charts
# =============================================================================
print("\n" + "=" * 70)
print("Example 2: Stratified XmR Charts Export (Killer Feature!)")
print("=" * 70)

df2 = make_sds1(K=4, T=10, n_min=2, n_max=3, seed=42)

spec2 = {
    'analysis_type': 'XmR',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y'
}

analysis2 = ad.Analysis(df2, spec2)
result2 = analysis2.calculate()

# Export stratified charts
result2.to_excel('example2_stratified_xmr.xlsx')

print("✓ Exported to: example2_stratified_xmr.xlsx")
print(f"  - Stratified: {result2.summary['is_stratified']}")
print(f"  - Number of charts: {result2.summary['n_charts']}")
print(f"  - Chart names: {', '.join(result2.all_charts)}")
print("  - Each group gets its own XmR chart with group-specific control limits!")

# =============================================================================
# Example 3: Complete Export with Full Dataset
# =============================================================================
print("\n" + "=" * 70)
print("Example 3: Complete Export (Including Full Dataset)")
print("=" * 70)

df3 = make_sds2(K=3, T=6, seed=42)

spec3 = {
    'analysis_type': 'Xbar',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y'
}

analysis3 = ad.Analysis(df3, spec3)
result3 = analysis3.calculate()

# Export everything including full dataset
result3.to_excel('example3_complete.xlsx', include_full_dataset=True)

print("✓ Exported to: example3_complete.xlsx")
print(f"  - Total observations: {result3.summary['n_observations']}")
print(f"  - SDS: {result3.sds}")
print(f"  - Variance decomposition: {result3.summary['variance_decomposition']}")
print("  - Tabs: Summary, Charts, Residuals, Effects, Interactions, Full_Dataset")

# =============================================================================
# Example 4: Minimal Export (Charts Only)
# =============================================================================
print("\n" + "=" * 70)
print("Example 4: Minimal Export (Charts Only)")
print("=" * 70)

df4 = make_sds1(K=2, T=5, n_min=3, n_max=3, seed=42)

spec4 = {
    'analysis_type': 'S',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y'
}

analysis4 = ad.Analysis(df4, spec4)
result4 = analysis4.calculate()

# Export only charts and summary (minimal config)
result4.to_excel(
    'example4_minimal.xlsx',
    include_residuals=False,
    include_effects=False,
    include_interactions=False,
    include_full_dataset=False
)

print("✓ Exported to: example4_minimal.xlsx")
print("  - Contains summary and chart tabs only")
print("  - Ideal for sharing control charts without detailed analysis")

# =============================================================================
# Example 5: Using ProcessBehavior (Frictionless API)
# =============================================================================
print("\n" + "=" * 70)
print("Example 5: Frictionless API with Auto-Export")
print("=" * 70)

# Create ProcessBehavior
pdf = ProcessBehavior(df1)

# Formulate and execute
study5 = pdf.formulate(
    response=pdf.cols.y,
    time=pdf.cols.time,
    factors=[pdf.cols.factor_1]
)
result5 = study5.execute()

# Export to Excel
result5.to_excel('example5_frictionless.xlsx')

print("✓ Exported to: example5_frictionless.xlsx")
print("  - Used auto-completion: pdf.cols.y, pdf.cols.time, etc.")
print("  - Auto-detected SDS and ran best analysis")
print("  - Exported results with one method call")

# =============================================================================
# Example 6: Custom Options
# =============================================================================
print("\n" + "=" * 70)
print("Example 6: Custom Export Options")
print("=" * 70)

df6 = make_sds3(K=3, T=5, n_per_kt=4, seed=42)

spec6 = {
    'analysis_type': 'Xbar',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y'
}

analysis6 = ad.Analysis(df6, spec6)
result6 = analysis6.calculate()

# Export with custom options
result6.to_excel(
    'example6_custom.xlsx',
    include_summary=True,
    include_charts=True,
    include_residuals=True,
    include_effects=True,
    include_interactions=True,
    include_full_dataset=False,
    format_cells=True  # Bold headers, freeze panes, auto-width columns
)

print("✓ Exported to: example6_custom.xlsx")
print("  - Professional formatting applied")
print("  - Bold headers, frozen top row, auto-sized columns")
print(f"  - Signal count: {result6.summary['n_signals_total']}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
All examples exported successfully!

Files created:
  1. example1_xbar_chart.xlsx      - Basic Xbar/S chart
  2. example2_stratified_xmr.xlsx  - Stratified XmR charts (killer feature!)
  3. example3_complete.xlsx        - Complete export with full dataset
  4. example4_minimal.xlsx         - Charts only
  5. example5_frictionless.xlsx    - Using ProcessBehavior API
  6. example6_custom.xlsx          - Custom formatting options

Each Excel file contains:
  ✓ Summary tab with SDS information and analysis metadata
  ✓ One tab per chart with data and control limits
  ✓ Residuals (R1-R5) if VAS was calculated
  ✓ Effects and interactions if available
  ✓ Professional formatting (bold headers, frozen panes, auto-width)

Usage pattern:
  result = analysis.calculate()
  result.to_excel('output.xlsx')

That's it! Simple, clean, and user-friendly.
""")

print("=" * 70)
print("Demo complete!")
print("=" * 70)
