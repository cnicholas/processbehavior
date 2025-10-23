"""
Quick verification of Excel export contents
"""

import pandas as pd

print("=" * 80)
print("EXCEL FILE VERIFICATION")
print("=" * 80)

excel_file = 'sds1_analysis_results.xlsx'

# Read the Excel file to see what sheets are included
xl_file = pd.ExcelFile(excel_file)

print(f"\n📁 File: {excel_file}")
print(f"\n📑 Sheets included ({len(xl_file.sheet_names)}):")
for i, sheet_name in enumerate(xl_file.sheet_names, 1):
    print(f"   {i}. {sheet_name}")

# Show a preview of key sheets
print("\n" + "=" * 80)
print("SHEET PREVIEWS")
print("=" * 80)

# Summary sheet
if 'Summary' in xl_file.sheet_names:
    print("\n📊 SUMMARY SHEET:")
    summary = pd.read_excel(excel_file, sheet_name='Summary')
    print(f"   Rows: {len(summary)}, Columns: {len(summary.columns)}")
    print("\n   First few rows:")
    print(summary.head(10).to_string(index=False))

# Xbar chart
if 'Xbar' in xl_file.sheet_names:
    print("\n\n📈 XBAR CHART:")
    xbar = pd.read_excel(excel_file, sheet_name='Xbar')
    print(f"   Rows: {len(xbar)}, Columns: {len(xbar.columns)}")
    print(f"   Columns: {list(xbar.columns)}")
    print("\n   Chart data:")
    print(xbar.to_string(index=False))

# Sbar chart
if 'Sbar' in xl_file.sheet_names:
    print("\n\n📉 SBAR CHART:")
    sbar = pd.read_excel(excel_file, sheet_name='Sbar')
    print(f"   Rows: {len(sbar)}, Columns: {len(sbar.columns)}")
    print(f"   Columns: {list(sbar.columns)}")
    print("\n   Chart data:")
    print(sbar.to_string(index=False))

# Residuals
if 'Residuals' in xl_file.sheet_names:
    print("\n\n🔬 RESIDUALS (VAS Decomposition):")
    residuals = pd.read_excel(excel_file, sheet_name='Residuals')
    print(f"   Rows: {len(residuals)}, Columns: {len(residuals.columns)}")
    print(f"   Columns: {list(residuals.columns)}")
    print("\n   First 10 rows:")
    print(residuals.head(10).to_string(index=False))

# Effects
if 'Effects' in xl_file.sheet_names:
    print("\n\n📊 MAIN EFFECTS:")
    effects = pd.read_excel(excel_file, sheet_name='Effects')
    print(f"   Rows: {len(effects)}, Columns: {len(effects.columns)}")
    print(f"   Columns: {list(effects.columns)}")
    print("\n   Effects summary:")
    print(effects.to_string(index=False))

# Interactions
if 'Interactions' in xl_file.sheet_names:
    print("\n\n🔄 INTERACTIONS:")
    interactions = pd.read_excel(excel_file, sheet_name='Interactions')
    print(f"   Rows: {len(interactions)}, Columns: {len(interactions.columns)}")
    print(f"   Columns: {list(interactions.columns)}")
    print("\n   Interaction effects:")
    print(interactions.to_string(index=False))

# Full dataset
if 'Full Dataset' in xl_file.sheet_names:
    print("\n\n📋 FULL DATASET:")
    dataset = pd.read_excel(excel_file, sheet_name='Full Dataset')
    print(f"   Rows: {len(dataset)}, Columns: {len(dataset.columns)}")
    print(f"   Columns: {list(dataset.columns)}")
    print("\n   First 10 rows:")
    print(dataset.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("✅ VERIFICATION COMPLETE")
print("=" * 80)
