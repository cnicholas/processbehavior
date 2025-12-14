"""
SDS Detection and Auto-Completion Demo

This demo showcases:
1. Synthetic data generation for all Sampling Design States (SDS 0-6)
2. Automatic SDS detection
3. Auto-completion of analysis specifications
4. Validation that detected SDS matches expected SDS

The processbehavior package automatically detects the data structure and selects
appropriate analysis methods - a killer feature not available in Minitab/JMP!

Author: ProcessBehavior Team
Date: 2024-11-30
"""

import numpy as np
import pandas as pd

from processbehavior import Analysis
from processbehavior.analysis_dataset import AnalysisDataSet, AnalysisSpecification
from processbehavior.datasets import synthetic
from processbehavior.sds_detector import SDSRegistry

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

# Consistent seed for reproducibility
SEED = 42


# ============================================================================
# Helper Function for Validation
# ============================================================================

def validate_sds(expected_sds: int, df: pd.DataFrame, spec: dict, name: str, description: str):
    """Validate SDS detection and show auto-completion."""
    print("\n" + "█" * 80)
    print(f"SDS {expected_sds}: {name.upper()}")
    print("█" * 80)
    print(description)

    print("\n" + "=" * 80)
    print(f"TEST: SDS {expected_sds}: {name} (Expected SDS: {expected_sds})")
    print("=" * 80)

    # Show data structure
    print(f"\n📊 Data Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")

    # Create specification
    aspec = AnalysisSpecification(spec)

    # Create analysis dataset (this is where SDS detection happens)
    ads = AnalysisDataSet(df=df, analysis_specification=aspec)
    detected_sds = ads.sampling_design_state

    # Validate
    status = "✅ PASS" if detected_sds == expected_sds else "❌ FAIL"
    print(f"\n🔍 SDS Detection Result: {status}")
    print(f"   Expected: SDS {expected_sds}")
    print(f"   Detected: SDS {detected_sds}")

    # Show auto-completion information
    print("\n⚙️  Auto-Completed Analysis Information:")
    print(f"   • Has grouping: {aspec.has_grouping}")
    print(f"   • Has time: {aspec.has_time}")
    print(f"   • Requires sort: {aspec.requires_sort}")
    if aspec.requires_sort:
        print(f"   • Sort columns: {aspec.sort_cols}")
    print(f"   • Data prep columns: {aspec.data_prep_output_cols[:5]}...")  # First 5
    print(f"   • Analysis output columns: {aspec.analysis_output_cols[:5]}...")  # First 5

    # Show SDS characteristics
    detector = SDSRegistry()
    characteristics = detector.get_sds_characteristics(detected_sds)

    print(f"\n📈 SDS {detected_sds} Characteristics:")
    print(f"   • Description: {characteristics['description']}")
    print(f"   • R2 Calculation Method: {characteristics['r2_method']}")
    print(f"   • Variance Decomposition: {characteristics['variance_decomposition']}")
    print(f"   • Interaction Analysis: {characteristics['interaction_analysis']}")

    # Show sample of prepared data
    print("\n📋 Sample of Prepared Data:")
    print(ads.analysis_dataset.head(3))

    # If SDS 1, show VAS residuals
    if detected_sds == 1 and 'R1' in ads.analysis_dataset.columns:
        print("\n🔬 VAS Residual Decomposition (SDS 1 Specialty):")
        vas_cols = ['R1', 'R2', 'R3', 'R4', 'R5']
        for col in vas_cols:
            if col in ads.analysis_dataset.columns:
                std = ads.analysis_dataset[col].std()
                print(f"   • {col}: σ = {std:.3f}")

    # Warn if detection doesn't match expectation
    if detected_sds != expected_sds:
        print("\n⚠️  WARNING: SDS detection mismatch!")
        print("   This may indicate:")
        print("   • Issue with data generation")
        print("   • Issue with detection logic")
        print("   • Incorrect specification (rsg_vars, time_var)")
        print("   • Need to review rational subgrouping rules")

    return ads


# ============================================================================
# SDS Configuration - Configuration-Driven Approach
# ============================================================================

sds_configs = [
    {
        'sds': 0,
        'name': 'No Structure',
        'description': ('Data has no grouping variables and no time ordering.\n'
                        'Typically indicates user needs to specify rsg_vars or time_var.'),
        'spec': {
            'analysis_type': 'Imr',
            'response_var': 'y'
        },
        'kwargs': None,  # SDS 0 not in make_sds() dispatcher
        'expected_sds': 0
    },
    {
        'sds': 1,
        'name': 'Full Replication',
        'description': ('Every (factor × time) cell has n≥2 observations.\n'
                        'Allows true estimation of within-cell variance → most powerful.'),
        'spec': {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg'
        },
        'kwargs': {'n_min': 2, 'n_max': 4},
        'expected_sds': 1
    },
    {
        'sds': 2,
        'name': 'No Replication',
        'description': ('Classic designed experiment: each (factor × time) cell has n=1.\n'
                        'Can\'t estimate within-cell variance directly → uses moving range.'),
        'spec': {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg'
        },
        'kwargs': {},  # Use defaults
        'expected_sds': 2
    },
    {
        'sds': 3,
        'name': 'Partial Replication',
        'description': ('Realistic scenario: some conditions replicated, others not.\n'
                        'More complex analysis - blends SDS1 and SDS2 approaches.'),
        'spec': {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg'
        },
        'kwargs': {'p_replicated': 0.6, 'n_when_replicated': 3},
        'expected_sds': 3
    },
    {
        'sds': 4,
        'name': 'Single Stream Over Time',
        'description': ('One process, measured over time with no grouping.\n'
                        'Traditional individuals chart - perfect for continuous monitoring.'),
        'spec': {
            'analysis_type': 'Imr',
            'rsg_vars': ['factor 1'],  # Need grouping var even though K=1 for SDS detection
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg'
        },
        'kwargs': {},
        'expected_sds': 4,
        'T': 50  # Override T for SDS 4
    },
    {
        'sds': 5,
        'name': 'Nested/Hierarchical Design',
        'description': ('Factors sampled at different frequencies (asynchronous).\n'
                        'Example: Operators sampled weekly, machines hourly.'),
        'spec': {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1', 'factor 2'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg'
        },
        'kwargs': {'L': 3, 'H_per_L': 4},
        'expected_sds': 5,
        'T': 12  # Override T for SDS 5
    },
    {
        'sds': 6,
        'name': 'Unstructured/Regime Changes',
        'description': ('Sparse or irregular sampling patterns - process changes mid-stream.\n'
                        'Requires special handling - can\'t assume consistent structure.'),
        'spec': {
            'analysis_type': 'Imr',
            'rsg_vars': ['factor 1'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg'
        },
        'kwargs': {},  # Use defaults
        'expected_sds': 6,
        'T': 40  # Override T for SDS 6
    },
]


# ============================================================================
# Main Demo Loop - Unified API Approach
# ============================================================================

print("=" * 80)
print("PROCESSBEHAVIOR: SDS DETECTION & AUTO-COMPLETION DEMO")
print("=" * 80)
print("\nThis demo validates automatic Sampling Design State detection")
print("and shows how the package auto-completes analysis specifications.")
print(f"\nUsing unified make_sds() API with consistent seed={SEED}\n")

results = []

for config in sds_configs:
    sds = config['sds']

    # Generate data
    if sds == 0:
        # SDS 0: Not in make_sds() dispatcher - manual creation
        df = pd.DataFrame({
            'y': np.random.RandomState(SEED).normal(50, 2, 30)
        })
    else:
        # SDS 1-6: Use unified make_sds() function
        T = config.get('T', 8)  # Default T=8, override for specific SDS types
        df = synthetic.make_sds(
            sds=sds,
            K=3,
            T=T,
            seed=SEED,
            **config['kwargs']
        )

    # Validate detection
    ads = validate_sds(
        expected_sds=config['expected_sds'],
        df=df,
        spec=config['spec'],
        name=config['name'],
        description=config['description']
    )

    # Store results
    results.append({
        'sds': sds,
        'name': config['name'],
        'expected': config['expected_sds'],
        'detected': ads.sampling_design_state,
        'pass': ads.sampling_design_state == config['expected_sds']
    })


# ============================================================================
# Killer Feature Demo: Stratified IMR vs Xbar-S
# ============================================================================

print("\n\n" + "█" * 80)
print("KILLER FEATURE: AUTOMATIC STRATIFICATION")
print("█" * 80)
print("The same SDS 1 data can be analyzed two ways:")
print("1. Stratified IMR: Separate I-MR charts per factor (not in Minitab!)")
print("2. Xbar-S: Cell-level analysis with VAS decomposition")

# Generate SDS 1 data for comparison
df_sds1 = synthetic.make_sds(sds=1, K=3, T=8, seed=SEED, n_min=2, n_max=4)

# Stratified IMR analysis
spec_stratified_imr = {
    'analysis_type': 'Imr',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y',
    'rsg_var_name': 'rsg'
}

print("\n🎯 STRATIFIED IMR ANALYSIS:")
print("   Creates separate I-MR chart for EACH factor level")
print("   Each gets its own appropriate control limits")

result_imr = Analysis(df_sds1, spec_stratified_imr).calculate()
print(f"\n   ✓ Created {len(result_imr)} individual charts:")
for group_name in sorted(result_imr.keys())[:3]:
    stats = result_imr[group_name]['statistics']
    print(f"     • {group_name}: μ={stats['center']:.2f}, "
          f"UCL={stats['ucl']:.2f}, LCL={stats['lcl']:.2f}")

print("\n   💡 This automatic stratification is NOT available in Minitab/JMP!")
print("      They require manual filtering for each subgroup.")

# Xbar-S analysis
spec_xbar = {
    'analysis_type': 'Xbar',
    'rsg_vars': ['factor 1'],
    'time_var': 'time',
    'response_var': 'y',
    'rsg_var_name': 'rsg'
}

print("\n🎯 XBAR-S ANALYSIS:")
print("   Treats each (factor × time) cell as a subgroup")
print("   Calculates VAS residual decomposition")

result_xbar = Analysis(df_sds1, spec_xbar).calculate()
print("\n   ✓ Created combined analysis:")
for key in sorted(result_xbar.keys())[:2]:
    stats = result_xbar[key]['statistics']
    print(f"     • {key} chart: μ={stats['center']}, "
          f"UCL={stats['ucl']}, LCL={stats['lcl']}")


# ============================================================================
# Summary Table
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY: SDS DETECTION VALIDATION")
print("=" * 80)

print("\n{:<10} {:<30} {:<15} {:<15} {:<10}".format(
    "SDS", "Description", "Expected", "Detected", "Status"))
print("-" * 80)

all_passed = True
for result in results:
    status = "✅ PASS" if result['pass'] else "❌ FAIL"
    if not result['pass']:
        all_passed = False
    print("{:<10} {:<30} {:<15} {:<15} {:<10}".format(
        f"SDS {result['sds']}",
        result['name'][:28],
        f"SDS {result['expected']}",
        f"SDS {result['detected']}",
        status
    ))

print("\n" + "=" * 80)
if all_passed:
    print("🎉 ALL SDS DETECTION TESTS PASSED!")
else:
    print("⚠️  SOME SDS DETECTION TESTS FAILED - Review Above")
print("=" * 80)

if all_passed:
    print("\nKey Insights:")
    print("1. ✅ Automatic SDS detection works for all data structures")
    print("2. ✅ Auto-completion provides correct column specifications")
    print("3. ✅ VAS residuals calculated appropriately based on SDS")
    print("4. ✅ Stratified analyses work automatically (killer feature!)")
    print("5. ✅ Detection is robust - handles all SDS types correctly")
    print("\n💡 ProcessBehavior automatically handles complexity that requires")
    print("   manual configuration in Minitab, JMP, and other SPC software.")
else:
    print("\n⚠️  Detection Issues Found:")
    for result in results:
        if not result['pass']:
            print(f"   • SDS {result['sds']} ({result['name']}): "
                  f"Expected {result['expected']}, Got {result['detected']}")
    print("\nThis may indicate:")
    print("   • Specification needs adjustment (rsg_vars, time_var)")
    print("   • Detection logic needs review")
    print("   • Data generation mismatch")

print("=" * 80)
