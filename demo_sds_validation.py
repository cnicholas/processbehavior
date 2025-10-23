"""
SDS Analysis Plan Validation Demo

This script demonstrates the new SDS diagnostic capabilities that enable
Tom Bishop to validate the implementation against his methodology.

Usage:
    python demo_sds_validation.py
"""

import pandas as pd

from processbehavior.sds_detector import SamplingDesignDetector


def demo_individual_plan():
    """Show detailed plan for a specific SDS."""
    print("\n" + "="*70)
    print("DEMO 1: Individual SDS Analysis Plan")
    print("="*70)
    print("\nQuerying SDS 1 (Full Factorial with Complete Replication):")
    print()

    plan = SamplingDesignDetector.get_analysis_plan(sds=1)
    print(plan)

    # Programmatic access to plan attributes
    print("\n" + "="*70)
    print("Programmatic access to plan attributes:")
    print("="*70)
    print(f"Valid charts: {plan.valid_charts}")
    print(f"VAS supported: {plan.vas_residuals_supported}")
    print(f"Available residuals: {plan.residuals_available}")
    print(f"R2 calculation method: {plan.residual_calculation_method}")
    print(f"Main effects: {plan.main_effects_supported}")
    print(f"Interactions: {plan.interaction_effects_supported}")


def demo_capability_matrix():
    """Show comparison matrix of all SDS capabilities."""
    print("\n" + "="*70)
    print("DEMO 2: SDS Capability Comparison Matrix")
    print("="*70)
    print()

    matrix = SamplingDesignDetector.get_capability_matrix()

    # Display with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    print(matrix)

    # Can export for external review
    print("\n" + "="*70)
    print("This matrix can be exported for Tom Bishop's review:")
    print("  matrix.to_excel('sds_validation.xlsx')")
    print("  matrix.to_csv('sds_validation.csv')")
    print("="*70)


def demo_all_plans():
    """Show complete analysis plans for all SDS."""
    print("\n" + "="*70)
    print("DEMO 3: Complete Analysis Plans for All SDS (0-6)")
    print("="*70)
    print("\nThis shows the authoritative 'recipe' for each SDS.")
    print("Tom Bishop can use this to verify line-by-line against his methodology.")
    print()

    # Print first 2 plans as example
    for sds in [0, 1]:
        plan = SamplingDesignDetector.get_analysis_plan(sds)
        print(plan)
        print()
        print()

    print("... (SDS 2-6 also available via SamplingDesignDetector.print_all_analysis_plans())")


def demo_validation_workflow():
    """Show how to use this for validation."""
    print("\n" + "="*70)
    print("DEMO 4: Validation Workflow")
    print("="*70)
    print()

    print("Validation checklist for each SDS:")
    print()

    for sds in range(7):
        plan = SamplingDesignDetector.get_analysis_plan(sds)
        print(f"SDS {sds}: {plan.name}")
        print(f"  ✓ Check: Valid charts = {plan.valid_charts}")
        print(f"  ✓ Check: VAS residuals = {plan.vas_residuals_supported}")
        if plan.vas_residuals_supported:
            print(f"  ✓ Check: R2 method = {plan.residual_calculation_method}")
        print(f"  ✓ Check: Main effects = {plan.main_effects_supported}")
        print(f"  ✓ Check: Interactions = {plan.interaction_effects_supported}")
        print(f"  ✓ Reference: {plan.bishop_reference}")
        print()


def demo_key_differences():
    """Highlight key differences between SDS."""
    print("\n" + "="*70)
    print("DEMO 5: Key Differences Between SDS")
    print("="*70)
    print()

    print("VAS Residual Support:")
    for sds in range(7):
        plan = SamplingDesignDetector.get_analysis_plan(sds)
        status = "✓ Supported" if plan.vas_residuals_supported else "✗ Not supported"
        method = f" ({plan.residual_calculation_method})" if plan.vas_residuals_supported else ""
        print(f"  SDS {sds}: {status}{method}")

    print("\nR2 Calculation Methods:")
    print("  • exact: All cells have n≥2, can calculate true within-cell variance")
    print("  • moving_average: No replication, use moving range approximation")
    print("  • hybrid: Mix of exact (where n≥2) and approximate (where n=1)")
    print("  • none: No VAS support")

    print("\nChart Restrictions:")
    for sds in range(7):
        plan = SamplingDesignDetector.get_analysis_plan(sds)
        if plan.invalid_charts:
            print(f"  SDS {sds}: Cannot use {', '.join(plan.invalid_charts)}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("SDS ANALYSIS PLAN VALIDATION SYSTEM")
    print("Diagnostic tool for Wheeler/Bishop methodology validation")
    print("="*70)

    # Run all demos
    demo_individual_plan()
    demo_capability_matrix()
    demo_all_plans()
    demo_validation_workflow()
    demo_key_differences()

    print("\n" + "="*70)
    print("NEXT STEPS FOR VALIDATION")
    print("="*70)
    print()
    print("1. Review each SDS plan against Wheeler/Bishop source material")
    print("2. Verify VAS residual calculation methods are correct")
    print("3. Confirm chart type restrictions align with methodology")
    print("4. Validate typical use cases and limitations")
    print("5. Export capability matrix for external review:")
    print("   >>> matrix = SamplingDesignDetector.get_capability_matrix()")
    print("   >>> matrix.to_excel('sds_validation.xlsx')")
    print()
    print("="*70)
