"""
Chart Type Selection Demo

This script demonstrates the new SDS-driven chart type selection with:
1. Auto-detection (frictionless default behavior)
2. Explicit selection with IDE auto-completion
3. Validation against SDS capabilities
4. Helpful error messages

Usage:
    python demo_chart_type_selection.py
"""

import pandas as pd
import numpy as np
from processbehavior import ProcessDataFrame
from processbehavior.sds_detector import SamplingDesignDetector


def demo_1_auto_detection():
    """Demo 1: Auto-detection (frictionless - Hadley principle)"""
    print("\n" + "="*70)
    print("DEMO 1: Auto-Detection (Frictionless Experience)")
    print("="*70)
    print("\nUser provides data with grouping variables.")
    print("System auto-detects SDS and selects recommended chart.\n")

    # Create SDS 1 data (complete replication)
    np.random.seed(42)
    operators = []
    machines = []
    times = []
    heights = []

    for op in ['A', 'B']:
        for machine in ['M1', 'M2']:
            for time in [1, 2, 3]:
                for rep in range(3):
                    operators.append(op)
                    machines.append(machine)
                    times.append(time)
                    heights.append(np.random.normal(100, 5))

    df = pd.DataFrame({
        'Height': heights,
        'Operator': operators,
        'Machine': machines,
        'Time': times
    })

    data = ProcessDataFrame(df)

    # No chart_type specified - system auto-selects
    result = data.analyze(
        response_var=data.columns.Height,
        time_var=data.columns.Time,
        grouping_vars=[data.columns.Operator, data.columns.Machine]
    )

    print(f"\nResult: Selected {result.spec.analysis_type} chart automatically")
    print(f"Available charts for future analyses: {data.charts}")


def demo_2_explicit_selection():
    """Demo 2: Explicit chart selection with auto-completion"""
    print("\n" + "="*70)
    print("DEMO 2: Explicit Chart Selection (Power Users)")
    print("="*70)
    print("\nUser explicitly chooses chart type.")
    print("IDE provides auto-completion of valid options.\n")

    # Same data as demo 1
    np.random.seed(42)
    operators = []
    machines = []
    times = []
    heights = []

    for op in ['A', 'B']:
        for machine in ['M1', 'M2']:
            for time in [1, 2, 3]:
                for rep in range(3):
                    operators.append(op)
                    machines.append(machine)
                    times.append(time)
                    heights.append(np.random.normal(100, 5))

    df = pd.DataFrame({
        'Height': heights,
        'Operator': operators,
        'Machine': machines,
        'Time': times
    })

    data = ProcessDataFrame(df)

    # First analyze to detect SDS and populate data.charts
    print("Step 1: Run initial analysis to detect SDS")
    result1 = data.analyze(
        response_var=data.columns.Height,
        time_var=data.columns.Time,
        grouping_vars=[data.columns.Operator, data.columns.Machine]
    )

    # Now data.charts has auto-completion!
    print(f"\nStep 2: Use data.charts for auto-completion")
    print(f"Available: {data.charts}")

    # User can now specify different chart
    print("\nStep 3: Explicitly request S chart instead of Xbar")
    result2 = data.analyze(
        response_var=data.columns.Height,
        time_var=data.columns.Time,
        grouping_vars=[data.columns.Operator, data.columns.Machine],
        chart_type=data.charts.S  # IDE auto-completes: Xbar, S, Imr
    )

    print(f"\nResult: Successfully created {result2.spec.analysis_type} chart")


def demo_3_validation_prevents_errors():
    """Demo 3: Validation prevents invalid chart selections"""
    print("\n" + "="*70)
    print("DEMO 3: Validation Prevents Errors")
    print("="*70)
    print("\nSystem validates chart_type against SDS capabilities.")
    print("Prevents users from requesting charts not supported by data structure.\n")

    # Create SDS 0 data (simple series - only supports IMR)
    np.random.seed(42)
    df = pd.DataFrame({
        'Temperature': np.random.normal(20, 2, 50)
    })

    data = ProcessDataFrame(df)

    print("Data: Simple series with no grouping (SDS 0)")
    print("This structure only supports IMR charts.\n")

    # Try to request Xbar (invalid for SDS 0)
    print("Attempting to request Xbar chart (invalid for SDS 0)...\n")

    try:
        result = data.analyze(
            response_var=data.columns.Temperature,
            chart_type='Xbar'  # This will fail - SDS 0 doesn't support Xbar
        )
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print("✓ System correctly prevented invalid chart selection")
        print(f"\nError message:\n{e}")


def demo_4_sds_driven_recommendations():
    """Demo 4: Different SDS → Different valid charts"""
    print("\n" + "="*70)
    print("DEMO 4: SDS-Driven Recommendations")
    print("="*70)
    print("\nDifferent data structures (SDS) support different chart types.\n")

    # SDS 0: Simple series
    df_sds0 = pd.DataFrame({'Value': np.random.normal(100, 10, 30)})

    # SDS 1: Complete replication
    np.random.seed(42)
    operators = []
    times = []
    values = []
    for op in ['A', 'B']:
        for time in [1, 2, 3]:
            for rep in range(3):
                operators.append(op)
                times.append(time)
                values.append(np.random.normal(100, 10))

    df_sds1 = pd.DataFrame({
        'Value': values,
        'Operator': operators,
        'Time': times
    })

    # Analyze each and show capabilities
    scenarios = [
        ("SDS 0: Simple Series", df_sds0, {}, 0),
        ("SDS 1: Complete Replication", df_sds1, {'time_var': 'Time', 'grouping_vars': ['Operator']}, 1)
    ]

    for name, df, extra_params, expected_sds in scenarios:
        print(f"\n{name}:")
        print("-" * 40)

        data = ProcessDataFrame(df)
        result = data.analyze(response_var='Value', **extra_params)

        plan = SamplingDesignDetector.get_analysis_plan(expected_sds)
        print(f"Valid charts: {plan.valid_charts}")
        print(f"Recommended: {plan.recommended_chart}")
        if plan.invalid_charts:
            print(f"Invalid: {', '.join([c.split('(')[0].strip() for c in plan.invalid_charts])}")


def demo_5_complete_workflow():
    """Demo 5: Complete workflow from data to chart selection"""
    print("\n" + "="*70)
    print("DEMO 5: Complete Workflow")
    print("="*70)
    print("\nTypical user journey with the new chart_type feature.\n")

    # Create data
    np.random.seed(42)
    batches = []
    days = []
    measurements = []

    for batch in ['Morning', 'Afternoon']:
        for day in range(1, 6):
            for rep in range(4):
                batches.append(batch)
                days.append(day)
                measurements.append(np.random.normal(75, 5))

    df = pd.DataFrame({
        'Measurement': measurements,
        'Batch': batches,
        'Day': days
    })

    print("Step 1: Create ProcessDataFrame")
    data = ProcessDataFrame(df)
    print(f"   Created with {len(df)} observations")

    print("\nStep 2: Run initial analysis (auto-detect)")
    result = data.analyze(
        response_var=data.columns.Measurement,
        time_var=data.columns.Day,
        grouping_vars=[data.columns.Batch]
    )
    print(f"   Auto-selected: {result.spec.analysis_type}")
    print(f"   Available charts: {list(data.charts._valid_charts)}")

    print("\nStep 3: Explore other valid charts")
    for chart in data.charts._valid_charts:
        if chart != result.spec.analysis_type:
            print(f"   Can also run: {chart} chart")

    print("\nStep 4: Re-analyze with different chart (if desired)")
    if 'S' in data.charts._valid_charts:
        result2 = data.analyze(
            response_var=data.columns.Measurement,
            time_var=data.columns.Day,
            grouping_vars=[data.columns.Batch],
            chart_type=data.charts.S
        )
        print(f"   Created {result2.spec.analysis_type} chart successfully")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CHART TYPE SELECTION DEMONSTRATION")
    print("SDS-Driven Chart Validation with Auto-Completion")
    print("="*70)

    # Run all demos
    demo_1_auto_detection()
    demo_2_explicit_selection()
    demo_3_validation_prevents_errors()
    demo_4_sds_driven_recommendations()
    demo_5_complete_workflow()

    print("\n" + "="*70)
    print("KEY FEATURES")
    print("="*70)
    print("\n1. ✓ Frictionless by default - auto-selects recommended chart")
    print("2. ✓ Explicit choice available - chart_type parameter")
    print("3. ✓ IDE auto-completion - data.charts accessor")
    print("4. ✓ SDS-driven validation - prevents invalid selections")
    print("5. ✓ Helpful error messages - explains why charts are invalid")
    print("6. ✓ Educational - shows available vs. invalid charts")
    print("\n" + "="*70 + "\n")
