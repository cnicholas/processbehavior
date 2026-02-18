#!/usr/bin/env python
"""
Demonstration of the ProcessBehavior API

This script shows how the API makes process behavior analysis
frictionless by auto-detecting the Sampling Design State (SDS) and
running the appropriate analysis.

Key features demonstrated:
1. Column name auto-completion via pb.cols
2. SDS-driven automatic analysis selection
3. XmR charts for simple series
4. Xbar/S charts for grouped data
5. Clear explanations of what's running and why
"""

import numpy as np
import pandas as pd

from processbehavior import ProcessBehavior

# Set random seed for reproducibility
np.random.seed(42)


def example1_simple_series():
    """Example 1: Simple time series -> XmR Chart (SDS 4)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Time Series")
    print("="*80)

    # Create simple measurement data
    df = pd.DataFrame({
        'Measurement': np.random.normal(100, 2, 30),
        'Time': range(1, 31)
    })

    print("\nData preview:")
    print(df.head())

    # Wrap in ProcessBehavior
    pb = ProcessBehavior(df)

    # Step 1: Formulate the study
    # Auto-completion for column names - typing `pb.cols.` shows Measurement and Time
    study = pb.formulate(
        response=pb.cols.Measurement,
        time=pb.cols.Time
    )

    # Inspect the study
    print(f"\nDetected SDS: {study.sds}")
    print(f"Valid charts: {study.valid_charts}")
    print(f"Recommended: {study.recommended_chart}")

    # Step 2: Execute the analysis
    result = study.execute()

    print("\nXmR Chart Results:")
    print(result.summary)


def example2_grouped_data():
    """Example 2: Manufacturing data with factors -> Xbar/S Charts"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Grouped Manufacturing Data")
    print("="*80)

    # Create manufacturing data with replication (SDS 1)
    # 2 operators x 3 time points x 5 replicates = 30 observations
    data = []
    for t in range(1, 4):
        for op in ['Alice', 'Bob']:
            for _ in range(5):
                data.append({
                    'Height': np.random.normal(50, 3),
                    'Operator': op,
                    'Time': t
                })
    df = pd.DataFrame(data)

    print("\nData preview:")
    print(df.head(10))

    pb = ProcessBehavior(df)

    # Formulate with factors
    study = pb.formulate(
        response=pb.cols.Height,
        factors=[pb.cols.Operator],
        time=pb.cols.Time
    )

    print(f"\nDetected SDS: {study.sds} ({study.sds_name})")
    print(f"Valid charts: {study.valid_charts}")
    print(f"Recommended: {study.recommended_chart}")

    # Execute analysis
    result = study.execute()

    print("\nXbar Chart Summary:")
    print(result.summary)


def example3_single_factor():
    """Example 3: Single factor with time -> Stratified XmR"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Single Factor Analysis (SDS 2)")
    print("="*80)

    # Create data with single observation per cell (SDS 2)
    df = pd.DataFrame({
        'Strength': np.random.normal(100, 5, 20),
        'Batch': ['A', 'B'] * 10,
        'Sequence': list(range(1, 11)) + list(range(1, 11))
    })

    print("\nData preview:")
    print(df.head(10))

    pb = ProcessBehavior(df)

    study = pb.formulate(
        response=pb.cols.Strength,
        factors=[pb.cols.Batch],
        time=pb.cols.Sequence
    )

    print(f"\nDetected SDS: {study.sds} ({study.sds_name})")
    print(f"Valid charts: {study.valid_charts}")
    print(f"Recommended: {study.recommended_chart}")

    result = study.execute()
    print("\nAnalysis Summary:")
    print(result.summary)


def example4_string_names():
    """Example 4: You can still use strings if you prefer"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Using String Names (backward compatible)")
    print("="*80)

    df = pd.DataFrame({
        'Value': np.random.normal(50, 2, 20),
        'Time': range(1, 21)
    })

    pb = ProcessBehavior(df)

    # Still works with plain strings (no auto-completion though)
    study = pb.formulate(
        response='Value',
        time='Time'
    )

    result = study.execute()
    print("\nResults:")
    print(result.summary)


def example5_chart_type_selection():
    """Example 5: Specifying chart type explicitly"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Explicit Chart Type Selection")
    print("="*80)

    # Create data with replication
    data = []
    for t in range(1, 6):
        for lane in ['L1', 'L2']:
            for _ in range(3):
                data.append({
                    'Weight': np.random.normal(100, 2),
                    'Lane': lane,
                    'Pull': t
                })
    df = pd.DataFrame(data)

    pb = ProcessBehavior(df)

    study = pb.formulate(
        response=pb.cols.Weight,
        factors=[pb.cols.Lane],
        time=pb.cols.Pull
    )

    print(f"\nDetected SDS: {study.sds}")
    print(f"Valid charts: {study.valid_charts}")

    # Use recommended chart
    result1 = study.execute()
    print(f"\nDefault (recommended={study.recommended_chart}):")
    print(result1.summary)

    # Or specify chart explicitly using auto-completion
    result2 = study.execute(chart=study.charts.XmR)
    print(f"\nWith chart=study.charts.XmR:")
    print(result2.summary)


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ProcessBehavior API Demonstration")
    print("Frictionless Process Behavior Analysis")
    print("="*80)

    example1_simple_series()
    example2_grouped_data()
    example3_single_factor()
    example4_string_names()
    example5_chart_type_selection()

    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. No more typos - use pb.cols.ColumnName for auto-completion")
    print("  2. No more wrong analysis types - system detects SDS automatically")
    print("  3. Two-step workflow: formulate() then execute()")
    print("  4. Clear explanations - always tells you what it's doing and why")
    print("  5. Follows your data - analysis adapts to data structure")


if __name__ == '__main__':
    main()
