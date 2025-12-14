#!/usr/bin/env python
"""
Demonstration of the new ProcessBehavior API

This script shows how the new API makes process behavior analysis
frictionless by auto-detecting the Sampling Design State (SDS) and
running the appropriate analysis.

Key features demonstrated:
1. Column name auto-completion
2. SDS-driven automatic analysis selection
3. IMR charts for simple series (like qcc)
4. Xbar/S charts for grouped data
5. Clear explanations of what's running and why
"""

import numpy as np
import pandas as pd

from processbehavior import ProcessBehavior

# Set random seed for reproducibility
np.random.seed(42)


def example1_simple_series():
    """Example 1: Simple time series → IMR Chart (SDS 0)"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Time Series")
    print("="*80)

    # Create simple measurement data
    df = pd.DataFrame({
        'Measurement': np.random.normal(100, 2, 30),
        'Time': pd.date_range('2024-01-01', periods=30, freq='D')
    })

    print("\nData preview:")
    print(df.head())

    # Wrap in ProcessBehavior
    data = ProcessBehavior(df)

    # Auto-completion for column names!
    # In an IDE, typing `data.cols.` will show Measurement and Time
    analysis = data.analyze(
        response_var=data.cols.Measurement,
        time_var=data.cols.Time
    )

    # System prints explanation of SDS detection and chosen analysis
    # Should show: "Detected SDS 0: Running IMR Chart"

    result = analysis.calculate()
    print("\nIMR Chart Results:")
    if isinstance(result, dict):
        print(result)
    else:
        print(result.head(10))


def example2_grouped_data():
    """Example 2: Manufacturing data with operators → Xbar/S Charts"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Grouped Manufacturing Data")
    print("="*80)

    # Create manufacturing data with rational subgroups
    n_obs = 60
    df = pd.DataFrame({
        'Height': np.random.normal(50, 3, n_obs),
        'Operator': np.random.choice(['Alice', 'Bob'], n_obs),
        'Machine': np.random.choice(['M1', 'M2'], n_obs),
        'ProductionTime': range(1, n_obs + 1)
    })

    print("\nData preview:")
    print(df.head(10))

    data = ProcessBehavior(df)

    # Auto-complete works for all columns!
    analysis = data.analyze(
        response_var=data.cols.Height,
        time_var=data.cols.ProductionTime,
        grouping_vars=[data.cols.Operator, data.cols.Machine]
    )

    # System should detect SDS 1, 2, or 3 and run Xbar/S charts
    # Prints explanation of detected SDS and why Xbar/S is appropriate

    result = analysis.calculate()
    print("\nXbar Chart (Subgroup Means):")
    print(result['Xbar']['data'].head())
    print("\nXbar Statistics:")
    print(result['Xbar']['statistics'])

    print("\nS Chart (Subgroup Variation):")
    print(result['Sbar']['data'].head())
    print("\nS Statistics:")
    print(result['Sbar']['statistics'])


def example3_single_factor():
    """Example 3: Single factor with time → Simplified grouping"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Single Factor Analysis")
    print("="*80)

    # Create data with single grouping factor
    df = pd.DataFrame({
        'Strength': np.random.normal(100, 5, 40),
        'Batch': ['A', 'B'] * 20,
        'Sequence': range(1, 41)
    })

    print("\nData preview:")
    print(df.head())

    data = ProcessBehavior(df)

    analysis = data.analyze(
        response_var=data.cols.Strength,
        time_var=data.cols.Sequence,
        grouping_vars=[data.cols.Batch]
    )

    result = analysis.calculate()
    print("\nXbar Results:")
    print(result['Xbar']['data'].head())


def example4_no_autocomplete():
    """Example 4: You can still use strings if you prefer"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Using String Names (backward compatible)")
    print("="*80)

    df = pd.DataFrame({
        'Value': np.random.normal(50, 2, 20),
        'Time': range(1, 21)
    })

    data = ProcessBehavior(df)

    # Still works with plain strings (no auto-completion though)
    analysis = data.analyze(
        response_var='Value',
        time_var='Time'
    )

    result = analysis.calculate()
    print("\nResults:")
    if isinstance(result, dict):
        print(result)
    else:
        print(result.head())


def example5_zero_centering():
    """Example 5: Zero-centered analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Zero-Centered Analysis")
    print("="*80)

    # Data with large offset
    df = pd.DataFrame({
        'Temperature': np.random.normal(1000, 5, 25),
        'Reading': range(1, 26)
    })

    print("\nOriginal data (centered around 1000):")
    print(df.head())

    data = ProcessBehavior(df)

    analysis = data.analyze(
        response_var=data.cols.Temperature,
        time_var=data.cols.Reading,
        zero_center=True  # Subtract mean to focus on variation
    )

    result = analysis.calculate()
    print("\nZero-centered results:")
    if isinstance(result, dict):
        print(result)
    else:
        print(result.head())
        print(f"\nData is now centered at: {result['mean'].iloc[0]:.3f}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("ProcessBehavior API Demonstration")
    print("Frictionless Process Behavior Analysis")
    print("="*80)

    example1_simple_series()
    example2_grouped_data()
    example3_single_factor()
    example4_no_autocomplete()
    example5_zero_centering()

    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. No more typos - use data.cols.ColumnName for auto-completion")
    print("  2. No more wrong analysis types - system detects SDS automatically")
    print("  3. Clear explanations - always tells you what it's doing and why")
    print("  4. Follows your data - analysis adapts to data structure")
    print("\nHappy analyzing! 📊")


if __name__ == '__main__':
    main()
