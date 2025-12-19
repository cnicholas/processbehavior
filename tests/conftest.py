"""
Shared pytest fixtures for processbehavior tests.

This module provides:
- Performance testing fixtures (large datasets)
- Common specification fixtures
- Module-scoped fixtures for expensive operations
"""

import numpy as np
import pytest

from processbehavior.datasets import synthetic

# ============================================================================
# Performance Data Fixtures
# ============================================================================


@pytest.fixture(scope='module')
def large_dataset_1m():
    """
    Generate 1M row dataset with 50 columns for performance testing.

    Structure: 100 × 2 factors × 500 times × 10 reps = 1,000,000 rows
    Columns: 4 core + 46 extra = 50 total

    This fixture is module-scoped to avoid regeneration between tests.
    """
    K1, K2, T, n = 100, 2, 500, 10
    df = synthetic.make_sds(1, K1=K1, K2=K2, T=T, n_min=n, n_max=n, seed=42)

    # Add 46 extra columns (noise columns not used in analysis)
    rng = np.random.default_rng(42)
    for i in range(46):
        df[f'extra_col_{i}'] = rng.normal(0, 1, len(df))

    return df


@pytest.fixture(scope='module')
def large_dataset_100k():
    """
    Medium dataset (100K rows) for faster iteration during development.

    Structure: 50 × 2 factors × 100 times × 10 reps = 100,000 rows
    Columns: 4 core + 46 extra = 50 total
    """
    K1, K2, T, n = 50, 2, 100, 10
    df = synthetic.make_sds(1, K1=K1, K2=K2, T=T, n_min=n, n_max=n, seed=42)

    rng = np.random.default_rng(42)
    for i in range(46):
        df[f'extra_col_{i}'] = rng.normal(0, 1, len(df))

    return df


@pytest.fixture(scope='module')
def large_dataset_10k():
    """
    Small-ish dataset (10K rows) for quick performance sanity checks.

    Structure: 10 × 2 factors × 50 times × 10 reps = 10,000 rows
    Columns: 4 core + 46 extra = 50 total
    """
    K1, K2, T, n = 10, 2, 50, 10
    df = synthetic.make_sds(1, K1=K1, K2=K2, T=T, n_min=n, n_max=n, seed=42)

    rng = np.random.default_rng(42)
    for i in range(46):
        df[f'extra_col_{i}'] = rng.normal(0, 1, len(df))

    return df


# ============================================================================
# Specification Fixtures
# ============================================================================


@pytest.fixture
def perf_spec():
    """Standard specification for performance tests (SDS 1 single factor)."""
    return {
        'analysis_type': 'Xbar',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y',
    }


@pytest.fixture
def perf_spec_imr():
    """Specification for IMR analysis performance tests."""
    return {
        'analysis_type': 'Imr',
        'rsg_vars': ['factor 1'],
        'time_var': 'time',
        'response_var': 'y',
    }


# ============================================================================
# Canonical Small Dataset Fixtures
# ============================================================================
# These fixtures provide small, reproducible datasets for common test patterns.
# Use these instead of inline pd.DataFrame({...}) for consistency.
#
# Naming convention:
#   sds{N}_small  - Minimal valid SDS N structure
#   *_data        - Purpose-specific test data


@pytest.fixture
def sds1_small():
    """
    SDS 1: Full replication - 2×2×2 grid with n=2 per cell.

    Structure: 2 factor1 levels × 2 factor2 levels × 2 time points × 2 reps
    Total rows: 16
    Columns: factor 1, factor 2, time, y

    Use for: Xbar/S chart tests, full factorial designs, VAS residual tests
    """
    return synthetic.make_sds(1, K1=2, K2=2, T=2, n_min=2, n_max=2, seed=42)


@pytest.fixture
def sds2_small():
    """
    SDS 2: No replication - 2×2×4 complete grid with n=1 per cell.

    Structure: 2 factor1 levels × 2 factor2 levels × 4 time points × 1 rep
    Total rows: 16
    Columns: factor 1, factor 2, time, y

    Use for: IMR chart tests, unreplicated factorial designs
    """
    return synthetic.make_sds(2, K1=2, K2=2, T=4, seed=42)


@pytest.fixture
def sds3_small():
    """
    SDS 3: Partial replication - mixed n=1 and n>=2 cells.

    Structure: 3 factor1 levels × 2 factor2 levels × 4 time points
    Total rows: ~36 (varies based on p_replicated)
    Columns: factor 1, factor 2, time, y

    Use for: Mixed replication scenarios, realistic production data
    """
    return synthetic.make_sds(3, K1=3, K2=2, T=4, p_replicated=0.5, seed=42)


@pytest.fixture
def sds4_small():
    """
    SDS 4: Single condition over time - simple time series.

    Structure: 1 condition × 20 time points
    Total rows: 20
    Columns: time, y

    Use for: IMR charts, time series analysis, trend detection
    """
    return synthetic.make_sds(4, T=20, seed=42)


@pytest.fixture
def sds4_minimal():
    """
    SDS 4: Minimal time series - 10 points.

    Structure: 1 condition × 10 time points
    Total rows: 10
    Columns: time, y

    Use for: Quick IMR tests, minimal valid data scenarios
    """
    return synthetic.make_sds(4, T=10, seed=42)


@pytest.fixture
def sds5_small():
    """
    SDS 5: Nested/hierarchical design.

    Structure: 2 locations × 3 units per location × 4 time points
    Total rows: ~24 (varies based on p_active)
    Columns: factor 1, factor 2, time, y

    Use for: Nested designs, multi-head/multi-location scenarios
    """
    return synthetic.make_sds(5, L=2, H_per_L=3, T=4, seed=42)


@pytest.fixture
def sds6_small():
    """
    SDS 6: Incomplete/irregular grid.

    Structure: Sparse 3×2 factorial over 8 time points
    Total rows: ~20 (incomplete grid)
    Columns: factor 1, factor 2, time, y

    Use for: Irregular sampling, missing combinations, stratified analysis
    """
    return synthetic.make_sds(6, T=8, K1=3, K2=2, p_sampled=0.5, seed=42)


# ============================================================================
# Purpose-Specific Data Fixtures
# ============================================================================


@pytest.fixture
def single_factor_data():
    """
    Single grouping factor with time - basic grouped IMR scenario.

    Structure: 3 levels × 6 time points × 2 reps = 36 rows
    Columns: factor 1, time, y

    Use for: Single-factor analysis, basic grouping tests
    """
    return synthetic.make_sds(1, K1=3, K2=1, T=6, n_min=2, n_max=2, seed=42)


@pytest.fixture
def two_factor_data():
    """
    Two grouping factors with time - standard factorial design.

    Structure: 3 factor1 × 2 factor2 × 4 time × 2 reps = 48 rows
    Columns: factor 1, factor 2, time, y

    Use for: Two-factor analysis, interaction effects, typical SPC scenarios
    """
    return synthetic.make_sds(1, K1=3, K2=2, T=4, n_min=2, n_max=2, seed=42)


@pytest.fixture
def imr_only_data():
    """
    Response-only data for IMR analysis (no factors, no explicit time).

    Structure: 15 observations
    Columns: y only (time implicit via obs_id after processing)

    Use for: Simplest IMR scenario, response-only formulation tests
    """
    import pandas as pd
    # Use make_sds4 but drop the time column to simulate response-only input
    df = synthetic.make_sds(4, T=15, seed=42)
    return pd.DataFrame({'y': df['y'].values})


# ============================================================================
# Simple Value Fixtures (for basic API tests)
# ============================================================================


@pytest.fixture
def simple_values():
    """
    Simple 5-value DataFrame for basic API tests.

    Use for: Testing Study properties, basic formulate() behavior
    """
    import pandas as pd
    return pd.DataFrame({'Value': [1, 2, 3, 4, 5]})


@pytest.fixture
def simple_values_10():
    """
    Simple 10-value DataFrame for tests needing slightly more data.

    Use for: Tests requiring valid IMR calculation (needs >2 points for mR)
    """
    import pandas as pd
    return pd.DataFrame({'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


@pytest.fixture
def simple_timeseries():
    """
    Simple time series with explicit time column.

    Structure: 30 observations with Time column
    Use for: Time series tests with explicit ordering
    """
    import pandas as pd
    np.random.seed(42)
    return pd.DataFrame({
        'Measurement': np.random.normal(100, 5, 30),
        'Time': range(1, 31)
    })


# ============================================================================
# Grouped Data Fixtures (for Xbar/S and residual tests)
# ============================================================================


@pytest.fixture
def grouped_single_factor():
    """
    Single-factor grouped data with replication.

    Structure: 2 batches × 5 time points × 3 reps = 30 rows (SDS 1)
    Columns: Value, Batch, Time

    Use for: Single-factor Xbar/S tests, Study properties with grouping
    """
    import pandas as pd
    np.random.seed(42)
    data_rows = []
    for batch in ['A', 'B']:
        for time in range(1, 6):
            for _ in range(3):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Batch': batch,
                    'Time': time
                })
    return pd.DataFrame(data_rows)


@pytest.fixture
def grouped_for_residuals():
    """
    Three-factor data for residual chart tests (R2/R3/R4/R5).

    Structure: 3 factors × 5 time points × 2 reps = 30 rows (SDS 1)
    Columns: Value, Factor, Time

    Use for: R4_Xbar, R4_S, R5_Xbar, R5_S, R3_Xbar, R3_S tests
    """
    import pandas as pd
    np.random.seed(42)
    data_rows = []
    for factor in ['A', 'B', 'C']:
        for time in range(1, 6):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })
    return pd.DataFrame(data_rows)


@pytest.fixture
def grouped_four_factors():
    """
    Four-factor data for comparing R4 vs R5 subgroup counts.

    Structure: 4 factors × 6 time points × 2 reps = 48 rows (SDS 1)
    Columns: Value, Factor, Time

    Use for: Tests verifying R4 has time-based subgroups, R5 has factor-based
    """
    import pandas as pd
    np.random.seed(42)
    data_rows = []
    for factor in ['A', 'B', 'C', 'D']:
        for time in range(1, 7):
            for _ in range(2):
                data_rows.append({
                    'Value': np.random.normal(50, 3),
                    'Factor': factor,
                    'Time': time
                })
    return pd.DataFrame(data_rows)
