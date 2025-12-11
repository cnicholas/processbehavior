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

    Structure: 100 factors × 1000 times × 10 reps = 1,000,000 rows
    Columns: 4 core + 46 extra = 50 total

    This fixture is module-scoped to avoid regeneration between tests.
    """
    K, T, n = 100, 1000, 10
    df = synthetic.make_sds1(K=K, T=T, n_min=n, n_max=n, seed=42)

    # Add 46 extra columns (noise columns not used in analysis)
    rng = np.random.default_rng(42)
    for i in range(46):
        df[f'extra_col_{i}'] = rng.normal(0, 1, len(df))

    return df


@pytest.fixture(scope='module')
def large_dataset_100k():
    """
    Medium dataset (100K rows) for faster iteration during development.

    Structure: 50 factors × 200 times × 10 reps = 100,000 rows
    Columns: 4 core + 46 extra = 50 total
    """
    K, T, n = 50, 200, 10
    df = synthetic.make_sds1(K=K, T=T, n_min=n, n_max=n, seed=42)

    rng = np.random.default_rng(42)
    for i in range(46):
        df[f'extra_col_{i}'] = rng.normal(0, 1, len(df))

    return df


@pytest.fixture(scope='module')
def large_dataset_10k():
    """
    Small-ish dataset (10K rows) for quick performance sanity checks.

    Structure: 10 factors × 100 times × 10 reps = 10,000 rows
    Columns: 4 core + 46 extra = 50 total
    """
    K, T, n = 10, 100, 10
    df = synthetic.make_sds1(K=K, T=T, n_min=n, n_max=n, seed=42)

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
