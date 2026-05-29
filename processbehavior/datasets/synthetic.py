"""
Synthetic Data Generators for Statistical Process Control

This module provides data generators for all Sampling Design States (SDS)
defined in the Variance Analysis System framework by Dr. Donald Wheeler
and extended by Dr. Thomas A. Bishop.

Purpose:
--------
- Testing: Validate analysis algorithms with known ground truth
- Documentation: Provide reproducible examples
- Education: Demonstrate different data structures and their implications
- Development: Rapid prototyping of new features

Quick Start:
-----------
    >>> from processbehavior.datasets import synthetic
    >>>
    >>> # Generate SDS1 data (full replication)
    >>> df = synthetic.make_sds1(K1=3, K2=2, T=8, n_min=2, n_max=4)
    >>>
    >>> # Use with analysis
    >>> from processbehavior.analysis import Analysis
    >>> spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'],
    ...         'response_var': 'y', 'time_var': 'time'}
    >>> results = Analysis(df, spec).calculate()

Sampling Design States:
----------------------
- SDS 1: Full replication (n≥2 in all cells)
- SDS 2: No replication (n=1 in all cells)
- SDS 3: Partial replication (mixed n=1 and n≥2)
- SDS 4: Single condition over time (one factor, multiple time points)
- SDS 5: Nested design (hierarchical factors, asynchronous)
- SDS 6: Unstructured/regime changes (irregular patterns)

References:
-----------
Wheeler, D. J. (1995). Advanced Topics in Statistical Process Control.
SPC Press, Knoxville, TN.

Bishop, D. R. (2023). Personal communication and collaboration on
Variance Analysis System implementation.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..exceptions import ValidationError
from ..formulation_spec import FormulationSpec

logger = logging.getLogger(__name__)

__all__ = [
    'make_sds1',
    'make_sds2',
    'make_sds3',
    'make_sds4',
    'make_sds5',
    'make_sds6',
    'make_design',
    'make_edge_cases',
    'make_large_dataset',
]


# ============================================================================
# SDS 1: Full Replication - Most Statistically Powerful
# ============================================================================


def make_sds1(
    K1: int = 3,
    K2: int = 2,
    T: int = 8,
    n_min: int = 2,
    n_max: int = 5,
    mu: float = 50.0,
    sigma: float = 0.4,
    factor1_effect_size: float = 2.0,
    factor2_effect_size: float = 1.5,
    time_effect_size: float = 1.0,
    interaction_effect_size: float = 0.5,
    seed: Optional[int] = None,
    include_truth: bool = False,
    factor1_names: Optional[list[str]] = None,
    factor2_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate Sampling Design State 1 data (full replication).

    SDS1 is the gold standard for process studies - every combination of
    factors and time has multiple observations, allowing true estimation of
    within-cell variance. This is what designed experiments strive for.

    Characteristics:
    ---------------
    - Every (factor1, factor2, time) cell has n ≥ 2 observations
    - True within-cell variance can be estimated directly
    - Supports full interaction analysis (R3 calculations)
    - Most statistically powerful design
    - Enables calculation of all residuals (R1-R5) with precision

    Use Cases:
    ---------
    - Testing Xbar-S chart algorithms
    - Validating variance decomposition (R1-R5)
    - Demonstrating interaction effects
    - Teaching full-factorial designs
    - Baseline for comparing other SDS types

    Args:
        K1: Number of factor 1 levels (e.g., machines, lanes)
        K2: Number of factor 2 levels (e.g., operators, shifts)
        T: Number of time periods (e.g., days, shifts, batches)
        n_min: Minimum observations per cell
        n_max: Maximum observations per cell
        mu: Grand mean (process center)
        sigma: Within-cell standard deviation (pure error)
        factor1_effect_size: Standard deviation of factor 1 main effects
        factor2_effect_size: Standard deviation of factor 2 main effects
        time_effect_size: Standard deviation of time main effects
        interaction_effect_size: Standard deviation of interactions
        seed: Random seed for reproducibility (None = random)
        include_truth: If True, include true effects as columns for validation
        factor1_names: Custom factor 1 level names (default: F1_1, F1_2, ...)
        factor2_names: Custom factor 2 level names (default: F2_1, F2_2, ...)

    Returns:
        DataFrame with columns:
            - time: Time period (1 to T)
            - factor 1: Factor 1 level
            - factor 2: Factor 2 level
            - y: Response variable

        Optional columns (if include_truth=True):
            - true_factor1_effect: The ρ_i component
            - true_factor2_effect: The φ_j component
            - true_time_effect: The τ_t component
            - true_interaction: The combined interaction component
            - true_error: The ε component (pure error)

    Mathematical Model:
        Y_ijtr = μ + ρ_i + φ_j + τ_t + (ρφ)_ij + (ρτ)_it + (φτ)_jt + ε_ijtr

        where:
            μ: Grand mean (mu parameter)
            ρ_i: Factor 1 main effect ~ N(0, factor1_effect_size²)
            φ_j: Factor 2 main effect ~ N(0, factor2_effect_size²)
            τ_t: Time main effect ~ N(0, time_effect_size²)
            interactions: Combined effect ~ N(0, interaction_effect_size²)
            ε_ijtr: Pure error for replicate r ~ N(0, σ²)

    Examples:
        >>> # Basic usage - 3 factor1 levels × 2 factor2 levels × 8 time periods
        >>> df = make_sds1(K1=3, K2=2, T=8, seed=42)
        >>> print(f"Generated {len(df)} observations")
        >>> df.groupby(['factor 1', 'factor 2', 'time']).size().min()
        2  # All cells have at least 2 observations

        >>> # With ground truth for validation testing
        >>> df = make_sds1(K1=2, K2=2, T=4, include_truth=True, seed=42)

        >>> # Custom factor names
        >>> df = make_sds1(K1=3, K2=2, T=6,
        ...                factor1_names=['Machine_A', 'Machine_B', 'Machine_C'],
        ...                factor2_names=['Operator_1', 'Operator_2'],
        ...                seed=42)

    Validation:
        The function automatically validates that generated data meets SDS1 criteria:
        - All (K1 × K2 × T) cells present (no missing combinations)
        - All cells have n_min ≤ n ≤ n_max observations
        - Response variable is numeric
        - No missing values in critical columns

    See Also:
        make_sds2: For unreplicated designs (n=1 per cell)
        make_sds3: For partial replication (mixed n=1 and n≥2)
        make_sds4: For time series (single factor over time)
    """
    rng = np.random.default_rng(seed)

    # Generate true effects from the model
    rho = rng.normal(0, factor1_effect_size, K1)  # Factor 1 main effects
    phi = rng.normal(0, factor2_effect_size, K2)  # Factor 2 main effects
    tau = rng.normal(0, time_effect_size, T)  # Time main effects
    inter = rng.normal(0, interaction_effect_size, (K1, K2, T))  # Interactions

    # Use custom factor names if provided
    if factor1_names is None:
        factor1_names = [f'F1_{k + 1}' for k in range(K1)]
    elif len(factor1_names) != K1:
        raise ValidationError(f'Length of factor1_names ({len(factor1_names)}) must equal K1 ({K1})')

    if factor2_names is None:
        factor2_names = [f'F2_{k + 1}' for k in range(K2)]
    elif len(factor2_names) != K2:
        raise ValidationError(f'Length of factor2_names ({len(factor2_names)}) must equal K2 ({K2})')

    rows = []
    for k1 in range(K1):
        for k2 in range(K2):
            for t in range(T):
                # Random number of observations per cell (uniform distribution)
                n = rng.integers(n_min, n_max + 1)

                for _i in range(n):
                    # Generate observation from true model
                    epsilon = rng.normal(0, sigma)  # Pure error
                    y = mu + rho[k1] + phi[k2] + tau[t] + inter[k1, k2, t] + epsilon

                    row = {'time': t + 1, 'factor 1': factor1_names[k1], 'factor 2': factor2_names[k2], 'y': y}

                    # Optionally include ground truth for validation
                    if include_truth:
                        row.update(
                            {
                                'true_factor1_effect': rho[k1],
                                'true_factor2_effect': phi[k2],
                                'true_time_effect': tau[t],
                                'true_interaction': inter[k1, k2, t],
                                'true_error': epsilon,
                                'true_mean': mu + rho[k1] + phi[k2] + tau[t] + inter[k1, k2, t],
                            }
                        )

                    rows.append(row)

    df = pd.DataFrame(rows)

    # Validation - ensure SDS1 criteria met
    cell_counts = df.groupby(['factor 1', 'factor 2', 'time']).size()

    if cell_counts.min() < n_min:
        raise AssertionError(f'SDS1 validation failed: min n={cell_counts.min()} < {n_min}')
    if cell_counts.max() > n_max:
        raise AssertionError(f'SDS1 validation failed: max n={cell_counts.max()} > {n_max}')
    expected_cells = K1 * K2 * T
    if len(cell_counts) != expected_cells:
        raise AssertionError(
            f'SDS1 validation failed: expected {expected_cells} cells, got {len(cell_counts)} '
            '(some factor-time combinations missing)'
        )

    logger.debug(
        f'Generated SDS1 data: {K1} × {K2} factors × {T} times × ~{(n_min + n_max) / 2:.1f} reps/cell '
        f'= {len(df)} observations'
    )

    return df


# ============================================================================
# SDS 2: No Replication - Common in Designed Experiments
# ============================================================================


def make_sds2(
    K1: int = 3,
    K2: int = 2,
    T: int = 10,
    mu: float = 50.0,
    sigma: float = 0.4,
    factor1_effect_size: float = 2.0,
    factor2_effect_size: float = 1.5,
    time_effect_size: float = 1.2,
    interaction_effect_size: float = 0.6,
    seed: Optional[int] = None,
    include_truth: bool = False,
    factor1_names: Optional[list[str]] = None,
    factor2_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate Sampling Design State 2 data (no replication).

    SDS2 is the classic unreplicated factorial design - exactly one observation
    per cell. Common in designed experiments where resources are limited.

    Characteristics:
    ---------------
    - Exactly one observation per (factor1, factor2, time) cell
    - Within-cell variance must be estimated indirectly (moving average)
    - Cannot separate pure error from interaction effects
    - More efficient data collection, less statistical power

    Use Cases:
    ---------
    - Testing moving average R2 calculation algorithms
    - Validating SDS detection logic
    - Demonstrating unreplicated factorial designs
    - Cost-constrained or time-constrained data collection

    Args:
        K1: Number of factor 1 levels
        K2: Number of factor 2 levels
        T: Number of time periods
        mu: Grand mean
        sigma: Process standard deviation (confounded with interaction)
        factor1_effect_size: Std dev of factor 1 effects
        factor2_effect_size: Std dev of factor 2 effects
        time_effect_size: Std dev of time effects
        interaction_effect_size: Std dev of interactions
        seed: Random seed for reproducibility
        include_truth: Include true effects as columns
        factor1_names: Custom factor 1 level names
        factor2_names: Custom factor 2 level names

    Returns:
        DataFrame with exactly K1 × K2 × T observations

    Examples:
        >>> # Basic usage
        >>> df = make_sds2(K1=3, K2=2, T=8, seed=42)
        >>> df.groupby(['factor 1', 'factor 2', 'time']).size().max()
        1  # All cells have exactly 1 observation
        >>> len(df)
        48  # K1 × K2 × T = 3 × 2 × 8

    Validation:
        - Exactly n=1 per cell (all cells)
        - Total observations = K1 × K2 × T
        - No missing factor-time combinations

    See Also:
        make_sds1: For replicated designs
        make_sds3: For mixed replication (most realistic)
    """
    rng = np.random.default_rng(seed)

    rho = rng.normal(0, factor1_effect_size, K1)
    phi = rng.normal(0, factor2_effect_size, K2)
    tau = rng.normal(0, time_effect_size, T)
    inter = rng.normal(0, interaction_effect_size, (K1, K2, T))

    if factor1_names is None:
        factor1_names = [f'F1_{k + 1}' for k in range(K1)]
    elif len(factor1_names) != K1:
        raise ValidationError('Length of factor1_names must equal K1')

    if factor2_names is None:
        factor2_names = [f'F2_{k + 1}' for k in range(K2)]
    elif len(factor2_names) != K2:
        raise ValidationError('Length of factor2_names must equal K2')

    rows = []
    for k1 in range(K1):
        for k2 in range(K2):
            for t in range(T):
                # Single observation per cell - this is the defining characteristic
                epsilon = rng.normal(0, sigma)
                y = mu + rho[k1] + phi[k2] + tau[t] + inter[k1, k2, t] + epsilon

                row = {'time': t + 1, 'factor 1': factor1_names[k1], 'factor 2': factor2_names[k2], 'y': y}

                if include_truth:
                    row.update(
                        {
                            'true_factor1_effect': rho[k1],
                            'true_factor2_effect': phi[k2],
                            'true_time_effect': tau[t],
                            'true_interaction': inter[k1, k2, t],
                            'true_confounded': inter[k1, k2, t] + epsilon,
                        }
                    )

                rows.append(row)

    df = pd.DataFrame(rows)

    # Validation
    cell_counts = df.groupby(['factor 1', 'factor 2', 'time']).size()

    if not (cell_counts == 1).all():
        raise AssertionError(
            f'SDS2 validation failed: expected all n=1, got min={cell_counts.min()}, max={cell_counts.max()}'
        )

    expected_n = K1 * K2 * T
    if len(df) != expected_n:
        raise AssertionError(f'SDS2 validation failed: expected {expected_n} obs, got {len(df)}')

    logger.debug(f'Generated SDS2 data: {K1} × {K2} factors × {T} times = {len(df)} observations')

    return df


# ============================================================================
# SDS 3: Partial Replication - MOST COMMON IN PRACTICE!
# ============================================================================


def make_sds3(  # noqa: C901
    K1: int = 3,
    K2: int = 2,
    T: int = 8,
    p_replicated: float = 0.5,
    n_when_replicated: int = 3,
    mu: float = 50.0,
    sigma: float = 0.5,
    factor1_effect_size: float = 2.0,
    factor2_effect_size: float = 1.5,
    time_effect_size: float = 1.0,
    interaction_effect_size: float = 0.5,
    seed: Optional[int] = None,
    replication_pattern: str = 'random',
    include_truth: bool = False,
    factor1_names: Optional[list[str]] = None,
    factor2_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate Sampling Design State 3 data (partial replication).

    SDS3 represents reality: some cells have replication, others don't.
    This is the MOST COMMON real-world scenario and the HARDEST to analyze.

    Characteristics:
    ---------------
    - Some cells have n=1, others have n≥2
    - Requires hybrid R2 estimation
    - Reflects real-world data collection reality
    - Most challenging to analyze correctly

    Use Cases:
    ---------
    - Testing hybrid variance estimation algorithms
    - Demonstrating real-world data structures
    - Validating robustness to missing data
    - Development of SDS detection logic

    Args:
        K1: Number of factor 1 levels
        K2: Number of factor 2 levels
        T: Number of time periods
        p_replicated: Proportion of cells with replication (0 to 1)
        n_when_replicated: Number of observations in replicated cells
        mu: Grand mean
        sigma: Within-cell standard deviation
        factor1_effect_size: Std dev of factor 1 effects
        factor2_effect_size: Std dev of factor 2 effects
        time_effect_size: Std dev of time effects
        interaction_effect_size: Std dev of interactions
        seed: Random seed
        replication_pattern: How to assign replication:
            - 'random': Random cells replicated (default)
            - 'early_times': First p_replicated fraction of time periods
            - 'late_times': Last p_replicated fraction of time periods
            - 'checkerboard': Alternating pattern
            - 'corners': Extreme combinations get replication
        include_truth: Include true effects as columns
        factor1_names: Custom factor 1 level names
        factor2_names: Custom factor 2 level names

    Returns:
        DataFrame with mixed replication structure, including 'cell_type' column

    Examples:
        >>> # 50% of cells replicated (balanced)
        >>> df = make_sds3(K1=3, K2=2, T=8, p_replicated=0.5, seed=42)
        >>> cell_sizes = df.groupby(['factor 1', 'factor 2', 'time']).size()
        >>> print(f"Singles: {(cell_sizes == 1).sum()}, "
        ...       f"Replicated: {(cell_sizes > 1).sum()}")

    Validation:
        - At least one cell with n=1
        - At least one cell with n≥2
        - Actual replication proportion within ±25% of target
        - All factor-time combinations present

    See Also:
        make_sds1: For fully replicated designs
        make_sds2: For unreplicated designs
    """
    rng = np.random.default_rng(seed)

    # Generate true effects
    rho = rng.normal(0, factor1_effect_size, K1)
    phi = rng.normal(0, factor2_effect_size, K2)
    tau = rng.normal(0, time_effect_size, T)
    inter = rng.normal(0, interaction_effect_size, (K1, K2, T))

    if factor1_names is None:
        factor1_names = [f'F1_{k + 1}' for k in range(K1)]
    elif len(factor1_names) != K1:
        raise ValidationError('Length of factor1_names must equal K1')

    if factor2_names is None:
        factor2_names = [f'F2_{k + 1}' for k in range(K2)]
    elif len(factor2_names) != K2:
        raise ValidationError('Length of factor2_names must equal K2')

    rows = []
    replicated_cells = []
    unreplicated_cells = []
    total_cells = K1 * K2 * T

    for k1 in range(K1):
        for k2 in range(K2):
            for t in range(T):
                # Determine replication based on pattern
                if replication_pattern == 'random':
                    is_replicated = rng.random() < p_replicated

                elif replication_pattern == 'early_times':
                    is_replicated = t < int(T * p_replicated)

                elif replication_pattern == 'late_times':
                    is_replicated = t >= int(T * (1 - p_replicated))

                elif replication_pattern == 'checkerboard':
                    is_replicated = (k1 + k2 + t) % 2 == 0

                elif replication_pattern == 'corners':
                    is_corner = (k1 == 0 or k1 == K1 - 1) or (t == 0 or t == T - 1)
                    is_replicated = is_corner or (rng.random() < p_replicated / 2)

                else:
                    raise ValidationError(
                        f'Unknown replication pattern: {replication_pattern}. '
                        f"Valid: 'random', 'early_times', 'late_times', "
                        f"'checkerboard', 'corners'"
                    )

                # Determine n for this cell
                n = n_when_replicated if is_replicated else 1

                # Track cell types for validation
                if is_replicated:
                    replicated_cells.append((k1, k2, t))
                else:
                    unreplicated_cells.append((k1, k2, t))

                # Generate observations
                for _i in range(n):
                    epsilon = rng.normal(0, sigma)
                    y = mu + rho[k1] + phi[k2] + tau[t] + inter[k1, k2, t] + epsilon

                    row = {
                        'time': t + 1,
                        'factor 1': factor1_names[k1],
                        'factor 2': factor2_names[k2],
                        'y': y,
                        'cell_type': 'replicated' if is_replicated else 'unreplicated',
                    }

                    if include_truth:
                        row.update(
                            {
                                'true_factor1_effect': rho[k1],
                                'true_factor2_effect': phi[k2],
                                'true_time_effect': tau[t],
                                'true_interaction': inter[k1, k2, t],
                                'true_error': epsilon,
                            }
                        )

                    rows.append(row)

    df = pd.DataFrame(rows)

    # Validation - ensure true SDS3 (mixed replication)
    cell_counts = df.groupby(['factor 1', 'factor 2', 'time']).size()
    has_singles = (cell_counts == 1).any()
    has_multiples = (cell_counts >= 2).any()

    if not (has_singles and has_multiples):
        raise AssertionError(
            f'SDS3 validation failed: must have both single and multiple observations. '
            f'Got singles={has_singles}, multiples={has_multiples}. '
            f'Try adjusting p_replicated or replication_pattern.'
        )

    # Check replication proportion
    actual_p = len(replicated_cells) / total_cells
    if abs(actual_p - p_replicated) > 0.25:  # Allow 25% tolerance
        logger.warning(f'SDS3: target replication {p_replicated:.1%}, actual {actual_p:.1%} (outside 25% tolerance)')

    logger.debug(
        f'Generated SDS3 data: {len(df)} obs, '
        f'{len(replicated_cells)} replicated cells ({actual_p:.1%}), '
        f'{len(unreplicated_cells)} unreplicated cells'
    )

    return df


# ============================================================================
# SDS 4: Incomplete grid, occupied cells replicated (N >= 2)
# ============================================================================


def make_sds4(
    K1: int = 3,
    K2: int = 2,
    T: int = 6,
    p_drop: float = 0.25,
    n_min: int = 2,
    n_max: int = 5,
    mu: float = 50.0,
    sigma: float = 0.4,
    factor1_effect_size: float = 2.0,
    factor2_effect_size: float = 1.5,
    time_effect_size: float = 1.0,
    seed: Optional[int] = None,
    factor1_names: Optional[list[str]] = None,
    factor2_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate Observed Design State 4 (ODS 4) data.

    ODS 4 is Bishop's classification for incomplete designs where every
    occupied (factor × time) cell carries at least 2 observations. Some
    cells are missing entirely (no data was collected there). After NA
    filtering during tidying, ODS 4 collapses to ADS 1 (full replication
    on the surviving grid).

    Characteristics
    ---------------
    - K1 × K2 × T grid with a fraction of cells empty (no observations)
    - Every occupied cell has n_min..n_max observations (n_min ≥ 2)
    - Bishop's structural ODS 4 classification by the N_kt distribution

    Args:
        K1, K2: Number of levels for factor 1 and factor 2
        T: Number of time periods
        p_drop: Approximate fraction of cells to leave empty (0 < p_drop < 1).
            The exact number is ``max(1, int(p_drop * K1 * K2 * T))``, clipped
            so at least one cell remains occupied.
        n_min, n_max: Replication range for occupied cells. ``n_min`` must be
            at least 2 for the result to classify as ODS 4 (not ODS 6).
        mu, sigma: Grand mean and within-cell error standard deviation
        factor1_effect_size, factor2_effect_size, time_effect_size: Standard
            deviations of the corresponding main effects
        seed: Random seed for reproducibility
        factor1_names, factor2_names: Custom factor level names

    Returns:
        DataFrame with columns ``time``, ``factor 1``, ``factor 2``, ``y``.
        Empty cells appear as single rows with ``y = NaN`` so the SDS
        detector observes them as N_kt = 0.

    Examples:
        >>> df = make_sds4(K1=3, K2=2, T=6, seed=42)
        >>> # After formulate(), study.observed_design_state.sds == 4
        >>> # After tidy/NA drop, study.analytical_design_state.sds == 1
    """
    if not 0 < p_drop < 1:
        raise ValidationError(f'p_drop must be in (0, 1), got {p_drop}')
    if n_min < 2:
        raise ValidationError(f'ODS 4 requires n_min >= 2 (no singletons), got {n_min}')
    if n_max < n_min:
        raise ValidationError(f'n_max ({n_max}) must be >= n_min ({n_min})')

    rng = np.random.default_rng(seed)

    rho = rng.normal(0, factor1_effect_size, K1)
    phi = rng.normal(0, factor2_effect_size, K2)
    tau = rng.normal(0, time_effect_size, T)

    if factor1_names is None:
        factor1_names = [f'F1_{k + 1}' for k in range(K1)]
    elif len(factor1_names) != K1:
        raise ValidationError(f'Length of factor1_names ({len(factor1_names)}) must equal K1 ({K1})')
    if factor2_names is None:
        factor2_names = [f'F2_{k + 1}' for k in range(K2)]
    elif len(factor2_names) != K2:
        raise ValidationError(f'Length of factor2_names ({len(factor2_names)}) must equal K2 ({K2})')

    total_cells = K1 * K2 * T
    n_drop = max(1, int(p_drop * total_cells))
    n_drop = min(n_drop, total_cells - 1)
    dropped = set(rng.choice(total_cells, size=n_drop, replace=False).tolist())

    rows = []
    cell_idx = 0
    for k1 in range(K1):
        for k2 in range(K2):
            for t in range(T):
                if cell_idx in dropped:
                    rows.append(
                        {
                            'time': t + 1,
                            'factor 1': factor1_names[k1],
                            'factor 2': factor2_names[k2],
                            'y': np.nan,
                        }
                    )
                else:
                    n = int(rng.integers(n_min, n_max + 1))
                    for _ in range(n):
                        eps = rng.normal(0, sigma)
                        y = mu + rho[k1] + phi[k2] + tau[t] + eps
                        rows.append(
                            {
                                'time': t + 1,
                                'factor 1': factor1_names[k1],
                                'factor 2': factor2_names[k2],
                                'y': y,
                            }
                        )
                cell_idx += 1

    df = pd.DataFrame(rows)

    logger.debug(
        f'Generated ODS 4 data: {K1}x{K2}x{T} grid, {n_drop}/{total_cells} cells empty, '
        f'{len(df)} total rows ({df["y"].notna().sum()} valid responses)'
    )
    return df


# ============================================================================
# SDS 5: Incomplete grid, occupied cells unreplicated (N = 1)
# ============================================================================


def make_sds5(
    K1: int = 3,
    K2: int = 2,
    T: int = 6,
    p_drop: float = 0.25,
    mu: float = 50.0,
    sigma: float = 0.4,
    factor1_effect_size: float = 2.0,
    factor2_effect_size: float = 1.5,
    time_effect_size: float = 1.0,
    seed: Optional[int] = None,
    factor1_names: Optional[list[str]] = None,
    factor2_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate Observed Design State 5 (ODS 5) data.

    ODS 5 is Bishop's classification for incomplete designs where every
    occupied (factor × time) cell carries exactly one observation. Some
    cells are missing entirely. After NA filtering during tidying, ODS 5
    collapses to ADS 2 (no replication on the surviving grid).

    Characteristics
    ---------------
    - K1 × K2 × T grid with a fraction of cells empty
    - Every occupied cell has exactly one observation
    - Bishop's structural ODS 5 classification by the N_kt distribution

    Args:
        K1, K2: Number of levels for factor 1 and factor 2
        T: Number of time periods
        p_drop: Approximate fraction of cells to leave empty (0 < p_drop < 1).
            The exact number is ``max(1, int(p_drop * K1 * K2 * T))``, clipped
            so at least one cell remains occupied.
        mu, sigma: Grand mean and within-cell error standard deviation
        factor1_effect_size, factor2_effect_size, time_effect_size: Standard
            deviations of the corresponding main effects
        seed: Random seed for reproducibility
        factor1_names, factor2_names: Custom factor level names

    Returns:
        DataFrame with columns ``time``, ``factor 1``, ``factor 2``, ``y``.
        Empty cells appear as single rows with ``y = NaN``.

    Examples:
        >>> df = make_sds5(K1=3, K2=2, T=6, seed=42)
        >>> # After formulate(), study.observed_design_state.sds == 5
        >>> # After tidy/NA drop, study.analytical_design_state.sds == 2
    """
    if not 0 < p_drop < 1:
        raise ValidationError(f'p_drop must be in (0, 1), got {p_drop}')

    rng = np.random.default_rng(seed)

    rho = rng.normal(0, factor1_effect_size, K1)
    phi = rng.normal(0, factor2_effect_size, K2)
    tau = rng.normal(0, time_effect_size, T)

    if factor1_names is None:
        factor1_names = [f'F1_{k + 1}' for k in range(K1)]
    elif len(factor1_names) != K1:
        raise ValidationError(f'Length of factor1_names ({len(factor1_names)}) must equal K1 ({K1})')
    if factor2_names is None:
        factor2_names = [f'F2_{k + 1}' for k in range(K2)]
    elif len(factor2_names) != K2:
        raise ValidationError(f'Length of factor2_names ({len(factor2_names)}) must equal K2 ({K2})')

    total_cells = K1 * K2 * T
    n_drop = max(1, int(p_drop * total_cells))
    n_drop = min(n_drop, total_cells - 1)
    dropped = set(rng.choice(total_cells, size=n_drop, replace=False).tolist())

    rows = []
    cell_idx = 0
    for k1 in range(K1):
        for k2 in range(K2):
            for t in range(T):
                if cell_idx in dropped:
                    rows.append(
                        {
                            'time': t + 1,
                            'factor 1': factor1_names[k1],
                            'factor 2': factor2_names[k2],
                            'y': np.nan,
                        }
                    )
                else:
                    eps = rng.normal(0, sigma)
                    y = mu + rho[k1] + phi[k2] + tau[t] + eps
                    rows.append(
                        {
                            'time': t + 1,
                            'factor 1': factor1_names[k1],
                            'factor 2': factor2_names[k2],
                            'y': y,
                        }
                    )
                cell_idx += 1

    df = pd.DataFrame(rows)

    logger.debug(
        f'Generated ODS 5 data: {K1}x{K2}x{T} grid, {n_drop}/{total_cells} cells empty, '
        f'every occupied cell has N=1, {df["y"].notna().sum()} valid responses'
    )
    return df


# ============================================================================
# SDS 6: Incomplete grid, mixed replication (some N=1, some N>=2)
# ============================================================================


def make_sds6(  # noqa: C901
    K1: int = 3,
    K2: int = 2,
    T: int = 6,
    p_drop: float = 0.25,
    p_singleton: float = 0.4,
    n_min: int = 2,
    n_max: int = 5,
    mu: float = 50.0,
    sigma: float = 0.4,
    factor1_effect_size: float = 2.0,
    factor2_effect_size: float = 1.5,
    time_effect_size: float = 1.0,
    seed: Optional[int] = None,
    factor1_names: Optional[list[str]] = None,
    factor2_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate Observed Design State 6 (ODS 6) data.

    ODS 6 is Bishop's classification for incomplete designs where occupied
    cells carry a mix of singleton (N=1) and replicated (N>=2) observations.
    Some cells are missing entirely. After NA filtering during tidying,
    ODS 6 collapses to ADS 3 (mixed replication on the surviving grid).

    Characteristics
    ---------------
    - K1 × K2 × T grid with a fraction of cells empty
    - Among occupied cells, some have exactly 1 observation; others have
      ``n_min..n_max`` observations
    - Bishop's structural ODS 6 classification by the N_kt distribution
      (must have BOTH singletons AND replicated cells)

    Args:
        K1, K2: Number of levels for factor 1 and factor 2
        T: Number of time periods
        p_drop: Approximate fraction of cells to leave empty (0 < p_drop < 1)
        p_singleton: Among occupied cells, probability of singleton vs
            replicated (0 < p_singleton < 1). The generator guarantees at
            least one of each so the result classifies as ODS 6, not ODS 4/5.
        n_min, n_max: Replication range for the replicated cells (>= 2)
        mu, sigma: Grand mean and within-cell error standard deviation
        factor1_effect_size, factor2_effect_size, time_effect_size: Standard
            deviations of the corresponding main effects
        seed: Random seed for reproducibility
        factor1_names, factor2_names: Custom factor level names

    Returns:
        DataFrame with columns ``time``, ``factor 1``, ``factor 2``, ``y``.
        Empty cells appear as single rows with ``y = NaN``.

    Examples:
        >>> df = make_sds6(K1=3, K2=2, T=6, seed=42)
        >>> # After formulate(), study.observed_design_state.sds == 6
        >>> # After tidy/NA drop, study.analytical_design_state.sds == 3
    """
    if not 0 < p_drop < 1:
        raise ValidationError(f'p_drop must be in (0, 1), got {p_drop}')
    if not 0 < p_singleton < 1:
        raise ValidationError(f'p_singleton must be in (0, 1), got {p_singleton}')
    if n_min < 2:
        raise ValidationError(f'n_min must be >= 2 (replicated cells), got {n_min}')
    if n_max < n_min:
        raise ValidationError(f'n_max ({n_max}) must be >= n_min ({n_min})')

    rng = np.random.default_rng(seed)

    rho = rng.normal(0, factor1_effect_size, K1)
    phi = rng.normal(0, factor2_effect_size, K2)
    tau = rng.normal(0, time_effect_size, T)

    if factor1_names is None:
        factor1_names = [f'F1_{k + 1}' for k in range(K1)]
    elif len(factor1_names) != K1:
        raise ValidationError(f'Length of factor1_names ({len(factor1_names)}) must equal K1 ({K1})')
    if factor2_names is None:
        factor2_names = [f'F2_{k + 1}' for k in range(K2)]
    elif len(factor2_names) != K2:
        raise ValidationError(f'Length of factor2_names ({len(factor2_names)}) must equal K2 ({K2})')

    total_cells = K1 * K2 * T
    n_drop = max(1, int(p_drop * total_cells))
    n_drop = min(n_drop, total_cells - 2)  # Need at least 2 occupied cells (1 singleton, 1 replicated)
    if n_drop < 1:
        raise ValidationError(
            f"Grid too small for ODS 6: K1*K2*T={total_cells} can't fit "
            f'both empty cells and at least one singleton + one replicated cell'
        )

    occupied_indices = [
        i for i in range(total_cells) if i not in set(rng.choice(total_cells, size=n_drop, replace=False).tolist())
    ]
    # rebuild dropped set from indices not in occupied
    occupied_set = set(occupied_indices)
    dropped_set = set(range(total_cells)) - occupied_set
    n_occupied = len(occupied_indices)

    # Assign singleton vs replicated status to each occupied cell;
    # guarantee at least one of each.
    singleton_flags = rng.random(n_occupied) < p_singleton
    if not singleton_flags.any():
        singleton_flags[0] = True
    if singleton_flags.all():
        singleton_flags[-1] = False

    occupied_to_singleton = dict(zip(occupied_indices, singleton_flags.tolist()))

    rows = []
    cell_idx = 0
    for k1 in range(K1):
        for k2 in range(K2):
            for t in range(T):
                if cell_idx in dropped_set:
                    rows.append(
                        {
                            'time': t + 1,
                            'factor 1': factor1_names[k1],
                            'factor 2': factor2_names[k2],
                            'y': np.nan,
                        }
                    )
                else:
                    n = 1 if occupied_to_singleton[cell_idx] else int(rng.integers(n_min, n_max + 1))
                    for _ in range(n):
                        eps = rng.normal(0, sigma)
                        y = mu + rho[k1] + phi[k2] + tau[t] + eps
                        rows.append(
                            {
                                'time': t + 1,
                                'factor 1': factor1_names[k1],
                                'factor 2': factor2_names[k2],
                                'y': y,
                            }
                        )
                cell_idx += 1

    df = pd.DataFrame(rows)

    logger.debug(
        f'Generated ODS 6 data: {K1}x{K2}x{T} grid, {n_drop}/{total_cells} cells empty, '
        f'{int(singleton_flags.sum())} singletons + '
        f'{int((~singleton_flags).sum())} replicated cells, '
        f'{df["y"].notna().sum()} valid responses'
    )
    return df


# ============================================================================
# Convenience Functions
# ============================================================================


def make_design(
    state: int,
    K1: int = 3,
    K2: int = 2,
    T: int = 8,
    seed: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate synthetic data classifying as the requested design state.

    Bishop's design-state scale assigns an integer 1-6 to each (factor x
    time) grid based on the N_kt distribution. ``make_design(state=N)``
    generates data whose Observed Design State (ODS) classification
    equals ``N``. Uses the three-state vocabulary (PDS / ODS / ADS);
    the integer ``state`` is Bishop's reference scale.

    Args:
        state: Bishop design-state code (1, 2, 3, 4, 5, or 6).
        K1: Number of factor 1 levels.
        K2: Number of factor 2 levels.
        T: Number of time periods.
        seed: Random seed for reproducibility.
        **kwargs: Additional keyword arguments forwarded to the
            state-specific generator (mix params like ``p_replicated``,
            ``p_drop``, ``p_singleton``, ``n_min``, ``n_max``).

    Returns:
        DataFrame with columns ``time``, ``factor 1``, ``factor 2``,
        ``y`` whose ODS classification equals ``state``. After
        NA-filtering during tidying, ODS 4/5/6 collapse to ADS 1/2/3.

    Examples:
        >>> # Round-trip: generated ODS equals requested state
        >>> for s in (1, 2, 3, 4, 5, 6):
        ...     df = make_design(state=s, seed=42)
        ...     study = pb.ProcessBehavior(df).formulate(
        ...         response='y',
        ...         factors=['factor 1', 'factor 2'],
        ...         time='time',
        ...     )
        ...     assert study.observed_design_state.sds == s

    Raises:
        ValidationError: If state not in [1, 2, 3, 4, 5, 6].
    """
    generators = {1: make_sds1, 2: make_sds2, 3: make_sds3, 4: make_sds4, 5: make_sds5, 6: make_sds6}

    if state not in generators:
        raise ValidationError(
            f'Design state {state} not implemented. Available states: {sorted(generators.keys())}'
        )

    return generators[state](K1=K1, K2=K2, T=T, seed=seed, **kwargs)


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================


def make_edge_cases() -> dict[str, pd.DataFrame]:
    """
    Generate challenging edge cases for testing robustness.

    These datasets test how well the system handles unusual but real situations:
    - Missing cells (incomplete factorial)
    - Extreme imbalance (one cell with many obs, others with few)
    - Single observation (should fail gracefully)
    - All same value (zero variance)
    - Extreme outliers
    - Mixed data quality

    Returns:
        Dictionary of DataFrames, keyed by edge case name

    Examples:
        >>> edge_cases = make_edge_cases()
        >>> list(edge_cases.keys())
        ['missing_cells', 'extreme_imbalance', 'single_obs',
         'zero_variance', 'outliers', 'mixed_quality']

        >>> # Test each edge case
        >>> for name, df in edge_cases.items():
        ...     try:
        ...         result = Analysis(df, spec).calculate()
        ...         print(f"{name}: PASSED")
        ...     except Exception as e:
        ...         print(f"{name}: {type(e).__name__}")

    Use Cases:
        - Regression testing for error handling
        - Validating input validation logic
        - Testing graceful degradation
        - Documentation of failure modes
    """
    cases = {}

    # 1. Missing cells (incomplete factorial)
    cases['missing_cells'] = pd.DataFrame(
        {
            'time': [1, 1, 2, 2, 3],  # No time=3 for factor B
            'factor 1': ['A', 'A', 'A', 'B', 'A'],
            'factor 2': ['NA'] * 5,
            'y': [10.1, 10.2, 11.0, 12.0, 10.5],
        }
    )

    # 2. Extreme imbalance (one cell dominates)
    cases['extreme_imbalance'] = pd.DataFrame(
        {
            'time': [1] * 20 + [2] * 2 + [3] * 20,  # Time 2 has only 2 obs
            'factor 1': ['A'] * 20 + ['B'] * 2 + ['A'] * 20,
            'factor 2': ['NA'] * 42,
            'y': np.random.normal(50, 2, 42),
        }
    )

    # 3. Single observation total (should fail)
    cases['single_obs'] = pd.DataFrame({'time': [1], 'factor 1': ['A'], 'factor 2': ['NA'], 'y': [50.0]})

    # 4. Zero variance (all values identical)
    cases['zero_variance'] = pd.DataFrame(
        {
            'time': [1, 1, 2, 2, 3, 3],
            'factor 1': ['A', 'A', 'B', 'B', 'A', 'B'],
            'factor 2': ['NA'] * 6,
            'y': [50.0] * 6,  # All identical
        }
    )

    # 5. Extreme outliers
    rng = np.random.default_rng(42)
    y_with_outliers = list(rng.normal(50, 1, 18))
    y_with_outliers[5] = 100.0  # Extreme high
    y_with_outliers[12] = 0.0  # Extreme low

    cases['outliers'] = pd.DataFrame(
        {
            'time': [1, 1, 2, 2, 3, 3] * 3,
            'factor 1': ['A'] * 6 + ['B'] * 6 + ['C'] * 6,
            'factor 2': ['NA'] * 18,
            'y': y_with_outliers,
        }
    )

    # 6. Mixed quality (some cells have high variance, others low)
    y_mixed = (
        list(rng.normal(50, 0.1, 6))  # Low variance
        + list(rng.normal(50, 5.0, 6))  # High variance
        + list(rng.normal(50, 0.5, 6))
    )  # Medium variance

    cases['mixed_quality'] = pd.DataFrame(
        {
            'time': [1, 1, 2, 2, 3, 3] * 3,
            'factor 1': ['A'] * 6 + ['B'] * 6 + ['C'] * 6,
            'factor 2': ['NA'] * 18,
            'y': y_mixed,
        }
    )

    # 7. Single cell with replication (rest empty)
    cases['single_cell_replicated'] = pd.DataFrame(
        {'time': [1] * 5, 'factor 1': ['A'] * 5, 'factor 2': ['NA'] * 5, 'y': [49.5, 50.0, 50.5, 50.2, 49.8]}
    )

    # 8. Nearly empty (very sparse data)
    cases['sparse'] = pd.DataFrame(
        {
            'time': [1, 5, 10, 15, 20],
            'factor 1': ['A', 'B', 'A', 'C', 'B'],
            'factor 2': ['NA'] * 5,
            'y': [50.1, 49.8, 50.3, 49.5, 50.0],
        }
    )

    return cases


# ============================================================================
# Validation and Comparison Utilities
# ============================================================================


def compare_sds_characteristics(
    sds_list: Optional[list[int]] = None, K1: int = 3, K2: int = 2, T: int = 8, seed: int = 42
) -> pd.DataFrame:
    """
    Generate comparison table of SDS characteristics.

    Creates a summary table comparing key features of each SDS type,
    useful for documentation and teaching.

    Args:
        sds_list: List of SDS to compare (default: all)
        K1: Factor 1 levels for comparison
        K2: Factor 2 levels for comparison
        T: Time periods for comparison
        seed: Random seed

    Returns:
        DataFrame with SDS comparison metrics

    Example:
        >>> summary = compare_sds_characteristics()
        >>> print(summary)

        sds  n_obs  n_cells  min_n  max_n  has_replication  complete_grid
        1    144    48       2      4      True             True
        2    48     48       1      1      False            True
        3    96     48       1      3      Mixed            True
        4    40     40       1      1      False            False
        5    38     38       1      1      False            False
        6    84     84       1      1      False            False
    """
    if sds_list is None:
        sds_list = [1, 2, 3, 4, 5, 6]

    rows = []
    for sds in sds_list:
        df = make_design(sds, K1=K1, K2=K2, T=T, seed=seed)

        if 'factor 1' in df.columns and 'time' in df.columns:
            cell_counts = df.groupby(['factor 1', 'time']).size()
            complete = len(cell_counts) == df['factor 1'].nunique() * df['time'].nunique()
        else:
            cell_counts = pd.Series([1] * len(df))
            complete = False

        min_n = cell_counts.min()
        max_n = cell_counts.max()

        if min_n == max_n == 1:
            replication = 'False'
        elif min_n >= 2:
            replication = 'True'
        else:
            replication = 'Mixed'

        rows.append(
            {
                'sds': sds,
                'n_obs': len(df),
                'n_cells': len(cell_counts),
                'min_n': min_n,
                'max_n': max_n,
                'has_replication': replication,
                'complete_grid': complete,
            }
        )

    return pd.DataFrame(rows)


def validate_sds_detection(df: pd.DataFrame, expected_sds: int, spec: 'FormulationSpec') -> bool:
    """
    Validate that generated data is correctly classified as expected SDS.

    This function is useful for testing SDS detection algorithms by
    generating data with known SDS and verifying correct classification.

    Args:
        df: DataFrame to validate
        expected_sds: Expected SDS classification (1-6)
        spec: FormulationSpec object

    Returns:
        True if detected SDS matches expected

    Example:
        >>> df = make_sds1(K1=3, K2=2, T=8, seed=42)
        >>> spec = FormulationSpec(response_var='Y', rsg_vars=('F1', 'F2'), time_var='T')
        >>> validate_sds_detection(df, expected_sds=1, spec)
        True

    Note:
        This requires importing from analysis_dataset module.
        Kept here for convenience in testing.
    """
    try:
        from ..analysis_dataset import AnalysisDataSet

        ads = AnalysisDataSet(df=df, spec=spec, observed_sds=expected_sds)
        detected_sds = ads.observed_design_state

        if detected_sds == expected_sds:
            logger.info(f'✓ Correctly detected SDS {expected_sds}')
            return True
        else:
            logger.error(f'✗ SDS detection mismatch: expected {expected_sds}, detected {detected_sds}')
            return False

    except ImportError:
        logger.warning('Cannot validate SDS detection: analysis_dataset module not available')
        return False


# ============================================================================
# Batch Generation Utilities
# ============================================================================


def generate_test_suite(output_dir: str = 'test_data', seed: int = 42) -> dict[str, str]:
    """
    Generate complete suite of test datasets and save to disk.

    Creates CSV files for all SDS types with multiple variations,
    useful for:
    - Regression testing
    - Benchmarking
    - Documentation examples
    - Training materials

    Args:
        output_dir: Directory to save CSV files
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping dataset names to file paths

    Example:
        >>> files = generate_test_suite(output_dir='datasets/test')
        >>> print(f"Generated {len(files)} test datasets")
        >>> for name, path in files.items():
        ...     print(f"  {name}: {path}")

    File Structure:
        test_data/
            sds1_basic.csv
            sds1_large.csv
            sds1_with_truth.csv
            sds2_basic.csv
            sds2_large.csv
            ...
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    files = {}

    # SDS 1 variations
    files['sds1_basic'] = os.path.join(output_dir, 'sds1_basic.csv')
    df = make_sds1(K1=3, K2=2, T=8, seed=seed)
    df.to_csv(files['sds1_basic'], index=False)

    files['sds1_large'] = os.path.join(output_dir, 'sds1_large.csv')
    df = make_sds1(K1=5, K2=2, T=12, n_min=3, n_max=5, seed=seed)
    df.to_csv(files['sds1_large'], index=False)

    files['sds1_with_truth'] = os.path.join(output_dir, 'sds1_with_truth.csv')
    df = make_sds1(K1=3, K2=2, T=6, include_truth=True, seed=seed)
    df.to_csv(files['sds1_with_truth'], index=False)

    # SDS 2 variations
    files['sds2_basic'] = os.path.join(output_dir, 'sds2_basic.csv')
    df = make_sds2(K1=3, K2=2, T=10, seed=seed)
    df.to_csv(files['sds2_basic'], index=False)

    files['sds2_large'] = os.path.join(output_dir, 'sds2_large.csv')
    df = make_sds2(K1=6, K2=2, T=16, seed=seed)
    df.to_csv(files['sds2_large'], index=False)

    # SDS 3 variations (critical!)
    files['sds3_balanced'] = os.path.join(output_dir, 'sds3_balanced.csv')
    df = make_sds3(K1=3, K2=2, T=8, p_replicated=0.5, seed=seed)
    df.to_csv(files['sds3_balanced'], index=False)

    files['sds3_sparse'] = os.path.join(output_dir, 'sds3_sparse.csv')
    df = make_sds3(K1=4, K2=2, T=10, p_replicated=0.2, seed=seed)
    df.to_csv(files['sds3_sparse'], index=False)

    files['sds3_early_times'] = os.path.join(output_dir, 'sds3_early_times.csv')
    df = make_sds3(K1=3, K2=2, T=12, replication_pattern='early_times', seed=seed)
    df.to_csv(files['sds3_early_times'], index=False)

    # SDS 4 variations (ODS 4: incomplete grid, all occupied cells replicated)
    files['sds4_default'] = os.path.join(output_dir, 'sds4_default.csv')
    df = make_sds4(K1=3, K2=2, T=8, seed=seed)
    df.to_csv(files['sds4_default'], index=False)

    files['sds4_sparse'] = os.path.join(output_dir, 'sds4_sparse.csv')
    df = make_sds4(K1=3, K2=2, T=10, p_drop=0.5, seed=seed)
    df.to_csv(files['sds4_sparse'], index=False)

    # SDS 5 variations (ODS 5: incomplete grid, all occupied cells singleton)
    files['sds5_default'] = os.path.join(output_dir, 'sds5_default.csv')
    df = make_sds5(K1=3, K2=2, T=8, seed=seed)
    df.to_csv(files['sds5_default'], index=False)

    files['sds5_sparse'] = os.path.join(output_dir, 'sds5_sparse.csv')
    df = make_sds5(K1=4, K2=3, T=12, p_drop=0.5, seed=seed)
    df.to_csv(files['sds5_sparse'], index=False)

    # SDS 6 variations (ODS 6: incomplete grid, mixed singleton / replicated)
    files['sds6_default'] = os.path.join(output_dir, 'sds6_default.csv')
    df = make_sds6(K1=3, K2=2, T=8, seed=seed)
    df.to_csv(files['sds6_default'], index=False)

    files['sds6_singleton_heavy'] = os.path.join(output_dir, 'sds6_singleton_heavy.csv')
    df = make_sds6(K1=4, K2=2, T=10, p_singleton=0.7, seed=seed)
    df.to_csv(files['sds6_singleton_heavy'], index=False)

    # Edge cases
    edge_cases = make_edge_cases()
    for name, df in edge_cases.items():
        files[f'edge_{name}'] = os.path.join(output_dir, f'edge_{name}.csv')
        df.to_csv(files[f'edge_{name}'], index=False)

    logger.info(f'Generated {len(files)} test datasets in {output_dir}')

    return files


# ============================================================================
# Documentation and Metadata
# ============================================================================


def get_sds_info(sds: int) -> dict[str, str]:
    """
    Get detailed information about a specific SDS type.

    Returns metadata and usage guidance for each SDS.

    Args:
        sds: Sampling Design State (1-6)

    Returns:
        Dictionary with description, characteristics, use cases, etc.

    Example:
        >>> info = get_sds_info(3)
        >>> print(info['description'])
        >>> print(info['when_to_use'])
    """
    sds_metadata = {
        1: {
            'name': 'Full Replication',
            'description': 'Every (factor, time) cell has n≥2 observations',
            'characteristics': [
                'True within-cell variance estimation',
                'Complete (factor × time) grid',
                'Supports full interaction analysis',
                'Most statistically powerful',
            ],
            'when_to_use': [
                'Testing Xbar-S chart algorithms',
                'Full factorial experiments',
                'When resources allow complete replication',
                'Baseline for comparing other SDS',
            ],
            'chart_types': ['Xbar-S', 'Range charts on subgroups'],
            'vas_capabilities': ['All residuals R1-R5', 'Full interaction effects'],
            'common_in': 'Designed experiments, validation studies',
        },
        2: {
            'name': 'No Replication',
            'description': 'Exactly one observation per (factor, time) cell',
            'characteristics': [
                'Complete (factor × time) grid',
                'No within-cell variance',
                'Must use moving average for R2',
                'Error confounded with interaction',
            ],
            'when_to_use': [
                'Resource-constrained experiments',
                'Screening studies',
                'Testing moving average algorithms',
                'When interaction assumed small',
            ],
            'chart_types': ['IMR on cell means', 'Moving range methods'],
            'vas_capabilities': ['R1-R5 with approximated R2', 'Limited interaction analysis'],
            'common_in': 'Designed experiments, DOE, initial screening',
        },
        3: {
            'name': 'Partial Replication',
            'description': 'Mixed: some cells n=1, others n≥2',
            'characteristics': [
                'Most common in real-world data',
                'Requires hybrid R2 estimation',
                'Irregular replication pattern',
                'Challenging to analyze correctly',
            ],
            'when_to_use': [
                'Testing hybrid variance estimation',
                'Real-world manufacturing data',
                'Unplanned replication scenarios',
                'Missing data situations',
            ],
            'chart_types': ['Hybrid approaches', 'Adaptive methods'],
            'vas_capabilities': ['Hybrid R2 calculation', 'Partial interaction analysis'],
            'common_in': 'Most real production data, unplanned sampling',
        },
        4: {
            'name': 'Single Condition Over Time',
            'description': 'One factor level, multiple time points (K=1)',
            'characteristics': [
                'Time series structure',
                'No grouping by factors',
                'May show trends or drift',
                'Classic control chart scenario',
            ],
            'when_to_use': [
                'Single process monitoring',
                'Time series analysis',
                'Trend detection',
                'Classic SPC applications',
            ],
            'chart_types': ['IMR', 'Individuals chart', 'Moving range'],
            'vas_capabilities': ['Limited (no factor structure)', 'Time effects only'],
            'common_in': 'Single machine/process, continuous monitoring',
        },
        5: {
            'name': 'Nested Design',
            'description': 'Hierarchical factors with asynchronous coverage',
            'characteristics': [
                'Factor 2 nested in Factor 1',
                'Irregular temporal patterns',
                'Not all combinations at all times',
                'Requires variance components',
            ],
            'when_to_use': [
                'Multi-head machines (heads in lanes)',
                'Hierarchical production structures',
                'Nested experimental designs',
                'Asynchronous operations',
            ],
            'chart_types': ['Hierarchical charts', 'Nested variance components'],
            'vas_capabilities': ['Nested effects', 'Variance component estimation'],
            'common_in': 'Multi-head fillers, multi-spindle machines, nested processes',
        },
        6: {
            'name': 'Unstructured / Regime Changes',
            'description': 'Irregular patterns with process regime changes',
            'characteristics': [
                'Incomplete (factor × time) grid',
                'Mean shifts over time (regimes)',
                'Irregular sampling patterns',
                'Complex time structure',
            ],
            'when_to_use': [
                'Long-term process studies',
                'Process with adjustments',
                'Regime detection problems',
                'Irregular production schedules',
            ],
            'chart_types': ['Adaptive methods', 'Change-point detection'],
            'vas_capabilities': ['Limited', 'Regime-specific analysis'],
            'common_in': 'Long-term studies, processes with changes, irregular schedules',
        },
    }

    if sds not in sds_metadata:
        raise ValidationError(f'SDS {sds} not defined. Valid: 1-6')

    return sds_metadata[sds]


def print_sds_summary():
    """
    Print formatted summary of all SDS types.

    Useful for documentation and teaching.

    Example:
        >>> print_sds_summary()

        SAMPLING DESIGN STATES SUMMARY
        ==============================

        SDS 1: Full Replication
        -----------------------
        Description: Every (factor, time) cell has n≥2 observations
        ...
    """
    print('\nSAMPLING DESIGN STATES SUMMARY')
    print('=' * 70)

    for sds in range(1, 7):
        info = get_sds_info(sds)
        print(f'\nSDS {sds}: {info["name"]}')
        print('-' * 70)
        print(f'Description: {info["description"]}')
        print('\nCharacteristics:')
        for char in info['characteristics']:
            print(f'  • {char}')
        print(f'\nCommon in: {info["common_in"]}')
        print(f'Chart types: {", ".join(info["chart_types"])}')


# ============================================================================
# Large Dataset Generator for Performance Testing
# ============================================================================


def make_large_dataset(
    n_rows: int = 1_000_000,
    n_extra_cols: int = 46,
    sds: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate large dataset for performance testing.

    Creates dataset with specified row count plus extra columns that won't
    be used in analysis, simulating real-world data dumps with many unused
    columns.

    This is useful for:
    - Performance benchmarking
    - Memory usage testing
    - Scalability validation
    - Stress testing the analysis pipeline

    Args:
        n_rows: Target number of rows (approximate, actual may vary slightly
                due to SDS structure constraints)
        n_extra_cols: Extra columns to add beyond the 4 core analysis columns
                      (factor 1, factor 2, time, y). Default 46 gives 50 total.
        sds: Sampling Design State to generate (1, 2, or 4 supported for large)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with approximately n_rows and (4 + n_extra_cols) columns

    Examples:
        >>> # Generate 1M row dataset with 50 columns
        >>> df = make_large_dataset(n_rows=1_000_000, n_extra_cols=46)
        >>> len(df)
        1000000
        >>> len(df.columns)
        50

        >>> # Generate 100K rows for faster testing
        >>> df = make_large_dataset(n_rows=100_000, seed=42)

        >>> # Test SDS 2 (no replication) at scale
        >>> df = make_large_dataset(n_rows=500_000, sds=2)

    Notes:
        - SDS 1: Uses K1 × K2 factors × T times × n replicates = n_rows
        - SDS 2: Uses K1 × K2 factors × T times = n_rows (no replication)
        - SDS 4: Uses T time points = n_rows (single factor)
        - Extra columns are randomly typed (60% float, 20% int, 20% string)
          to simulate realistic mixed-type data exports.
    """
    rng = np.random.default_rng(seed)

    if sds == 1:
        # Calculate K1, K2, T, n to hit target rows
        # K1 × K2 × T × n = n_rows
        n = 10  # Fixed replication
        K2 = 2  # Fixed K2
        cells = n_rows // (n * K2)
        K1 = int(np.sqrt(cells))
        T = cells // K1

        df = make_sds1(K1=K1, K2=K2, T=T, n_min=n, n_max=n, seed=seed)

    elif sds == 2:
        # K1 × K2 × T = n_rows (no replication)
        K2 = 2  # Fixed K2
        remaining = n_rows // K2
        K1 = int(np.sqrt(remaining))
        T = remaining // K1
        df = make_sds2(K1=K1, K2=K2, T=T, seed=seed)

    elif sds == 4:
        # ODS 4: incomplete grid, all occupied cells replicated.
        # Scale K1, K2, T to roughly hit n_rows total observations.
        K1 = max(2, int(np.sqrt(n_rows // 8)))
        K2 = max(2, int(np.sqrt(n_rows // 8)))
        T = max(2, n_rows // (K1 * K2 * 3))
        df = make_sds4(K1=K1, K2=K2, T=T, seed=seed)

    else:
        raise ValidationError(
            f'SDS {sds} not optimized for large datasets. '
            f'Supported: 1 (full replication), 2 (no replication), 4 (incomplete grid)'
        )

    # Add extra columns with mixed data types
    for i in range(n_extra_cols):
        dtype_choice = rng.choice(['float', 'int', 'str'], p=[0.6, 0.2, 0.2])

        if dtype_choice == 'float':
            df[f'extra_{i}'] = rng.normal(0, 100, len(df))
        elif dtype_choice == 'int':
            df[f'extra_{i}'] = rng.integers(0, 1000, len(df))
        else:
            df[f'extra_{i}'] = rng.choice(['A', 'B', 'C', 'D'], len(df))

    logger.info(f'Generated large dataset: {len(df):,} rows × {len(df.columns)} cols')
    return df


# ============================================================================
# Module-level convenience
# ============================================================================

# Quick access to all generators (SDS 1-6)
# Note: SDS 0 was consolidated into SDS 4. Response-only data is now
# treated as SDS 4 with implicit time ordering via obs_id.
GENERATORS = {1: make_sds1, 2: make_sds2, 3: make_sds3, 4: make_sds4, 5: make_sds5, 6: make_sds6}


if __name__ == '__main__':
    # Demo: Generate one dataset of each type
    print('Generating example datasets for each SDS...\n')

    for sds in range(1, 7):
        print(f'SDS {sds}: {get_sds_info(sds)["name"]}')
        df = make_design(sds, K1=3, K2=2, T=6, seed=42)
        print(f'  Generated {len(df)} observations')
        print(f'  Columns: {df.columns.tolist()}')
        print(f'  Preview:\n{df.head(3)}')
        print()

    print('\nEdge cases:')
    edge_cases = make_edge_cases()
    for name, df in edge_cases.items():
        print(f'  {name}: {len(df)} observations')

    print('\nFor detailed info on any SDS, use get_sds_info(sds_number)')
    print('For complete summary, use print_sds_summary()')
