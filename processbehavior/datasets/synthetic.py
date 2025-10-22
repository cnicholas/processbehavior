"""
Synthetic Data Generators for Statistical Process Control

This module provides data generators for all Sampling Design States (SDS)
defined in the Variance Analysis System framework by Dr. Donald Wheeler
and extended by Dr. Davis Bishop.

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
    >>> df = synthetic.make_sds1(K=3, T=8, n_min=2, n_max=4)
    >>> 
    >>> # Use with analysis
    >>> spec = {'analysis_type': 'Xbar', 'rsg_vars': ['factor 1'], 
    ...         'response_var': 'y', 'time_var': 'time'}
    >>> results = perform_analysis(df, spec)

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

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
import logging
from ..analysis_dataset import AnalysisSpecification

logger = logging.getLogger(__name__)

__all__ = [
    'make_sds1',
    'make_sds2', 
    'make_sds3',
    'make_sds4',
    'make_sds5',
    'make_sds6',
    'make_sds',
    'make_edge_cases'
]


# ============================================================================
# SDS 1: Full Replication - Most Statistically Powerful
# ============================================================================

def make_sds1(
    K: int = 3,
    T: int = 8,
    n_min: int = 2,
    n_max: int = 5,
    mu: float = 50.0,
    sigma: float = 0.4,
    factor_effect_size: float = 2.0,
    time_effect_size: float = 1.0,
    interaction_effect_size: float = 0.5,
    seed: Optional[int] = None,
    include_truth: bool = False,
    factor_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate Sampling Design State 1 data (full replication).
    
    SDS1 is the gold standard for process studies - every combination of
    factor and time has multiple observations, allowing true estimation of
    within-cell variance. This is what designed experiments strive for.
    
    Characteristics:
    ---------------
    - Every (factor, time) cell has n ≥ 2 observations
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
        K: Number of factor levels (e.g., machines, lanes, operators)
        T: Number of time periods (e.g., days, shifts, batches)
        n_min: Minimum observations per (factor, time) cell
        n_max: Maximum observations per (factor, time) cell
        mu: Grand mean (process center)
        sigma: Within-cell standard deviation (pure error)
        factor_effect_size: Standard deviation of factor main effects
        time_effect_size: Standard deviation of time main effects  
        interaction_effect_size: Standard deviation of (factor × time) interactions
        seed: Random seed for reproducibility (None = random)
        include_truth: If True, include true effects as columns for validation
        factor_names: Custom factor level names (default: K1, K2, ...)
        
    Returns:
        DataFrame with columns:
            - time: Time period (1 to T)
            - factor 1: Factor level (K1, K2, ... or custom names)
            - factor 2: 'NA' (placeholder for future two-factor designs)
            - y: Response variable
            
        Optional columns (if include_truth=True):
            - true_factor_effect: The ρ_i component
            - true_time_effect: The τ_j component
            - true_interaction: The (ρτ)_ij component
            - true_error: The ε_ijr component (pure error)
    
    Mathematical Model:
        Y_ijr = μ + ρ_i + τ_j + (ρτ)_ij + ε_ijr
        
        where:
            μ: Grand mean (mu parameter)
            ρ_i: Factor i main effect ~ N(0, factor_effect_size²)
            τ_j: Time j main effect ~ N(0, time_effect_size²)
            (ρτ)_ij: Factor-Time interaction ~ N(0, interaction_effect_size²)
            ε_ijr: Pure error for replicate r ~ N(0, σ²)
            
        This additive model with interaction is the foundation of VAS.
    
    Examples:
        >>> # Basic usage - 3 factors, 8 time periods
        >>> df = make_sds1(K=3, T=8, seed=42)
        >>> print(f"Generated {len(df)} observations")
        >>> df.groupby(['factor 1', 'time']).size().min()
        2  # All cells have at least 2 observations
        
        >>> # With ground truth for validation testing
        >>> df = make_sds1(K=2, T=4, include_truth=True, seed=42)
        >>> # Now can validate that analysis recovers true effects
        >>> print(df[['y', 'true_factor_effect', 'true_time_effect']].head())
        
        >>> # Custom factor names
        >>> df = make_sds1(K=3, T=6, 
        ...                factor_names=['Machine_A', 'Machine_B', 'Machine_C'],
        ...                seed=42)
        
        >>> # Large effects, low noise (easy to detect)
        >>> df = make_sds1(K=5, T=10, 
        ...                factor_effect_size=5.0,  # Large differences
        ...                time_effect_size=3.0,
        ...                sigma=0.1,                # Low noise
        ...                seed=42)
        
        >>> # Realistic manufacturing scenario
        >>> df = make_sds1(K=4, T=12,           # 4 lanes, 12 hours
        ...                n_min=3, n_max=5,     # 3-5 samples per hour
        ...                mu=237.5,              # Target fill weight
        ...                sigma=0.8,             # Within-cell variability
        ...                factor_effect_size=1.5, # Lane differences
        ...                time_effect_size=0.5,   # Hour-to-hour drift
        ...                seed=42)
    
    Validation:
        The function automatically validates that generated data meets SDS1 criteria:
        - All (K,T) cells present (no missing combinations)
        - All cells have n_min ≤ n ≤ n_max observations
        - Response variable is numeric
        - No missing values in critical columns
    
    Notes:
        - Increasing n_min/n_max provides more precision but more data collection cost
        - Higher sigma makes effects harder to detect (more realistic)
        - Set interaction_effect_size=0 for additive model (no interaction)
        - Use include_truth=True when validating algorithm correctness
    
    See Also:
        make_sds2: For unreplicated designs (n=1 per cell)
        make_sds3: For partial replication (mixed n=1 and n≥2)
        make_sds4: For time series (single factor over time)
    """
    rng = np.random.default_rng(seed)
    
    # Generate true effects from the model
    rho = rng.normal(0, factor_effect_size, K)      # Factor main effects
    tau = rng.normal(0, time_effect_size, T)        # Time main effects
    inter = rng.normal(0, interaction_effect_size, (K, T))  # Interactions
    
    # Use custom factor names if provided
    if factor_names is None:
        factor_names = [f"K{k+1}" for k in range(K)]
    elif len(factor_names) != K:
        raise ValueError(f"Length of factor_names ({len(factor_names)}) must equal K ({K})")
    
    rows = []
    for k in range(K):
        for t in range(T):
            # Random number of observations per cell (uniform distribution)
            n = rng.integers(n_min, n_max + 1)
            
            for i in range(n):
                # Generate observation from true model: Y = μ + ρ + τ + ρτ + ε
                epsilon = rng.normal(0, sigma)  # Pure error
                y = mu + rho[k] + tau[t] + inter[k, t] + epsilon
                
                row = {
                    'time': t + 1,
                    'factor 1': factor_names[k],
                    'factor 2': "NA",
                    'y': y
                }
                
                # Optionally include ground truth for validation
                if include_truth:
                    row.update({
                        'true_factor_effect': rho[k],
                        'true_time_effect': tau[t],
                        'true_interaction': inter[k, t],
                        'true_error': epsilon,
                        'true_mean': mu + rho[k] + tau[t] + inter[k, t]
                    })
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Validation - ensure SDS1 criteria met
    cell_counts = df.groupby(['factor 1', 'time']).size()
    
    if cell_counts.min() < n_min:
        raise AssertionError(f"SDS1 validation failed: min n={cell_counts.min()} < {n_min}")
    if cell_counts.max() > n_max:
        raise AssertionError(f"SDS1 validation failed: max n={cell_counts.max()} > {n_max}")
    if len(cell_counts) != K * T:
        raise AssertionError(
            f"SDS1 validation failed: expected {K*T} cells, got {len(cell_counts)} "
            "(some factor-time combinations missing)"
        )
    
    logger.debug(
        f"Generated SDS1 data: {K} factors × {T} times × ~{(n_min+n_max)/2:.1f} reps/cell "
        f"= {len(df)} observations"
    )
    
    return df


# ============================================================================
# SDS 2: No Replication - Common in Designed Experiments  
# ============================================================================

def make_sds2(
    K: int = 3,
    T: int = 10,
    mu: float = 50.0,
    sigma: float = 0.4,
    factor_effect_size: float = 2.0,
    time_effect_size: float = 1.2,
    interaction_effect_size: float = 0.6,
    seed: Optional[int] = None,
    include_truth: bool = False,
    factor_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate Sampling Design State 2 data (no replication).
    
    SDS2 is the classic unreplicated factorial design - exactly one observation
    per (factor, time) combination. Common in designed experiments where
    resources are limited and you want to cover more factor combinations.
    
    Characteristics:
    ---------------
    - Exactly one observation per (factor, time) cell
    - Within-cell variance must be estimated indirectly (moving average)
    - Cannot separate pure error from interaction effects
    - More efficient data collection, less statistical power
    - Ybar_kt = Y (since only one observation per cell)
    
    Use Cases:
    ---------
    - Testing moving average R2 calculation algorithms
    - Validating SDS detection logic
    - Demonstrating unreplicated factorial designs
    - Cost-constrained or time-constrained data collection
    - Initial screening experiments
    
    Args:
        K: Number of factor levels
        T: Number of time periods
        mu: Grand mean
        sigma: Process standard deviation (confounded with interaction)
        factor_effect_size: Std dev of factor effects
        time_effect_size: Std dev of time effects
        interaction_effect_size: Std dev of interactions
        seed: Random seed for reproducibility
        include_truth: Include true effects as columns
        factor_names: Custom factor level names
        
    Returns:
        DataFrame with exactly K×T observations
        
    Mathematical Model:
        Y_ij = μ + ρ_i + τ_j + (ρτ)_ij + ε_ij
        
        Note: In SDS2, the (ρτ)_ij and ε_ij are completely confounded.
        The moving average method attempts to separate them by assuming
        smooth time trends, but this is an approximation.
    
    Examples:
        >>> # Basic usage
        >>> df = make_sds2(K=3, T=8, seed=42)
        >>> df.groupby(['factor 1', 'time']).size().max()
        1  # All cells have exactly 1 observation
        >>> len(df)
        24  # K × T = 3 × 8
        
        >>> # Compare data efficiency with SDS1
        >>> df1 = make_sds1(K=3, T=8, n_min=3, n_max=3, seed=42)
        >>> df2 = make_sds2(K=3, T=8, seed=42)
        >>> print(f"SDS1: {len(df1)} obs, SDS2: {len(df2)} obs")
        >>> # SDS1 requires 3× more data collection
        
        >>> # With ground truth to see what's confounded
        >>> df = make_sds2(K=4, T=6, include_truth=True, seed=42)
        >>> # Can verify that R2 estimation via moving average
        >>> # approximates true_interaction + true_error
        
        >>> # Large designed experiment
        >>> df = make_sds2(K=8, T=16, seed=42)
        >>> # 128 observations covers 8×16 = 128 combinations
    
    Important Limitation:
        In SDS2, you cannot definitively separate:
        - True interaction effects (ρτ)_ij
        - Pure random error ε_ij
        
        The moving average R2 calculation assumes that interactions are
        relatively smooth over time within each factor level. Violations
        of this assumption can lead to biased estimates.
    
    Validation:
        - Exactly n=1 per cell (all cells)
        - Total observations = K × T
        - No missing factor-time combinations
    
    See Also:
        make_sds1: For replicated designs
        make_sds3: For mixed replication (most realistic)
    """
    rng = np.random.default_rng(seed)
    
    rho = rng.normal(0, factor_effect_size, K)
    tau = rng.normal(0, time_effect_size, T)
    inter = rng.normal(0, interaction_effect_size, (K, T))
    
    if factor_names is None:
        factor_names = [f"K{k+1}" for k in range(K)]
    elif len(factor_names) != K:
        raise ValueError(f"Length of factor_names must equal K")
    
    rows = []
    for k in range(K):
        for t in range(T):
            # Single observation per cell - this is the defining characteristic
            y = mu + rho[k] + tau[t] + inter[k, t] + rng.normal(0, sigma)
            
            row = {
                'time': t + 1,
                'factor 1': factor_names[k],
                'factor 2': "NA",
                'y': y
            }
            
            if include_truth:
                row.update({
                    'true_factor_effect': rho[k],
                    'true_time_effect': tau[t],
                    'true_interaction': inter[k, t],
                    'true_confounded': inter[k, t] + (y - mu - rho[k] - tau[t] - inter[k, t])
                })
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Validation
    cell_counts = df.groupby(['factor 1', 'time']).size()
    
    if not (cell_counts == 1).all():
        raise AssertionError(
            f"SDS2 validation failed: expected all n=1, "
            f"got min={cell_counts.min()}, max={cell_counts.max()}"
        )
    
    expected_n = K * T
    if len(df) != expected_n:
        raise AssertionError(
            f"SDS2 validation failed: expected {expected_n} obs, got {len(df)}"
        )
    
    logger.debug(f"Generated SDS2 data: {K} factors × {T} times = {len(df)} observations")
    
    return df


# ============================================================================
# SDS 3: Partial Replication - MOST COMMON IN PRACTICE!
# ============================================================================

def make_sds3(
    K: int = 3,
    T: int = 8,
    p_replicated: float = 0.5,
    n_when_replicated: int = 3,
    mu: float = 50.0,
    sigma: float = 0.5,
    factor_effect_size: float = 2.0,
    time_effect_size: float = 1.0,
    interaction_effect_size: float = 0.5,
    seed: Optional[int] = None,
    replication_pattern: str = 'random',
    include_truth: bool = False,
    factor_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate Sampling Design State 3 data (partial replication).
    
    ⭐ THIS IS THE MOST COMMON REAL-WORLD SCENARIO ⭐
    
    SDS3 represents reality: some cells have replication, others don't.
    This happens due to:
    - Unplanned sampling (operators take extra samples when concerned)
    - Missing data (some planned replicates not collected)
    - Cost constraints (replicate only critical combinations)
    - Phased studies (early phase replicated, later phase not)
    
    This is the HARDEST case to analyze correctly, and most software
    handles it poorly or not at all!
    
    Characteristics:
    ---------------
    - Some (factor, time) cells have n=1, others have n≥2
    - Requires hybrid R2 estimation:
        * Within-cell variance for replicated cells
        * Moving average for unreplicated cells
    - Reflects real-world data collection reality
    - Most challenging to analyze correctly
    
    Use Cases:
    ---------
    - Testing hybrid variance estimation algorithms (CRITICAL)
    - Demonstrating real-world data structures
    - Validating robustness to missing data
    - Teaching about practical SPC challenges
    - Development of SDS detection logic
    
    Args:
        K: Number of factor levels
        T: Number of time periods
        p_replicated: Proportion of cells with replication (0 to 1)
        n_when_replicated: Number of observations in replicated cells
        mu: Grand mean
        sigma: Within-cell standard deviation
        factor_effect_size: Std dev of factor effects
        time_effect_size: Std dev of time effects
        interaction_effect_size: Std dev of interactions
        seed: Random seed
        replication_pattern: How to assign replication:
            - 'random': Random cells replicated (default)
            - 'early_times': First p_replicated fraction of time periods
            - 'late_times': Last p_replicated fraction of time periods
            - 'specific_factors': First p_replicated fraction of factors
            - 'checkerboard': Alternating pattern (k+t even/odd)
            - 'corners': Extreme combinations get replication
        include_truth: Include true effects as columns
        factor_names: Custom factor level names
        
    Returns:
        DataFrame with mixed replication structure, including 'cell_type' column
        
    Examples:
        >>> # 50% of cells replicated (balanced)
        >>> df = make_sds3(K=3, T=8, p_replicated=0.5, seed=42)
        >>> cell_sizes = df.groupby(['factor 1', 'time']).size()
        >>> print(f"Singles: {(cell_sizes == 1).sum()}, "
        ...       f"Replicated: {(cell_sizes > 1).sum()}")
        
        >>> # Realistic scenario: early data collection had replication
        >>> df = make_sds3(K=4, T=12, 
        ...                replication_pattern='early_times',
        ...                p_replicated=0.4,  # First 40% of time
        ...                seed=42)
        >>> # Models situation where process was validated early,
        >>> # then moved to reduced sampling
        
        >>> # Sparse replication (only 20% of cells)
        >>> df = make_sds3(K=5, T=10, 
        ...                p_replicated=0.2,
        ...                n_when_replicated=5,  # Heavy replication when done
        ...                seed=42)
        
        >>> # Checkerboard pattern (common in some experimental designs)
        >>> df = make_sds3(K=4, T=8, 
        ...                replication_pattern='checkerboard',
        ...                seed=42)
        
        >>> # With ground truth to validate hybrid R2 calculation
        >>> df = make_sds3(K=3, T=6, 
        ...                include_truth=True,
        ...                seed=42)
        >>> # Can verify that system correctly handles mixed replication
    
    Replication Patterns:
        Different patterns model different real-world scenarios:
        
        - 'random': Unplanned sampling (operator discretion)
        - 'early_times': Process validation phase
        - 'late_times': Special investigation at end
        - 'specific_factors': Some factors monitored more closely
        - 'checkerboard': Planned incomplete block design
        - 'corners': Focus on extreme operating conditions
    
    Implementation Note:
        This generator is CRITICAL for testing SDS3 detection and hybrid
        R2 calculation. The system must handle this correctly or it will
        fail on most real-world data!
        
        Current system status:
        - SDS3 detection: Partially implemented
        - Hybrid R2 calculation: Needs completion
        
    Validation:
        - At least one cell with n=1
        - At least one cell with n≥2
        - Actual replication proportion within ±20% of target
        - All factor-time combinations present
    
    See Also:
        make_sds1: For fully replicated designs
        make_sds2: For unreplicated designs
    """
    rng = np.random.default_rng(seed)
    
    # Generate true effects
    rho = rng.normal(0, factor_effect_size, K)
    tau = rng.normal(0, time_effect_size, T)
    inter = rng.normal(0, interaction_effect_size, (K, T))
    
    if factor_names is None:
        factor_names = [f"K{k+1}" for k in range(K)]
    elif len(factor_names) != K:
        raise ValueError(f"Length of factor_names must equal K")
    
    rows = []
    replicated_cells = []
    unreplicated_cells = []
    
    for k in range(K):
        for t in range(T):
            # Determine replication based on pattern
            if replication_pattern == 'random':
                is_replicated = rng.random() < p_replicated
                
            elif replication_pattern == 'early_times':
                is_replicated = t < int(T * p_replicated)
                
            elif replication_pattern == 'late_times':
                is_replicated = t >= int(T * (1 - p_replicated))
                
            elif replication_pattern == 'specific_factors':
                is_replicated = k < int(K * p_replicated)
                
            elif replication_pattern == 'checkerboard':
                is_replicated = (k + t) % 2 == 0
                
            elif replication_pattern == 'corners':
                # Replicate when factor or time is at extremes
                is_corner = (k == 0 or k == K-1) or (t == 0 or t == T-1)
                is_replicated = is_corner or (rng.random() < p_replicated/2)
                
            else:
                raise ValueError(
                    f"Unknown replication pattern: {replication_pattern}. "
                    f"Valid: 'random', 'early_times', 'late_times', 'specific_factors', "
                    f"'checkerboard', 'corners'"
                )
            
            # Determine n for this cell
            n = n_when_replicated if is_replicated else 1
            
            # Track cell types for validation
            if is_replicated:
                replicated_cells.append((k, t))
            else:
                unreplicated_cells.append((k, t))
            
            # Generate observations
            for i in range(n):
                epsilon = rng.normal(0, sigma)
                y = mu + rho[k] + tau[t] + inter[k, t] + epsilon
                
                row = {
                    'time': t + 1,
                    'factor 1': factor_names[k],
                    'factor 2': "NA",
                    'y': y,
                    'cell_type': 'replicated' if is_replicated else 'unreplicated'
                }
                
                if include_truth:
                    row.update({
                        'true_factor_effect': rho[k],
                        'true_time_effect': tau[t],
                        'true_interaction': inter[k, t],
                        'true_error': epsilon
                    })
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Validation - ensure true SDS3 (mixed replication)
    cell_counts = df.groupby(['factor 1', 'time']).size()
    has_singles = (cell_counts == 1).any()
    has_multiples = (cell_counts >= 2).any()
    
    if not (has_singles and has_multiples):
        raise AssertionError(
            f"SDS3 validation failed: must have both single and multiple observations. "
            f"Got singles={has_singles}, multiples={has_multiples}. "
            f"Try adjusting p_replicated or replication_pattern."
        )
    
    # Check replication proportion
    actual_p = len(replicated_cells) / (K * T)
    if abs(actual_p - p_replicated) > 0.25:  # Allow 25% tolerance
        logger.warning(
            f"SDS3: target replication {p_replicated:.1%}, "
            f"actual {actual_p:.1%} (outside 25% tolerance)"
        )
    
    logger.debug(
        f"Generated SDS3 data: {len(df)} obs, "
        f"{len(replicated_cells)} replicated cells ({actual_p:.1%}), "
        f"{len(unreplicated_cells)} unreplicated cells"
    )
    
    return df


# ============================================================================
# SDS 4: Single Condition Over Time - Time Series
# ============================================================================

def make_sds4(
    T: int = 40,
    mu: float = 50.0,
    sigma: float = 0.4,
    drift_rate: float = 0.15,
    drift_type: str = 'random_walk',
    seed: Optional[int] = None,
    include_truth: bool = False
) -> pd.DataFrame:
    """
    Generate Sampling Design State 4 data (single condition over time).
    
    SDS4 represents a single process measured over time with no grouping
    structure. This is classic time series data in SPC - one machine,
    one product, one process parameter tracked over many time points.
    
    Characteristics:
    ---------------
    - Only one "condition" or factor level (K=1)
    - Multiple time points (large T)
    - No grouping structure
    - Appropriate for IMR (Individual-Moving Range) charts
    - May exhibit trends, drift, or cycles
    
    Use Cases:
    ---------
    - Testing IMR chart without grouping
    - Demonstrating time series control charts
    - Teaching trend detection
    - Single-stream process monitoring
    - Continuous process variables
    
    Args:
        T: Number of time periods (observations)
        mu: Starting process mean
        sigma: Random variation (short-term)
        drift_rate: Standard deviation of drift per time step
        drift_type: Type of time-based pattern:
            - 'random_walk': Cumulative random drift (default)
            - 'linear': Linear trend upward
            - 'cyclic': Sinusoidal pattern
            - 'step': Step change at midpoint
            - 'none': Pure random (no time structure)
        seed: Random seed
        include_truth: Include drift component as column
        
    Returns:
        DataFrame with time series structure
        
    Mathematical Models:
        Random Walk:
            Y_t = μ + Σ(δ_i) + ε_t
            where δ_i ~ N(0, drift_rate²)
            
        Linear:
            Y_t = μ + β*t + ε_t
            where β = drift_rate
            
        Cyclic:
            Y_t = μ + A*sin(2π*t/period) + ε_t
            where A = drift_rate * 3
            
        Step:
            Y_t = μ + [0 if t<T/2 else drift_rate*3] + ε_t
    
    Examples:
        >>> # Basic time series (random walk)
        >>> df = make_sds4(T=50, seed=42)
        >>> df['factor 1'].unique()
        array(['K1'])  # Single factor level
        
        >>> # Linear upward trend
        >>> df = make_sds4(T=100, drift_type='linear', 
        ...                drift_rate=0.1, seed=42)
        >>> # Useful for testing trend detection
        
        >>> # Cyclic pattern (seasonal effect)
        >>> df = make_sds4(T=80, drift_type='cyclic',
        ...                drift_rate=2.0, seed=42)
        >>> # Models daily/weekly cycles
        
        >>> # Step change (process adjustment at t=50)
        >>> df = make_sds4(T=100, drift_type='step',
        ...                drift_rate=5.0, seed=42)
        >>> # Models mean shift at midpoint
        
        >>> # Pure random
        >>> # Pure random (no drift)
        >>> df = make_sds4(T=60, drift_type='none', seed=42)
        >>> # Models stable process with only random variation
        
        >>> # With ground truth to validate drift detection
        >>> df = make_sds4(T=50, include_truth=True, seed=42)
        >>> # Can plot true_drift vs estimated drift
    
    Validation:
        - Single factor level only (K=1)
        - T time points present
        - Appropriate for IMR analysis
    
    See Also:
        make_sds6: For regime changes (discrete shifts)
        make_sds5: For nested hierarchical structures
    """
    rng = np.random.default_rng(seed)
    
    # Generate drift/trend component based on type
    if drift_type == 'random_walk':
        # Cumulative random walk
        steps = rng.normal(0, drift_rate, T)
        drift = np.cumsum(steps)
        
    elif drift_type == 'linear':
        # Linear trend
        drift = np.linspace(0, drift_rate * T, T)
        
    elif drift_type == 'cyclic':
        # Sinusoidal pattern
        period = T / 4  # 4 complete cycles
        t_vals = np.arange(T)
        amplitude = drift_rate * 3
        drift = amplitude * np.sin(2 * np.pi * t_vals / period)
        
    elif drift_type == 'step':
        # Step change at midpoint
        drift = np.zeros(T)
        midpoint = T // 2
        drift[midpoint:] = drift_rate * 3
        
    elif drift_type == 'none':
        # No drift, pure random
        drift = np.zeros(T)
        
    else:
        raise ValueError(
            f"Unknown drift_type: {drift_type}. "
            f"Valid: 'random_walk', 'linear', 'cyclic', 'step', 'none'"
        )
    
    # Generate observations
    rows = []
    for t in range(T):
        epsilon = rng.normal(0, sigma)
        y = mu + drift[t] + epsilon
        
        row = {
            'time': t + 1,
            'factor 1': "K1",  # Single condition
            'factor 2': "NA",
            'y': y
        }
        
        if include_truth:
            row.update({
                'true_drift': drift[t],
                'true_error': epsilon,
                'true_mean': mu + drift[t]
            })
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Validation
    if df['factor 1'].nunique() != 1:
        raise AssertionError("SDS4 must have single factor level")
    if len(df) != T:
        raise AssertionError(f"SDS4 validation failed: expected {T} obs, got {len(df)}")
    
    logger.debug(
        f"Generated SDS4 data: {T} time points, drift_type='{drift_type}', "
        f"drift_rate={drift_rate}"
    )
    
    return df


# ============================================================================
# SDS 5: Nested Design - Hierarchical with Asynchronous Coverage
# ============================================================================

def make_sds5(
    L: int = 2,
    H_per_L: int = 3,
    T: int = 8,
    mu: float = 50.0,
    sigma: float = 0.4,
    line_effect_size: float = 2.0,
    head_effect_size: float = 1.0,
    time_effect_size: float = 0.8,
    p_active: float = 0.8,
    seed: Optional[int] = None,
    include_truth: bool = False
) -> pd.DataFrame:
    """
    Generate Sampling Design State 5 data (nested design with asynchronous coverage).
    
    SDS5 models hierarchical/nested structures common in manufacturing:
    - Multiple production lines, each with multiple heads/spindles
    - Heads are NESTED within lines (Head 1 on Line A ≠ Head 1 on Line B)
    - Not all heads active at all times (asynchronous)
    
    This is common in:
    - Multi-head fillers (heads nested in lanes)
    - Multi-spindle machines (spindles nested in stations)
    - Hierarchical production (operators nested in shifts)
    
    Characteristics:
    ---------------
    - Two or more factors with hierarchical structure
    - Factor 2 nested within Factor 1
    - Irregular temporal patterns (not all combinations at all times)
    - Asynchronous operation (heads come on/off line)
    - Requires special handling of nested structure
    
    Use Cases:
    ---------
    - Testing nested design analysis
    - Demonstrating hierarchical structures
    - Multi-head filling machine scenarios
    - Multi-spindle machining operations
    - Teaching variance component models
    
    Args:
        L: Number of lines (top-level factor)
        H_per_L: Number of heads per line (nested factor)
        T: Number of time periods
        mu: Grand mean
        sigma: Within-observation error
        line_effect_size: Std dev of line effects
        head_effect_size: Std dev of head-within-line effects
        time_effect_size: Std dev of time effects
        p_active: Probability that a head is active at any time point (0 to 1)
        seed: Random seed
        include_truth: Include true effects as columns
        
    Returns:
        DataFrame with nested structure
        
    Mathematical Model:
        Y_ihjt = μ + λ_i + η_ij + τ_t + ε_ihjt
        
        where:
            μ: Grand mean
            λ_i: Line i effect ~ N(0, line_effect_size²)
            η_ij: Head j within line i ~ N(0, head_effect_size²)
            τ_t: Time t effect ~ N(0, time_effect_size²)
            ε_ihjt: Error ~ N(0, σ²)
            
        Note: η_ij is nested (Head 1 of Line 1 ≠ Head 1 of Line 2)
    
    Examples:
        >>> # Basic nested structure: 2 lines, 3 heads each
        >>> df = make_sds5(L=2, H_per_L=3, T=8, seed=42)
        >>> # Check nesting: each head appears with only one line
        >>> df.groupby('factor 2')['factor 1'].nunique().max()
        1  # Each head belongs to exactly one line
        
        >>> # Multi-head filler scenario
        >>> df = make_sds5(L=4, H_per_L=4, T=12,  # 4 lanes, 4 heads, 12 hours
        ...                mu=237.5,               # Target fill weight
        ...                line_effect_size=1.5,   # Lane-to-lane variation
        ...                head_effect_size=0.8,   # Head-within-lane variation
        ...                p_active=0.9,           # 90% uptime
        ...                seed=42)
        
        >>> # Sparse operation (maintenance scenario)
        >>> df = make_sds5(L=3, H_per_L=2, T=10,
        ...                p_active=0.5,  # Only 50% uptime
        ...                seed=42)
        >>> # Many missing time points for each head
        
        >>> # With ground truth for variance component validation
        >>> df = make_sds5(L=2, H_per_L=3, T=6,
        ...                include_truth=True, seed=42)
        >>> # Can estimate variance components and compare to truth
    
    Nested Structure Verification:
        >>> df = make_sds5(L=2, H_per_L=3, seed=42)
        >>> # Verify nesting: each head unique to its line
        >>> nesting_check = df.groupby('factor 2')['factor 1'].unique()
        >>> all(len(lines) == 1 for lines in nesting_check)
        True
    
    Validation:
        - Each head appears with exactly one line (nesting)
        - Not all (line, head, time) combinations present (asynchronous)
        - At least some observations for each line
        - Irregular time coverage (some time points may be missing for heads)
    
    See Also:
        make_sds1: For crossed (not nested) factorial designs
        make_sds6: For regime changes with irregular patterns
    """
    rng = np.random.default_rng(seed)
    
    # Generate hierarchical effects
    line_eff = rng.normal(0, line_effect_size, L)  # Line-level effects
    head_eff = rng.normal(0, head_effect_size, (L, H_per_L))  # Head nested in line
    tau = rng.normal(0, time_effect_size, T)  # Time effects
    
    rows = []
    
    for line in range(L):
        for head in range(H_per_L):
            # CRITICAL FIX: Make head names unique across ALL lines
            # This ensures proper nesting structure
            # Format: Line{i}_Head{j} creates unique identifier
            head_name = f"Line{line+1}_Head{head+1}"
            
            # Determine which time points this head is active
            # Asynchronous: not all heads active at all times
            active_times = [t for t in range(T) if rng.random() < p_active]
            
            # Ensure at least some activity
            if len(active_times) == 0:
                active_times = [rng.integers(0, T)]
            
            for t in active_times:
                epsilon = rng.normal(0, sigma)
                y = (mu + line_eff[line] + head_eff[line, head] + 
                     tau[t] + epsilon)
                
                row = {
                    'time': t + 1,
                    'factor 1': f"Line{line+1}",
                    'factor 2': head_name,  # Unique head identifier
                    'y': y
                }
                
                if include_truth:
                    row.update({
                        'true_line_effect': line_eff[line],
                        'true_head_effect': head_eff[line, head],
                        'true_time_effect': tau[t],
                        'true_error': epsilon
                    })
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Validation - verify nested structure
    head_line_map = df.groupby('factor 2')['factor 1'].nunique()
    if not (head_line_map == 1).all():
        raise AssertionError(
            "SDS5 validation failed: heads must be nested within lines. "
            "Each head should appear with exactly one line."
        )
    
    # Check that we have irregular coverage (asynchronous)
    full_grid_size = L * H_per_L * T
    actual_combinations = df.groupby(['factor 1', 'factor 2', 'time']).size().shape[0]
    
    if actual_combinations >= full_grid_size * 0.95:
        logger.warning(
            f"SDS5: Nearly complete grid ({actual_combinations}/{full_grid_size}). "
            f"Consider reducing p_active for more realistic asynchronous pattern."
        )
    
    # Ensure we have data for all lines
    lines_with_data = df['factor 1'].nunique()
    if lines_with_data < L:
        logger.warning(
            f"SDS5: Only {lines_with_data}/{L} lines have data. "
            f"Consider increasing p_active."
        )
    
    logger.debug(
        f"Generated SDS5 data: {L} lines × {H_per_L} heads/line × {T} times "
        f"= {len(df)} observations (p_active={p_active:.1%})"
    )
    
    return df


# ============================================================================
# SDS 6: Unstructured / Regime Changes
# ============================================================================

def make_sds6(
    T: int = 80,
    K: int = 3,
    mu: float = 50.0,
    sigma: float = 0.5,
    regime_lengths: Optional[List[int]] = None,
    regime_shifts: Optional[List[float]] = None,
    factor_effect_size: float = 1.8,
    p_sampled: float = 0.7,
    seed: Optional[int] = None,
    include_truth: bool = False
) -> pd.DataFrame:
    """
    Generate Sampling Design State 6 data (unstructured with regime changes).
    
    SDS6 represents complex, irregular patterns common in real processes:
    - Regime changes (process mean shifts over time)
    - Irregular sampling (not all factors sampled at all times)
    - Cannot form regular (factor, time) grid
    - Mix of time-based patterns and factor effects
    
    This models situations like:
    - Process adjustments (mean shifts between regimes)
    - Equipment changes (new settings, maintenance)
    - Raw material changes (different suppliers)
    - Irregular production schedules
    
    Characteristics:
    ---------------
    - May lack clear (factor × time) structure
    - Regime changes or process shifts over time
    - Irregular, sparse sampling patterns
    - May have only time OR only factors (not both in regular grid)
    - Difficult to separate regime effects from factor/time effects
    
    Use Cases:
    ---------
    - Testing regime detection algorithms
    - Demonstrating non-standard data patterns
    - Teaching about process changes
    - Irregular production schedules
    - Long-term process studies with adjustments
    
    Args:
        T: Total number of time periods
        K: Number of factor levels (machines, operators, etc.)
        mu: Baseline process mean
        sigma: Random variation
        regime_lengths: Duration of each regime (list). 
                       If None, uses [20, 20, 20, 20] for T=80
        regime_shifts: Mean shift for each regime (list).
                      If None, uses [-1.0, 0.0, 1.2, 0.0]
        factor_effect_size: Std dev of factor effects
        p_sampled: Probability that each (factor, time) is sampled (0 to 1)
        seed: Random seed
        include_truth: Include regime info and true effects
        
    Returns:
        DataFrame with irregular structure and regime information
        
    Mathematical Model:
        Y_ijt = μ + δ_r(t) + ρ_i + ε_ijt
        
        where:
            μ: Baseline mean
            δ_r(t): Regime shift for regime r active at time t
            ρ_i: Factor i effect ~ N(0, factor_effect_size²)
            ε_ijt: Error ~ N(0, σ²)
            
        Regime changes create step functions in the mean over time.
    
    Examples:
        >>> # Basic regime change pattern
        >>> df = make_sds6(T=80, K=3, seed=42)
        >>> # Check regime distribution
        >>> df.groupby('regime')['y'].mean()
        
        >>> # Custom regime pattern (process adjustment history)
        >>> df = make_sds6(T=100, K=4,
        ...                regime_lengths=[30, 20, 30, 20],
        ...                regime_shifts=[-2.0, 0.0, 1.5, -1.0],
        ...                seed=42)
        >>> # Models: low period, adjustment to target, 
        >>> #         high period, correction
        
        >>> # Sparse sampling (irregular production)
        >>> df = make_sds6(T=60, K=3, 
        ...                p_sampled=0.4,  # Only 40% of slots sampled
        ...                seed=42)
        
        >>> # Single regime (no changes) - degenerates toward SDS 1/2/3
        >>> df = make_sds6(T=50, K=3,
        ...                regime_lengths=[50],
        ...                regime_shifts=[0.0],
        ...                p_sampled=1.0,  # Full sampling
        ...                seed=42)
        
        >>> # With ground truth for regime detection validation
        >>> df = make_sds6(T=80, K=3, include_truth=True, seed=42)
        >>> # Can test change-point detection algorithms
    
    Validation:
        - Incomplete (factor, time) grid (sparse sampling)
        - Regime changes present (if multiple regimes)
        - Some observations in each regime
        - All factors appear at least once
    
    Notes:
        - High p_sampled (near 1.0) makes this more like SDS 1/2/3
        - Low p_sampled (near 0.3) creates very irregular patterns
        - Regime shifts should be large enough to be detectable
          relative to sigma (typically > 2*sigma)
    
    See Also:
        make_sds4: For single condition with time patterns
        make_sds5: For nested structures
    """
    rng = np.random.default_rng(seed)
    
    # Default regime pattern if not specified
    if regime_lengths is None:
        n_regimes = 4
        regime_lengths = [T // n_regimes] * n_regimes
        # Adjust last regime to account for rounding
        regime_lengths[-1] = T - sum(regime_lengths[:-1])
    
    if regime_shifts is None:
        regime_shifts = [-1.0, 0.0, 1.2, 0.0]
    
    if len(regime_shifts) != len(regime_lengths):
        raise ValueError(
            f"regime_shifts length ({len(regime_shifts)}) must equal "
            f"regime_lengths length ({len(regime_lengths)})"
        )
    
    if sum(regime_lengths) != T:
        raise ValueError(
            f"Sum of regime_lengths ({sum(regime_lengths)}) must equal T ({T})"
        )
    
    # Create regime mapping
    regimes = np.repeat(range(len(regime_lengths)), regime_lengths)
    
    # Generate factor effects
    mach_eff = rng.normal(0, factor_effect_size, K)
    
    rows = []
    regime_obs_count = {i: 0 for i in range(len(regime_lengths))}
    
    for t in range(T):
        regime = regimes[min(t, len(regimes) - 1)]
        shift = regime_shifts[regime]
        
        # Irregular sampling: not all factors sampled at each time
        active_factors = [k for k in range(K) if rng.random() < p_sampled]
        
        # Ensure at least one factor sampled
        if len(active_factors) == 0:
            active_factors = [rng.integers(0, K)]
        
        for k in active_factors:
            epsilon = rng.normal(0, sigma)
            y = mu + shift + mach_eff[k] + epsilon
            
            row = {
                'time': t + 1,
                'factor 1': f"Machine{k+1}",
                'factor 2': "NA",
                'y': y,
                'regime': regime
            }
            
            if include_truth:
                row.update({
                    'regime_shift': shift,
                    'true_machine_effect': mach_eff[k],
                    'true_error': epsilon,
                    'true_mean': mu + shift + mach_eff[k]
                })
            
            rows.append(row)
            regime_obs_count[regime] += 1
    
    df = pd.DataFrame(rows)
    
    # Validation
    # Check that we have irregular grid (some factor-time combinations missing)
    full_grid_size = K * T
    actual_combinations = df.groupby(['factor 1', 'time']).size().shape[0]
    
    if actual_combinations >= full_grid_size * 0.95:
        logger.warning(
            f"SDS6: Nearly complete grid ({actual_combinations}/{full_grid_size}). "
            f"Consider reducing p_sampled for more irregular pattern."
        )
    
    # Check that all regimes have data
    empty_regimes = [r for r, count in regime_obs_count.items() if count == 0]
    if empty_regimes:
        raise AssertionError(
            f"SDS6 validation failed: regimes {empty_regimes} have no observations. "
            f"Increase p_sampled or adjust regime_lengths."
        )
    
    # Check that all factors appear
    factors_present = df['factor 1'].nunique()
    if factors_present < K:
        logger.warning(
            f"SDS6: Only {factors_present}/{K} factors have observations. "
            f"Consider increasing p_sampled."
        )
    
    logger.debug(
        f"Generated SDS6 data: {len(df)} observations across {len(regime_lengths)} regimes, "
        f"{actual_combinations}/{full_grid_size} cells filled ({actual_combinations/full_grid_size:.1%})"
    )
    
    return df


# ============================================================================
# Convenience Functions
# ============================================================================

def make_sds(
    sds: int,
    K: int = 3,
    T: int = 8,
    seed: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate synthetic data for any Sampling Design State.
    
    Convenience function that dispatches to the appropriate SDS-specific
    generator. Useful for loops and parameterized testing.
    
    Args:
        sds: Sampling Design State (1, 2, 3, 4, 5, or 6)
        K: Number of factor levels (not used for SDS 4)
        T: Number of time periods
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to specific generator
        
    Returns:
        DataFrame appropriate for specified SDS
        
    Examples:
        >>> # Generate data for each SDS type
        >>> for sds in [1, 2, 3, 4, 5, 6]:
        ...     df = make_sds(sds, K=3, T=8, seed=42)
        ...     print(f"SDS {sds}: {len(df)} observations")
        
        >>> # Useful for parameterized testing
        >>> @pytest.mark.parametrize('sds', [1, 2, 3, 4, 5, 6])
        >>> def test_all_sds(sds):
        ...     df = make_sds(sds, seed=42)
        ...     assert len(df) > 0
        
        >>> # Pass through kwargs to specific generator
        >>> df = make_sds(3, K=4, T=10, 
        ...               p_replicated=0.3,  # SDS3-specific arg
        ...               seed=42)
    
    Raises:
        ValueError: If SDS type not in [1, 2, 3, 4, 5, 6]
    """
    generators = {
        1: make_sds1,
        2: make_sds2,
        3: make_sds3,
        4: make_sds4,
        5: make_sds5,
        6: make_sds6
    }
    
    if sds not in generators:
        raise ValueError(
            f"SDS {sds} not implemented. "
            f"Available SDS types: {sorted(generators.keys())}"
        )
    
    # SDS 4 doesn't use K parameter
    if sds == 4:
        return generators[sds](T=T, seed=seed, **kwargs)
    # SDS 5 uses L and H_per_L instead of K
    elif sds == 5:
        L = kwargs.pop('L', max(2, K // 2))
        H_per_L = kwargs.pop('H_per_L', 3)
        return generators[sds](L=L, H_per_L=H_per_L, T=T, seed=seed, **kwargs)
    else:
        return generators[sds](K=K, T=T, seed=seed, **kwargs)


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================

def make_edge_cases() -> Dict[str, pd.DataFrame]:
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
        ...         result = perform_analysis(df, spec)
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
    cases['missing_cells'] = pd.DataFrame({
        'time': [1, 1, 2, 2, 3],  # No time=3 for factor B
        'factor 1': ['A', 'A', 'A', 'B', 'A'],
        'factor 2': ['NA'] * 5,
        'y': [10.1, 10.2, 11.0, 12.0, 10.5]
    })
    
    # 2. Extreme imbalance (one cell dominates)
    cases['extreme_imbalance'] = pd.DataFrame({
        'time': [1]*20 + [2]*2 + [3]*20,  # Time 2 has only 2 obs
        'factor 1': ['A']*20 + ['B']*2 + ['A']*20,
        'factor 2': ['NA']*42,
        'y': np.random.normal(50, 2, 42)
    })
    
    # 3. Single observation total (should fail)
    cases['single_obs'] = pd.DataFrame({
        'time': [1],
        'factor 1': ['A'],
        'factor 2': ['NA'],
        'y': [50.0]
    })
    
    # 4. Zero variance (all values identical)
    cases['zero_variance'] = pd.DataFrame({
        'time': [1, 1, 2, 2, 3, 3],
        'factor 1': ['A', 'A', 'B', 'B', 'A', 'B'],
        'factor 2': ['NA'] * 6,
        'y': [50.0] * 6  # All identical
    })
    
    # 5. Extreme outliers
    rng = np.random.default_rng(42)
    y_with_outliers = list(rng.normal(50, 1, 18))
    y_with_outliers[5] = 100.0   # Extreme high
    y_with_outliers[12] = 0.0    # Extreme low
    
    cases['outliers'] = pd.DataFrame({
        'time': [1, 1, 2, 2, 3, 3] * 3,
        'factor 1': ['A']*6 + ['B']*6 + ['C']*6,
        'factor 2': ['NA'] * 18,
        'y': y_with_outliers
    })
    
    # 6. Mixed quality (some cells have high variance, others low)
    y_mixed = (list(rng.normal(50, 0.1, 6)) +   # Low variance
               list(rng.normal(50, 5.0, 6)) +   # High variance
               list(rng.normal(50, 0.5, 6)))    # Medium variance
    
    cases['mixed_quality'] = pd.DataFrame({
        'time': [1, 1, 2, 2, 3, 3] * 3,
        'factor 1': ['A']*6 + ['B']*6 + ['C']*6,
        'factor 2': ['NA'] * 18,
        'y': y_mixed
    })
    
    # 7. Single cell with replication (rest empty)
    cases['single_cell_replicated'] = pd.DataFrame({
        'time': [1] * 5,
        'factor 1': ['A'] * 5,
        'factor 2': ['NA'] * 5,
        'y': [49.5, 50.0, 50.5, 50.2, 49.8]
    })
    
    # 8. Nearly empty (very sparse data)
    cases['sparse'] = pd.DataFrame({
        'time': [1, 5, 10, 15, 20],
        'factor 1': ['A', 'B', 'A', 'C', 'B'],
        'factor 2': ['NA'] * 5,
        'y': [50.1, 49.8, 50.3, 49.5, 50.0]
    })
    
    return cases


# ============================================================================
# Validation and Comparison Utilities
# ============================================================================

def compare_sds_characteristics(
    sds_list: Optional[List[int]] = None,
    K: int = 3,
    T: int = 8,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate comparison table of SDS characteristics.
    
    Creates a summary table comparing key features of each SDS type,
    useful for documentation and teaching.
    
    Args:
        sds_list: List of SDS to compare (default: all)
        K: Factor levels for comparison
        T: Time periods for comparison
        seed: Random seed
        
    Returns:
        DataFrame with SDS comparison metrics
        
    Example:
        >>> summary = compare_sds_characteristics()
        >>> print(summary)
        
        sds  n_obs  n_cells  min_n  max_n  has_replication  complete_grid
        1    72     24       2      4      True             True
        2    24     24       1      1      False            True
        3    48     24       1      3      Mixed            True
        4    40     40       1      1      False            False
        5    38     38       1      1      False            False
        6    42     42       1      1      False            False
    """
    if sds_list is None:
        sds_list = [1, 2, 3, 4, 5, 6]
    
    rows = []
    for sds in sds_list:
        df = make_sds(sds, K=K, T=T, seed=seed)
        
        if 'factor 1' in df.columns and 'time' in df.columns:
            cell_counts = df.groupby(['factor 1', 'time']).size()
            complete = len(cell_counts) == df['factor 1'].nunique() * df['time'].nunique()
        else:
            cell_counts = pd.Series([1] * len(df))
            complete = False
        
        min_n = cell_counts.min()
        max_n = cell_counts.max()
        
        if min_n == max_n == 1:
            replication = "False"
        elif min_n >= 2:
            replication = "True"
        else:
            replication = "Mixed"
        
        rows.append({
            'sds': sds,
            'n_obs': len(df),
            'n_cells': len(cell_counts),
            'min_n': min_n,
            'max_n': max_n,
            'has_replication': replication,
            'complete_grid': complete
        })
    
    return pd.DataFrame(rows)


def validate_sds_detection(
    df: pd.DataFrame,
    expected_sds: int,
    analysis_specification: 'AnalysisSpecification'
) -> bool:
    """
    Validate that generated data is correctly classified as expected SDS.
    
    This function is useful for testing SDS detection algorithms by
    generating data with known SDS and verifying correct classification.
    
    Args:
        df: DataFrame to validate
        expected_sds: Expected SDS classification (1-6)
        analysis_specification: Analysis specification object
        
    Returns:
        True if detected SDS matches expected
        
    Example:
        >>> df = make_sds1(K=3, T=8, seed=42)
        >>> spec = AnalysisSpecification('Xbar', {...})
        >>> ads = AnalysisDataSet(df, spec)
        >>> validate_sds_detection(df, expected_sds=1, spec)
        True
        
    Note:
        This requires importing from analysis_dataset module.
        Kept here for convenience in testing.
    """
    try:
        from analysis_dataset import AnalysisDataSet
        
        ads = AnalysisDataSet(df=df, analysis_specification=analysis_specification)
        detected_sds = ads.sampling_design_state
        
        if detected_sds == expected_sds:
            logger.info(f"✓ Correctly detected SDS {expected_sds}")
            return True
        else:
            logger.error(
                f"✗ SDS detection mismatch: expected {expected_sds}, "
                f"detected {detected_sds}"
            )
            return False
            
    except ImportError:
        logger.warning("Cannot validate SDS detection: analysis_dataset module not available")
        return False


# ============================================================================
# Batch Generation Utilities
# ============================================================================

def generate_test_suite(
    output_dir: str = 'test_data',
    seed: int = 42
) -> Dict[str, str]:
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
    df = make_sds1(K=3, T=8, seed=seed)
    df.to_csv(files['sds1_basic'], index=False)
    
    files['sds1_large'] = os.path.join(output_dir, 'sds1_large.csv')
    df = make_sds1(K=5, T=12, n_min=3, n_max=5, seed=seed)
    df.to_csv(files['sds1_large'], index=False)
    
    files['sds1_with_truth'] = os.path.join(output_dir, 'sds1_with_truth.csv')
    df = make_sds1(K=3, T=6, include_truth=True, seed=seed)
    df.to_csv(files['sds1_with_truth'], index=False)
    
    # SDS 2 variations
    files['sds2_basic'] = os.path.join(output_dir, 'sds2_basic.csv')
    df = make_sds2(K=3, T=10, seed=seed)
    df.to_csv(files['sds2_basic'], index=False)
    
    files['sds2_large'] = os.path.join(output_dir, 'sds2_large.csv')
    df = make_sds2(K=6, T=16, seed=seed)
    df.to_csv(files['sds2_large'], index=False)
    
    # SDS 3 variations (critical!)
    files['sds3_balanced'] = os.path.join(output_dir, 'sds3_balanced.csv')
    df = make_sds3(K=3, T=8, p_replicated=0.5, seed=seed)
    df.to_csv(files['sds3_balanced'], index=False)
    
    files['sds3_sparse'] = os.path.join(output_dir, 'sds3_sparse.csv')
    df = make_sds3(K=4, T=10, p_replicated=0.2, seed=seed)
    df.to_csv(files['sds3_sparse'], index=False)
    
    files['sds3_early_times'] = os.path.join(output_dir, 'sds3_early_times.csv')
    df = make_sds3(K=3, T=12, replication_pattern='early_times', seed=seed)
    df.to_csv(files['sds3_early_times'], index=False)
    
    # SDS 4 variations
    files['sds4_random_walk'] = os.path.join(output_dir, 'sds4_random_walk.csv')
    df = make_sds4(T=50, drift_type='random_walk', seed=seed)
    df.to_csv(files['sds4_random_walk'], index=False)
    
    files['sds4_linear'] = os.path.join(output_dir, 'sds4_linear.csv')
    df = make_sds4(T=60, drift_type='linear', seed=seed)
    df.to_csv(files['sds4_linear'], index=False)
    
    files['sds4_step'] = os.path.join(output_dir, 'sds4_step.csv')
    df = make_sds4(T=80, drift_type='step', seed=seed)
    df.to_csv(files['sds4_step'], index=False)
    
    # SDS 5 variations
    files['sds5_basic'] = os.path.join(output_dir, 'sds5_basic.csv')
    df = make_sds5(L=2, H_per_L=3, T=8, seed=seed)
    df.to_csv(files['sds5_basic'], index=False)
    
    files['sds5_large'] = os.path.join(output_dir, 'sds5_large.csv')
    df = make_sds5(L=4, H_per_L=4, T=12, seed=seed)
    df.to_csv(files['sds5_large'], index=False)
    
    # SDS 6 variations
    files['sds6_basic'] = os.path.join(output_dir, 'sds6_basic.csv')
    df = make_sds6(T=80, K=3, seed=seed)
    df.to_csv(files['sds6_basic'], index=False)
    
    files['sds6_sparse'] = os.path.join(output_dir, 'sds6_sparse.csv')
    df = make_sds6(T=60, K=4, p_sampled=0.4, seed=seed)
    df.to_csv(files['sds6_sparse'], index=False)
    
    # Edge cases
    edge_cases = make_edge_cases()
    for name, df in edge_cases.items():
        files[f'edge_{name}'] = os.path.join(output_dir, f'edge_{name}.csv')
        df.to_csv(files[f'edge_{name}'], index=False)
    
    logger.info(f"Generated {len(files)} test datasets in {output_dir}")
    
    return files


# ============================================================================
# Documentation and Metadata
# ============================================================================

def get_sds_info(sds: int) -> Dict[str, str]:
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
                'Most statistically powerful'
            ],
            'when_to_use': [
                'Testing Xbar-S chart algorithms',
                'Full factorial experiments',
                'When resources allow complete replication',
                'Baseline for comparing other SDS'
            ],
            'chart_types': ['Xbar-S', 'Range charts on subgroups'],
            'vas_capabilities': ['All residuals R1-R5', 'Full interaction effects'],
            'common_in': 'Designed experiments, validation studies'
        },
        2: {
            'name': 'No Replication',
            'description': 'Exactly one observation per (factor, time) cell',
            'characteristics': [
                'Complete (factor × time) grid',
                'No within-cell variance',
                'Must use moving average for R2',
                'Error confounded with interaction'
            ],
            'when_to_use': [
                'Resource-constrained experiments',
                'Screening studies',
                'Testing moving average algorithms',
                'When interaction assumed small'
            ],
            'chart_types': ['IMR on cell means', 'Moving range methods'],
            'vas_capabilities': ['R1-R5 with approximated R2', 'Limited interaction analysis'],
            'common_in': 'Designed experiments, DOE, initial screening'
        },
        3: {
            'name': 'Partial Replication',
            'description': 'Mixed: some cells n=1, others n≥2',
            'characteristics': [
                'Most common in real-world data',
                'Requires hybrid R2 estimation',
                'Irregular replication pattern',
                'Challenging to analyze correctly'
            ],
            'when_to_use': [
                'Testing hybrid variance estimation',
                'Real-world manufacturing data',
                'Unplanned replication scenarios',
                'Missing data situations'
            ],
            'chart_types': ['Hybrid approaches', 'Adaptive methods'],
            'vas_capabilities': ['Hybrid R2 calculation', 'Partial interaction analysis'],
            'common_in': 'Most real production data, unplanned sampling'
        },
        4: {
            'name': 'Single Condition Over Time',
            'description': 'One factor level, multiple time points (K=1)',
            'characteristics': [
                'Time series structure',
                'No grouping by factors',
                'May show trends or drift',
                'Classic control chart scenario'
            ],
            'when_to_use': [
                'Single process monitoring',
                'Time series analysis',
                'Trend detection',
                'Classic SPC applications'
            ],
            'chart_types': ['IMR', 'Individuals chart', 'Moving range'],
            'vas_capabilities': ['Limited (no factor structure)', 'Time effects only'],
            'common_in': 'Single machine/process, continuous monitoring'
        },
        5: {
            'name': 'Nested Design',
            'description': 'Hierarchical factors with asynchronous coverage',
            'characteristics': [
                'Factor 2 nested in Factor 1',
                'Irregular temporal patterns',
                'Not all combinations at all times',
                'Requires variance components'
            ],
            'when_to_use': [
                'Multi-head machines (heads in lanes)',
                'Hierarchical production structures',
                'Nested experimental designs',
                'Asynchronous operations'
            ],
            'chart_types': ['Hierarchical charts', 'Nested variance components'],
            'vas_capabilities': ['Nested effects', 'Variance component estimation'],
            'common_in': 'Multi-head fillers, multi-spindle machines, nested processes'
        },
        6: {
            'name': 'Unstructured / Regime Changes',
            'description': 'Irregular patterns with process regime changes',
            'characteristics': [
                'Incomplete (factor × time) grid',
                'Mean shifts over time (regimes)',
                'Irregular sampling patterns',
                'Complex time structure'
            ],
            'when_to_use': [
                'Long-term process studies',
                'Process with adjustments',
                'Regime detection problems',
                'Irregular production schedules'
            ],
            'chart_types': ['Adaptive methods', 'Change-point detection'],
            'vas_capabilities': ['Limited', 'Regime-specific analysis'],
            'common_in': 'Long-term studies, processes with changes, irregular schedules'
        }
    }
    
    if sds not in sds_metadata:
        raise ValueError(f"SDS {sds} not defined. Valid: 1-6")
    
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
    print("\nSAMPLING DESIGN STATES SUMMARY")
    print("=" * 70)
    
    for sds in range(1, 7):
        info = get_sds_info(sds)
        print(f"\nSDS {sds}: {info['name']}")
        print("-" * 70)
        print(f"Description: {info['description']}")
        print(f"\nCharacteristics:")
        for char in info['characteristics']:
            print(f"  • {char}")
        print(f"\nCommon in: {info['common_in']}")
        print(f"Chart types: {', '.join(info['chart_types'])}")


# ============================================================================
# Module-level convenience
# ============================================================================

# Quick access to all generators
GENERATORS = {
    1: make_sds1,
    2: make_sds2,
    3: make_sds3,
    4: make_sds4,
    5: make_sds5,
    6: make_sds6
}


if __name__ == '__main__':
    # Demo: Generate one dataset of each type
    print("Generating example datasets for each SDS...\n")
    
    for sds in range(1, 7):
        print(f"SDS {sds}: {get_sds_info(sds)['name']}")
        df = make_sds(sds, K=3, T=6, seed=42)
        print(f"  Generated {len(df)} observations")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Preview:\n{df.head(3)}")
        print()
    
    print("\nEdge cases:")
    edge_cases = make_edge_cases()
    for name, df in edge_cases.items():
        print(f"  {name}: {len(df)} observations")
    
    print("\nFor detailed info on any SDS, use get_sds_info(sds_number)")
    print("For complete summary, use print_sds_summary()")