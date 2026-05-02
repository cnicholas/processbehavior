"""
Statistical Process Control constants and formulas.

This module contains domain knowledge for SPC (Statistical Process Control):
- Control chart constants (c4, b3, b4, d2, d3)
- Control limit calculations for various chart types
- Signal detection rules

References
----------
- Wheeler, D. J. (1995). Advanced Topics in Statistical Process Control
- ISO 7870-2:2013 Control charts - Part 2: Shewhart control charts
- Montgomery, D. C. (2009). Statistical Quality Control
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ============================================================================
# Statistical Control Chart Constants
# ============================================================================

# Control limit multiplier (3-sigma limits are standard in SPC)
SIGMA_MULTIPLIER = 3

# E2 constant for XmR charts (n=2, moving range of 2 consecutive observations)
# Used for calculating control limits on individual values
# E2 = d2 / d3 for n=2, where d2 = 1.128 and d3 = 0.8525
XMR_LIMIT_MULTIPLIER = 2.66

# D4 constant for R charts (n=2, range of 2 consecutive observations)
# Used for upper control limit on moving range
# D4 = 1 + 3(d3/d2) for n=2
R_UPPER_LIMIT_MULTIPLIER = 3.268


# ============================================================================
# Bias Correction Constants
# ============================================================================

def c4(n: int) -> float:
    """
    Calculate c4 bias constant for Xbar and S charts.

    The c4 constant corrects for bias in the standard deviation estimate
    when using subgroups. It approaches 1.0 as n increases.

    Parameters
    ----------
    n : int
        Subgroup size (must be >= 2)

    Returns
    -------
    float
        c4 constant for given subgroup size

    Examples
    --------
    >>> c4(2)
    0.7978845608028654
    >>> c4(5)
    0.9399856029866252
    >>> c4(25)
    0.9896050420253648

    Notes
    -----
    Formula: c4(n) = sqrt(2/(n-1)) * Γ(n/2) / Γ((n-1)/2)

    For small samples (n < 2), the standard deviation is heavily biased.
    The c4 constant corrects this bias.

    References
    ----------
    Wheeler (1995), Advanced Topics in SPC, Chapter 3
    """
    if n < 2:
        raise ValueError(f"Subgroup size must be >= 2, got {n}")

    out = np.sqrt(2 / (n - 1)) * (
        np.exp(math.lgamma(n / 2) - math.lgamma((n - 1) / 2))
    )
    return out


def b3(n: int, sigma_multiplier: float = 3) -> float:
    """
    Calculate b3 lower control limit constant for S charts.

    The b3 constant is used to calculate the lower process limit (LPL)
    for the standard deviation (S) chart.

    Parameters
    ----------
    n : int
        Subgroup size (must be >= 2)
    sigma_multiplier : float, default 3
        Sigma multiplier for control limits (default 3-sigma)

    Returns
    -------
    float
        b3 constant for given subgroup size. Returns 0 if calculated value < 0.

    Examples
    --------
    >>> b3(2)
    0
    >>> b3(5)
    0
    >>> b3(10)
    0.284...

    Notes
    -----
    Formula: b3(n) = 1 - sigma_multiplier/c4(n) * sqrt(1 - c4(n)²)

    For small subgroup sizes (n < 6), the raw b3 formula yields negative
    values. Since standard deviations cannot be negative, these values are
    clamped to 0.

    References
    ----------
    Wheeler (1995), Advanced Topics in SPC, Chapter 3
    """
    if n < 2:
        raise ValueError(f"Subgroup size must be >= 2, got {n}")

    c4_n = c4(n)
    out = 1 - (sigma_multiplier / c4_n * math.sqrt(1 - math.pow(c4_n, 2)))

    return 0 if out < 0 else out


def b4(n: int, sigma_multiplier: float = 3) -> float:
    """
    Calculate b4 upper control limit constant for S charts.

    The b4 constant is used to calculate the upper process limit (UPL)
    for the standard deviation (S) chart.

    Parameters
    ----------
    n : int
        Subgroup size (must be >= 2)
    sigma_multiplier : float, default 3
        Sigma multiplier for control limits (default 3-sigma)

    Returns
    -------
    float
        b4 constant for given subgroup size

    Examples
    --------
    >>> b4(2)
    3.2667284695806667
    >>> b4(5)
    2.088586704330583
    >>> b4(25)
    1.4355113780646384

    Notes
    -----
    Formula: b4(n) = 1 + sigma_multiplier/(c4(n)) * sqrt(1 - c4(n)^2)

    The b4 constant decreases as subgroup size increases, approaching the
    value of 1 + 3*sqrt(1-1) = 1 for very large subgroups.

    References
    ----------
    Wheeler (1995), Advanced Topics in SPC, Chapter 3
    """
    if n < 2:
        raise ValueError(f"Subgroup size must be >= 2, got {n}")

    c4_n = c4(n)
    out = 1 + (sigma_multiplier / c4_n * math.sqrt(1 - math.pow(c4_n, 2)))

    return out


# ============================================================================
# Control Limit Calculations
# ============================================================================

def calculate_limits(
    limits_type: str,
    mean: float = None,
    sd: float = None,
    N: int = None,
    mR: float = None,
    round_to: int = 3,
    sigma_multiplier: float = 3
) -> pd.Series:
    """
    Calculate control limits for various chart types.

    Supports Xbar, S, XmR (individuals/moving range), and R charts.

    Parameters
    ----------
    limits_type : str
        Type of control chart: 'Xbar', 'S', 'XmR', or 'R'
    mean : float, optional
        Grand mean or subgroup mean (required for Xbar, XmR)
    sd : float, optional
        Standard deviation (required for Xbar, S)
    N : int, optional
        Subgroup size (required for Xbar, S)
    mR : float, optional
        Mean moving range (required for XmR, R)
    round_to : int, default=3
        Number of decimal places for rounding (currently unused)

    Returns
    -------
    pd.Series
        Series with 'lpl' and 'upl' keys (Lower/Upper Process Limits)

    Examples
    --------
    Calculate Xbar chart limits:
    >>> calculate_limits(limits_type='Xbar', mean=10.0, sd=0.5, N=5)
    lpl    9.366...
    upl   10.633...
    dtype: float64

    Calculate S chart limits:
    >>> calculate_limits(limits_type='S', sd=0.5, N=5)
    lpl    0.0
    upl    1.044...
    dtype: float64

    Calculate XmR chart limits:
    >>> calculate_limits(limits_type='XmR', mean=10.0, mR=0.3)
    lpl    9.202
    upl   10.798
    dtype: float64

    Calculate R chart limits:
    >>> calculate_limits(limits_type='R', mR=0.3)
    lpl    0.000
    upl    0.980...
    dtype: float64

    Raises
    ------
    ValueError
        If required parameters are missing or limits_type is invalid

    Notes
    -----
    Xbar limits: X̄ ± (3 * Wd) / sqrt(n), where Wd = S / c4(n)
    S limits: S * b3(n) to S * b4(n)
    XmR limits: X̄ ± (E2 * mR), where E2 = 2.66
    R limits: 0 to mR * D4, where D4 = 3.268

    References
    ----------
    Wheeler (1995), Advanced Topics in SPC, Chapters 3-5
    """
    if limits_type == "Xbar":
        if None in [sd, mean, N]:
            raise ValueError(
                f"The limits calculation for {limits_type} requires (mean, sd, and N). "
                f"Got: mean={mean}, sd={sd}, N={N}"
            )

        # Wd = S / c4(n) - within-subgroup standard deviation
        Wd = sd / c4(N)

        # Studentized Process Limits for sub-group means
        # LPLx = X̄ - (sigma_multiplier * Wd) / √n
        # UPLx = X̄ + (sigma_multiplier * Wd) / √n
        lpl = mean - ((sigma_multiplier * Wd) / math.sqrt(N))
        upl = mean + ((sigma_multiplier * Wd) / math.sqrt(N))

    elif limits_type == "S":
        if None in [sd, N]:
            raise ValueError(
                f"The limits calculation for {limits_type} requires (sd, and N). "
                f"Got: sd={sd}, N={N}"
            )

        # LPL = S * b3(N)
        # UPL = S * b4(N)
        lpl = sd * b3(N, sigma_multiplier)
        upl = sd * b4(N, sigma_multiplier)

    elif limits_type == "XmR":
        if None in [mean, mR]:
            raise ValueError(
                f"The limits calculation for {limits_type} requires (mean, and mR). "
                f"Got: mean={mean}, mR={mR}"
            )

        # LPL = X̄ - (E2 * mR)
        # UPL = X̄ + (E2 * mR)
        lpl = mean - (XMR_LIMIT_MULTIPLIER * mR)
        upl = mean + (XMR_LIMIT_MULTIPLIER * mR)

    elif limits_type == "R":
        if mR is None:
            raise ValueError(
                f"The limits calculation for {limits_type} requires (mR). "
                f"Got: mR={mR}"
            )

        # LPL = 0 (ranges cannot be negative)
        # UPL = mR * D4
        lpl = 0
        upl = mR * R_UPPER_LIMIT_MULTIPLIER

    else:
        raise ValueError(
            f"The limits type '{limits_type}' is not supported. "
            f"Supported types: 'Xbar', 'S', 'XmR', 'R'"
        )

    return pd.Series({'lpl': lpl, 'upl': upl}, index=['lpl', 'upl'])


# ============================================================================
# Signal Detection
# ============================================================================

def detect_beyond_limits(x: float, lpl: float, upl: float) -> int:
    """
    Detect if a point is beyond process limits.

    This is the most basic signal detection rule (Rule 1 in Western Electric
    rules): a single point beyond the 3-sigma process limits.

    Parameters
    ----------
    x : float
        Value to check
    lpl : float
        Lower process limit
    upl : float
        Upper process limit

    Returns
    -------
    int
        -1 if below LPL, 1 if above UPL, 0 if within limits

    Examples
    --------
    >>> detect_beyond_limits(10.5, lpl=9.0, upl=11.0)
    0
    >>> detect_beyond_limits(8.5, lpl=9.0, upl=11.0)
    -1
    >>> detect_beyond_limits(11.5, lpl=9.0, upl=11.0)
    1

    Notes
    -----
    This implements "Test 1" from Western Electric rules:
    - One point beyond the 3-sigma process limits

    Additional tests (runs, trends, etc.) are not implemented here.

    References
    ----------
    Western Electric (1956). Statistical Quality Control Handbook
    Wheeler & Chambers (1992). Understanding Statistical Process Control
    """
    if x < lpl:
        return -1
    elif x > upl:
        return 1
    else:
        return 0


# ============================================================================
# Chart Name Constants
# ============================================================================

# Valid base chart types for syntactic validation.
# API uses focal-chart naming: 'X' (individual) and 'mR' (moving range).
# Use companion=True to get the paired chart (X↔mR).
VALID_BASE_CHARTS = {"Xbar", "S", "X", "mR", "Histogram"}

# Case-insensitive canonical chart name mapping
CHART_NAME_CANONICAL: dict[str, str] = {
    name.lower(): name for name in VALID_BASE_CHARTS
}

# Removed chart names with migration hints
_REMOVED_CHART_NAMES: dict[str, str] = {
    "xmr": "Use chart='X' for the individual chart, or chart='X' with companion=True for both X and mR.",
    "r": "Use chart='mR' for the moving range chart.",
}


def normalize_chart_name(name: str) -> str:
    """Normalize chart name to canonical form (case-insensitive).

    Returns the canonical name if recognized, otherwise returns input unchanged.
    This allows stratum names (e.g. 'Alice') to pass through unmodified.

    Raises
    ------
    ValueError
        If the name matches a removed chart name (e.g., 'XmR').
    """
    lower = name.lower()
    if lower in CHART_NAME_CANONICAL:
        return CHART_NAME_CANONICAL[lower]
    if lower in _REMOVED_CHART_NAMES:
        raise ValueError(
            f"Chart name '{name}' has been removed. "
            f"{_REMOVED_CHART_NAMES[lower]}"
        )
    return name

# Human-readable residual aliases
# Maps alias -> {id, label, description}
RESIDUAL_ALIASES = {
    "mean_removed": {
        "id": "R1",
        "label": "Mean Removed",
        "description": "Residual after removing grand mean"
    },
    "within_cell": {
        "id": "R2",
        "label": "Within Cell",
        "description": "Within-cell variation (per VAS definition)"
    },
    "structure_removed": {
        "id": "R3",
        "label": "Structure Removed",
        "description": "Residual after removing factor structure"
    },
    "time_structure_removed": {
        "id": "R4",
        "label": "Time Structure Removed",
        "description": "Residual after removing time structure"
    },
    "noise": {
        "id": "R5",
        "label": "Noise",
        "description": "Unexplained variation"
    },
}

# Reverse lookup: "R5" -> "noise"
RESIDUAL_ID_TO_ALIAS = {v["id"]: k for k, v in RESIDUAL_ALIASES.items()}
