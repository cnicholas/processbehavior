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
import scipy.special

# ============================================================================
# Statistical Control Chart Constants
# ============================================================================

# Control limit multiplier (3-sigma limits are standard in SPC)
SIGMA_MULTIPLIER = 3

# E2 constant for IMR charts (n=2, moving range of 2 consecutive observations)
# Used for calculating control limits on individual values
# E2 = d2 / d3 for n=2, where d2 = 1.128 and d3 = 0.8525
IMR_LIMIT_MULTIPLIER = 2.66

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
        np.exp(scipy.special.loggamma(n / 2) - scipy.special.loggamma((n - 1) / 2))
    )
    return out


def b3(n: int) -> float:
    """
    Calculate b3 lower control limit constant for S charts.

    The b3 constant is used to calculate the lower control limit (LCL)
    for the standard deviation (S) chart.

    Parameters
    ----------
    n : int
        Subgroup size (must be >= 2)

    Returns
    -------
    float
        b3 constant for given subgroup size (returns 0 if calculated value < 0)

    Examples
    --------
    >>> b3(2)
    0
    >>> b3(5)
    0
    >>> b3(6)
    0.029769698109093037

    Notes
    -----
    Formula: b3(n) = max(0, 1 - 3/(c4(n)) * sqrt(1 - c4(n)^2))

    For small subgroup sizes (n < 6), b3 is typically 0, meaning there is
    no lower control limit on the S chart.

    References
    ----------
    Wheeler (1995), Advanced Topics in SPC, Chapter 3
    """
    if n < 2:
        raise ValueError(f"Subgroup size must be >= 2, got {n}")

    c4_n = c4(n)
    out = 1 - (SIGMA_MULTIPLIER / c4_n * math.sqrt(1 - math.pow(c4_n, 2)))

    return 0 if out < 0 else out


def b4(n: int) -> float:
    """
    Calculate b4 upper control limit constant for S charts.

    The b4 constant is used to calculate the upper control limit (UCL)
    for the standard deviation (S) chart.

    Parameters
    ----------
    n : int
        Subgroup size (must be >= 2)

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
    Formula: b4(n) = 1 + 3/(c4(n)) * sqrt(1 - c4(n)^2)

    The b4 constant decreases as subgroup size increases, approaching the
    value of 1 + 3*sqrt(1-1) = 1 for very large subgroups.

    References
    ----------
    Wheeler (1995), Advanced Topics in SPC, Chapter 3
    """
    if n < 2:
        raise ValueError(f"Subgroup size must be >= 2, got {n}")

    c4_n = c4(n)
    out = 1 + (SIGMA_MULTIPLIER / c4_n * math.sqrt(1 - math.pow(c4_n, 2)))

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
    round_to: int = 3
) -> pd.Series:
    """
    Calculate control limits for various chart types.

    Supports Xbar, S, Imr (individuals/moving range), and R charts.

    Parameters
    ----------
    limits_type : str
        Type of control chart: 'Xbar', 'S', 'Imr', or 'R'
    mean : float, optional
        Grand mean or subgroup mean (required for Xbar, Imr)
    sd : float, optional
        Standard deviation (required for Xbar, S)
    N : int, optional
        Subgroup size (required for Xbar, S)
    mR : float, optional
        Mean moving range (required for Imr, R)
    round_to : int, default=3
        Number of decimal places for rounding (currently unused)

    Returns
    -------
    pd.Series
        Series with 'lcl' and 'ucl' keys

    Examples
    --------
    Calculate Xbar chart limits:
    >>> calculate_limits(limits_type='Xbar', mean=10.0, sd=0.5, N=5)
    lcl    9.366...
    ucl   10.633...
    dtype: float64

    Calculate S chart limits:
    >>> calculate_limits(limits_type='S', sd=0.5, N=5)
    lcl    0.0
    ucl    1.044...
    dtype: float64

    Calculate IMR chart limits:
    >>> calculate_limits(limits_type='Imr', mean=10.0, mR=0.3)
    lcl    9.202
    ucl   10.798
    dtype: float64

    Calculate R chart limits:
    >>> calculate_limits(limits_type='R', mR=0.3)
    lcl    0.000
    ucl    0.980...
    dtype: float64

    Raises
    ------
    ValueError
        If required parameters are missing or limits_type is invalid

    Notes
    -----
    Xbar limits: X̄ ± (3 * Wd) / sqrt(n), where Wd = S / c4(n)
    S limits: S * b3(n) to S * b4(n)
    IMR limits: X̄ ± (E2 * mR), where E2 = 2.66
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

        # Studentized Control Limits for sub-group means
        # LCLx = X̄ - (3 * Wd) / √n
        # UCLx = X̄ + (3 * Wd) / √n
        lcl = mean - ((SIGMA_MULTIPLIER * Wd) / math.sqrt(N))
        ucl = mean + ((SIGMA_MULTIPLIER * Wd) / math.sqrt(N))

    elif limits_type == "S":
        if None in [sd, N]:
            raise ValueError(
                f"The limits calculation for {limits_type} requires (sd, and N). "
                f"Got: sd={sd}, N={N}"
            )

        # LCL = S * b3(N)
        # UCL = S * b4(N)
        lcl = sd * b3(N)
        ucl = sd * b4(N)

    elif limits_type == "Imr":
        if None in [mean, mR]:
            raise ValueError(
                f"The limits calculation for {limits_type} requires (mean, and mR). "
                f"Got: mean={mean}, mR={mR}"
            )

        # LCL = X̄ - (E2 * mR)
        # UCL = X̄ + (E2 * mR)
        lcl = mean - (IMR_LIMIT_MULTIPLIER * mR)
        ucl = mean + (IMR_LIMIT_MULTIPLIER * mR)

    elif limits_type == "R":
        if mR is None:
            raise ValueError(
                f"The limits calculation for {limits_type} requires (mR). "
                f"Got: mR={mR}"
            )

        # LCL = 0 (ranges cannot be negative)
        # UCL = mR * D4
        lcl = 0
        ucl = mR * R_UPPER_LIMIT_MULTIPLIER

    else:
        raise ValueError(
            f"The limits type '{limits_type}' is not supported. "
            f"Supported types: 'Xbar', 'S', 'Imr', 'R'"
        )

    return pd.Series({'lcl': lcl, 'ucl': ucl}, index=['lcl', 'ucl'])


# ============================================================================
# Signal Detection
# ============================================================================

def detect_beyond_limits(x: float, lcl: float, ucl: float) -> int:
    """
    Detect if a point is beyond control limits.

    This is the most basic signal detection rule (Rule 1 in Western Electric
    rules): a single point beyond the 3-sigma control limits.

    Parameters
    ----------
    x : float
        Value to check
    lcl : float
        Lower control limit
    ucl : float
        Upper control limit

    Returns
    -------
    int
        -1 if below LCL, 1 if above UCL, 0 if within limits

    Examples
    --------
    >>> detect_beyond_limits(10.5, lcl=9.0, ucl=11.0)
    0
    >>> detect_beyond_limits(8.5, lcl=9.0, ucl=11.0)
    -1
    >>> detect_beyond_limits(11.5, lcl=9.0, ucl=11.0)
    1

    Notes
    -----
    This implements "Test 1" from Western Electric rules:
    - One point beyond the 3-sigma control limits

    Additional tests (runs, trends, etc.) are not implemented here.

    References
    ----------
    Western Electric (1956). Statistical Quality Control Handbook
    Wheeler & Chambers (1992). Understanding Statistical Process Control
    """
    if x < lcl:
        return -1
    elif x > ucl:
        return 1
    else:
        return 0
