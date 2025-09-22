"""
Control limits calculations for SPC analysis.

This module contains functions for calculating control limits and statistical
constants used in process behavior charts.
"""

import math

import pandas as pd
import scipy.special


def c4(n: int) -> float:
    """Calculate bias constant for Xbar and S charts.

    Args:
        n: Sample size

    Returns:
        c4 constant value
    """
    return math.sqrt(2 / (n - 1)) * (
        math.exp(scipy.special.loggamma(n / 2) - scipy.special.loggamma((n - 1) / 2))
    )


def b3(n: int) -> float:
    """Calculate B3 constant for S chart lower control limit.

    Args:
        n: Sample size

    Returns:
        B3 constant value (0 if calculated value is negative)
    """
    out = 1 - (3 / c4(n) * math.sqrt(1 - math.pow(c4(n), 2)))
    return 0 if out < 0 else out


def b4(n: int) -> float:
    """Calculate B4 constant for S chart upper control limit.

    Args:
        n: Sample size

    Returns:
        B4 constant value
    """
    return 1 + (3 / c4(n) * math.sqrt(1 - math.pow(c4(n), 2)))


def calculate_limits(
    limits_type: str,
    mean: float = None,
    sd: float = None,
    N: int = None,
    mR: float = None,
    round_to: int = 3,
) -> pd.Series:
    """Calculate control limits for different chart types.

    Args:
        limits_type: Type of control chart ("Xbar", "S", "Imr", "R")
        mean: Process mean (required for Xbar and Imr charts)
        sd: Standard deviation (required for Xbar and S charts)
        N: Sample size (required for Xbar and S charts)
        mR: Moving range (required for Imr and R charts)
        round_to: Number of decimal places to round to

    Returns:
        Series with 'lcl' and 'ucl' values

    Raises:
        ValueError: If required parameters are missing or limits_type is unsupported
    """
    if limits_type == 'Xbar':
        if None in [sd, mean, N]:
            raise ValueError(
                f'The limits calculation for {limits_type}: requires (mean, sd, and N)'
            )

        # Wd = S / c4n
        Wd = sd / c4(N)
        # Studentized Control Limits for sub-group means
        # LCLx = Xbar - (3 * Wd) / sqrt(n)
        lcl = mean + (-1 * ((3 * Wd) / math.sqrt(N)))
        ucl = mean + ((3 * Wd) / math.sqrt(N))

    elif limits_type == 'S':
        if None in [sd, N]:
            raise ValueError(f'The limits calculation for {limits_type}: requires (sd, and N)')

        # lcl - use B3 - S*B3(N)
        lcl = sd * b3(N)
        # ucl - use B4
        ucl = sd * b4(N)

    elif limits_type == 'Imr':
        if None in [mean, mR]:
            raise ValueError(f'The limits calculation for {limits_type}: requires (mean, and mR)')

        lcl = mean + (-1.0 * (2.66 * mR))
        ucl = mean + (2.66 * mR)

    elif limits_type == 'R':
        if mR is None:
            raise ValueError(f'The limits calculation for {limits_type}: requires (mR)')

        lcl = 0
        ucl = mR * 3.268

    else:
        raise ValueError(
            f'The limits type: {limits_type}: is not supported- provide (Xbar, S, Imr, or R)'
        )

    return pd.Series({'lcl': lcl, 'ucl': ucl}, index=['lcl', 'ucl'])


def detect_beyond_limits(x: float, lcl: float, ucl: float) -> int:
    """Detect if a point is beyond control limits.

    Args:
        x: Value to check
        lcl: Lower control limit
        ucl: Upper control limit

    Returns:
        -1 if below LCL, 1 if above UCL, 0 if within limits
    """
    if x < lcl:
        return -1
    elif x > ucl:
        return 1
    else:
        return 0
