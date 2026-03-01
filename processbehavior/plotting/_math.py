"""Pure Python normal distribution functions (scipy replacement)."""

from __future__ import annotations

import math


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Calculate normal probability density function.

    Pure Python implementation to avoid scipy dependency.

    Parameters
    ----------
    x : float
        Point at which to evaluate the PDF
    mu : float, default 0.0
        Mean of the distribution
    sigma : float, default 1.0
        Standard deviation of the distribution

    Returns
    -------
    float
        PDF value at x
    """
    coefficient = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * math.exp(exponent)


def normal_ppf(p: float) -> float:
    """
    Calculate normal percent point function (inverse CDF / quantile function).

    Uses Acklam's algorithm for rational approximation to the
    inverse cumulative normal distribution. Accurate to approximately
    1.15e-9 in absolute error for p in (0, 1).

    Parameters
    ----------
    p : float
        Probability value in (0, 1)

    Returns
    -------
    float
        Quantile (z-score) corresponding to probability p

    References
    ----------
    Acklam, P. J. (2010). An algorithm for computing the inverse normal
    cumulative distribution function.
    https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
    """
    # Coefficients for rational approximation
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    # Define break-points
    p_low = 0.02425
    p_high = 1 - p_low

    if p <= 0 or p >= 1:
        if p == 0:
            return float('-inf')
        elif p == 1:
            return float('inf')
        else:
            raise ValueError(f"p must be in (0, 1), got {p}")

    # Rational approximation for lower region
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    # Rational approximation for upper region
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    # Rational approximation for central region
    q = p - 0.5
    r = q * q
    return (
        ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]
    ) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
