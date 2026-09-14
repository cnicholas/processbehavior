"""
Series-length precision of the sigma estimate.

One fact for the design report: where the sigma behind the natural process
limits comes from at the observed structure, and how precise that estimate is
at the observed series length. No threshold, label, or warning is attached;
the analyst weighs it (Bishop: judgment belongs to the analyst, not to a rule).

Two sources of sigma:

- Replicated cells (ADS 1): within-cell deviation. Precision is a matter of the
  within-cell degrees of freedom, sum over cells of (N_kt - 1). The number of
  time periods is irrelevant; T = 1 is a complete study.
- Any singleton cell (ADS 2, ADS 3): the 2-point moving range over the ordered
  sequence, T - 1 ranges for T time points. Its precision at short T is
  tabulated in ``MR_SIGMA_INTERVAL_80`` (see #114).
"""

from __future__ import annotations

from dataclasses import dataclass

from .spc_constants import MR_SIGMA_INTERVAL_80

_TABLE_MAX = max(MR_SIGMA_INTERVAL_80)


@dataclass(frozen=True)
class SeriesLengthPrecision:
    """
    Where sigma comes from, and how precise it is, at the observed series length.

    Attributes
    ----------
    T : int or None
        Distinct time points on the analysis dataset; None when no time variable.
    n_moving_ranges : int or None
        T - 1 when the moving range carries sigma and T >= 2; else None.
    within_cell_df : int
        Sum over cells of (N_kt - 1); 0 when no cell is replicated.
    sigma_from_time : bool
        True when sigma rests on moving ranges (any singleton cell); False when
        it rests on within-cell replication.
    mr_interval_80 : tuple of (float, float) or None
        (p10, p90) of the moving-range sigma estimate relative to the truth on a
        stable process, from ``MR_SIGMA_INTERVAL_80``; only when
        ``sigma_from_time`` and 3 <= T <= 30.
    description : str
        The sentence printed in the design report. Empty when there is nothing
        to say (no time variable and no replication).
    """

    T: int | None
    n_moving_ranges: int | None
    within_cell_df: int
    sigma_from_time: bool
    mr_interval_80: tuple[float, float] | None
    description: str


def assess_series_length(T: int | None, within_cell_df: int, sigma_from_time: bool) -> SeriesLengthPrecision:
    """
    Build the precision statement for a study.

    Parameters
    ----------
    T : int or None
        Distinct time points on the analysis dataset (None if no time variable).
    within_cell_df : int
        Sum over cells of (N_kt - 1) on the analysis dataset.
    sigma_from_time : bool
        True when the limits rest on the moving range (any singleton cell).

    Returns
    -------
    SeriesLengthPrecision
    """
    if not sigma_from_time:
        n_mr = None
        interval = None
        if T is None:
            description = (
                f'Sigma rests on within-cell replication ({within_cell_df} degrees of freedom); no time sequence.'
            )
        else:
            description = (
                f'T={T}. Sigma rests on within-cell replication ({within_cell_df} degrees of freedom), '
                f'not on the time sequence.'
            )
        return SeriesLengthPrecision(T, n_mr, within_cell_df, False, interval, description)

    if T is None:
        return SeriesLengthPrecision(None, None, within_cell_df, True, None, '')

    n_mr = T - 1 if T >= 2 else 0
    interval = MR_SIGMA_INTERVAL_80.get(T) if T <= _TABLE_MAX else None

    if T < 2:
        description = f'T={T}. Sigma for X/mR rests on the moving range, which needs at least 2 time points.'
    elif T == 2:
        description = 'T=2. Sigma for X/mR rests on a single moving range.'
    elif interval is not None:
        lo, hi = interval
        description = (
            f'T={T}. Sigma for X/mR rests on {n_mr} moving ranges; on a stable process, '
            f'80% of such estimates fall between {lo:.2f}x and {hi:.2f}x the true sigma.'
        )
    else:
        lo, hi = MR_SIGMA_INTERVAL_80[_TABLE_MAX]
        description = (
            f'T={T}. Sigma for X/mR rests on {n_mr} moving ranges; on a stable process, '
            f'80% of such estimates fall within {lo:.2f}x to {hi:.2f}x the true sigma or narrower.'
        )
    if within_cell_df > 0 and description:
        # Partial replication (ADS 3): X/mR limits rest on the moving range, while the
        # replicated cells still carry within-cell degrees of freedom for Xbar/S.
        description += f' Replicated cells also carry {within_cell_df} within-cell degrees of freedom for Xbar/S.'
    return SeriesLengthPrecision(T, n_mr, within_cell_df, True, interval, description)
