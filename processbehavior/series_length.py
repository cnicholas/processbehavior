"""
Series-length precision of the sigma estimate.

One fact for the design report: where the sigma behind the natural process
limits comes from at the observed structure and series length. No threshold,
label, or warning is attached; the analyst weighs it (Bishop: judgment belongs
to the analyst, not to a rule).

Two sources of sigma:

- Every cell replicated (ADS 1): within-cell deviation. The number of time
  periods does not enter; T = 1 is a complete study.
- Any singleton cell (ADS 2, ADS 3): the 2-point moving range over the ordered
  sequence, T - 1 ranges for T time points.

The sentence names the source and the count. The within-cell degrees of freedom
and the 80% interval of the moving-range estimate at this T
(``MR_SIGMA_INTERVAL_80``, from #114) ride on the result for callers who want
them; the report does not print them.
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
        The sentence printed in the design report, e.g.
        ``T=4. Sigma for X/mR rests on 3 moving ranges.`` Empty when there is
        nothing to say (no time variable and no replication).
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

    The sentence names only the source of sigma at this structure and T. The
    numbers behind it (``within_cell_df``, ``mr_interval_80``) ride on the
    result for callers who want them; the report does not print them.

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
        if T is None:
            description = 'Sigma rests on within-cell replication.'
        else:
            description = f'T={T}. Sigma rests on within-cell replication.'
        return SeriesLengthPrecision(T, None, within_cell_df, False, None, description)

    if T is None:
        return SeriesLengthPrecision(None, None, within_cell_df, True, None, '')

    n_mr = T - 1 if T >= 2 else 0
    interval = MR_SIGMA_INTERVAL_80.get(T) if T <= _TABLE_MAX else None

    if T < 2:
        description = f'T={T}. Sigma for X/mR rests on the moving range, which needs at least 2 time points.'
    elif T == 2:
        description = 'T=2. Sigma for X/mR rests on a single moving range.'
    else:
        description = f'T={T}. Sigma for X/mR rests on {n_mr} moving ranges.'
    return SeriesLengthPrecision(T, n_mr, within_cell_df, True, interval, description)
