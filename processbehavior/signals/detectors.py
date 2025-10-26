"""
Vectorized rule detection functions.

All detectors use pandas/numpy for performance.
Each detector returns a boolean Series indicating violations.
"""

from __future__ import annotations

import pandas as pd


def detect_beyond_limits(
    data: pd.DataFrame,
    stats: dict,
    value_col: str
) -> pd.Series:
    """
    Rule 1: Points beyond control limits.

    Detects any point beyond ±3σ from centerline.

    Parameters
    ----------
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics with 'ucl' and 'lcl' keys
    value_col : str
        Name of value column

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]
    return (values > stats['ucl']) | (values < stats['lcl'])


def detect_zone_a_2_of_3(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    zones: dict
) -> pd.Series:
    """
    Rule 2: 2 of 3 consecutive points in Zone A.

    Detects when 2 out of 3 consecutive points fall in Zone A
    (between 2σ and 3σ from centerline), same side.

    Parameters
    ----------
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics
    value_col : str
        Name of value column
    zones : dict
        Zone boundaries from ZoneDefinition.get_boundaries()

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]

    # Determine which zone each point is in
    in_upper_a = (values > zones['A_upper'][0]) & (values <= zones['A_upper'][1])
    in_lower_a = (values >= zones['A_lower'][0]) & (values < zones['A_lower'][1])

    # Rolling window count
    upper_count = in_upper_a.astype(int).rolling(window=3, min_periods=3).sum()
    lower_count = in_lower_a.astype(int).rolling(window=3, min_periods=3).sum()

    # Flag if 2 or more in window
    violations = (upper_count >= 2) | (lower_count >= 2)

    return violations.fillna(False)


def detect_zone_b_4_of_5(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    zones: dict
) -> pd.Series:
    """
    Rule 3: 4 of 5 consecutive points in Zone B or beyond.

    Detects when 4 out of 5 consecutive points fall in Zone B or beyond
    (beyond 1σ from centerline), same side.

    Parameters
    ----------
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics
    value_col : str
        Name of value column
    zones : dict
        Zone boundaries

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]

    # In Zone B or beyond (same side)
    upper = values > zones['B_upper'][0]
    lower = values < zones['B_lower'][1]

    upper_count = upper.astype(int).rolling(window=5, min_periods=5).sum()
    lower_count = lower.astype(int).rolling(window=5, min_periods=5).sum()

    violations = (upper_count >= 4) | (lower_count >= 4)
    return violations.fillna(False)


def detect_run(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    length: int = 8
) -> pd.Series:
    """
    Rule 4: Run of points on same side of centerline.

    Detects when 8 or more consecutive points fall on the same side
    of the centerline.

    Parameters
    ----------
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics with 'Mean' key
    value_col : str
        Name of value column
    length : int, default 8
        Number of consecutive points required

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]
    center = stats['Mean']

    # Above or below center
    above = (values > center).astype(int)
    below = (values < center).astype(int)

    # Count consecutive
    above_streak = above.rolling(window=length, min_periods=length).sum()
    below_streak = below.rolling(window=length, min_periods=length).sum()

    violations = (above_streak == length) | (below_streak == length)
    return violations.fillna(False)


def detect_trend(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    length: int = 6,
    direction: str = 'both'
) -> pd.Series:
    """
    Rule 5: Trending sequence.

    Detects when 6 or more consecutive points are steadily
    increasing or decreasing.

    Parameters
    ----------
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics (not used but kept for consistency)
    value_col : str
        Name of value column
    length : int, default 6
        Number of consecutive points required
    direction : {'up', 'down', 'both'}, default 'both'
        Which direction trends to detect

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]

    # Calculate differences
    diffs = values.diff()

    up_trend = pd.Series(False, index=values.index)
    down_trend = pd.Series(False, index=values.index)

    if direction in ['up', 'both']:
        increasing = (diffs > 0).astype(int).rolling(
            window=length, min_periods=length
        ).sum()
        up_trend = increasing == length

    if direction in ['down', 'both']:
        decreasing = (diffs < 0).astype(int).rolling(
            window=length, min_periods=length
        ).sum()
        down_trend = decreasing == length

    violations = up_trend | down_trend
    return violations.fillna(False)


def detect_oscillation(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    length: int = 14
) -> pd.Series:
    """
    Rule 6: Alternating up-down pattern.

    Detects when 14 or more consecutive points alternate up and down.

    Parameters
    ----------
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics (not used but kept for consistency)
    value_col : str
        Name of value column
    length : int, default 14
        Number of consecutive points required

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]

    # Detect direction changes
    diffs = values.diff()
    changes = (diffs * diffs.shift(1) < 0).astype(int)

    # Count consecutive changes
    change_count = changes.rolling(window=length - 1, min_periods=length - 1).sum()

    violations = change_count == (length - 1)
    return violations.fillna(False)


def detect_reduced_variation(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    zones: dict,
    length: int = 15
) -> pd.Series:
    """
    Rule 7: Points within Zone C (reduced variation).

    Detects when 15 or more consecutive points fall within Zone C
    (within ±1σ of centerline), indicating reduced variation.

    Parameters
    ----------
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics
    value_col : str
        Name of value column
    zones : dict
        Zone boundaries
    length : int, default 15
        Number of consecutive points required

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]

    # Within Zone C
    in_zone_c = (
        (values > zones['C_lower'][0]) &
        (values < zones['C_upper'][1])
    )

    in_c_count = in_zone_c.astype(int).rolling(
        window=length, min_periods=length
    ).sum()

    violations = in_c_count == length
    return violations.fillna(False)


def detect_avoiding_center(
    data: pd.DataFrame,
    stats: dict,
    value_col: str,
    zones: dict,
    length: int = 8
) -> pd.Series:
    """
    Rule 8: Points avoiding Zone C.

    Detects when 8 or more consecutive points fall outside Zone C
    (beyond ±1σ from centerline).

    Parameters
    ----------
    data : DataFrame
        Chart data
    stats : dict
        Chart statistics
    value_col : str
        Name of value column
    zones : dict
        Zone boundaries
    length : int, default 8
        Number of consecutive points required

    Returns
    -------
    Series of bool
        True for violations
    """
    values = data[value_col]

    # Outside Zone C
    outside_c = (
        (values <= zones['C_lower'][0]) |
        (values >= zones['C_upper'][1])
    )

    outside_count = outside_c.astype(int).rolling(
        window=length, min_periods=length
    ).sum()

    violations = outside_count == length
    return violations.fillna(False)
