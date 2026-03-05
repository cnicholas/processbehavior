"""
VAS (Variance Analysis System) residual calculations for process behavior analysis.

This module calculates the Wheeler/Bishop VAS residuals (R1-R5) that decompose
total variation into interpretable components:

- R1: Total deviation from grand mean (pure algebra)
- R2: Within-cell (unexplained) variation (structure-dependent)
- R3: Interaction effects (factor × time) (pure algebra)
- R4: Time effects + unexplained (pure algebra given R2)
- R5: Factor effects + unexplained (pure algebra given R2)

R2 is the ONLY residual whose calculation varies by structure:
- exact: R2 = Y - Ȳ_kt (Eq 59), when all cells have n >= 2
- ma2: R2 = (Y_j - Y_{j-1}) / 2 (Eq 66), when all cells have n = 1
- hybrid: exact where n >= 2, MA2 where n = 1

All other residuals (R1, R3, R4, R5) are pure algebraic transformations
that don't depend on structure once the means are defined.

Module structure:
- Pure mean functions: calculate_grand_mean, calculate_factor_means,
  calculate_time_means, calculate_cell_means
- Pure residual functions: calculate_r1_residual, calculate_r3_residual,
  calculate_r4_residual, calculate_r5_residual
- Consolidated R2: calculate_r2(df, y, r2_method, n_per_cell)
- Orchestration: calculate_vas_residuals(df, spec, r2_method, ...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .sds_detector import R2Method

if TYPE_CHECKING:
    from .formulation_spec import FormulationSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Pure Functions: Means (Ybar calculations)
# ============================================================================

def calculate_grand_mean(df: pd.DataFrame, response_var: str) -> float:
    """
    Calculate grand mean (Ȳ) - average of all observations.

    This is the baseline for R1 residuals.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable column

    Returns
    -------
    float
        Grand mean (Ȳ)

    Examples
    --------
    >>> df = pd.DataFrame({'weight': [10.0, 10.5, 9.5, 10.0]})
    >>> calculate_grand_mean(df, 'weight')
    10.0
    """
    return df[response_var].mean()


def calculate_factor_means(
    df: pd.DataFrame,
    response_var: str,
    rsg_var: str
) -> pd.Series:
    """
    Calculate factor-level means (Ȳ_k) - average for each factor level.

    Broadcasts the mean for each factor level to all rows in that level.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable
    rsg_var : str
        Name of rational subgroup variable

    Returns
    -------
    Series
        Factor means, same length as df (broadcast to rows)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'lane': ['A', 'A', 'B', 'B'],
    ...     'weight': [10.0, 10.5, 9.0, 9.5]
    ... })
    >>> calculate_factor_means(df, 'weight', 'lane')
    0    10.25
    1    10.25
    2     9.25
    3     9.25
    Name: weight, dtype: float64
    """
    return df.groupby(rsg_var, observed=True)[response_var].transform('mean')


def calculate_time_means(
    df: pd.DataFrame,
    response_var: str,
    time_var: str
) -> pd.Series:
    """
    Calculate time-level means (Ȳ_t) - average for each time point.

    Broadcasts the mean for each time point to all rows at that time.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable
    time_var : str
        Name of time variable

    Returns
    -------
    Series
        Time means, same length as df (broadcast to rows)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'pull': [1, 1, 2, 2],
    ...     'weight': [10.0, 10.5, 9.0, 9.5]
    ... })
    >>> calculate_time_means(df, 'weight', 'pull')
    0    10.25
    1    10.25
    2     9.25
    3     9.25
    Name: weight, dtype: float64
    """
    return df.groupby(time_var, observed=True)[response_var].transform('mean')


def calculate_cell_means(
    df: pd.DataFrame,
    response_var: str,
    rsg_var: str,
    time_var: str
) -> pd.Series:
    """
    Calculate cell means (Ȳ_kt) - average for each (factor × time) cell.

    Broadcasts the mean for each cell to all rows in that cell.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable
    rsg_var : str
        Name of rational subgroup variable
    time_var : str
        Name of time variable

    Returns
    -------
    Series
        Cell means, same length as df (broadcast to rows)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'lane': ['A', 'A', 'B', 'B'],
    ...     'pull': [1, 1, 2, 2],
    ...     'weight': [10.0, 10.5, 9.0, 9.5]
    ... })
    >>> calculate_cell_means(df, 'weight', 'lane', 'pull')
    0    10.25
    1    10.25
    2     9.25
    3     9.25
    Name: weight, dtype: float64
    """
    return df.groupby([rsg_var, time_var], observed=True)[response_var].transform('mean')


# ============================================================================
# Pure Functions: Residuals (R1, R3, R4, R5)
# ============================================================================

def calculate_r1_residual(
    df: pd.DataFrame,
    response_var: str,
    grand_mean: float
) -> pd.Series:
    """
    Calculate R1 residual: total deviation from grand mean.

    R1 = Y - Ȳ  (Equation 56 from Wheeler)

    R1 represents the total variation of each observation around the
    overall average. It's the foundation for all other residuals.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable column
    grand_mean : float
        Grand mean (Ȳ)

    Returns
    -------
    Series
        R1 residuals, same length as df

    Examples
    --------
    >>> df = pd.DataFrame({'weight': [10.1, 10.3, 9.9]})
    >>> r1 = calculate_r1_residual(df, 'weight', 10.1)
    >>> r1.tolist()
    [0.0, 0.19999999999999929, -0.20000000000000107]

    Notes
    -----
    R1 is the simplest residual - it just shows how far each observation
    is from the overall average. The sum of all R1 values is always zero.
    """
    return df[response_var] - grand_mean


def calculate_r3_residual(
    df: pd.DataFrame,
    response_var: str,
    factor_means: pd.Series,
    time_means: pd.Series,
    grand_mean: float
) -> pd.Series:
    """
    Calculate R3 residual: interaction effects (factor × time).

    R3 = Y - Ȳ_k - Ȳ_t + Ȳ  (Equation 66 from Wheeler)

    R3 captures the interaction between factors and time. It represents
    variation that can't be explained by factor effects or time effects alone.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable
    factor_means : Series
        Factor means (Ȳ_k)
    time_means : Series
        Time means (Ȳ_t)
    grand_mean : float
        Grand mean (Ȳ)

    Returns
    -------
    Series
        R3 residuals

    Examples
    --------
    >>> df = pd.DataFrame({'y': [10.0, 10.5, 9.0, 9.5]})
    >>> factor_means = pd.Series([10.25, 10.25, 9.25, 9.25])
    >>> time_means = pd.Series([9.75, 9.75, 10.0, 10.0])
    >>> r3 = calculate_r3_residual(df, 'y', factor_means, time_means, 9.875)
    """
    return df[response_var] - factor_means - time_means + grand_mean


def calculate_r4_residual(
    time_means: pd.Series,
    grand_mean: float,
    r2: pd.Series
) -> pd.Series:
    """
    Calculate R4 residual: time effects + unexplained.

    R4 = Ȳ_t - Ȳ + R2  (Equation 72 from Wheeler)

    R4 represents time effects plus within-cell variation. Used to
    assess if time contributes meaningful variation.

    Parameters
    ----------
    time_means : Series
        Time means (Ȳ_t)
    grand_mean : float
        Grand mean (Ȳ)
    r2 : Series
        R2 residuals

    Returns
    -------
    Series
        R4 residuals
    """
    return time_means - grand_mean + r2


def calculate_r5_residual(
    factor_means: pd.Series,
    grand_mean: float,
    r2: pd.Series
) -> pd.Series:
    """
    Calculate R5 residual: factor effects + unexplained.

    R5 = Ȳ_k - Ȳ + R2  (Equation 75 from Wheeler)

    R5 represents factor effects plus within-cell variation. Used to
    assess if factors contribute meaningful variation and to calculate
    main effects.

    Parameters
    ----------
    factor_means : Series
        Factor means (Ȳ_k)
    grand_mean : float
        Grand mean (Ȳ)
    r2 : Series
        R2 residuals

    Returns
    -------
    Series
        R5 residuals
    """
    return factor_means - grand_mean + r2


# ============================================================================
# R2: Structure-dependent residual (consolidated)
# ============================================================================

def calculate_r2(
    df: pd.DataFrame,
    y: str,
    r2_method: R2Method,
    n_per_cell: pd.Series | None = None
) -> pd.Series:
    """
    Calculate R2 residual using the specified method.

    R2 is the ONLY structure-dependent residual. The method is determined by
    observed cell sizes (via SDSRegistry.get_r2_method()):

    - exact: R2 = Y - Ȳ_kt (Eq 59), all cells have n >= 2
    - ma2: R2 = (Y_j - Y_{j-1}) / 2 (Eq 66), all cells have n = 1
    - hybrid: exact where n >= 2, MA2 where n = 1

    Parameters
    ----------
    df : pd.DataFrame
        Input data with 'cell_key', 'rsg_key', 'sort_key', 'Ybar_kt' columns
    y : str
        Name of response variable
    r2_method : R2Method
        'exact', 'ma2', or 'hybrid'
    n_per_cell : pd.Series, optional
        Pre-computed observations per cell. Pass from ADS to avoid recomputation.

    Returns
    -------
    pd.Series
        R2 residuals with name="R2"

    Notes
    -----
    For hybrid: MA2 is computed on the FULL ordered stream within each rsg_key,
    then selected only for singleton cells. This is methodologically required —
    filtering to singletons first would break the consecutive-observation
    assumption in Wheeler's (Y_j - Y_{j-1})/2 formula.
    """
    if r2_method == "exact":
        return pd.Series(df[y] - df["Ybar_kt"], index=df.index, name="R2")

    if r2_method == "ma2":
        df_sorted = df.sort_values("sort_key")
        r2 = df_sorted.groupby("rsg_key", observed=True)[y].transform(
            lambda s: (s - s.shift(1)) / 2
        ).fillna(0.0)
        return pd.Series(r2.loc[df.index], index=df.index, name="R2")

    # hybrid: exact where n >= 2, MA2 where n = 1
    if n_per_cell is None:
        n_per_cell = df.groupby("cell_key", observed=True)[y].transform("size")

    r2_exact = df[y] - df["Ybar_kt"]

    # MA2 on full ordered stream (not filtered to singletons)
    df_sorted = df.sort_values("sort_key")
    r2_ma2 = df_sorted.groupby("rsg_key", observed=True)[y].transform(
        lambda s: (s - s.shift(1)) / 2
    ).fillna(0.0)
    r2_ma2 = r2_ma2.loc[df.index]

    out = np.where(n_per_cell >= 2, r2_exact, r2_ma2)
    return pd.Series(out, index=df.index, name="R2")


# ============================================================================
# Orchestration: calculate_vas_residuals
# ============================================================================

def _validate_prerequisites(df: pd.DataFrame) -> None:
    """
    Ensure required keys exist before computing residuals.

    Parameters
    ----------
    df : pd.DataFrame
        Input data

    Raises
    ------
    ValueError
        If required columns are missing
    """
    required = {"rsg_key", "obs_id", "cell_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for VAS residuals: {missing}. "
            f"Ensure data_preparation.build_keys() was called."
        )


def calculate_vas_residuals(
    df: pd.DataFrame,
    spec: FormulationSpec,
    r2_method: R2Method,
    n_per_cell: pd.Series | None = None,
    ybar_kt: pd.Series | None = None
) -> pd.DataFrame:
    """
    Calculate all VAS residuals using structure-driven R2 method.

    This is the primary entry point. The r2_method should be determined by
    SDSRegistry.get_r2_method() based on observed cell sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (prepared, with keys: rsg_key, obs_id, cell_key)
    spec : FormulationSpec
        Analysis specification
    r2_method : R2Method
        'exact', 'ma2', or 'hybrid' - determines R2 calculation
    n_per_cell : pd.Series, optional
        Pre-computed observations per cell. Pass from ADS to avoid recomputation.
    ybar_kt : pd.Series, optional
        Pre-computed cell means (Ȳ_kt). Pass from ADS to avoid recomputation.

    Returns
    -------
    pd.DataFrame
        Input data with added columns:
        - Ybar, Ybar_k, Ybar_t, Ybar_kt (means)
        - R1, R2, R3, R4, R5 (residuals)

    Raises
    ------
    ValueError
        If required columns are missing or no grouping structure
    """
    _validate_prerequisites(df)

    if not spec.has_grouping:
        raise ValueError(
            "VAS residuals require grouping structure.\n"
            "No grouping variables specified in spec."
        )

    out = df.copy()
    y = spec.response_var

    # Step 1: Compute cell means FIRST (foundation for everything)
    logger.debug("Calculating means (Ybar_kt, then unweighted Ybar, Ybar_k, Ybar_t)")
    out['Ybar_kt'] = ybar_kt if ybar_kt is not None else calculate_cell_means(
        out, y, spec.rsg_var_name, spec.time_var
    )

    # Step 2: Derive marginal means from cell means (unweighted means analysis)
    # Wheeler/Bishop VAS uses mean of cell means, giving each experimental
    # condition equal weight regardless of sample size within cells.
    cell_means_unique = out.groupby(
        [spec.rsg_var_name, spec.time_var], observed=True
    )['Ybar_kt'].first()

    grand_mean = cell_means_unique.mean()
    out['Ybar'] = grand_mean

    factor_means = cell_means_unique.groupby(level=0, observed=True).mean()
    out['Ybar_k'] = out[spec.rsg_var_name].map(factor_means).astype(float)

    time_means = cell_means_unique.groupby(level=1, observed=True).mean()
    out['Ybar_t'] = out[spec.time_var].map(time_means).astype(float)

    # Step 3: Calculate R1 (pure algebra)
    logger.debug("Calculating R1 residual")
    out['R1'] = calculate_r1_residual(out, y, grand_mean)

    # Step 4: Calculate R2 (structure-dependent)
    logger.debug(f"Calculating R2 residual (method: {r2_method})")
    out['R2'] = calculate_r2(out, y, r2_method, n_per_cell=n_per_cell)

    # Step 5: Calculate R3 (pure algebra)
    logger.debug("Calculating R3 residual")
    out['R3'] = calculate_r3_residual(
        out, y, out['Ybar_k'], out['Ybar_t'], grand_mean
    )

    # Step 6: Calculate R4 (pure algebra given R2)
    logger.debug("Calculating R4 residual")
    out['R4'] = calculate_r4_residual(out['Ybar_t'], grand_mean, out['R2'])

    # Step 7: Calculate R5 (pure algebra given R2)
    logger.debug("Calculating R5 residual")
    out['R5'] = calculate_r5_residual(out['Ybar_k'], grand_mean, out['R2'])

    logger.debug("VAS residuals calculated successfully")
    return out
