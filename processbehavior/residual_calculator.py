"""
VAS (Variance Analysis System) residual calculations for process behavior analysis.

This module calculates the Bishop VAS residuals (R1-R5) that decompose
total variation into interpretable components:

- R1: Total deviation from grand mean (pure algebra)
- R2: Within-cell (unexplained) variation (structure-dependent)
- R3: Interaction effects (factor × time) (pure algebra)
- R4: Time effects + unexplained (pure algebra given R2)
- R5: Factor effects + unexplained (pure algebra given R2)

R2 is the ONLY residual whose calculation varies by structure:
- exact (state 1): R2 = Y - Ȳ_kt (Eq 59), when all cells have n >= 2
- ma2 (states 2 & 3): R2 = (Y_j - Y_{j-1}) / 2 (Eq 13.8-13.9),
  when any cell has n = 1 — applied to ALL observations across the
  entire sorted stream, no grouping

All other residuals (R1, R3, R4, R5) are pure algebraic transformations
that don't depend on structure once the means are defined.

Module structure:
- Pure mean functions: calculate_grand_mean, calculate_factor_means,
  calculate_time_means, calculate_cell_means
- Pure residual functions: calculate_r1_residual, calculate_r3_residual,
  calculate_r4_residual, calculate_r5_residual
- Consolidated R2: calculate_r2(df, y, r2_method, n_per_cell)
- Orchestration: calculate_vas_residuals(df, spec, r2_method, ...)
- Request residuals: resolve_r6_groupby, calculate_r6_residuals — R6/RCR6 are
  computed per execute() request from that request's by=, never stored on the
  study-level frame (see spc_constants.REQUEST_RESIDUALS)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from .exceptions import ValidationError
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


def calculate_factor_means(df: pd.DataFrame, response_var: str, rsg_var_name: str) -> pd.Series:
    """
    Calculate factor-level means (Ȳ_k) - average for each factor level.

    Broadcasts the mean for each factor level to all rows in that level.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable
    rsg_var_name : str
        Name of the composite rational subgroup column (e.g., 'rsg')

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
    return df.groupby(rsg_var_name, observed=True)[response_var].transform('mean')


def calculate_time_means(df: pd.DataFrame, response_var: str, time_var: str) -> pd.Series:
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


def calculate_cell_means(df: pd.DataFrame, response_var: str, rsg_var_name: str, time_var: str) -> pd.Series:
    """
    Calculate cell means (Ȳ_kt) - average for each (factor × time) cell.

    Broadcasts the mean for each cell to all rows in that cell.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable
    rsg_var_name : str
        Name of the composite rational subgroup column (e.g., 'rsg')
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
    return df.groupby([rsg_var_name, time_var], observed=True)[response_var].transform('mean')


# ============================================================================
# Pure Functions: Residuals (R1, R3, R4, R5)
# ============================================================================


def calculate_r1_residual(df: pd.DataFrame, response_var: str, grand_mean: float) -> pd.Series:
    """
    Calculate R1 residual: total deviation from grand mean.

    R1 = Y - Ȳ  (Bishop Equation 56)

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
    df: pd.DataFrame, response_var: str, factor_means: pd.Series, time_means: pd.Series, grand_mean: float
) -> pd.Series:
    """
    Calculate R3 residual: interaction effects (factor × time).

    R3 = Y - Ȳ_k - Ȳ_t + Ȳ  (Bishop Equation 66)

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


def calculate_r4_residual(time_means: pd.Series, grand_mean: float, r2: pd.Series) -> pd.Series:
    """
    Calculate R4 residual: time effects + unexplained.

    R4 = Ȳ_t - Ȳ + R2  (Bishop Equation 72)

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


def calculate_r5_residual(factor_means: pd.Series, grand_mean: float, r2: pd.Series) -> pd.Series:
    """
    Calculate R5 residual: factor effects + unexplained.

    R5 = Ȳ_k - Ȳ + R2  (Bishop Equation 75)

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


def calculate_r2(df: pd.DataFrame, y: str, r2_method: R2Method, n_per_cell: pd.Series | None = None) -> pd.Series:
    """
    Calculate R2 residual using the specified method.

    R2 is the ONLY structure-dependent residual. The method is determined by
    observed cell sizes (via SDSRegistry.get_r2_method()):

    - exact (state 1): R2 = Y - Ȳ_kt (Eq 59), all cells have n >= 2
    - ma2 (states 2 & 3): R2 = (Y_j - Y_{j-1}) / 2 (Eq 13.8-13.9),
      any cell has n = 1

    When any cell has n=1, MA2 is applied to ALL observations across the
    entire canonical-sorted stream — no grouping by rsg_key, no hybrid
    per-cell selection. Bishop Eq 13.7-13.9 specify j=2,...,J with no
    grouping; only j=1 gets R2=0.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with 'cell_key', 'sort_key', 'Ybar_kt' columns
    y : str
        Name of response variable
    r2_method : R2Method
        'exact', 'ma2', or 'hybrid' — retained for API compatibility.
        Branching is driven by n_per_cell.
    n_per_cell : pd.Series, optional
        Pre-computed observations per cell. Pass from ADS to avoid recomputation.

    Returns
    -------
    pd.Series
        R2 residuals with name="R2"
    """
    if n_per_cell is None:
        n_per_cell = df.groupby('cell_key', observed=True)[y].transform('size')

    # State 1: all cells replicated → exact (Eq 59)
    if (n_per_cell >= 2).all():
        return pd.Series(df[y] - df['Ybar_kt'], index=df.index, name='R2')

    # States 2 & 3: any singletons → MA2 for ALL observations (Eq 13.8-13.9)
    # MA2 runs across the entire canonical-sorted stream — no grouping.
    # j=1 has no predecessor → R2 is NaN (Bishop leaves it blank).
    df_sorted = df.sort_values('sort_key')
    y_sorted = df_sorted[y]
    r2 = (y_sorted - y_sorted.shift(1)) / 2
    return pd.Series(r2.loc[df.index], index=df.index, name='R2')


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
    required = {'rsg_key', 'obs_id', 'cell_key'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f'Missing required columns for VAS residuals: {missing}. Ensure data_preparation.build_keys() was called.'
        )


def calculate_vas_residuals(
    df: pd.DataFrame,
    spec: FormulationSpec,
    r2_method: R2Method,
    n_per_cell: pd.Series | None = None,
    ybar_kt: pd.Series | None = None,
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
        raise ValueError('VAS residuals require grouping structure.\nNo grouping variables specified in spec.')

    out = df.copy()
    y = spec.response_var

    # Step 1: Compute cell means FIRST (foundation for everything)
    logger.debug('Calculating means (Ybar_kt, then unweighted Ybar, Ybar_k, Ybar_t)')
    out['Ybar_kt'] = ybar_kt if ybar_kt is not None else calculate_cell_means(out, y, spec.rsg_var_name, spec.time_var)

    # Step 2: Derive marginal means from cell means (unweighted means analysis)
    # Bishop VAS uses mean of cell means, giving each experimental
    # condition equal weight regardless of sample size within cells.
    cell_means_unique = out.groupby([spec.rsg_var_name, spec.time_var], observed=True)['Ybar_kt'].first()

    grand_mean = cell_means_unique.mean()
    out['Ybar'] = grand_mean

    factor_means = cell_means_unique.groupby(level=0, observed=True).mean()
    out['Ybar_k'] = out[spec.rsg_var_name].map(factor_means).astype(float)

    time_means = cell_means_unique.groupby(level=1, observed=True).mean()
    out['Ybar_t'] = out[spec.time_var].map(time_means).astype(float)

    # Step 3: Calculate R1 (pure algebra)
    logger.debug('Calculating R1 residual')
    out['R1'] = calculate_r1_residual(out, y, grand_mean)

    # Step 4: Calculate R2 (structure-dependent)
    logger.debug(f'Calculating R2 residual (method: {r2_method})')
    out['R2'] = calculate_r2(out, y, r2_method, n_per_cell=n_per_cell)

    # Step 5: Calculate R3 (interaction effects)
    # Unified formula: Ybar_kt - Ybar_k - Ybar_t + Ybar + R2
    # For exact (state 1): R2 = Y - Ybar_kt, so this simplifies to
    #   Y - Ybar_k - Ybar_t + Ybar (algebraically identical to old formula).
    # For MA2 (states 2-3): adds R2 to each row per Bishop.
    logger.debug('Calculating R3 residual')
    out['R3'] = out['Ybar_kt'] - out['Ybar_k'] - out['Ybar_t'] + grand_mean + out['R2']

    # Step 6: Calculate R4 (pure algebra given R2)
    logger.debug('Calculating R4 residual')
    out['R4'] = calculate_r4_residual(out['Ybar_t'], grand_mean, out['R2'])

    # Step 7: Calculate R5 (pure algebra given R2)
    logger.debug('Calculating R5 residual')
    out['R5'] = calculate_r5_residual(out['Ybar_k'], grand_mean, out['R2'])

    logger.debug('VAS residuals calculated successfully')
    return out


# ============================================================================
# Request Residuals: R6 / RCR6 (computed per execute() request, never stored)
# ============================================================================


def resolve_r6_groupby(by: Sequence[str] | None, factors: Sequence[str]) -> str | list[str]:
    """Resolve which factor(s) an R6 request groups on, or raise the reason it cannot.

    One rule, two callers: ``Study._validate_execute_request`` calls this for its
    ValidationErrors (validation stays pure — no frame is touched), and
    ``Analysis._build_request_frame`` calls it again to compute. They cannot drift.

    Parameters
    ----------
    by : Sequence[str] | None
        The request's ``by=`` (list or ChartRequest tuple).
    factors : Sequence[str]
        The study's factors (``spec.rsg_vars_list``).

    Returns
    -------
    str | list[str]
        A single factor name, or the list of factors when ``by`` names several —
        the exact shape ``DataFrame.groupby`` receives.
    """
    if not factors:
        raise ValidationError('R6 requires factors. No factors defined in this study.')

    # Determine which factor(s) from by
    if by is not None:
        by_factors = [b for b in by if b in factors]
        if not by_factors:
            raise ValidationError(
                f'R6 requires at least one factor in by=.\n'
                f'Available factors: {factors}\n'
                f"Example: study.execute(chart='Xbar', value='R6', by=['{factors[0]}'])"
            )
        return by_factors if len(by_factors) > 1 else by_factors[0]
    if len(factors) == 1:
        return factors[0]
    raise ValidationError(
        f'R6 requires by=[factor(s)] to specify which factor(s).\n'
        f'Available factors: {factors}\n'
        f"Example: study.execute(chart='Xbar', value='R6', by=['{factors[0]}'])"
    )


def calculate_r6_residuals(df: pd.DataFrame, groupby_key: str | list[str], recentered: bool) -> pd.DataFrame:
    """Return a new frame carrying R6 (and RCR6 when recentered). Never mutates ``df``.

    R6 = α_i + R2 where α_i = mean(R5 | factor level(s)).

    Bit-identity constraints — this math moved verbatim from the old
    ``Study._compute_r6`` and must keep producing identical floats:

    - ``groupby`` keeps its default kwargs (no ``observed=``, no ``dropna=``) to
      preserve row alignment and values exactly.
    - RCR6 is ``Ybar + alpha + R2`` evaluated left-to-right. Do not rewrite it as
      ``Ybar + R6``: float addition is non-associative, so the two differ in the
      last ulp.

    A recentered request carries **both** columns: the mR-of-RCR6 path reads the
    plain R6 column (``_resolve_mr_source_column('RCR6') == 'R6'``).
    """
    df = df.copy()
    alpha = df.groupby(groupby_key)['R5'].transform('mean')

    df['R6'] = alpha + df['R2']
    if recentered:
        df['RCR6'] = df['Ybar'] + alpha + df['R2']
    return df
