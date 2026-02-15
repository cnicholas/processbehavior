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

Follows the Pythonic Hadley philosophy:
- Pure functions (no mutation)
- Explicit inputs and outputs
- Type hints everywhere
- Comprehensive docstrings with examples
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
# Pure Functions: Residuals (R1-R5)
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


def calculate_r2_residual_sds1(
    df: pd.DataFrame,
    response_var: str,
    cell_means: pd.Series
) -> pd.Series:
    """
    Calculate R2 for SDS 1 (full replication): within-cell variation.

    R2 = Y - Ȳ_kt  (Equation 58 from Wheeler)

    With full replication (all cells have n≥2), we can directly estimate
    within-cell variance from the deviations within each cell.

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable
    cell_means : Series
        Cell means (Ȳ_kt), broadcast to rows

    Returns
    -------
    Series
        R2 residuals

    Examples
    --------
    >>> df = pd.DataFrame({'y': [10.0, 10.5, 9.0, 9.5]})
    >>> cell_means = pd.Series([10.25, 10.25, 9.25, 9.25])
    >>> r2 = calculate_r2_residual_sds1(df, 'y', cell_means)
    >>> r2.tolist()
    [-0.25, 0.25, -0.25, 0.25]
    """
    return df[response_var] - cell_means


def calculate_r2_residual_sds2(
    df: pd.DataFrame,
    response_var: str,
    rsg_var: str
) -> pd.Series:
    """
    Calculate R2 for SDS 2 (no replication): backward moving average.

    Per Tom Bishop Equation 65-66:
    Y_ma_j = (Y_j + Y_{j-1}) / 2
    R2_j = Y_j - Y_ma_j = (Y_j - Y_{j-1}) / 2 = MR_j / 2

    With no replication (all cells n=1), R2 = Y - Ȳ_kt would give R2=0
    (no information about within-subgroup variation). Instead, use a
    backward-looking 2-point moving average to approximate the local mean
    and extract the "unexplained" variation.

    Parameters
    ----------
    df : DataFrame
        Input data (must be sorted by [rsg_var, time])
    response_var : str
        Name of response variable
    rsg_var : str
        Rational subgroup variable

    Returns
    -------
    Series
        R2 residuals where R2_j = (Y_j - Y_{j-1}) / 2 for j ≥ 2
        First observation in each group is NaN (no lag available)

    Notes
    -----
    This is Tom Bishop's method for SDS 2 and 6 (Equations 64-66).
    The backward MA smooths the data to remove PDC and PT effects,
    leaving the unexplained variation (R2 ≈ λ + η + ε).

    Mathematical result: R2_j = (Y_j - Y_{j-1}) / 2
    This equals half of the moving range, connecting to IMR methodology.

    References
    ----------
    Tom Bishop, "Understanding Statistical Process Control", Section 20.2.1
    """
    # Backward-looking moving average: (Y_j + Y_{j-1}) / 2
    # This is the current value + the lagged value, divided by 2
    ma2 = df.groupby(rsg_var, observed=True)[response_var].transform(
        lambda s: (s + s.shift(1)) / 2.0
    )

    # R2 = Y - Y_ma
    # Note: First observation in each group will have NaN (no lag)
    return df[response_var] - ma2


def calculate_r2_residual_sds3(
    df: pd.DataFrame,
    response_var: str,
    cell_means: pd.Series,
    rsg_var: str,
    time_var: str
) -> pd.Series:
    """
    Calculate R2 for SDS 3 (partial replication): hybrid approach.

    R2 = Y - Ȳ_kt  for cells with n > 1
    R2 = 0         for cells with n = 1

    With partial replication, some cells have n≥2 (can estimate variance)
    and some have n=1 (cannot). For n=1 cells, set R2=0 (no within-cell
    variance estimable).

    Parameters
    ----------
    df : DataFrame
        Input data
    response_var : str
        Name of response variable
    cell_means : Series
        Cell means (Ȳ_kt)
    rsg_var : str
        Rational subgroup variable
    time_var : str
        Time variable

    Returns
    -------
    Series
        R2 residuals (hybrid)

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'rsg': ['A', 'A', 'B'],  # A has n=2, B has n=1
    ...     'time': [1, 1, 1],
    ...     'y': [10.0, 10.5, 9.0]
    ... })
    >>> # Cell means would be: [10.25, 10.25, 9.0]
    >>> # n per cell: A×1 has n=2, B×1 has n=1
    >>> # R2: [-0.25, 0.25, 0.0]  (B gets 0 because n=1)
    """
    # Count observations per cell
    n_per_cell = df.groupby([rsg_var, time_var], observed=True)[response_var].transform('count')

    # Calculate within-cell deviation
    r2_within = df[response_var] - cell_means

    # Use within-cell deviation only for n>1, otherwise 0
    # Return as Series to match expected type
    # NOTE: This is DEPRECATED - use calculate_r2_hybrid instead which correctly
    # uses MA2 for singleton cells instead of zeroing them out.
    return pd.Series(
        np.where(n_per_cell > 1, r2_within, 0.0),
        index=df.index
    )


def calculate_r2_hybrid(
    df: pd.DataFrame,
    y: str,
    n_per_cell: pd.Series | None = None
) -> pd.Series:
    """
    Calculate R2 using hybrid method: exact where replicated, MA2 where singleton.

    This is the correct implementation for partial replication (SDS 3/5) that
    uses MA2 for singleton cells instead of zeroing them out.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with 'cell_key', 'rsg_key', 'obs_id', 'Ybar_kt' columns
    y : str
        Name of response variable
    n_per_cell : pd.Series, optional
        Pre-computed observations per cell. If None, will be computed.
        Pass this when available to avoid recomputation.

    Returns
    -------
    pd.Series
        R2 residuals with name="R2"

    Notes
    -----
    For cells with n >= 2: R2 = Y - Ȳ_kt (exact method, Eq 59)
    For cells with n = 1:  R2 = MA2 (moving average, Eq 66)

    MA2 is computed on the full ordered series within each rsg_key,
    then selected for singleton cells. This ensures proper variance
    estimation even for cells without replication.
    """
    # Use passed n_per_cell if available (computed once in ADS), else compute
    if n_per_cell is None:
        n_per_cell = df.groupby("cell_key", observed=True)[y].transform("size")

    # Exact: Y - Ybar_kt
    r2_exact = df[y] - df["Ybar_kt"]

    # MA2 on full ordered series (use canonical sort_key)
    df_sorted = df.sort_values("sort_key")
    r2_ma2 = df_sorted.groupby("rsg_key", observed=True)[y].transform(
        lambda s: (s - s.shift(1)) / 2
    )
    r2_ma2 = r2_ma2.loc[df.index]  # restore original order by index

    # Combine: exact where n >= 2, MA2 where n = 1
    out = np.where(n_per_cell >= 2, r2_exact, r2_ma2)
    return pd.Series(out, index=df.index, name="R2")


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
# Orchestration Class
# ============================================================================

class ResidualCalculator:
    """
    Calculates VAS residuals (R1-R5) using structure-driven R2 method.

    This class orchestrates the pure residual calculation functions,
    adapting the R2 calculation based on observed cell structure and
    adding all necessary mean columns to the dataset.

    The primary entry point is :meth:`calculate_vas_residuals`, which
    accepts an R2Method ('exact', 'ma2', or 'hybrid') determined by
    SDSRegistry.get_r2_method() based on observed cell sizes.

    Examples
    --------
    Calculate residuals using structure-driven R2 method:

    >>> calc = ResidualCalculator()
    >>> df_with_residuals = calc.calculate_vas_residuals(df, spec, r2_method='exact')
    >>> # df now has: Ybar, Ybar_k, Ybar_t, Ybar_kt, R1, R2, R3, R4, R5
    """

    def calculate_residuals(
        self,
        df: pd.DataFrame,
        spec: FormulationSpec,
        sds: int
    ) -> pd.DataFrame:
        """
        Calculate all VAS residuals and add to DataFrame.

        .. deprecated::
            Use :meth:`calculate_vas_residuals` instead, which uses
            structure-driven R2 method selection (R2Method) rather than
            SDS-based dispatch. The new method correctly handles hybrid
            R2 (MA2 for singleton cells instead of zeroing them out).

        Parameters
        ----------
        df : DataFrame
            Input data (prepared, with keys)
        spec : FormulationSpec
            Analysis specification
        sds : int
            Sampling Design State (1-6)

        Returns
        -------
        DataFrame
            Input data with added columns:
            - Ybar, Ybar_k, Ybar_t, Ybar_kt (means)
            - R1, R2, R3, R4, R5 (residuals)

        Raises
        ------
        ValueError
            If SDS doesn't support VAS residuals (0)
            If required columns missing
        """
        import warnings
        warnings.warn(
            "calculate_residuals() is deprecated. Use calculate_vas_residuals() "
            "with an R2Method parameter instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if not spec.has_grouping:
            raise ValueError(
                "VAS residuals require grouping structure.\n"
                "No grouping variables specified in spec."
            )

        if sds not in [1, 2, 3, 4, 5, 6]:
            raise ValueError(
                f"VAS residuals not supported for SDS {sds}.\n"
                f"Valid SDS values: 1 (full replication), "
                f"2 (no replication), 3 (partial replication), "
                f"4 (replicated design), 5 (nested), 6 (sparse design).\n"
                f"Current SDS: {sds}"
            )

        out = df.copy()
        y = spec.response_var

        # Step 1: Calculate all means
        logger.debug("Calculating means (Ybar, Ybar_k, Ybar_t, Ybar_kt)")

        grand_mean = calculate_grand_mean(out, y)
        out['Ybar'] = grand_mean

        out['Ybar_k'] = calculate_factor_means(out, y, spec.rsg_var_name)

        out['Ybar_t'] = calculate_time_means(out, y, spec.time_var)

        out['Ybar_kt'] = calculate_cell_means(out, y, spec.rsg_var_name, spec.time_var)

        # Step 2: Calculate R1 (always the same)
        logger.debug("Calculating R1 residual")
        out['R1'] = calculate_r1_residual(out, y, grand_mean)

        # Step 3: Calculate R2 (SDS-dependent)
        logger.debug(f"Calculating R2 residual (SDS {sds} method)")
        out['R2'] = self._calculate_r2_for_sds(out, spec, sds)

        # Step 4: Calculate R3
        logger.debug("Calculating R3 residual")
        out['R3'] = calculate_r3_residual(
            out, y, out['Ybar_k'], out['Ybar_t'], grand_mean
        )

        # Step 5: Calculate R4
        logger.debug("Calculating R4 residual")
        out['R4'] = calculate_r4_residual(out['Ybar_t'], grand_mean, out['R2'])

        # Step 6: Calculate R5
        logger.debug("Calculating R5 residual")
        out['R5'] = calculate_r5_residual(out['Ybar_k'], grand_mean, out['R2'])

        logger.debug("VAS residuals calculated successfully")
        return out

    def _calculate_r2_for_sds(
        self,
        df: pd.DataFrame,
        spec: FormulationSpec,
        sds: int
    ) -> pd.Series:
        """
        Calculate R2 using appropriate method for the SDS.

        .. deprecated::
            Used only by the deprecated :meth:`calculate_residuals`.
            Prefer :meth:`_calculate_r2` with R2Method for new code.
        """
        y = spec.response_var

        if sds == 1:
            # Full replication: direct within-cell variance
            # R2 = Y - Ȳ_kt (Equation 59)
            logger.debug("SDS 1: R2 = Y - Ybar_kt")
            return calculate_r2_residual_sds1(df, y, df['Ybar_kt'])

        elif sds == 2:
            # No replication (all N_kt = 1): moving average approximation
            # R2 = Y - Y_ma where Y_ma = (Y_j + Y_{j-1}) / 2 (Equations 64-66)
            logger.debug("SDS 2: R2 = Y - MA2 (moving average method)")
            # Must sort for moving average to work correctly
            df_sorted = df.sort_values([spec.rsg_var_name, spec.time_var]).copy()
            r2 = calculate_r2_residual_sds2(df_sorted, y, spec.rsg_var_name)
            # Re-index to match original order
            return r2.reindex(df.index)

        elif sds == 3:
            # Partial replication: hybrid approach
            # R2 = Y - Ȳ_kt for cells with n > 1, else 0
            logger.debug("SDS 3: R2 = Y - Ybar_kt (n>1), else 0")
            return calculate_r2_residual_sds3(
                df, y, df['Ybar_kt'], spec.rsg_var_name, spec.time_var
            )

        elif sds == 4:
            # Replicated design: same as SDS 1, 3, 5
            # R2 = Y - Ȳ_kt (Equation 59)
            logger.debug("SDS 4: R2 = Y - Ybar_kt")
            return calculate_r2_residual_sds3(
                df, y, df['Ybar_kt'], spec.rsg_var_name, spec.time_var
            )

        elif sds == 5:
            # Nested: use hybrid approach
            # R2 = Y - Ȳ_kt (Equation 59)
            logger.debug("SDS 5: R2 = Y - Ybar_kt (hybrid approach)")
            return calculate_r2_residual_sds3(
                df, y, df['Ybar_kt'], spec.rsg_var_name, spec.time_var
            )

        elif sds == 6:
            # Sparse design (some N_kt = 0, rest N_kt = 1): moving average method
            # Same as SDS 2 - R2 = Y - Y_ma (Equations 64-66)
            logger.debug("SDS 6: R2 = Y - MA2 (moving average method, sparse design)")
            # Must sort for moving average to work correctly
            df_sorted = df.sort_values([spec.rsg_var_name, spec.time_var]).copy()
            r2 = calculate_r2_residual_sds2(df_sorted, y, spec.rsg_var_name)
            # Re-index to match original order
            return r2.reindex(df.index)

        else:
            raise ValueError(f"R2 calculation not defined for SDS {sds}")

    def _calculate_r2(
        self,
        df: pd.DataFrame,
        spec: FormulationSpec,
        r2_method: R2Method,
        n_per_cell: pd.Series | None = None
    ) -> pd.Series:
        """
        Calculate R2 using the specified method.

        This is the new structure-driven approach that replaces the SDS-based
        `_calculate_r2_for_sds` method. R2 method is determined by observed
        cell sizes, not SDS label.

        Parameters
        ----------
        df : pd.DataFrame
            Prepared data with required columns
        spec : FormulationSpec
            Analysis specification
        r2_method : R2Method
            'exact', 'ma2', or 'hybrid'
        n_per_cell : pd.Series, optional
            Pre-computed observations per cell. Pass from ADS to avoid recomputation.

        Returns
        -------
        pd.Series
            R2 residuals
        """
        y = spec.response_var

        if r2_method == "exact":
            logger.debug("R2 method: exact (Y - Ybar_kt)")
            return pd.Series(df[y] - df["Ybar_kt"], index=df.index, name="R2")

        elif r2_method == "ma2":
            logger.debug("R2 method: ma2 (moving average)")
            df_sorted = df.sort_values("sort_key")
            r2 = df_sorted.groupby("rsg_key", observed=True)[y].transform(
                lambda s: (s - s.shift(1)) / 2
            )
            return pd.Series(r2.loc[df.index], index=df.index, name="R2")

        else:  # hybrid = exact-else-MA2
            logger.debug("R2 method: hybrid (exact where n>=2, MA2 where n=1)")
            return calculate_r2_hybrid(df, y, n_per_cell=n_per_cell)

    def _validate_prerequisites(self, df: pd.DataFrame) -> None:
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
        self,
        df: pd.DataFrame,
        spec: FormulationSpec,
        r2_method: R2Method,
        n_per_cell: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Calculate all VAS residuals using structure-driven R2 method.

        This is the new entry point that uses R2Method instead of SDS.
        The r2_method should be determined by SDSRegistry.get_r2_method()
        based on observed cell sizes.

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

        Returns
        -------
        pd.DataFrame
            Input data with added columns:
            - Ybar, Ybar_k, Ybar_t, Ybar_kt (means)
            - R1, R2, R3, R4, R5 (residuals)

        Raises
        ------
        ValueError
            If required columns are missing
        """
        # Validate prerequisites
        self._validate_prerequisites(df)

        if not spec.has_grouping:
            raise ValueError(
                "VAS residuals require grouping structure.\n"
                "No grouping variables specified in spec."
            )

        out = df.copy()
        y = spec.response_var

        # Step 1: Calculate all means
        logger.debug("Calculating means (Ybar, Ybar_k, Ybar_t, Ybar_kt)")

        grand_mean = calculate_grand_mean(out, y)
        out['Ybar'] = grand_mean

        out['Ybar_k'] = calculate_factor_means(out, y, spec.rsg_var_name)
        out['Ybar_t'] = calculate_time_means(out, y, spec.time_var)
        out['Ybar_kt'] = calculate_cell_means(out, y, spec.rsg_var_name, spec.time_var)

        # Step 2: Calculate R1 (pure algebra)
        logger.debug("Calculating R1 residual")
        out['R1'] = calculate_r1_residual(out, y, grand_mean)

        # Step 3: Calculate R2 (structure-dependent)
        logger.debug(f"Calculating R2 residual (method: {r2_method})")
        out['R2'] = self._calculate_r2(out, spec, r2_method, n_per_cell=n_per_cell)

        # Step 4: Calculate R3 (pure algebra)
        logger.debug("Calculating R3 residual")
        out['R3'] = calculate_r3_residual(
            out, y, out['Ybar_k'], out['Ybar_t'], grand_mean
        )

        # Step 5: Calculate R4 (pure algebra given R2)
        logger.debug("Calculating R4 residual")
        out['R4'] = calculate_r4_residual(out['Ybar_t'], grand_mean, out['R2'])

        # Step 6: Calculate R5 (pure algebra given R2)
        logger.debug("Calculating R5 residual")
        out['R5'] = calculate_r5_residual(out['Ybar_k'], grand_mean, out['R2'])

        logger.debug("VAS residuals calculated successfully")
        return out
