"""
Effects and interactions calculator for VAS (Variance Analysis System).

This module calculates:
- Main effects for each factor (mean of R5 per level)
- Time effects (mean of R1 per time point)
- Interaction effects (factor × time, factor × factor)
- Main effect scores (R2 + Main Effect for each factor)

These calculations help identify which factors and time periods contribute
most to process variation.

Follows the Pythonic Hadley philosophy:
- Pure functions where possible
- Clear, descriptive names
- Comprehensive documentation
- Type hints everywhere
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from .exceptions import ValidationError

if TYPE_CHECKING:
    from .formulation_spec import FormulationSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Pure Functions: Main Effects
# ============================================================================


def calculate_factor_main_effects(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """
    Calculate main effect for a single factor.

    Main Effect = mean(R5) per factor level

    The main effect shows the average impact of each factor level on
    the response, after accounting for within-cell variation.

    Parameters
    ----------
    df : DataFrame
        Data with R5 column
    factor : str
        Factor column name

    Returns
    -------
    DataFrame
        Two columns: [factor, 'Main_Effect']
        One row per factor level

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'lane': ['A', 'A', 'B', 'B'],
    ...     'R5': [0.2, 0.3, -0.2, -0.3]
    ... })
    >>> me = calculate_factor_main_effects(df, 'lane')
    >>> me
      lane  Main_Effect
    0    A         0.25
    1    B        -0.25

    Notes
    -----
    Large positive main effects indicate factor levels that consistently
    produce higher values. Large negative effects indicate lower values.
    """
    if 'R5' not in df.columns:
        raise ValidationError(
            f'Cannot calculate main effects - R5 column missing.\n'
            f'Available columns: {df.columns.tolist()}\n'
            f'Fix: Calculate VAS residuals first'
        )

    if factor not in df.columns:
        raise ValidationError(f"Factor '{factor}' not found in data.\nAvailable columns: {df.columns.tolist()}")

    me = df.groupby(factor, sort=False, observed=True)['R5'].mean().rename('Main_Effect').reset_index()

    # Validate uniqueness
    if me.duplicated(subset=[factor]).any():
        raise ValidationError(f"Duplicate levels found for factor '{factor}'")

    return me[[factor, 'Main_Effect']]


def calculate_time_main_effects(df: pd.DataFrame, time_var: str) -> pd.DataFrame:
    """
    Calculate main effect for time.

    Time Effect = mean(R4) per time point

    R4 = Ȳ_t - Ȳ + R2 preserves the time effect plus within-cell noise,
    so mean(R4) per time point gives the time main effect (analogous to
    mean(R5) per factor level for factor main effects).

    Parameters
    ----------
    df : DataFrame
        Data with R4 column
    time_var : str
        Time variable name

    Returns
    -------
    DataFrame
        Two columns: [time_var, 'PT_ME']
        One row per time point

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'pull': [1, 1, 2, 2],
    ...     'R4': [0.1, 0.2, -0.1, -0.2]
    ... })
    >>> te = calculate_time_main_effects(df, 'pull')
    >>> te
       pull  PT_ME
    0     1   0.15
    1     2  -0.15
    """
    if 'R4' not in df.columns:
        raise ValidationError('Cannot calculate time effects - R4 column missing.\nCalculate VAS residuals first.')

    te = df.groupby(time_var, sort=False, observed=True)['R4'].mean().rename('PT_ME').reset_index()

    return te[[time_var, 'PT_ME']]


def calculate_main_effect_scores(df: pd.DataFrame, factor: str, main_effects: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate main effect scores: R2 + Main Effect per row.

    MEs = R2 + Main_Effect(factor_level)

    This combines within-cell variation (R2) with the factor's main
    effect to show each observation's contribution to factor variation.

    Parameters
    ----------
    df : DataFrame
        Data with R2 column and factor column
    factor : str
        Factor name
    main_effects : DataFrame
        Main effects from calculate_factor_main_effects()
        Must have columns: [factor, 'Main_Effect']

    Returns
    -------
    DataFrame
        Two columns: [factor, '{factor}_MEs']
        Same number of rows as df

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'lane': ['A', 'A', 'B'],
    ...     'R2': [0.1, -0.1, 0.05]
    ... })
    >>> me = pd.DataFrame({
    ...     'lane': ['A', 'B'],
    ...     'Main_Effect': [0.25, -0.25]
    ... })
    >>> mes = calculate_main_effect_scores(df, 'lane', me)
    >>> mes
      lane  lane_MEs
    0    A      0.35
    1    A      0.15
    2    B     -0.20
    """
    if 'R2' not in df.columns:
        raise ValidationError('Cannot calculate MEs - R2 column missing')

    if factor not in df.columns:
        raise ValidationError(f"Factor '{factor}' not in data")

    if not isinstance(main_effects, pd.DataFrame):
        raise TypeError('main_effects must be a DataFrame')

    required_cols = {factor, 'Main_Effect'}
    if not required_cols.issubset(main_effects.columns):
        raise ValidationError(
            f'main_effects missing required columns.\nRequired: {required_cols}\nFound: {set(main_effects.columns)}'
        )

    # Merge to add Main_Effect column
    merged = df[[factor, 'R2']].merge(main_effects, on=factor, how='left', validate='many_to_one')

    # Calculate score
    label = f'{factor}_MEs'
    merged[label] = merged['R2'] + merged['Main_Effect']

    return merged[[factor, label]]


# ============================================================================
# Pure Functions: Interactions
# ============================================================================


def calculate_interaction_cell_means(df: pd.DataFrame, rsg_vars: list[str], time_var: str) -> pd.Series:
    """
    Calculate cell-level interaction effect (factor × time).

    Interaction = mean(R3) per cell

    For SDS 1 (full replication), the cell-average of R3 equals the
    interaction effect: Ȳ_kt - Ȳ_k - Ȳ_t + Ȳ

    Parameters
    ----------
    df : DataFrame
        Data with R3 column
    rsg_vars : list of str
        Factor variable names
    time_var : str
        Time variable name

    Returns
    -------
    Series
        Interaction effects broadcast to all rows
        Length equals length of input DataFrame
        Each row gets the mean R3 value for its factor×time cell

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'lane': ['A', 'A', 'B', 'B'],
    ...     'pull': [1, 1, 2, 2],
    ...     'R3': [0.1, 0.15, -0.1, -0.05]
    ... })
    >>> int_eff = calculate_interaction_cell_means(df, ['lane'], 'pull')
    >>> int_eff  # Broadcast to all rows
    0    0.125
    1    0.125
    2   -0.075
    3   -0.075
    Name: R3, dtype: float64
    """
    if 'R3' not in df.columns:
        raise ValidationError('Cannot calculate interactions - R3 column missing')

    keys = list(rsg_vars) + [time_var]

    # Use transform to broadcast cell means back to all rows
    return df.groupby(keys, sort=False, observed=True)['R3'].transform('mean')


def calculate_pdc_by_time_sds2(
    df: pd.DataFrame, ybar_kt: pd.Series, ybar_k: pd.Series, ybar_t: pd.Series, ybar: float
) -> pd.Series:
    """
    Calculate process disruption component (PDC) for SDS 2.

    PDC = Ȳ_kt - Ȳ_k - Ȳ_t + Ȳ

    For SDS 2 (no replication), we calculate this directly from means
    rather than from R3 averages.

    Parameters
    ----------
    df : DataFrame
        Input data (for index/length)
    ybar_kt : Series
        Cell means
    ybar_k : Series
        Factor means
    ybar_t : Series
        Time means
    ybar : float
        Grand mean

    Returns
    -------
    Series
        PDC values per row
    """
    return ybar_kt - ybar_k - ybar_t + ybar


def calculate_factor_interaction_effects(
    df: pd.DataFrame, rsg_vars: list[str], main_effects: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Calculate two-factor interaction effects (F1 × F2).

    Rx = mean(R5)_ij - ME_i - ME_j

    Where:
    - mean(R5)_ij is the average R5 for factor combination i,j
    - ME_i is the main effect of factor 1 level i
    - ME_j is the main effect of factor 2 level j

    Parameters
    ----------
    df : DataFrame
        Data with R5 column and factor columns
    rsg_vars : list of str
        Factor names (at least 2 required)
    main_effects : dict
        Dict mapping factor name to its main effects DataFrame

    Returns
    -------
    DataFrame
        Columns: [factor1, factor2, 'Rx']

    Notes
    -----
    Only calculates for first 2 factors if more than 2 provided.
    Higher-order interactions (3+) are not currently supported.

    Examples
    --------
    >>> # With lanes A,B and heads 1,2
    >>> # If lane A adds 0.2, head 1 adds 0.1, but A×1 adds 0.4 total
    >>> # Then Rx = 0.4 - 0.2 - 0.1 = 0.1 (synergistic interaction)
    """
    if len(rsg_vars) < 2:
        logger.debug(f'Only {len(rsg_vars)} factor(s) - factor interaction requires at least 2. Skipping.')
        return pd.DataFrame()

    if 'R5' not in df.columns:
        raise ValidationError('Cannot calculate interactions - R5 column missing')

    # Use first 2 factors only
    if len(rsg_vars) > 2:
        logger.warning(f'More than 2 factors in RSG - calculating interaction for first 2: {rsg_vars[:2]}')

    factor1, factor2 = rsg_vars[0], rsg_vars[1]

    # Calculate mean R5 for each factor combination
    rsg_r5 = df.groupby(list(rsg_vars[:2]), as_index=False, observed=True)['R5'].mean()

    # Add main effects for each factor
    for _i, factor in enumerate([factor1, factor2]):
        me = main_effects.get(factor)
        if me is None or not isinstance(me, pd.DataFrame):
            raise ValidationError(
                f"Main effects for '{factor}' not found or invalid.\nAvailable: {list(main_effects.keys())}"
            )

        # Validate structure
        if not {factor, 'Main_Effect'}.issubset(me.columns):
            raise ValidationError(f"Main effects for '{factor}' missing required columns")

        rsg_r5 = rsg_r5.merge(me, on=factor, how='left', validate='many_to_one')
        # Rename to distinguish
        rsg_r5 = rsg_r5.rename(columns={'Main_Effect': f'ME_{factor}'})

    # Calculate interaction residual
    rsg_r5['Rx'] = rsg_r5['R5'] - rsg_r5[f'ME_{factor1}'] - rsg_r5[f'ME_{factor2}']

    return rsg_r5[[factor1, factor2, 'Rx']]


def calculate_factor_interaction_scores(
    df: pd.DataFrame, rsg_vars: list[str], factor_interactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate factor interaction scores per row.

    FIE = R2 + Rx

    Where Rx is the two-factor interaction effect.

    Parameters
    ----------
    df : DataFrame
        Data with R2 column
    rsg_vars : list of str
        Factor names (first 2 used)
    factor_interactions : DataFrame
        From calculate_factor_interaction_effects()
        Must have columns: [factor1, factor2, 'Rx']

    Returns
    -------
    DataFrame
        Columns: [factor1, factor2, 'factor_interaction_effects']
    """
    if factor_interactions.empty:
        return pd.DataFrame()

    if 'R2' not in df.columns:
        raise ValidationError('Cannot calculate interaction scores - R2 missing')

    factor1, factor2 = rsg_vars[0], rsg_vars[1]

    # Merge to add Rx to each row
    merged = df[[factor1, factor2, 'R2']].merge(
        factor_interactions, on=[factor1, factor2], how='left', validate='many_to_one'
    )

    merged['factor_interaction_effects'] = merged['R2'] + merged['Rx']

    return merged[[factor1, factor2, 'factor_interaction_effects']]


# ============================================================================
# Orchestration Class
# ============================================================================


class EffectsCalculator:
    """
    Calculates main effects and interactions from VAS residuals.

    This class coordinates calculation of:
    - Main effects (factor-level averages)
    - Time effects (time-point averages)
    - Main effect scores (per-row contributions)
    - Interaction effects (factor × time, factor × factor)

    Requires VAS residuals (R1, R2, R5) to be present in data.

    Examples
    --------
    Calculate all effects:

    >>> calc = EffectsCalculator()
    >>> effects_dict = calc.calculate_all_effects(df, spec)
    >>> effects_dict.keys()
    dict_keys(['main_effect', 'lane', 'head', 'time', 'lane_MEs', ...])
    """

    def calculate_all_effects(self, df: pd.DataFrame, spec: FormulationSpec) -> dict:
        """
        Calculate all main effects.

        Returns a dictionary containing:
        - 'main_effect': Main effects at RSG level (backward compatibility)
        - '{factor}': Main effects for each factor
        - 'time': Time main effects (if time variable specified)
        - '{factor}_MEs': Main effect scores for each factor

        Note: Interactions (factor_time, factor_factor) are calculated
        separately via calculate_interactions().

        Parameters
        ----------
        df : DataFrame
            Data with VAS residuals (R1, R2, R5)
        spec : FormulationSpec
            Analysis specification

        Returns
        -------
        dict
            Dictionary of effects DataFrames

        Examples
        --------
        >>> calc = EffectsCalculator()
        >>> effects = calc.calculate_all_effects(df, spec)
        >>> effects['lane']  # Main effects for lane factor
          lane  Main_Effect
        0    A         0.25
        1    B        -0.25
        >>> effects['time']  # Time main effects
           pull  PT_ME
        0     1   0.15
        1     2  -0.15
        """
        effects = {}

        if not spec.rsg_vars:
            logger.debug('No grouping variables - skipping effects calculation')
            return effects

        # Validate residuals present
        self._validate_residuals(df)

        # Calculate main effects for each factor
        logger.debug('Calculating factor main effects')
        for factor in spec.rsg_vars:
            me = calculate_factor_main_effects(df, factor)
            effects[factor] = me

        # Calculate RSG-level main effect (for backward compatibility)
        if spec.rsg_var_name:
            rsg_me = calculate_factor_main_effects(df, spec.rsg_var_name)
            effects['main_effect'] = rsg_me

        # Calculate time main effects
        if spec.has_time:
            logger.debug('Calculating time main effects')
            te = calculate_time_main_effects(df, spec.time_var)
            effects['time'] = te

        # Calculate main effect scores for each factor
        logger.debug('Calculating main effect scores')
        for factor in spec.rsg_vars:
            me = effects[factor]
            mes = calculate_main_effect_scores(df, factor, me)
            effects[f'{factor}_MEs'] = mes

        # Calculate factor interaction effects (if 2+ factors)
        # Note: factor × factor interaction is stored in interactions dict, not effects
        if len(spec.rsg_vars) >= 2:
            logger.debug('Calculating factor interaction effects')
            fi = calculate_factor_interaction_effects(df, spec.rsg_vars, effects)
            if not fi.empty:
                # Calculate per-row scores (kept in effects for backward compatibility)
                fie = calculate_factor_interaction_scores(df, spec.rsg_vars, fi)
                if not fie.empty:
                    effects['factor_interaction_effects'] = fie

        logger.debug('All effects calculated successfully')
        return effects

    def calculate_interactions(
        self, df: pd.DataFrame, spec: FormulationSpec, sds: int, effects: dict | None = None
    ) -> dict:
        """
        Calculate interaction effects.

        Calculates two types of interactions:
        - factor_time: Factor × time interaction (PDC - Process Disruption Component)
        - factor_factor: Factor × factor interaction (if 2+ factors)

        Parameters
        ----------
        df : DataFrame
            Data with VAS residuals
        spec : FormulationSpec
            Analysis specification
        sds : int
            Sampling Design State
        effects : dict, optional
            Effects dict from calculate_all_effects() (needed for factor_factor)

        Returns
        -------
        dict
            Dictionary with:
            - 'factor_time': Factor × time interaction (PDC)
            - 'factor_factor': Factor × factor interaction (if 2+ factors)
        """
        interactions = {}

        # Factor × time interaction (requires grouping + time)
        if spec.has_grouping and spec.has_time:
            if 'R3' not in df.columns:
                logger.warning('R3 not found - cannot calculate factor × time interaction')
            else:
                logger.debug('Calculating factor × time interaction (PDC)')

                if sds == 1:
                    # Full replication: use cell averages of R3
                    pdc = calculate_interaction_cell_means(df, spec.rsg_vars, spec.time_var)
                    interactions['factor_time'] = pdc

                elif sds == 2:
                    # No replication: direct calculation
                    pdc = calculate_pdc_by_time_sds2(
                        df,
                        df['Ybar_kt'],
                        df['Ybar_k'],
                        df['Ybar_t'],
                        df['Ybar'].iloc[0],  # Grand mean (constant)
                    )
                    interactions['factor_time'] = pdc

                else:
                    # SDS 3+: use SDS 1 approach
                    logger.debug(f'SDS {sds}: Using cell-average approach for PDC')
                    pdc = calculate_interaction_cell_means(df, spec.rsg_vars, spec.time_var)
                    interactions['factor_time'] = pdc

        # Factor × factor interaction (requires 2+ factors)
        if spec.rsg_vars and len(spec.rsg_vars) >= 2 and effects is not None:
            logger.debug('Calculating factor × factor interaction')
            fi = calculate_factor_interaction_effects(df, spec.rsg_vars, effects)
            if not fi.empty:
                interactions['factor_factor'] = fi

        return interactions

    def _validate_residuals(self, df: pd.DataFrame) -> None:
        """Validate that required residuals are present."""
        required = {'R1', 'R2', 'R5'}
        missing = required - set(df.columns)

        if missing:
            raise ValidationError(
                f'Cannot calculate effects - missing residuals: {missing}\n'
                f'Available columns: {df.columns.tolist()}\n'
                f'Fix: Calculate VAS residuals before calling calculate_all_effects()'
            )
