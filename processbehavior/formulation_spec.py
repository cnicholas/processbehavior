"""
FormulationSpec and ChartRequest - Frozen configuration for process behavior analysis.

FormulationSpec replaces DataPrepConfig + the structural half of AnalysisSpecification.
It is the single config object that flows through the system:
    formulate() -> AnalysisDataSet -> DataPreparation/ResidualCalculator/EffectsCalculator

ChartRequest replaces the behavioral half of AnalysisSpecification.
It is ephemeral, created in Study.execute() and consumed by Analysis.

Together they eliminate dict intermediaries and the DataPrepConfig -> AnalysisSpecification
inheritance chain.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormulationSpec:
    """Structural knowledge about the data. Created once in formulate(), immutable.

    This is the single config object that flows through the system:
    formulate() -> AnalysisDataSet -> DataPreparation/ResidualCalculator/EffectsCalculator

    Parameters
    ----------
    response_var : str
        Response variable (measurement) being analyzed.
    rsg_vars : tuple[str, ...] or None
        Rational subgrouping variables. Tuple for immutability.
    time_var : str or None
        Time/sequence variable for ordering observations.
    round_to : int
        Decimal places for rounding. Default 3.
    rsg_var_name : str
        Label for the composite rational subgroup column. Default 'rsg'.
    rsg_var_delim : str
        Delimiter for multi-variable grouping. Default '_'.
    unit_of_analysis : str or None
        The fundamental entity being measured (informational only).

    Raises
    ------
    ValueError
        If response_var is not provided.

    Examples
    --------
    >>> spec = FormulationSpec(
    ...     response_var='Height',
    ...     rsg_vars=('Operator', 'Machine'),
    ...     time_var='Time'
    ... )
    >>> spec.has_grouping
    True
    >>> spec.response_var
    'Height'
    """

    response_var: str
    rsg_vars: tuple[str, ...] | None = None
    time_var: str | None = None
    round_to: int = 3
    rsg_var_name: str = 'rsg'
    rsg_var_delim: str = '_'
    unit_of_analysis: str | None = None

    def __post_init__(self):
        if self.response_var is None:
            raise ValueError('A response variable is required!')

    @property
    def has_grouping(self) -> bool:
        """Return True if rational subgrouping variables are defined."""
        return self.rsg_vars is not None

    @property
    def has_time(self) -> bool:
        """Return True if time variable is defined."""
        return self.time_var is not None

    @property
    def rsg_vars_list(self) -> list[str]:
        """rsg_vars as a mutable list for pandas operations. Empty list if None."""
        return list(self.rsg_vars) if self.rsg_vars else []

    @property
    def requires_sort(self) -> bool:
        """Return True if data needs sorting (has time variable)."""
        return self.has_time


@dataclass(frozen=True)
class ChartRequest:
    """What execute() needs to run a specific chart. Ephemeral per execute() call.

    Created in Study.execute(), consumed by Analysis. Never stored long-term.

    Parameters
    ----------
    chart : str
        Chart type to produce ('Xbar', 'S', 'XmR', 'R', 'Histogram').
    by : tuple[str, ...] or None
        Factors to group/stratify by.
    value_col : str or None
        Explicit column to chart (response_var or residual column).
    residual : str or None
        Residual identifier (e.g., 'R2', 'R5') if charting a residual.
    residual_chart_type : str or None
        Base chart type when charting a residual.
    recentered : bool
        Whether to use re-centered residuals (RCR columns).
    companion : bool
        Whether to return both companion charts (Xbar+S or XmR+R).
    bins : int
        Number of bins for Histogram chart.
    phased : bool
        Whether to compute per-phase limits for collapsed factors.
    """

    chart: str
    by: tuple[str, ...] | None = None
    value_col: str | None = None
    residual: str | None = None
    residual_chart_type: str | None = None
    recentered: bool = False
    companion: bool = False
    bins: int = 10
    phased: bool = False
    n_sigma: float = 3.0
    n_mode: str = "actual"
