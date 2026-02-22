"""
Study class for process behavior analysis formulation.

The Study object represents a formulated analysis - it knows what data structure
you have (SDS), what charts are valid, and guides you toward correct analysis.

This is the "teaching" layer of the API that helps users understand their data
before running calculations.

Design Philosophy (Pythonic Hadley):
- Human-first: Rich __repr__ teaches users about their data
- Pit of success: Valid charts shown, invalid charts explained
- Composability: study.execute() returns AnalysisResult for chaining
- Immutable: Frozen dataclass, different formulations create new objects
"""

from __future__ import annotations

import difflib
import functools
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .exceptions import ChartNotAvailableError, FactorNotFoundError
from .spc_constants import RESIDUAL_ALIASES, VALID_BASE_CHARTS, normalize_chart_name

if TYPE_CHECKING:
    import pandas as pd

    from .analysis_dataset import AnalysisDataSet
    from .analysis_result import AnalysisResult
    from .capability import CapabilityResult, SpecLimits
    from .formulation_spec import FormulationSpec
    from .process_behavior import ProcessBehavior
    from .sds_detector import SDSAnalysisPlan, SDSResult


# Maximum number of combo strings to return in missing_combos/extra_combos
MAX_COMBO_DISPLAY = 100


@dataclass
class DesignReport:
    """
    Compares sampling plan to observed data.

    Returned by study.design(). Provides insight into the experimental
    design structure and any mismatches between plan and observation.

    Attributes
    ----------
    factors : pd.DataFrame
        Factor-level summary table with columns: factor, planned, observed,
        missing_levels, extra_levels
    missing_levels : dict[str, list]
        Levels in plan but not observed, per factor
    extra_levels : dict[str, list]
        Levels observed but not in plan, per factor
    K : int
        Planned K (product of factor level counts) or K_observed if no plan
    K_observed : int
        Observed K (nunique of rsg from analysis dataset)
    K_missing : int
        Count of missing RSG combinations
    T : int | None
        Planned T (time points) if specified
    T_observed : int | None
        Observed unique time points
    T_missing : int | None
        T - T_observed if T specified
    R : int | None
        Total cells: K × T (planned) or R_observed if no plan
    R_observed : int | None
        Actual unique (rsg, time) cells in data
    R_missing : int | None
        R - R_observed if R specified
    N : int | None
        Planned N (observations per cell) if specified
    N_observed : tuple[int, float, int] | None
        Observed (min, median, max) cell sizes
    missing_combos : list[str]
        RSG combinations in plan but not observed (capped at 100)
    extra_combos : list[str]
        RSG combinations observed but not in plan (capped at 100)
    extra_count : int
        Total count of extras (use when list is capped)
    sds_reason : str | None
        SDS classification reason from detector
    unit_of_analysis : str | None
        The fundamental entity being measured (if specified)
    structure_summary : str
        Summary of structure discrepancies

    Examples
    --------
    >>> design = study.design()
    >>> design
    DesignReport(2 factors, with plan)
      SDS reason: incomplete_with_replication
      K: planned=12, observed=8, missing=4
      T: planned=10, observed=8, missing=2
      R: planned=120, observed=64, missing=56
      N: planned=2, observed=(min=1, median=2.0, max=3)

      Factors:
        Lane: planned=[1,2,3,4], observed=[1,2,3,4]
        Phase: planned=[1,2,3], observed=[1,2], missing=[3]

      Structure: Incomplete: 4 RSG groups missing; 2 time points missing

    >>> design.factors
       factor     planned      observed  missing_levels  extra_levels
    0    Lane  [1,2,3,4]    [1,2,3,4]              []            []
    1   Phase    [1,2,3]        [1,2]             [3]            []

    >>> design.missing_levels
    {'Lane': [], 'Phase': [3]}

    >>> design.K
    12
    >>> design.K_observed
    8
    >>> design.missing_combos
    ['1_3', '2_3', '3_3', '4_3']
    """

    _sampling_plan: dict[str, list] | None
    _observed_levels: dict[str, list]
    _factors: list[str]  # Must be spec.rsg_vars order
    _delim: str = '_'  # RSG delimiter (from spec.rsg_var_delim)
    _T: int | None = None  # Planned T
    _N: int | None = None  # Planned N
    _T_observed: int | None = None  # nunique(time) from analysis dataset
    _N_observed: tuple[int, float, int] | None = None  # (min, median, max) cell sizes
    _observed_rsg_values: list[str] | None = None  # Actual rsg strings from analysis dataset
    _sds_reason: str | None = None  # From SDSResult.reason
    _unit_of_analysis: str | None = None  # Fundamental entity being measured
    _R_observed: int | None = None  # Count of unique (rsg, time) cells

    @property
    def factors(self) -> pd.DataFrame:
        """
        Factor-level summary table.

        Returns DataFrame with columns:
        - factor: Factor column name
        - planned: Levels in plan (or observed if no plan)
        - observed: Levels actually in data
        - missing_levels: Levels in plan but not in data
        - extra_levels: Levels in data but not in plan
        """
        import pandas as pd

        rows = []
        for factor in self._factors:
            observed = self._observed_levels.get(factor, [])
            planned = self._sampling_plan.get(factor, observed) if self._sampling_plan is not None else observed

            planned_set = set(planned)
            observed_set = set(observed)

            missing = self._safe_sort(list(planned_set - observed_set))
            extra = self._safe_sort(list(observed_set - planned_set))

            rows.append({
                'factor': factor,
                'planned': planned,
                'observed': observed,
                'missing_levels': missing,
                'extra_levels': extra
            })

        return pd.DataFrame(rows)

    @property
    def missing_levels(self) -> dict[str, list]:
        """Levels in plan but not observed, per factor."""
        result: dict[str, list] = {}
        for factor in self._factors:
            observed = set(self._observed_levels.get(factor, []))
            planned = set(self._sampling_plan.get(factor, [])) if self._sampling_plan is not None else observed
            result[factor] = self._safe_sort(list(planned - observed))
        return result

    @property
    def extra_levels(self) -> dict[str, list]:
        """Levels observed but not in plan, per factor."""
        result: dict[str, list] = {}
        for factor in self._factors:
            observed = set(self._observed_levels.get(factor, []))
            planned = set(self._sampling_plan.get(factor, [])) if self._sampling_plan is not None else observed
            result[factor] = self._safe_sort(list(observed - planned))
        return result

    @property
    def has_plan(self) -> bool:
        """Whether a sampling plan was provided."""
        return self._sampling_plan is not None

    @staticmethod
    def _safe_sort(items: list) -> list:
        """Sort items safely, handling mixed types."""
        try:
            return sorted(items)
        except TypeError:
            return list(items)

    @functools.cached_property
    def _expected_rsgs(self) -> set[str]:
        """
        Generate expected RSG strings from sampling plan.

        Uses encode_rsg() to safely generate strings from factor level tuples,
        avoiding the need to parse RSG strings back (which fails when factor
        values contain the delimiter).
        """
        if not self._sampling_plan:
            return set()
        from itertools import product

        from processbehavior.data_preparation import encode_rsg

        factor_levels = [self._sampling_plan[f] for f in self._factors]
        return {encode_rsg(combo, self._delim) for combo in product(*factor_levels)}

    def _rsg_in_plan(self, rsg: str) -> bool:
        """
        Check if an rsg string matches the sampling plan.

        Uses set membership against expected RSGs (O(1) lookup).
        This avoids parsing RSG strings, which fails when factor values
        contain the delimiter.
        """
        return rsg in self._expected_rsgs

    # =========================================================================
    # K, T, N Properties
    # =========================================================================

    @property
    def K(self) -> int:
        """
        Planned K: product of factor level counts.

        If no plan is provided, returns K_observed.
        Computed without materializing the cartesian product.
        """
        if not self._sampling_plan:
            return self.K_observed
        from math import prod
        return prod(len(self._sampling_plan[f]) for f in self._factors)

    @property
    def K_observed(self) -> int:
        """
        Observed K: actual unique RSG groups in data (nunique).

        This is the count of unique RSG values in the analysis dataset,
        NOT the product of observed factor levels (which can overstate
        sparse designs).
        """
        return len(self._observed_rsg_values) if self._observed_rsg_values else 0

    @property
    def K_missing(self) -> int:
        """
        Missing RSG groups: K - (observed combos that are in plan).

        Computed efficiently without generating the full cartesian product.
        Handles sparse designs where observed may have extra combos not in plan.
        """
        if not self._sampling_plan:
            return 0
        # Count observed rsg values that match the plan
        observed_in_plan = sum(
            1 for rsg in (self._observed_rsg_values or [])
            if self._rsg_in_plan(rsg)
        )
        return max(0, self.K - observed_in_plan)

    @property
    def T(self) -> int | None:
        """Planned time points (if specified in the plan)."""
        return self._T

    @property
    def T_observed(self) -> int | None:
        """Observed unique time points from analysis dataset."""
        return self._T_observed

    @property
    def T_missing(self) -> int | None:
        """
        Missing time points: T - T_observed.

        Returns None if T was not specified in the plan.
        """
        if self._T is None or self._T_observed is None:
            return None
        return max(0, self._T - self._T_observed)

    @property
    def R(self) -> int | None:
        """
        Total cells in design: K × T.

        With a plan: Returns K × T (planned total cells).
        Without a plan: Returns R_observed.
        Returns None if no time variable specified.
        """
        if not self._sampling_plan:
            return self.R_observed
        if self._T is None:
            return None
        return self.K * self._T

    @property
    def R_observed(self) -> int | None:
        """
        Actual unique (rsg, time) cells observed in data.

        This is the count of filled cells, not K_observed × T_observed
        (which would overstate sparse designs).
        """
        return self._R_observed

    @property
    def R_missing(self) -> int | None:
        """
        Missing cells: R - R_observed.

        Returns None if R is not specified (no time variable).
        """
        if self.R is None or self.R_observed is None:
            return None
        return max(0, self.R - self.R_observed)

    @property
    def N(self) -> int | None:
        """Planned observations per cell (if specified in the plan)."""
        return self._N

    @property
    def N_observed(self) -> tuple[int, float, int] | None:
        """
        Observed (min, median, max) cell sizes.

        Median is computed over cell sizes (groupby result), answering
        "typical replication per cell."
        """
        return self._N_observed

    @property
    def sds_reason(self) -> str | None:
        """
        SDS classification reason from detector.

        Possible values: 'no_structure', 'full_replication', 'no_replication',
        'partial_replication', 'single_condition', 'nested',
        'incomplete_with_replication', 'incomplete_no_replication'.
        """
        return self._sds_reason

    @property
    def sds_reason_detail(self) -> str | None:
        """
        Human-readable SDS classification with explanation.

        Adds context about why this SDS was assigned based on observed
        data structure. This describes what analysis approaches are valid.

        Examples:
            'full_replication (min n >= 2)'
            'no_replication (all cells n = 1)'
            'partial_replication (mixed cell sizes)'
        """
        if not self._sds_reason:
            return None

        if not self._N_observed:
            return self._sds_reason

        min_n, _, max_n = self._N_observed

        explanations = {
            'full_replication': f'min n = {min_n} >= 2',
            'no_replication': 'all cells n = 1',
            'partial_replication': f'cell sizes range {min_n} to {max_n}',
            'single_condition': 'single factor level over time',
            'implicit_single_condition': 'no factors defined',
            'nested': 'hierarchical factor structure',
            'incomplete_with_replication': f'coverage < 95%, has replication (min n = {min_n})',
            'incomplete_no_replication': 'coverage < 95%, all cells n = 1',
        }

        explanation = explanations.get(self._sds_reason, '')
        if explanation:
            return f"{self._sds_reason} ({explanation})"
        return self._sds_reason

    @property
    def plan_adherence(self) -> str | None:
        """
        Describes how well data collection matched the sampling plan.

        This is separate from SDS classification (which determines valid
        analysis approaches). Plan adherence answers: "Did data collection
        go as planned?"

        Returns None if no plan was provided.

        Examples:
            'complete'
            'underreplicated (min n = 2 < planned N = 4)'
            'incomplete_time (observed 8 of 10 time points)'
            'incomplete_factors (observed 5 of 6 factor combinations)'
        """
        if not self._sampling_plan:
            return None

        issues = []

        # Check replication adherence
        if self._N is not None and self._N_observed is not None:
            min_n, _, _ = self._N_observed
            if min_n < self._N:
                issues.append(f"underreplicated (min n = {min_n} < planned N = {self._N})")

        # Check time completeness
        if self._T is not None and self._T_observed is not None and self._T_observed < self._T:
            issues.append(f"incomplete_time (observed {self._T_observed} of {self._T} time points)")

        # Check factor completeness
        k_observed = self.K_observed
        k_planned = self.K
        if k_observed < k_planned:
            issues.append(f"incomplete_factors (observed {k_observed} of {k_planned} factor combinations)")

        if not issues:
            return "complete"

        return "; ".join(issues)

    @property
    def unit_of_analysis(self) -> str | None:
        """
        The fundamental entity being measured.

        For example, 'filled cup' or 'loan contract'. Returns None if not specified.
        """
        return self._unit_of_analysis

    @property
    def missing_combos(self) -> list[str]:
        """
        RSG combinations in plan but not observed (as rsg strings).

        Returns at most MAX_COMBO_DISPLAY items to prevent memory issues
        with large K. Use K_missing for the total count.

        Uses natural sort so '1_10' comes after '1_2'.
        """
        if not self._sampling_plan:
            return []
        from itertools import product

        from processbehavior.data_preparation import encode_rsg, natural_sort_key

        # Generate expected combos using _factors order (= spec.rsg_vars)
        factor_levels = [self._sampling_plan[f] for f in self._factors]
        # Use same delimiter as dataset
        expected = {encode_rsg(combo, self._delim) for combo in product(*factor_levels)}

        # Get observed combos
        observed = set(self._observed_rsg_values) if self._observed_rsg_values else set()

        # Natural sort (so 1_10 comes after 1_2, not before)
        result = sorted(expected - observed, key=natural_sort_key)
        return result[:MAX_COMBO_DISPLAY]

    @property
    def extra_count(self) -> int:
        """
        Count of RSG combinations observed but not in plan.

        Computed efficiently without generating full cartesian product.
        """
        if not self._sampling_plan:
            return 0
        return sum(
            1 for rsg in (self._observed_rsg_values or [])
            if not self._rsg_in_plan(rsg)
        )

    @property
    def extra_combos(self) -> list[str]:
        """
        RSG combinations observed but not in plan (as rsg strings).

        Returns at most MAX_COMBO_DISPLAY items. Use extra_count for total.

        Uses natural sort so '1_10' comes after '1_2'.
        """
        if not self._sampling_plan:
            return []

        from processbehavior.data_preparation import natural_sort_key

        # Efficient: check each observed value against plan without cartesian product
        extras = [
            rsg for rsg in (self._observed_rsg_values or [])
            if not self._rsg_in_plan(rsg)
        ]

        result = sorted(extras, key=natural_sort_key)
        return result[:MAX_COMBO_DISPLAY]

    @property
    def structure_summary(self) -> str:
        """
        Summary of structure discrepancy (distinct from SDS reason).

        Returns 'Complete structure' if plan matches observation,
        otherwise lists what's missing or extra.
        """
        issues = []
        if self.K_missing > 0:
            issues.append(f"{self.K_missing} RSG groups missing")
        if self.extra_count > 0:
            issues.append(f"{self.extra_count} extra RSG groups in data")
        if self.T_missing and self.T_missing > 0:
            issues.append(f"{self.T_missing} time points missing")
        if self._N is not None and self._N_observed:
            min_n, _, _ = self._N_observed
            if min_n < self._N:
                issues.append(f"some cells have n={min_n} < planned n={self._N}")

        if not issues:
            return "Complete structure"
        return "Incomplete: " + "; ".join(issues)

    def _repr_ktrn_lines(self) -> list[str]:
        """Build K/T/R/N summary lines for __repr__."""
        lines = []

        # K summary
        if self.has_plan:
            lines.append(f"  K: planned={self.K}, observed={self.K_observed}, missing={self.K_missing}")
        else:
            lines.append(f"  K: observed={self.K_observed}")

        # T summary
        if self._T is not None:
            lines.append(f"  T: planned={self._T}, observed={self._T_observed}, missing={self.T_missing}")
        elif self._T_observed is not None:
            lines.append(f"  T: observed={self._T_observed}")

        # R = K × T (total cells)
        if self.R is not None:
            if self.has_plan and self._T is not None:
                lines.append(f"  R: planned={self.R}, observed={self.R_observed}, missing={self.R_missing}")
            elif self.R_observed is not None:
                lines.append(f"  R: observed={self.R_observed}")

        # N summary
        if self._N is not None and self._N_observed is not None:
            min_n, med_n, max_n = self._N_observed
            lines.append(f"  N: planned={self._N}, observed=(min={min_n}, median={med_n}, max={max_n})")
        elif self._N_observed is not None:
            min_n, med_n, max_n = self._N_observed
            lines.append(f"  N: observed=(min={min_n}, median={med_n}, max={max_n})")

        return lines

    def __repr__(self) -> str:
        """Nice summary showing plan vs observed per factor with K/T/N."""
        plan_status = "with plan" if self.has_plan else "observed only"
        lines = [f"DesignReport({len(self._factors)} factors, {plan_status})"]

        # Unit of analysis (if specified)
        if self._unit_of_analysis:
            lines.append(f"  Unit of analysis: {self._unit_of_analysis}")

        # SDS classification reason with detail
        if self.sds_reason_detail:
            lines.append(f"  SDS reason: {self.sds_reason_detail}")
        elif self._sds_reason:
            lines.append(f"  SDS reason: {self._sds_reason}")

        # Plan adherence (only shown when plan is provided)
        if self.plan_adherence:
            lines.append(f"  Plan adherence: {self.plan_adherence}")

        # K/T/R/N summary
        lines.extend(self._repr_ktrn_lines())

        # Factor details
        lines.append("")
        lines.append("  Factors:")
        for factor in self._factors:
            observed = self._observed_levels.get(factor, [])
            planned = self._sampling_plan.get(factor, observed) if self._sampling_plan is not None else observed

            planned_set = set(planned)
            observed_set = set(observed)
            missing = self._safe_sort(list(planned_set - observed_set))

            line = f"    {factor}: observed={observed}"
            if self.has_plan:
                line = f"    {factor}: planned={planned}, observed={observed}"
            if missing:
                line += f", missing={missing}"

            lines.append(line)

        # Structure summary (discrepancy details)
        lines.append("")
        lines.append(f"  Structure: {self.structure_summary}")

        return '\n'.join(lines)


class StudyChartAccessor:
    """
    Provides IDE auto-completion for primary chart types in a Study.

    This class dynamically creates attributes for each valid primary chart type,
    enabling IDE auto-completion and preventing invalid chart selections.

    Usage:
        study = pb.formulate(response='weight', factors=['lane'])

        # IDE auto-completes valid primary charts
        result = study.execute(chart=study.charts.Xbar)

        # Chart residuals using value parameter
        result = study.execute(chart=study.charts.Xbar, value='R5')

    Attributes are set dynamically based on SDS-specific valid charts.
    """

    def __init__(self, valid_charts: list[str]):
        """
        Initialize accessor with valid primary chart types.

        Parameters
        ----------
        valid_charts : list of str
            Primary chart types (Xbar, S, XmR, R)
        """
        self._valid_charts = valid_charts

        # Dynamically add each valid chart as an attribute
        for chart in self._valid_charts:
            # Convert chart names to valid Python identifiers
            attr_name = chart.replace(':', '_').replace('-', '_')
            setattr(self, attr_name, chart)

    def __repr__(self) -> str:
        """Display available primary chart types."""
        return f"StudyChartAccessor({', '.join(self._valid_charts)})"

    def __dir__(self) -> list[str]:
        """Support for tab-completion in IPython/Jupyter."""
        return [c.replace(':', '_').replace('-', '_') for c in self._valid_charts]


@dataclass(frozen=True)
class Study:
    """
    A Study formulation for process behavior analysis.

    The Study object represents a complete formulation of how to analyze
    process behavior data. It encapsulates:

    - The data structure (Sampling Design State)
    - Valid and recommended chart types
    - Available residual analyses
    - Guidance on methodology
    - Pre-calculated dataset with residuals (R1-R5, RCR1-RCR5)

    This is an immutable object - to change the formulation, create a new Study.

    Parameters
    ----------
    _pdf : ProcessBehavior
        Reference to the data source
    _spec : FormulationSpec
        Structural configuration (from formulate())
    _plan : SDSAnalysisPlan
        Analysis plan based on detected SDS
    _ads : AnalysisDataSet
        Pre-calculated AnalysisDataSet with rsg, means, and residuals (R1-R5)

    Examples
    --------
    Create a study formulation:

    >>> pb = ProcessBehavior(df)
    >>> study = pb.formulate(response='weight', factors=['lane'], time='pull')
    >>> print(study)  # Rich display of formulation

    Check what's available:

    >>> study.sds  # 1-6
    >>> study.valid_charts  # ['Xbar', 'S', ...]
    >>> study.recommended_chart  # 'Xbar'

    Access the prepared dataset:

    >>> study.dataset  # DataFrame with rsg, R1-R5, RCR1-RCR5

    Run the analysis:

    >>> result = study.execute()  # Uses recommended chart
    >>> result = study.execute(chart='Xbar')  # Explicit chart
    >>> result = study.execute(chart=study.charts.Xbar)  # Via accessor

    See Also
    --------
    ProcessBehavior.formulate : Create a Study from data
    AnalysisResult : Result of study.execute()
    """
    _pdf: ProcessBehavior
    _spec: FormulationSpec
    _plan: SDSAnalysisPlan
    _ads: AnalysisDataSet
    _sampling_plan: dict[str, list] | None = None
    _factor_order: list[str] | None = None
    _T: int | None = None  # Planned time points
    _N: int | None = None  # Planned observations per cell
    _sds_result: SDSResult | None = None  # For accessing .reason in DesignReport

    # =========================================================================
    # User-Facing Properties (Clean Names)
    # =========================================================================

    @property
    def response(self) -> str:
        """
        The response variable being analyzed.

        This is the measurement or outcome variable that will be charted.
        """
        return self._spec.response_var

    @property
    def factors(self) -> list[str] | None:
        """
        Grouping factors defining rational subgroups.

        These are the categorical variables (like Lane, Operator, Machine)
        that define how observations are grouped. Returns None if no
        factors are specified.
        """
        rsg_vars = self._spec.rsg_vars
        return list(rsg_vars) if rsg_vars else None

    @property
    def time(self) -> str | None:
        """
        Time variable for ordering observations.

        This defines the sequence of measurements (like Pull, Day, Hour).
        Returns None if no time variable is specified.
        """
        return self._spec.time_var

    @property
    def precision(self) -> int:
        """
        Decimal precision for output values.

        Statistics and chart values will be rounded to this many decimal places.
        """
        return self._spec.round_to

    @property
    def unit_of_analysis(self) -> str | None:
        """
        The fundamental entity being measured.

        For example, in a manufacturing process producing cups filled with yogurt,
        the unit of analysis is 'filled cup'. In a loan collection process, it
        would be 'loan contract'. Returns None if not specified.
        """
        return self._spec.unit_of_analysis

    @property
    def dataset(self) -> pd.DataFrame:
        """
        Full analysis dataset with means and residuals.

        This is the pre-calculated dataset produced during formulate().
        It includes:
        - rsg: Rational subgroup identifier
        - Ybar, Ybar_k, Ybar_t, Ybar_kt: Hierarchical means
        - R1-R5: VAS residuals (where applicable for the SDS)
        - RCR1-RCR5: Re-centered residuals (Y reconstructed from components)

        Returns a copy to preserve immutability.

        Returns
        -------
        pd.DataFrame
            Copy of the analysis dataset

        Examples
        --------
        >>> study = pdf.formulate(response='weight', factors=['lane'], time='pull')
        >>> df = study.dataset
        >>> df[['rsg', 'weight', 'R1', 'R2', 'R3', 'R4', 'R5']].head()
        """
        return self._ads.analysis_dataset.copy()

    # =========================================================================
    # SDS Properties (Sampling Design State)
    # =========================================================================

    @property
    def sds(self) -> int:
        """
        Sampling Design State (0-6).

        The SDS classifies your data structure based on:
        - Whether you have grouping factors
        - Whether you have a time variable
        - Whether you have replication (multiple observations per cell)

        SDS determines which charts are valid and how residuals are calculated.

        Returns
        -------
        int
            SDS value from 0 to 6

        See Also
        --------
        sds_name : Human-readable name for the SDS
        sds_description : Detailed explanation
        """
        return self._plan.sds

    @property
    def sds_name(self) -> str:
        """
        Human-readable name for the detected Sampling Design State.

        Examples: "Full Factorial with Replication", "Time Series Only"
        """
        return self._plan.name

    @property
    def sds_description(self) -> str:
        """
        Detailed description of the detected data structure.

        Explains what the SDS means in terms of your data structure
        and what analysis approaches are appropriate.
        """
        return self._plan.description

    # =========================================================================
    # Chart Properties
    # =========================================================================

    @property
    def valid_charts(self) -> list[str]:
        """
        Chart types that are valid for this data structure.

        These are the primary control charts that can be created
        based on the detected SDS.

        Returns
        -------
        list of str
            Valid chart types (e.g., ['Xbar', 'S', 'XmR'])
        """
        return self._plan.valid_charts

    @property
    def recommended_chart(self) -> str:
        """
        The recommended chart type for this data structure.

        This is the chart type that best suits your data based on
        Wheeler & Bishop methodology.
        """
        return self._plan.recommended_chart

    @property
    def residual_charts(self) -> list[str]:
        """
        Available residual chart types for VAS analysis.

        Residual charts help diagnose sources of variation:
        - R2: Within-subgroup variation (measurement noise)
        - R3: Interaction effects (factor × time)
        - R4: Time effects (trends, shifts over time)
        - R5: Factor effects (differences between levels)

        Returns
        -------
        list of str
            Available residual chart types (e.g., ['R2_S', 'R3_XmR'])
        """
        return self._plan.residual_charts

    @property
    def charts(self) -> StudyChartAccessor:
        """
        Accessor for IDE auto-completion of primary chart types.

        Primary charts are the standard process behavior charts:
        Xbar, S, XmR, R

        For residual charts (VAS analysis), use study.residuals instead.

        Usage:
            study.charts.Xbar  # Auto-completes valid primary charts
            study.charts.XmR

        Returns
        -------
        StudyChartAccessor
            Object with primary chart types as attributes
        """
        return StudyChartAccessor(self.valid_charts)

    @property
    def available_residuals(self) -> list[str]:
        """
        Available residual types (R1-R5) for this study.

        Returns unique residual identifiers that can be used with the `value`
        parameter in execute(). Residuals decompose sources of variation:

        - R1: Total residual (y - grand mean)
        - R2: Within-subgroup variation (measurement noise)
        - R3: Interaction effects (factor × time)
        - R4: Time effects (trends, shifts over time)
        - R5: Factor effects (differences between levels)

        Usage:
            study.execute(chart='Xbar', value='R5')  # Factor effects
            study.execute(chart='XmR', value='R4')   # Time effects

        Returns
        -------
        list[str]
            Available residual identifiers (e.g., ['R1', 'R2', 'R3', 'R4', 'R5'])

        See Also
        --------
        residual_charts : Full list of residual+chart combinations
        """
        # Extract unique residual IDs from residual_charts (e.g., R2_S -> R2)
        residual_ids = sorted(set(
            chart.split('_')[0] for chart in self.residual_charts
        ))
        return residual_ids

    @property
    def support(self) -> pd.DataFrame:
        """
        Chart support matrix for this study.

        Returns a DataFrame with one row per chart type showing availability,
        recommendations, and explanations. This is the single source of truth
        for chart capabilities.

        Returns
        -------
        pd.DataFrame
            Columns: chart, category, available, recommended, reason, question

        Examples
        --------
        >>> study.support
               chart  category  available  recommended  ...
        0       Xbar   primary       True         True  ...
        1          S   primary       True        False  ...

        >>> study.support[study.support['available']]  # Filter to available
        >>> study.support.query("category == 'residual'")  # Residual charts
        """
        import pandas as pd

        from .sds_detector import SDSAnalysisPlan

        rows = []

        # All possible primary charts
        ALL_PRIMARY = ['Xbar', 'S', 'XmR', 'R']

        # Build invalid_reasons dict from _plan.invalid_charts
        invalid_reasons = self._parse_invalid_charts()

        for chart in ALL_PRIMARY:
            rows.append({
                'chart': chart,
                'category': 'primary',
                'available': chart in self.valid_charts,
                'recommended': chart == self.recommended_chart,
                'reason': invalid_reasons.get(chart),
                'question': SDSAnalysisPlan.CHART_QUESTIONS.get(chart, '')
            })

        # All possible residual charts
        ALL_RESIDUALS = [
            'R2_S', 'R2_XmR',
            'R3_Xbar', 'R3_S', 'R3_XmR',
            'R4_Xbar', 'R4_S', 'R4_XmR',
            'R5_Xbar', 'R5_S', 'R5_XmR'
        ]

        for chart in ALL_RESIDUALS:
            available = chart in self.residual_charts
            rows.append({
                'chart': chart,
                'category': 'residual',
                'available': available,
                'recommended': False,
                'reason': None if available else 'Not available for this SDS',
                'question': SDSAnalysisPlan.CHART_QUESTIONS.get(chart, '')
            })

        return pd.DataFrame(rows)

    def _parse_invalid_charts(self) -> dict[str, str]:
        """
        Parse invalid_charts list into dict of chart → reason.

        The _plan.invalid_charts format is: ['S (requires n≥2 per subgroup)']
        This parses to: {'S': 'requires n≥2 per subgroup'}
        """
        result = {}
        for entry in self._plan.invalid_charts:
            # Format: 'ChartType (reason)'
            if '(' in entry and entry.endswith(')'):
                chart = entry.split('(')[0].strip()
                reason = entry[entry.index('(') + 1:-1]
                result[chart] = reason
        return result

    # =========================================================================
    # Guidance Methods
    # =========================================================================

    def why_not(self, chart: str) -> str:
        """
        Explain why a chart type is or isn't available for this study.

        This is a teaching method - it helps users understand the
        methodology by explaining constraints. Uses the support DataFrame
        as the single source of truth.

        Parameters
        ----------
        chart : str
            Chart type to check (e.g., 'XmR', 'S', 'R2_S')

        Returns
        -------
        str
            Explanation of availability with the question the chart answers

        Examples
        --------
        >>> study.why_not('S')
        "'S' unavailable: requires n≥2 per subgroup"

        >>> study.why_not('Xbar')
        "'Xbar' IS available. Are subgroup means stable over time?"
        """
        df = self.support
        row = df[df['chart'] == chart]

        if row.empty:
            return f"'{chart}' is not a recognized chart type. Use study.support to see all options."

        row = row.iloc[0]
        if row['available']:
            return f"'{chart}' IS available. {row['question']}"
        else:
            return f"'{chart}' unavailable: {row['reason']}"

    def design(self) -> DesignReport:
        """
        Get design report comparing sampling plan to observed data.

        Returns a DesignReport showing factors, levels, and any mismatches
        between planned and observed structure. Works with or without a
        sampling plan:

        - With plan: Shows planned vs observed, highlights missing/extra levels
        - Without plan: Shows observed structure only

        Note: Time Handling
        -------------------
        The sampling plan specifies factor levels only, not time levels.
        For SDS detection, observed unique time values are used as the
        "planned" time set. This means coverage detects missing factor
        combos within observed time blocks, but time points not in the
        data are not considered "missing."

        Returns
        -------
        DesignReport
            Object with:
            - factors: DataFrame with planned/observed/missing/extra per factor
            - missing_levels: dict of levels in plan but not observed
            - extra_levels: dict of levels observed but not in plan
            - has_plan: whether a sampling plan was provided
            - K, K_observed, K_missing: RSG group counts
            - T, T_observed, T_missing: Time point counts (if T specified)
            - N, N_observed: Cell size info (if N specified)
            - missing_combos, extra_combos: RSG string lists
            - sds_reason, structure_summary: Diagnostic info

        Examples
        --------
        >>> study = pb.formulate(
        ...     response=pb.cols.Weight,
        ...     time=pb.cols.Pull,
        ...     plan={
        ...         'factors': {pb.cols.Lane: [1,2,3,4], pb.cols.Phase: [1,2,3]},
        ...         'T': 10,
        ...         'N': 2
        ...     }
        ... )
        >>> design = study.design()
        >>> design
        DesignReport(2 factors, with plan)
          SDS reason: incomplete_with_replication
          K: planned=12, observed=8, missing=4
          T: planned=10, observed=8, missing=2
          N: planned=2, observed=(min=1, median=2.0, max=3)

          Factors:
            Lane: planned=[1,2,3,4], observed=[1,2,3,4]
            Phase: planned=[1,2,3], observed=[1,2], missing=[3]

          Structure: Incomplete: 4 RSG groups missing; 2 time points missing

        >>> design.missing_levels
        {'Lane': [], 'Phase': [3]}

        >>> design.K
        12
        >>> design.missing_combos
        ['1_3', '2_3', '3_3', '4_3']

        Without a plan (shows observed structure):

        >>> study = pb.formulate(response='weight', factors=['lane', 'phase'])
        >>> study.design()
        DesignReport(2 factors, observed only)
          K: observed=4
          ...
        """
        # _factors MUST be spec.rsg_vars order (not plan dict order)
        # to ensure correct cartesian product encoding
        factors = self._spec.rsg_vars_list

        # Use analysis dataset (post-filtering, NA-handled) for all observed values
        ads_df = self._ads.analysis_dataset

        # Get observed levels from analysis dataset
        observed_levels: dict[str, list] = {}
        for factor in factors:
            if factor in ads_df.columns:
                values = ads_df[factor].dropna().unique()
                try:
                    observed_levels[factor] = sorted(values.tolist())
                except TypeError:
                    observed_levels[factor] = list(values)
            else:
                observed_levels[factor] = []

        # Compute T_observed from analysis dataset
        T_observed = None
        if self._spec.time_var and self._spec.time_var in ads_df.columns:
            T_observed = int(ads_df[self._spec.time_var].nunique())

        # Compute observed RSG values, N_observed, and R_observed
        N_observed = None
        R_observed = None
        observed_rsg_values: list[str] = []
        rsg_col = self._spec.rsg_var_name

        if rsg_col and rsg_col in ads_df.columns:
            # Get actual rsg values from analysis dataset
            observed_rsg_values = ads_df[rsg_col].unique().tolist()

            # Cell sizes: group by (rsg, time) or just rsg
            if self._spec.time_var and self._spec.time_var in ads_df.columns:
                cell_sizes = ads_df.groupby(
                    [rsg_col, self._spec.time_var], observed=True
                ).size()
                # R_observed = count of unique (rsg, time) cells
                R_observed = len(cell_sizes)
            else:
                cell_sizes = ads_df.groupby([rsg_col], observed=True).size()

            # N_observed = (min, median, max)
            if len(cell_sizes) > 0:
                N_observed = (
                    int(cell_sizes.min()),
                    float(cell_sizes.median()),
                    int(cell_sizes.max())
                )

        return DesignReport(
            _sampling_plan=self._sampling_plan,
            _observed_levels=observed_levels,
            _factors=factors,
            _delim=self._spec.rsg_var_delim,
            _T=self._T,
            _N=self._N,
            _T_observed=T_observed,
            _N_observed=N_observed,
            _observed_rsg_values=observed_rsg_values,
            _sds_reason=self._sds_result.reason if self._sds_result else None,
            _unit_of_analysis=self._spec.unit_of_analysis,
            _R_observed=R_observed,
        )

    def capability(
        self,
        specs: SpecLimits | None = None,
        *,
        usl: float | None = None,
        lsl: float | None = None,
        target: float | None = None,
    ) -> CapabilityResult:
        """
        Assess process capability against specification limits (Ch. 16).

        Parameters
        ----------
        specs : SpecLimits, optional
            Pre-built specification limits.  If provided, keyword arguments
            are ignored.
        usl : float, optional
            Upper specification limit.
        lsl : float, optional
            Lower specification limit.
        target : float, optional
            Target value.

        Returns
        -------
        CapabilityResult
            Frozen dataclass with Pp/Ppk, Cp/Cpk (when R2 available),
            Z-scores, and empirical percent outside.

        Examples
        --------
        >>> cap = study.capability(usl=250.5, lsl=249.5, target=250.0)
        >>> cap.ppk
        1.42

        >>> cap2 = study.capability(usl=251.0, lsl=249.0)  # wider specs
        """
        from .capability import SpecLimits as _SpecLimits
        from .capability import assess_capability

        if specs is None:
            specs = _SpecLimits(usl=usl, lsl=lsl, target=target)
        return assess_capability(self._ads, specs, round_to=self._spec.round_to)

    def execute(
        self,
        chart: str | None = None,
        by: list[str] | None = None,
        value: str | None = None,
        recentered: bool = False,
        bins: int | None = None,
        paired: bool = False,
        staged: bool = False,
        n_sigma: float = 3.0,
        n_mode: str = "actual",
    ) -> AnalysisResult:
        """
        Run the analysis and return results.

        This executes the formulated study using the specified chart type
        (or the recommended chart if none specified). The `by` parameter
        controls how data is grouped/stratified, and `value` specifies
        what to chart (response or residuals).

        IMPORTANT: `by` creates a VIEW over the immutable analytic dataset.
        It does NOT recompute residuals or change the decomposition of variation.
        Residual values are identical regardless of `by`.

        Parameters
        ----------
        chart : str, optional
            Chart type to use. If None, uses recommended_chart.
            Can use study.charts.Xbar for IDE auto-completion.

            Valid charts: 'Xbar', 'S', 'XmR', 'R'

        by : list[str], optional
            Factors to group/stratify by. Controls view granularity.

            - by=None: Default for chart type (full rsg_key for Xbar/S,
              ERROR for XmR/R with factors - must be explicit)
            - by=[]: Collapse all factors (single overall chart/group)
            - by=['factor']: Stratify/aggregate by single factor
            - by=['f1', 'f2']: Stratify/aggregate by factor combination

            For Xbar/S: by controls aggregation (groups on x-axis)
            For XmR/R: by controls stratification (separate charts)

        value : str, optional
            What to chart. Options:

            - None or 'response': Chart the response variable (default)
            - 'R1' through 'R5': Chart the specified residual

        recentered : bool, default False
            For residual charts only. If True, uses re-centered residuals
            (RCR2, RCR3, etc.) which add back the appropriate mean for
            easier interpretation. See Tom Bishop Equation 80.

        paired : bool, default False
            When True, returns both Xbar and S charts together (or both XmR
            and R charts) regardless of which chart is requested. This follows
            Wheeler's methodology of always analyzing paired charts together.

            - paired=False (default): Returns only the requested chart (SRP-compliant)
            - paired=True: Returns both paired charts (Xbar+S or XmR+R)

        staged : bool, default False
            When True, computes per-stage center lines and control limits.
            Each stage is a contiguous run of the same collapsed factor
            combination (the groups demarcated by vertical lane boundaries).
            Within each stage, the moving range is computed independently.

            - staged=False (default): Global limits across entire chart
            - staged=True: Per-stage limits (requires by=[])

            Single-point stages yield zero-width limits because mR is
            imputed as 0; count available in metadata['single_point_stages'].

        n_sigma : float, default 3.0
            Sigma multiplier for Xbar/S chart control limits. Standard SPC
            uses 3-sigma limits; smaller values give narrower limits (more
            sensitive), larger values give wider limits (fewer false signals).
            Only valid for Xbar/S charts.

        n_mode : str, default "actual"
            How to determine subgroup size N for Xbar/S limit calculations.

            - "actual": Use each subgroup's actual size N_k (default).
              When sizes vary, limits vary per subgroup ("Varies" in statistics).
            - "average": Use the mean subgroup size N-bar for all subgroups.
              Produces constant limits even when subgroup sizes vary.

            Only valid for Xbar/S charts.

        Returns
        -------
        AnalysisResult
            Complete analysis results with charts, statistics, residuals.
            Use result.plot() to visualize or result.to_excel() to export.

        Raises
        ------
        ValueError
            If specified chart is not valid for this SDS
            If by contains invalid factors
            If XmR/R with factors but by not specified

        Examples
        --------
        Use recommended chart:

        >>> result = study.execute()

        Specify chart explicitly:

        >>> result = study.execute(chart='Xbar')
        >>> result = study.execute(chart=study.charts.Xbar)

        Use by parameter for views:

        >>> # Xbar aggregated by factor 1
        >>> result = study.execute(chart='Xbar', by=['factor 1'])

        >>> # IMR stratified by all factors (separate chart per combo)
        >>> result = study.execute(chart='XmR', by=['factor 1', 'factor 2'])

        >>> # IMR single overall chart
        >>> result = study.execute(chart='XmR', by=[])

        Chart residuals:

        >>> result = study.execute(chart='Xbar', value='R5')  # Factor effects
        >>> result = study.execute(chart='XmR', value='R4', recentered=True)

        Chain to visualization:

        >>> study.execute().plot()
        """
        # Import here to avoid circular imports
        from .analysis import Analysis

        # Determine chart type
        chart_request = chart or self.recommended_chart

        # Parse and validate chart request (returns base chart type only)
        base_chart = self._parse_chart_request(chart_request)

        # Validate by parameter (may raise ValueError)
        by_validated = self._validate_by_parameter(by, base_chart)

        # Recentered validation - only R1-R5 allowed with recentered=True
        if recentered:
            RECENTERABLE_VALUES = {'R1', 'R2', 'R3', 'R4', 'R5'}
            if value is None:
                from .exceptions import ValidationError
                raise ValidationError(
                    "recentered=True requires a residual value (R1-R5), got value=None"
                )
            if value.upper() not in RECENTERABLE_VALUES:
                from .exceptions import ValidationError
                raise ValidationError(
                    f"recentered=True requires a residual value (R1-R5), got '{value}'"
                )

        # Staged limits validation
        if staged:
            if base_chart not in ('XmR', 'R'):
                from .exceptions import ValidationError
                raise ValidationError(
                    f"staged=True is only valid for XmR or R charts, "
                    f"got '{base_chart}'."
                )
            if by_validated is None:
                from .exceptions import ValidationError
                raise ValidationError(
                    "staged=True requires an explicit by= parameter."
                )
            if not self._spec.rsg_vars_list:
                from .exceptions import ValidationError
                raise ValidationError(
                    "staged=True requires factors to define stages (rsg_key). "
                    "This study has no factors; staged limits do not apply."
                )
            if len(by_validated) > 0:
                all_factors = set(self._spec.rsg_vars_list)
                by_set = set(by_validated)
                if not (all_factors - by_set):
                    from .exceptions import ValidationError
                    raise ValidationError(
                        "staged=True requires collapsed factors to define stages. "
                        f"by={list(by_validated)} includes all factors; "
                        "no factors remain to create stage boundaries. "
                        "Remove factors from by= or set staged=False."
                    )

        # n_sigma / n_mode validation
        self._validate_n_sigma_n_mode(n_sigma, n_mode, base_chart)

        # For Histogram chart, default by=[] if not specified (full distribution)
        if base_chart == 'Histogram' and by is None:
            by_validated = []

        # Primary chart validation
        if base_chart not in self.valid_charts:
            available_list = list(self.valid_charts)
            raise ChartNotAvailableError(
                f"Chart type '{base_chart}' is not valid for SDS {self.sds}.\n"
                f"Valid charts: {', '.join(available_list)}\n"
                f"Recommended: {self.recommended_chart}\n"
                f"Use study.why_not('{base_chart}') for explanation.",
                chart=base_chart,
                available=available_list
            )

        # Resolve value column (response or residual)
        value_col = self._resolve_value_column(value, recentered)

        # Validate residual availability if charting a residual
        # Skip validation for Histogram - histograms can plot any available numeric column
        if value is not None and value.upper().startswith('R') and base_chart != 'Histogram':
            # Extract residual identifier (R1-R5, RCR1-RCR5)
            residual_id = value.upper()
            if residual_id.startswith('RCR'):
                residual_id = f"R{residual_id[3:]}"

            # Check availability using canonical name
            canonical_name = f"{residual_id}_{base_chart}"
            if canonical_name not in self.residual_charts:
                available_list = list(self.residual_charts) if self.residual_charts else []
                available_str = ', '.join(available_list) if available_list else 'None'
                raise ChartNotAvailableError(
                    f"Residual chart '{canonical_name}' is not available for SDS {self.sds}.\n"
                    f"Available residual charts: {available_str}\n"
                    f"Use study.residual_charts to see available options.",
                    chart=canonical_name,
                    available=available_list
                )

        # Determine if this is a residual chart
        is_residual = value is not None and value.upper().startswith('R')

        # Build chart request (ephemeral, per-execute)
        from .formulation_spec import ChartRequest

        request = ChartRequest(
            chart=base_chart,
            by=tuple(by_validated) if by_validated is not None else None,
            value_col=value_col,
            residual=value.upper() if is_residual else None,
            residual_chart_type=base_chart if is_residual else None,
            recentered=recentered,
            paired=paired,
            bins=bins if bins is not None else 10,
            staged=staged,
            n_sigma=n_sigma,
            n_mode=n_mode,
        )

        # Create and run analysis using pre-calculated AnalysisDataSet
        # This makes execute() cheap - the expensive residual calculation was done in formulate()
        analysis = Analysis(self._spec, request, analysis_dataset=self._ads)
        return analysis.calculate()

    def _parse_chart_request(self, chart: str) -> str:
        """
        Parse chart string to validate base chart type.

        Parameters
        ----------
        chart : str
            Chart request string. Only accepts base charts: 'Xbar', 'S', 'XmR', 'R'

        Returns
        -------
        str
            Validated base chart type

        Raises
        ------
        ValueError
            For invalid chart types or old residual chart syntax
        """
        if not chart or not isinstance(chart, str):
            raise ValueError("Chart name must be a non-empty string")

        # Normalize case (e.g. "xmr" -> "XmR", "XBAR" -> "Xbar")
        chart = normalize_chart_name(chart)

        # Base chart - valid
        if chart in VALID_BASE_CHARTS:
            return chart

        # Detect old residual chart syntax and provide migration guidance
        if "_" in chart:
            # Old syntax: R5_Xbar, noise_Xbar, rc_R5_Xbar, RCR5_Xbar
            self._raise_old_syntax_error(chart)

        # R1-R5 without base chart
        if re.match(r'^R\d+$', chart) or re.match(r'^RCR\d+$', chart):
            raise ValueError(
                f"'{chart}' is a residual identifier, not a chart type.\n"
                f"Use: study.execute(chart='Xbar', value='{chart}')\n"
                f"Or: study.execute(chart='XmR', value='{chart}')"
            )

        # Alias without base chart
        if chart in RESIDUAL_ALIASES:
            residual_id = RESIDUAL_ALIASES[chart]["id"]
            raise ValueError(
                f"'{chart}' is a residual alias for {residual_id}, not a chart type.\n"
                f"Use: study.execute(chart='Xbar', value='{residual_id}')\n"
                f"Or: study.execute(chart='XmR', value='{residual_id}')"
            )

        raise ValueError(
            f"Unknown chart '{chart}'. "
            f"Valid chart types: {', '.join(sorted(VALID_BASE_CHARTS))}"
        )

    def _raise_old_syntax_error(self, chart: str) -> None:
        """Raise helpful error for old residual chart syntax."""
        # Strip rc_ prefix if present
        working = chart
        recentered = False
        if working.startswith("rc_"):
            recentered = True
            working = working[3:]

        # Try to parse residual and base chart
        if "_" in working:
            residual_part, base_chart = working.rsplit("_", 1)

            # Check for RCR prefix
            if residual_part.startswith("RCR"):
                residual_id = f"R{residual_part[3:]}"
                recentered = True
            elif residual_part.startswith("R") and residual_part[1:].isdigit():
                residual_id = residual_part
            elif residual_part in RESIDUAL_ALIASES:
                residual_id = RESIDUAL_ALIASES[residual_part]["id"]
            else:
                residual_id = residual_part

            if base_chart in VALID_BASE_CHARTS:
                recentered_hint = ", recentered=True" if recentered else ""
                raise ValueError(
                    f"Old residual chart syntax '{chart}' is no longer supported.\n"
                    f"Use: study.execute(chart='{base_chart}', value='{residual_id}'{recentered_hint})"
                )

        raise ValueError(
            f"Invalid chart name '{chart}'. "
            f"Valid chart types: {', '.join(sorted(VALID_BASE_CHARTS))}"
        )

    def _validate_by_parameter(  # noqa: C901
        self,
        by: list[str] | None,
        base_chart: str
    ) -> list[str] | None:
        """
        Validate and normalize the `by` parameter.

        Parameters
        ----------
        by : list[str] | None
            User-specified by parameter
        base_chart : str
            Base chart type being requested (Xbar, S, XmR, R)

        Returns
        -------
        list[str] | None
            Normalized by parameter

        Raises
        ------
        ValueError
            If by contains factors not in rsg_vars
            If by=None for IMR/R with factors (must be explicit)
            If by contains time variable for IMR/R charts
        """
        factors = self._spec.rsg_vars_list
        time_var = self._spec.time_var
        is_time_series_chart = base_chart in ('XmR', 'R')

        # No factors case: by=None is fine, by=[] is also fine
        if not factors:
            if by is not None and by != []:
                raise ValueError(
                    f"Invalid by={by}. No factors defined in this study. "
                    f"Use by=None or by=[] for single stream."
                )
            return []  # Normalize to empty list for no-factor case

        # Has factors case
        if by is None:
            if is_time_series_chart:
                # IMR/R with factors requires explicit by
                factor_str = ', '.join(f"'{f}'" for f in factors)
                raise ValueError(
                    f"IMR/R charts with factors require explicit 'by' parameter.\n"
                    f"Specify how to stratify:\n"
                    f"  by=[{factor_str}] for {self._get_factor_combinations()} charts (one per factor combination)\n"
                    f"  by=['{factors[0]}'] for fewer charts (stratify by single factor)\n"
                    f"  by=[] for single overall chart"
                )
            # Xbar/S: by=None means full rsg_key (current behavior)
            return None

        # Normalize string to list
        if isinstance(by, str):
            by = [by]

        # Validate by is subset of valid dimensions
        # - IMR/R: factors only (time is x-axis)
        # - Xbar/S: factors + time (cell_key dimensions)
        by_set = set(by)
        factor_set = set(factors)
        invalid = by_set - factor_set

        if invalid and time_var and time_var in invalid:  # noqa: SIM102
            # Check if user tried to use time
                if is_time_series_chart:
                    raise ValueError(
                        f"Cannot use time variable '{time_var}' in by for {base_chart} charts. "
                        f"Time is the x-axis for {base_chart} charts, not a stratification dimension. "
                        f"Valid by dimensions: {sorted(factors)}"
                    )
                # For Xbar/S, time in by means group by time (one point per time)
                # This is valid - remove from invalid set
                invalid = invalid - {time_var}

        if invalid:
            # Build valid dimensions list based on chart type
            valid_dims = sorted(factors)
            if time_var and not is_time_series_chart:
                valid_dims = sorted(factors + [time_var])

            # Build helpful error message with suggestions
            suggestions = []
            first_invalid = None
            first_suggestion = None

            for inv in sorted(invalid):
                # Check for case-insensitive match first
                case_match = next(
                    (v for v in valid_dims if v.lower() == inv.lower()),
                    None
                )
                if case_match:
                    suggestions.append(
                        f"'{inv}' is not a valid by variable.\n"
                        f"Did you mean '{case_match}'? (names are case-sensitive)"
                    )
                    if first_invalid is None:
                        first_invalid = inv
                        first_suggestion = case_match
                else:
                    # Try fuzzy matching
                    close = difflib.get_close_matches(inv, valid_dims, n=1, cutoff=0.6)
                    if close:
                        suggestions.append(
                            f"'{inv}' is not a valid by variable.\n"
                            f"Did you mean '{close[0]}'?"
                        )
                        if first_invalid is None:
                            first_invalid = inv
                            first_suggestion = close[0]
                    else:
                        suggestions.append(f"'{inv}' is not a valid by variable.")
                        if first_invalid is None:
                            first_invalid = inv

            msg = "\n".join(suggestions)
            msg += f"\nValid: {', '.join(valid_dims)}"
            raise FactorNotFoundError(
                msg,
                factor=first_invalid,
                suggestion=first_suggestion,
                available=valid_dims
            )

        return list(by)

    @staticmethod
    def _validate_n_sigma_n_mode(
        n_sigma: float, n_mode: str, base_chart: str
    ) -> None:
        """Validate n_sigma and n_mode parameters for execute()."""
        from .exceptions import ValidationError

        if (n_sigma != 3.0 or n_mode != "actual") and base_chart not in ('Xbar', 'S'):
            raise ValidationError(
                f"n_sigma and n_mode are only supported for Xbar/S charts, "
                f"got '{base_chart}'."
            )

        if base_chart in ('Xbar', 'S'):
            if not (isinstance(n_sigma, (int, float)) and math.isfinite(n_sigma) and n_sigma > 0):
                raise ValidationError(
                    f"n_sigma must be a finite number > 0; got {n_sigma!r}"
                )
            if n_mode not in ("actual", "average"):
                raise ValidationError(
                    f"n_mode must be 'actual' or 'average'; got {n_mode!r}"
                )

    def _get_factor_combinations(self) -> int:
        """Get count of unique factor combinations in the dataset."""
        if not self._spec.rsg_vars:
            return 1
        return self._ads.analysis_dataset[self._spec.rsg_var_name].nunique()

    def _resolve_value_column(
        self,
        value: str | None,
        recentered: bool
    ) -> str:
        """
        Resolve the value parameter to a column name.

        Parameters
        ----------
        value : str | None
            User-specified value ('response', 'R1', 'R2', etc.) or None
        recentered : bool
            Whether to use recentered residuals (RCR columns)

        Returns
        -------
        str
            Column name to chart (response_var, R1-R5, or RCR1-RCR5)

        Raises
        ------
        ValueError
            If value specifies unavailable residual
        """
        if value is None or value.lower() == 'response':
            return self._spec.response_var

        # Parse residual specification
        value_upper = value.upper()
        if value_upper.startswith('RCR'):
            # Already recentered format
            col_name = value_upper
        elif value_upper.startswith('R') and len(value_upper) >= 2 and value_upper[1:].isdigit():
            residual_num = value_upper[1:]
            prefix = 'RCR' if recentered else 'R'
            col_name = f'{prefix}{residual_num}'
        else:
            raise ValueError(
                f"Invalid value '{value}'. "
                f"Valid options: 'response', 'R1', 'R2', 'R3', 'R4', 'R5' "
                f"(or 'RCR1'-'RCR5' for recentered)."
            )

        # Validate column exists in analysis dataset
        if col_name not in self._ads.analysis_dataset.columns:
            available = [c for c in self._ads.analysis_dataset.columns
                        if c.startswith('R') and len(c) == 2 and c[1].isdigit()]
            raise ValueError(
                f"Residual column '{col_name}' not available for SDS {self.sds}. "
                f"Available: {available}"
            )

        return col_name

    # =========================================================================
    # Display Methods
    # =========================================================================

    def __repr__(self) -> str:
        """
        Concise study summary.

        Shows formulation, SDS, and available charts in minimal format.
        Use study.support for the full chart availability DataFrame.
        """
        # 1-line formulation summary
        factors_str = ', '.join(self.factors) if self.factors else 'None'
        time_str = self.time or 'None'

        lines = [
            f"Study(response='{self.response}', factors=[{factors_str}], time='{time_str}', sds={self.sds})",
        ]
        if self.unit_of_analysis:
            lines.append(f"  Unit of analysis: {self.unit_of_analysis}")
        lines.append(f"  Valid: {', '.join(self.valid_charts)} | Recommended: {self.recommended_chart}")
        if self.available_residuals:
            lines.append(f"  Residuals: {', '.join(self.available_residuals)}")
        lines.append("  → study.execute() or study.support for details")

        return '\n'.join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        factors_str = ', '.join(self.factors) if self.factors else 'None'
        time_str = self.time or 'None'

        residuals = self.available_residuals
        residual_html = f"<br><strong>Residuals:</strong> {', '.join(residuals)}" if residuals else ""
        uoa_html = f"<br><strong>Unit of analysis:</strong> {self.unit_of_analysis}" if self.unit_of_analysis else ""

        style = "font-family: monospace; padding: 8px; border: 1px solid #ccc; background: #f9f9f9"
        valid_charts_str = ', '.join(self.valid_charts)
        html = f"""
        <div style="{style}">
            <code>Study(response='{self.response}', factors=[{factors_str}], time='{time_str}', sds={self.sds})</code>
            {uoa_html}
            <br><strong>Valid:</strong> {valid_charts_str} | <strong>Recommended:</strong> {self.recommended_chart}
            {residual_html}
            <br><em>→ study.execute() or study.support for details</em>
        </div>
        """
        return html
