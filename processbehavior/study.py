"""
Study class for process behavior analysis formulation.

The Study object represents a formulated analysis - it knows the
design-state lineage of your data (PDS / ODS / ADS), what charts are
valid, and guides you toward correct analysis.

This is the "teaching" layer of the API that helps users understand their data
before running calculations.

Design Philosophy (Pythonic Hadley):
- Human-first: Rich __repr__ teaches users about their data
- Pit of success: Valid charts shown, invalid charts explained
- Composability: study.execute() returns AnalysisResult for chaining
- Immutable: Frozen dataclass, different formulations create new objects
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .exceptions import ChartNotAvailableError, FactorNotFoundError, ValidationError
from .sds_detector import SDSRegistry
from .spc_constants import (
    ALL_RESIDUALS,
    RESIDUAL_ALIASES,
    RESIDUAL_LABELS,
    VALID_BASE_CHARTS,
    normalize_chart_name,
)

if TYPE_CHECKING:
    import pandas as pd

    from .analysis_dataset import AnalysisDataSet
    from .analysis_result import AnalysisResult
    from .calibration import Calibration
    from .capability import CapabilityResult, SpecLimits
    from .derivations import Derivation
    from .formulation_spec import FormulationSpec
    from .loss_function import LossResult
    from .maximum_information import MaximumInformationResult
    from .process_behavior import ProcessBehavior
    from .sds_detector import SDSAnalysisPlan, SDSResult


# Maximum number of combo strings to return in missing_combos/extra_combos
MAX_COMBO_DISPLAY = 100

# The VAS residual codes a user may name in execute(value=...) — stored plus request; see
# spc_constants for what separates them. Membership only: whether a given code is *available*
# for this study is a different question (Study.residuals), and whether it can pair with a
# given chart is a third (Study._residual_pair_problem).
# Chart names have an equivalent in spc_constants.VALID_BASE_CHARTS, imported above.
RESIDUAL_CODES = ALL_RESIDUALS


def _base_residual_code(value: str) -> str:
    """'RCR5' -> 'R5', 'r2' -> 'R2'. Recentring does not change which chart a residual pairs with."""
    upper = value.upper()
    return upper[2:] if upper.startswith('RCR') else upper


@dataclass
class DesignReport:
    """
    Compares sampling plan to observed data.

    Returned by study.design(). Provides insight into the experimental
    design structure and any mismatches between plan and observation.

    Attributes
    ----------
    factors_table : pd.DataFrame
        Factor-level summary table with columns: factor, planned, observed,
        missing_levels, extra_levels. (Aliased as ``factors`` for backward
        compatibility.)
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
      SDS reason: incomplete_no_singletons
      K: planned=12, observed=8, missing=4
      T: planned=10, observed=8, missing=2
      R: planned=120, observed=64, missing=56
      N: planned=2, observed=(min=1, median=2.0, max=3)

      Factors:
        Lane: planned=[1,2,3,4], observed=[1,2,3,4]
        Phase: planned=[1,2,3], observed=[1,2], missing=[3]

      Structure: Incomplete: 4 RSG groups missing; 2 time points missing

    >>> design.factors_table
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
    _min_cell_size: int | None = None  # From SDSResult (raw data, pre-filtering)
    _n_empty_cells: int | None = None  # From SDSResult (raw data, pre-filtering)
    _pds_result: SDSResult | None = None  # Plan Design State
    _ods_result: SDSResult | None = None  # Observed Design State
    _ads_result: SDSResult | None = None  # Analytical Design State

    @property
    def factors_table(self) -> pd.DataFrame:
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

            rows.append(
                {
                    'factor': factor,
                    'planned': ', '.join(str(v) for v in planned),
                    'observed': ', '.join(str(v) for v in observed),
                    'missing_levels': ', '.join(str(v) for v in missing) if missing else '',
                    'extra_levels': ', '.join(str(v) for v in extra) if extra else '',
                }
            )

        return pd.DataFrame(rows)

    @property
    def factors(self) -> pd.DataFrame:
        """Backwards-compatible alias for ``factors_table``.

        Returns the same DataFrame as ``factors_table``. Prefer the new
        name to avoid confusion with ``Study.factors`` (which is a
        ``list[str]`` of column names, not a DataFrame).
        """
        return self.factors_table

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
        observed_in_plan = sum(1 for rsg in (self._observed_rsg_values or []) if self._rsg_in_plan(rsg))
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

        Possible values (per SDSReasonType):
        - 'full_replication': SDS 1 (all N_kt >= 2)
        - 'no_replication': SDS 2 (all N_kt = 1)
        - 'partial_replication': SDS 3 (mixed N_kt)
        - 'incomplete_no_singletons': SDS 4 (empty cells, all observed replicated)
        - 'incomplete_no_replication': SDS 5 (empty cells, all observed n=1)
        - 'incomplete_with_singletons': SDS 6 (empty cells + mixed)
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
            'incomplete_with_singletons': f'empty cells, mixed replication (min n = {min_n})',
            'incomplete_no_singletons': f'empty cells, all observed cells replicated (min n = {min_n})',
            'incomplete_no_replication': 'empty cells, all observed cells n = 1',
        }

        explanation = explanations.get(self._sds_reason, '')
        if explanation:
            return f'{self._sds_reason} ({explanation})'
        return self._sds_reason

    @property
    def min_cell_size(self) -> int | None:
        """
        Minimum observations per cell from SDS detection (on raw data).

        This is the structural gate input that determines SDS classification.
        Note: this may differ from N_observed (which is computed from the
        analysis dataset after NA filtering).
        """
        return self._min_cell_size

    @property
    def n_empty_cells(self) -> int | None:
        """
        Count of factor x time cells with zero observations (from SDS detection).

        Cells where all response values are NA or garbage are counted as empty.
        A value > 0 triggers SDS 4-6 classification.
        """
        return self._n_empty_cells

    @property
    def coverage(self) -> float | None:
        """
        Ratio of observed cells to planned cells (0.0 to 1.0).

        Only meaningful when a sampling plan is provided and time is specified.
        Returns None if no plan, no time variable, or planned R is zero.
        """
        if not self.has_plan or self._T is None:
            return None
        r = self.R
        r_obs = self.R_observed
        if r is None or r_obs is None or r == 0:
            return None
        return r_obs / r

    @property
    def remediation(self) -> str | None:
        """
        Actionable guidance for improving the sampling design.

        Returns a deterministic sentence based on SDS classification
        describing what to change to enable richer analysis.
        Returns None when no remediation is needed (SDS 1 / full replication)
        or when the reason is not recognized.
        """
        if not self._sds_reason:
            return None

        hints: dict[str, str | None] = {
            'full_replication': None,
            'no_replication': (
                'To enable within-cell variance estimation, collect >= 2 observations per factor x time cell.'
            ),
            'partial_replication': (
                'To enable consistent variance estimation, ensure all cells have >= 2 observations.'
            ),
            'incomplete_with_singletons': (
                'To complete the design, fill missing factor x time cells and ensure >= 2 observations per cell.'
            ),
            'incomplete_no_singletons': ('To complete the design, fill missing factor x time cells.'),
            'incomplete_no_replication': (
                'To enable Xbar/S charts and within-cell residuals, '
                'collect >= 2 observations per cell and fill missing cells.'
            ),
        }
        return hints.get(self._sds_reason)

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
                issues.append(f'underreplicated (min n = {min_n} < planned N = {self._N})')

        # Check time completeness
        if self._T is not None and self._T_observed is not None and self._T_observed < self._T:
            issues.append(f'incomplete_time (observed {self._T_observed} of {self._T} time points)')

        # Check factor completeness
        k_observed = self.K_observed
        k_planned = self.K
        if k_observed < k_planned:
            issues.append(f'incomplete_factors (observed {k_observed} of {k_planned} factor combinations)')

        if not issues:
            return 'complete'

        return '; '.join(issues)

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
        return sum(1 for rsg in (self._observed_rsg_values or []) if not self._rsg_in_plan(rsg))

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
        extras = [rsg for rsg in (self._observed_rsg_values or []) if not self._rsg_in_plan(rsg)]

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
            issues.append(f'{self.K_missing} RSG groups missing')
        if self.extra_count > 0:
            issues.append(f'{self.extra_count} extra RSG groups in data')
        if self.T_missing and self.T_missing > 0:
            issues.append(f'{self.T_missing} time points missing')
        if self._N is not None and self._N_observed:
            min_n, _, _ = self._N_observed
            if min_n < self._N:
                issues.append(f'some cells have n={min_n} < planned n={self._N}')

        if not issues:
            return 'Complete structure'
        return 'Incomplete: ' + '; '.join(issues)

    def _repr_metrics_line(self) -> str:
        """Build compact K/T/R/N metrics line for __repr__."""
        parts = []

        # Min cell size
        if self._min_cell_size is not None and self._min_cell_size > 0:
            parts.append(f'Min cell size: {self._min_cell_size}')

        # K summary
        if self.has_plan:
            parts.append(f'K: planned={self.K}, observed={self.K_observed}')
        else:
            parts.append(f'K: {self.K_observed}')

        # T summary
        if self._T is not None:
            parts.append(f'T: planned={self._T}, observed={self._T_observed}')
        elif self._T_observed is not None:
            parts.append(f'T: {self._T_observed}')

        # R = K × T (total cells)
        if self.R is not None:
            if self.has_plan and self._T is not None:
                parts.append(f'R: planned={self.R}, observed={self.R_observed}')
            elif self.R_observed is not None:
                parts.append(f'R: {self.R_observed}')

        # N summary
        if self._N is not None and self._N_observed is not None:
            min_n, med_n, max_n = self._N_observed
            parts.append(f'N: planned={self._N}, observed=(min={min_n}, median={med_n}, max={max_n})')
        elif self._N_observed is not None:
            min_n, med_n, max_n = self._N_observed
            parts.append(f'N: (min={min_n}, median={med_n}, max={max_n})')

        return '  ' + ' | '.join(parts)

    _REASON_DISPLAY = {
        'full_replication': 'Full Replication',
        'no_replication': 'No Replication',
        'partial_replication': 'Partial Replication',
        'incomplete_no_singletons': 'Incomplete, No Singletons',
        'incomplete_no_replication': 'Incomplete, No Replication',
        'incomplete_with_singletons': 'Incomplete, With Singletons',
    }

    def _humanize_reason(self, reason: str) -> str:
        """Convert machine reason token to human-readable label."""
        return self._REASON_DISPLAY.get(reason, reason)

    def __repr__(self) -> str:
        """Nice summary showing plan vs observed per factor with K/T/N."""
        lines = [f'Design Report ({len(self._factors)} factors)']

        # Unit of analysis (if specified)
        if self._unit_of_analysis:
            lines.append(f'  Unit of analysis: {self._unit_of_analysis}')

        # Design lineage
        ods = self._ods_result
        ads = self._ads_result
        pds = self._pds_result

        if ods and ads:
            lines.append('  Design-state lineage:')
            if pds is not None:
                lines.append(f'    PDS (Planned):    {pds.sds} ({self._humanize_reason(pds.reason)})')
            else:
                lines.append('    PDS (Planned):    no plan supplied')
            empty_detail = f' — {ods.n_empty_cells} empty cells' if ods.n_empty_cells > 0 else ''
            ods_reason = self._humanize_reason(ods.reason)
            lines.append(f'    ODS (Observed):   {ods.sds} ({ods_reason}){empty_detail}')
            lines.append(f'    ADS (Analytical): {ads.sds} ({self._humanize_reason(ads.reason)})')
        elif self.sds_reason_detail:
            lines.append(f'  Classification reason: {self.sds_reason_detail}')
        elif self._sds_reason:
            lines.append(f'  Classification reason: {self._sds_reason}')

        # Plan adherence (only shown when plan is provided)
        if self.plan_adherence:
            lines.append(f'  Plan adherence: {self.plan_adherence}')

        # Compact metrics line (includes min cell size)
        lines.append(self._repr_metrics_line())

        # Factor details
        lines.append('')
        lines.append('  Factors:')
        for factor in self._factors:
            observed = self._observed_levels.get(factor, [])
            planned = self._sampling_plan.get(factor, observed) if self._sampling_plan is not None else observed

            planned_set = set(planned)
            observed_set = set(observed)
            missing = self._safe_sort(list(planned_set - observed_set))

            line = f'    {factor}: observed={observed}'
            if self.has_plan:
                line = f'    {factor}: planned={planned}, observed={observed}'
            if missing:
                line += f', missing={missing}'

            lines.append(line)

        # Structure summary (discrepancy details)
        lines.append('')
        lines.append(f'  Structure: {self.structure_summary}')

        # Available analyses (derived from ADS)
        if self._ads_result and self._ads_result.sds > 0:
            lines.extend(self._repr_available_analyses())

        return '\n'.join(lines)

    def _repr_available_analyses(self) -> list[str]:
        """Compact chart×value availability section derived from ADS."""
        ads = self._ads_result
        plan = SDSRegistry.get_analysis_plan(sds=ads.sds, min_cell_size=ads.min_cell_size)

        lines = ['', f'  Available analyses (ADS {ads.sds}):']

        # Primary charts with recommended marker
        primaries = []
        for chart in plan.valid_charts:
            label = f'{chart} *' if chart == plan.recommended_chart else chart
            primaries.append(label)
        lines.append(f'    Primary: {", ".join(primaries)}')

        # Residual charts grouped by residual
        from collections import defaultdict

        residual_groups: dict[str, list[str]] = defaultdict(list)
        for chart_type, residual in plan.residual_charts:
            residual_groups[residual].append(chart_type)

        for residual in ['R2', 'R3', 'R4', 'R5', 'R6']:
            if residual in residual_groups:
                charts = ', '.join(residual_groups[residual])
                lines.append(f'    {residual}: {charts}')

        # Analysis methods
        lines.append('    Methods: Capability, Loss Function, Maximum Information')

        return lines


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
            Primary chart types (Xbar, S, X, mR)
        """
        self._valid_charts = valid_charts

        # Dynamically add each valid chart as an attribute
        for chart in self._valid_charts:
            # Convert chart names to valid Python identifiers
            attr_name = chart.replace(':', '_').replace('-', '_')
            setattr(self, attr_name, chart)

    def __repr__(self) -> str:
        """Display available primary chart types."""
        return f'StudyChartAccessor({", ".join(self._valid_charts)})'

    def __dir__(self) -> list[str]:
        """Support for tab-completion in IPython/Jupyter."""
        return [c.replace(':', '_').replace('-', '_') for c in self._valid_charts]


class StudyResidualAccessor:
    """
    Provides IDE auto-completion for available VAS residuals.

    Dynamically creates attributes for each residual whose columns
    were actually computed in the analysis dataset.

    Usage:
        study = pb.formulate(response='y', factors=['lane'], time='t')

        # IDE auto-completes available residuals
        result = study.execute(chart=study.charts.Xbar, value=study.residuals.R2)

        # With recentered
        result = study.execute(
            chart=study.charts.X, value=study.residuals.R4, recentered=True
        )

    Attributes are set dynamically based on which residuals were computed.
    """

    def __init__(self, available: list[str]):
        self._available = available
        for r in available:
            setattr(self, r, r)

    def __repr__(self) -> str:
        if not self._available:
            return 'StudyResidualAccessor(none — requires factors + time)'
        return f'StudyResidualAccessor({", ".join(self._available)})'

    def __dir__(self) -> list[str]:
        return list(self._available)

    def __iter__(self):
        return iter(self._available)

    def __len__(self):
        return len(self._available)

    def __contains__(self, item):
        return item in self._available


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

    >>> study.observed_design_state.sds  # 1-6
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
    _sds_result: SDSResult | None = None  # ODS result for accessing .reason in DesignReport
    _pds_result: SDSResult | None = None  # PDS result (None when no plan)
    calibrations: Mapping[str, Calibration] = field(default_factory=dict)  # label -> Calibration
    # Resolved (fit-frozen) derived-variable specs materialized into this study's
    # analytic dataset. Empty when no derived variables were attached. Provenance
    # + serialization target; the frozen `fitted` values make binning reproducible.
    derivations: tuple[Derivation, ...] = ()

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
    # Design State Properties (PDS / ODS / ADS)
    # =========================================================================

    @property
    def plan_design_state(self) -> SDSResult | None:
        """
        Plan Design State (PDS) — what the user intended to collect.

        Computed from plan parameters (K×T×N). Always SDS 1 (N>=2) or
        SDS 2 (N==1). Returns None when no plan is provided.

        Returns
        -------
        SDSResult or None
        """
        return self._pds_result

    @property
    def observed_design_state(self) -> SDSResult:
        """
        Observed Design State (ODS) — what was actually collected.

        Detected on raw data (before NA filtering). Diagnostic/lineage only;
        the ADS drives analysis decisions.

        Returns
        -------
        SDSResult
        """
        if self._sds_result is None:
            raise RuntimeError(
                'Study has no observed_design_state — it was not built via '
                'ProcessBehavior.formulate(). Use pb.formulate(...) to '
                'construct Study; direct construction is not supported.'
            )
        return self._sds_result

    @property
    def analytical_design_state(self) -> SDSResult:
        """
        Analytical Design State (ADS) — what is fit for analysis.

        Computed on tidy data (after NA filtering). Drives valid charts,
        residual availability, R2 method, and interaction method selection.

        Returns
        -------
        SDSResult
        """
        return self._ads.analytical_design_state

    @property
    def ads_reason(self) -> str | None:
        """
        Machine-readable reason token for the analytical design state.

        Returns SDSReasonType value, e.g. 'full_replication', 'no_replication'.
        Returns None when ADS=0 (no valid observations after data cleaning).
        """
        return self.analytical_design_state.reason

    @property
    def ads_description(self) -> str:
        """
        Human-readable description of the analytical design state.

        Returns prose from SDSRegistry, e.g.
        'Full replication (all cells n>=2)'.

        Reflects the analyzable structure after data cleaning.
        """
        chars = SDSRegistry().get_sds_characteristics(self.analytical_design_state.sds)
        return chars['description']

    # =========================================================================
    # Chart Properties
    # =========================================================================

    @property
    def valid_charts(self) -> list[str]:
        """
        Chart types that are valid for this data structure.

        This is *chart-family* validity, decided by the Analytical Design
        State: whether the chart has any legitimate use on this study. It is
        not the same question as "will ``execute(chart=...)`` succeed".

        On ADS 2, for instance, ``'Xbar'`` is listed — an Xbar of a residual
        (``value='R6'``) or of a pooled subgroup (``by=[factor]``) computes
        fine — while charting the *response* has nothing to subgroup and
        raises. For that narrower question use :attr:`support` (which carries
        a ``reason``) or :meth:`why_not`.

        Returns
        -------
        list of str
            Valid chart types (e.g., ['Histogram', 'Xbar', 'S', 'X', 'mR']).
            Empty list when ADS=0 (no valid observations after cleaning).

        See Also
        --------
        support : Availability matrix including data-aware reasons.
        why_not : Explains a specific refusal.
        """
        if self.analytical_design_state.sds == 0:
            return []
        return self._plan.valid_charts

    @property
    def recommended_chart(self) -> str | None:
        """
        The recommended chart type for this data structure.

        This is the chart type that best suits your data based on
        Bishop methodology. Returns None when ADS=0
        (no valid observations after data cleaning).
        """
        if self.analytical_design_state.sds == 0:
            return None
        return self._plan.recommended_chart

    @property
    def residual_charts(self) -> list[tuple[str, str]]:
        """
        Available residual chart types for VAS analysis.

        Returns (chart_type, residual) tuples matching the ``execute()``
        signature: ``study.execute(chart=chart_type, value=residual)``.

        Returns
        -------
        list of tuple[str, str]
            Available residual charts, e.g. ``[('S', 'R2'), ('Xbar', 'R5')]``

        Examples
        --------
        >>> for chart, value in study.residual_charts:
        ...     result = study.execute(chart=chart, value=value)
        """
        ads_cols = set(self._ads.analysis_dataset.columns)
        # R6 is computed on-the-fly from R5+R2, so check prerequisites instead of column
        r6_available = {'R5', 'R2'}.issubset(ads_cols)
        return [
            (chart, value)
            for chart, value in self._plan.residual_charts
            if value in ads_cols or (value == 'R6' and r6_available)
        ]

    @property
    def charts(self) -> StudyChartAccessor:
        """
        Accessor for IDE auto-completion of primary chart types.

        Primary charts are the standard process behavior charts:
        Xbar, S, X, mR

        For residual charts (VAS analysis), use study.residuals instead.

        Usage:
            study.charts.Xbar  # Auto-completes valid primary charts
            study.charts.X

        Returns
        -------
        StudyChartAccessor
            Object with primary chart types as attributes
        """
        return StudyChartAccessor(self.valid_charts)

    @property
    def residuals(self) -> StudyResidualAccessor:
        """
        Available VAS residuals for this study.

        Provides IDE auto-completion for residual types that can be
        passed to execute(value=...). Only includes residuals whose
        columns were actually computed (requires factors + time).

        - R1: Total residual (y - grand mean)
        - R2: Within-subgroup variation (measurement noise)
        - R3: Interaction effects (factor × time)
        - R4: Time effects (trends, shifts over time)
        - R5: Factor effects (differences between levels)

        Usage:
            study.execute(chart=study.charts.Xbar, value=study.residuals.R5)
            study.execute(chart='X', value=study.residuals.R4, recentered=True)

        Returns
        -------
        StudyResidualAccessor
            Accessor with available residuals as attributes (e.g., .R2, .R5)

        See Also
        --------
        residual_charts : Full list of residual+chart combinations (internal)
        """
        ads_cols = set(self._ads.analysis_dataset.columns)
        available = sorted(r for r in ('R1', 'R2', 'R3', 'R4', 'R5') if r in ads_cols)
        if self._spec.rsg_vars_list and 'R5' in ads_cols:
            available.append('R6')
        return StudyResidualAccessor(available)

    @property
    def available_analysis_methods(self) -> pd.DataFrame:
        """
        Analysis methods available for this study beyond charts.

        Returns a DataFrame listing Capability, Loss Function, and
        Maximum Information with availability based on the ADS.

        Returns
        -------
        pd.DataFrame
            Columns: method, available, description
        """
        import pandas as pd

        ads_cols = set(self._ads.analysis_dataset.columns)
        vas_available = 'R2' in ads_cols

        rows = [
            {
                'method': 'Capability',
                'available': self.analytical_design_state.sds > 0,
                'description': 'Process capability against specification limits (Ch. 16)',
            },
            {
                'method': 'Loss Function',
                'available': vas_available,
                'description': 'Taguchi loss decomposition into 5 components (Ch. 15)',
            },
            {
                'method': 'Maximum Information',
                'available': vas_available,
                'description': 'Noise floor analysis via R2 X + histogram',
            },
        ]
        return pd.DataFrame(rows)

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
            Columns: chart, value, category, available, recommended, reason, question

            For primary charts, ``value`` is None. For residual charts,
            ``value`` is the residual name (e.g. 'R2', 'R5').

        Examples
        --------
        >>> study.support
               chart value  category  available  recommended  ...
        0       Xbar  None   primary       True         True  ...
        1          S  None   primary       True        False  ...
        2       Xbar    R5  residual       True        False  ...

        >>> study.support[study.support['available']]  # Filter to available
        >>> study.support.query("category == 'residual'")  # Residual charts
        """
        import pandas as pd

        from .sds_detector import SDSAnalysisPlan

        rows = []

        # All possible primary charts
        ALL_PRIMARY = ['Xbar', 'S', 'X', 'mR', 'Histogram']

        # All possible residual charts as (chart_type, residual) tuples. Named _PAIRS to
        # keep it distinct from spc_constants.ALL_RESIDUALS, which is a set of residual
        # codes — different concept, and now one import away.
        # R1 is location-only (Xbar/X): it is the response shifted by -Ybar, so an S or mR
        # chart of R1 is numerically identical to the response's and adds no diagnostic.
        ALL_RESIDUAL_PAIRS = [
            ('Xbar', 'R1'),
            ('X', 'R1'),
            ('S', 'R2'),
            ('X', 'R2'),
            ('Xbar', 'R3'),
            ('S', 'R3'),
            ('X', 'R3'),
            ('Xbar', 'R4'),
            ('S', 'R4'),
            ('X', 'R4'),
            ('Xbar', 'R5'),
            ('S', 'R5'),
            ('X', 'R5'),
            ('Xbar', 'R6'),
            ('S', 'R6'),
        ]

        # ADS=0 guard: all charts unavailable
        if self.analytical_design_state.sds == 0:
            for chart in ALL_PRIMARY:
                rows.append(
                    {
                        'chart': chart,
                        'value': None,
                        'category': 'primary',
                        'available': False,
                        'recommended': False,
                        'reason': 'No valid observations after data cleaning',
                        'question': SDSAnalysisPlan.CHART_QUESTIONS.get(chart, ''),
                    }
                )
            for chart, value in ALL_RESIDUAL_PAIRS:
                rows.append(
                    {
                        'chart': chart,
                        'value': value,
                        'category': 'residual',
                        'available': False,
                        'recommended': False,
                        'reason': 'No valid observations after data cleaning',
                        'question': SDSAnalysisPlan.CHART_QUESTIONS.get((chart, value), ''),
                    }
                )
            return pd.DataFrame(rows)

        # Build invalid_reasons dict from _plan.invalid_charts
        invalid_reasons = self._parse_invalid_charts()

        for chart in ALL_PRIMARY:
            # Availability here is about charting *the response*, which is what
            # a bare execute(chart=...) does — so it must account for the
            # replication rule execute() enforces, not just chart-family
            # validity. See _response_pair_problem.
            response_problem = self._response_pair_problem(chart) if chart in self.valid_charts else None
            rows.append(
                {
                    'chart': chart,
                    'value': None,
                    'category': 'primary',
                    'available': chart in self.valid_charts and response_problem is None,
                    'recommended': chart == self.recommended_chart,
                    'reason': invalid_reasons.get(chart) or response_problem,
                    'question': SDSAnalysisPlan.CHART_QUESTIONS.get(chart, ''),
                }
            )

        for chart, value in ALL_RESIDUAL_PAIRS:
            available = (chart, value) in self.residual_charts
            rows.append(
                {
                    'chart': chart,
                    'value': value,
                    'category': 'residual',
                    'available': available,
                    'recommended': False,
                    'reason': None if available else 'Not available for this SDS',
                    'question': SDSAnalysisPlan.CHART_QUESTIONS.get((chart, value), ''),
                }
            )

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
                reason = entry[entry.index('(') + 1 : -1]
                result[chart] = reason
        return result

    # =========================================================================
    # Guidance Methods
    # =========================================================================

    def why_not(self, chart: str, value: str | None = None) -> str:
        """
        Explain why a chart type is or isn't available for this study.

        This is a teaching method - it helps users understand the
        methodology by explaining constraints. Uses the support DataFrame
        as the single source of truth.

        Parameters
        ----------
        chart : str
            Chart type to check (e.g., 'X', 'S', 'Xbar')
        value : str, optional
            Residual to check (e.g., 'R2', 'R5'). Required for residual
            chart queries.

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

        >>> study.why_not('Xbar', value='R5')
        "'Xbar' (R5) IS available. Do factors have a significant effect on the mean?"

        A recognised chart in an invalid pairing names the charts that DO work — the same
        answer ``execute()`` gives, because both read one predicate:

        >>> study.why_not('Xbar', value='R2')
        "'Xbar' with value='R2' is not valid for ADS 1.\\nValid charts for R2: S, X"
        """
        # Handle old fused syntax: why_not('R5_Xbar') -> why_not('Xbar', value='R5')
        if '_' in chart and value is None:
            parts = chart.split('_', 1)
            if parts[0].startswith('R') and len(parts[0]) >= 2 and parts[0][1:].isdigit():
                value, chart = parts[0], parts[1]

        # ADS=0: no valid observations after cleaning
        if self.analytical_design_state.sds == 0:
            label = f"'{chart}'" if value is None else f"'{chart}' ({value})"
            return f'{label} unavailable: no valid observations after data cleaning'

        label = f"'{chart}'" if value is None else f"'{chart}' ({value})"

        # Unrecognised chart name — distinct from a recognised chart in an invalid
        # combination, which is the case this method used to conflate it with.
        if chart not in VALID_BASE_CHARTS:
            return (
                f"'{chart}' is not a recognized chart type. "
                f'Valid charts: {", ".join(sorted(VALID_BASE_CHARTS))}.'
            )

        if value is not None:
            base_residual = _base_residual_code(value)
            if base_residual not in RESIDUAL_CODES:
                return (
                    f"'{value}' is not a recognized residual. "
                    f'Valid values: {", ".join(RESIDUAL_CODES)} '
                    f"(prefix with 'RC' for the recentered form, e.g. 'RCR5')."
                )

            # Same predicate execute() validates against, so the two cannot disagree.
            problem = self._residual_pair_problem(chart, value)
            if problem:
                return problem
            if chart == 'mR':
                return (
                    f'{label} IS available. Is the moving range of {value} stable? '
                    f"Add companion=True to get the X chart alongside it."
                )
            from .sds_detector import SDSAnalysisPlan

            question = SDSAnalysisPlan.CHART_QUESTIONS.get((chart, base_residual), '')
            return f'{label} IS available. {question}'.rstrip()

        # Primary chart: the support row carries availability, reason and question.
        df = self.support
        row = df[(df['chart'] == chart) & (df['value'].isna())]
        if row.empty:
            return f'{label} is not a recognized chart type. Use study.support to see all options.'

        row = row.iloc[0]
        if row['available']:
            return f'{label} IS available. {row["question"]}'.rstrip()
        else:
            return f'{label} unavailable: {row["reason"]}'

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
          SDS reason: incomplete_no_singletons
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
                cell_sizes = ads_df.groupby([rsg_col, self._spec.time_var], observed=True).size()
                # R_observed = count of unique (rsg, time) cells
                R_observed = len(cell_sizes)
            else:
                cell_sizes = ads_df.groupby([rsg_col], observed=True).size()

            # N_observed = (min, median, max)
            if len(cell_sizes) > 0:
                N_observed = (int(cell_sizes.min()), float(cell_sizes.median()), int(cell_sizes.max()))

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
            _min_cell_size=self._sds_result.min_cell_size if self._sds_result else None,
            _n_empty_cells=self._sds_result.n_empty_cells if self._sds_result else None,
            _pds_result=self._pds_result,
            _ods_result=self._sds_result,
            _ads_result=self._ads.analytical_design_state,
        )

    def capability(
        self,
        specs: SpecLimits | None = None,
        *,
        usl: float | None = None,
        lsl: float | None = None,
        target: float | None = None,
        window: tuple | None = None,
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
        window : tuple, optional
            Time-based subset ``(start, end)`` in the units of this study's
            designated time variable (an int for a sequence, a date/datetime for
            a date axis); half-open ``start <= t < end``, either bound ``None``
            for open. Default ``None`` = full data (unchanged behaviour). This is
            a view over the frozen analytic dataset: current capability and the
            potential centering use the windowed observed values, while the
            potential noise floor (σ̂_R2) stays the full-study pooled residual
            basis. See ``assess_capability`` for the windowing rules/guards.

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

        Before vs. after a change at time 25 (integer time axis):

        >>> before = study.capability(usl=240, target=180, window=(None, 25))
        >>> after = study.capability(usl=240, target=180, window=(25, None))
        """
        from .capability import SpecLimits as _SpecLimits
        from .capability import assess_capability

        if specs is None:
            specs = _SpecLimits(usl=usl, lsl=lsl, target=target)
        return assess_capability(self._ads, specs, round_to=self._spec.round_to, window=window)

    def loss_function(self, target: float | None = None) -> LossResult:
        """
        Assess Taguchi Loss Function decomposition (Ch. 15).

        Decomposes expected loss into 5 components: centering, unexplained,
        PDC, time, and PDC×time interaction.

        Parameters
        ----------
        target : float, optional
            Target value. Defaults to grand mean (centering = 0).

        Returns
        -------
        LossResult
            Frozen dataclass with loss decomposition and Pareto percentages.

        Examples
        --------
        >>> result = study.loss_function(target=237.0)
        >>> result.pct_interaction  # largest driver?
        43.3
        """
        from .loss_function import assess_loss

        return assess_loss(self._ads, target=target, round_to=self._spec.round_to)

    def maximum_information(self) -> MaximumInformationResult:
        """
        Maximum information analysis of R2 residuals (Bishop).

        Examines the noise floor via an X chart and percentage histogram.

        Returns
        -------
        MaximumInformationResult
            Frozen dataclass with noise-floor statistics and ``.plot()`` method.

        Examples
        --------
        >>> mi = study.maximum_information()
        >>> mi.plot()                    # Combined X + histogram
        >>> mi.plot(view='histogram')    # Percentage histogram only
        """
        from .maximum_information import assess_maximum_information

        return assess_maximum_information(self._ads, round_to=self._spec.round_to)

    def execute(
        self,
        chart: Literal['Histogram', 'Xbar', 'S', 'X', 'mR'] | str | None = None,
        by: list[str] | None = None,
        value: Literal['response', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6'] | str | None = None,
        recentered: bool = False,
        bins: int | None = None,
        companion: bool = False,
        phased: bool = False,
        n_sigma: float = 3.0,
        n_mode: Literal['actual', 'average'] = 'actual',
        calibration: Calibration | str | None = None,
        stratify_by: list[str] | None = None,
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
            Chart type. If None, uses recommended_chart.
            Valid charts: 'Xbar', 'S', 'X', 'mR', 'Histogram'.
            X is the focal (individual values) chart; mR is its companion
            (moving range). Use companion=True to get both together.

        by : list[str], optional
            Factors to group/stratify by. Controls view granularity.

            - by=None: Default for chart type (full rsg_key for Xbar/S,
              ERROR for X/mR with factors - must be explicit)
            - by=[]: For X/mR: single overall chart (collapse all factors).
              For Xbar/S: equivalent to by=None (cell-level grouping).
            - by=['factor']: Stratify/aggregate by single factor
            - by=['f1', 'f2']: Stratify/aggregate by factor combination

            For Xbar/S: by controls aggregation (groups on x-axis).
              Special case: by=[time_var] with factors stratifies by
              factor combinations, producing one chart per combo with
              time on x-axis. Only applies to response (not residuals).
            For X/mR: by controls stratification (separate charts)

        value : str, optional
            What to chart. Options:

            - None or 'response': Chart the response variable (default)
            - 'R1' through 'R5': Chart the specified residual

        recentered : bool, default False
            For residual charts only. If True, charts the re-centered
            residual RCRk = Rk + baseline_k, shifting from a zero-centered
            scale to the response scale for easier interpretation. The
            baseline for each residual is defined by the VAS decomposition
            (Bishop VAS §15.5). Requires VAS decomposition
            (factors + time).

        bins : int, optional
            Number of bins for Histogram charts. Defaults to 10.
            Only applicable to Histogram; providing bins for other chart types
            raises ValidationError.

        companion : bool, default False
            Returns both the location chart and its companion range chart
            in a single result, regardless of which is requested
            (e.g., chart='mR', companion=True returns both X and mR).
            Each chart is accessed separately via get_chart().
            Not applicable to Histogram.

        phased : bool, default False
            When True, computes per-phase center lines and control limits.
            Each phase is a contiguous run of the same collapsed factor
            combination (the groups demarcated by vertical lane boundaries).
            Within each phase, the moving range is computed independently.

            - phased=False (default): Global limits across entire chart
            - phased=True: Per-phase limits (requires by=[])

            Single-point phases yield zero-width limits because mR is
            imputed as 0; count available in metadata['single_point_phases'].

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

        calibration : Calibration or str, optional
            Apply standards-given limits using a frozen mean/sigma instead of
            data-derived estimates. Accepts a :class:`Calibration` object or the
            label of one previously attached via :meth:`with_calibration`.
            Location charts (X, Xbar, response, residuals) place limits at
            ``center ± n_sigma·sigma`` (Xbar divides by ``√N``); dispersion
            charts (S, mR) use the standards-given sampling-distribution form.
            X/mR remain 3-sigma, so a calibrated X/mR rejects a non-default
            ``n_sigma`` (calibration is not a back door to non-3-sigma limits).

        Notes
        -----
        **Xbar center line (Bishop VAS unweighted).** For both response and
        residual Xbar charts, the center line is the mean of (factor x time)
        cell means on the charted column — equal weight per experimental
        condition, regardless of how many observations fall in each cell. This
        is the VAS-canonical grand mean and matches Bishop's Minitab reference.
        It is **not** the observation-weighted mean and **not** the mean of the
        plotted subgroup means. On balanced designs the three coincide; on
        unbalanced designs they differ slightly, but per Bishop the practical
        difference is negligible — the methodology still requires the
        unweighted form.

        Returns
        -------
        AnalysisResult
            Complete analysis results with charts, statistics, residuals.
            Use result.plot() to visualize or result.to_excel() to export.

        Raises
        ------
        ChartNotAvailableError
            If specified chart is not valid for this SDS
        ValidationError
            If by contains invalid factors, or if X/mR with factors
            requires an explicit by parameter that wasn't supplied

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

        >>> # X chart stratified by all factors (separate chart per combo)
        >>> result = study.execute(chart='X', by=['factor 1', 'factor 2'])

        >>> # X chart single overall
        >>> result = study.execute(chart='X', by=[])

        >>> # Stratified Xbar by time (one chart per factor combination)
        >>> result = study.execute(chart='Xbar', by=['PRODUCTION TIME'])

        Chart residuals:

        >>> result = study.execute(chart='Xbar', value='R5')  # Factor effects
        >>> result = study.execute(chart='X', value='R4', recentered=True)

        Chain to visualization:

        >>> study.execute().plot()
        """
        # Import here to avoid circular imports
        from .analysis import Analysis
        from .formulation_spec import ChartRequest

        # 0a. `stratify_by` is an explicit spelling of `by` for the X/mR regime, where the
        # argument means "split into separate charts" rather than "compose the subgroup".
        # Same parameter underneath — the alias exists so the call site says which of the
        # two operations is intended, since the return shape differs between them.
        if stratify_by is not None:
            if by is not None:
                raise ValidationError(
                    "Pass either 'by' or 'stratify_by', not both — they set the same "
                    "parameter.\n"
                    "'stratify_by' is the explicit spelling for X/mR (separate chart per "
                    "stratum); 'by' also composes subgroups for Xbar/S."
                )
            by = stratify_by

        # 0. Resolve calibration (label string -> attached Calibration object).
        calibration_obj = self._resolve_calibration(calibration)

        # 1. Resolve chart name (None → recommended, basic shape check).
        base_chart = self._resolve_execute_chart(chart)

        # 2. Run all parameter validation; resolve by/value/is_residual.
        by_validated, value_col, is_residual = self._validate_execute_request(
            base_chart=base_chart,
            by=by,
            value=value,
            recentered=recentered,
            companion=companion,
            bins=bins,
            phased=phased,
            n_sigma=n_sigma,
            n_mode=n_mode,
        )

        # 3. Build ephemeral request and dispatch to Analysis.
        request = ChartRequest(
            chart=base_chart,
            by=tuple(by_validated) if by_validated is not None else None,
            value_col=value_col,
            residual=value.upper() if is_residual else None,
            residual_chart_type=base_chart if is_residual else None,
            recentered=recentered,
            companion=companion,
            bins=bins,
            phased=phased,
            n_sigma=n_sigma,
            n_mode=n_mode,
            calibration=calibration_obj,
        )
        analysis = Analysis(self._spec, request, analysis_dataset=self._ads)
        return analysis.calculate()

    def supports_calibration(
        self,
        chart: Literal['Histogram', 'Xbar', 'S', 'X', 'mR'] | str | None = None,
        by: list[str] | None = None,
        value: Literal['response', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6'] | str | None = None,
        recentered: bool = False,
        bins: int | None = None,
        companion: bool = False,
        phased: bool = False,
        n_sigma: float = 3.0,
        n_mode: Literal['actual', 'average'] = 'actual',
        stratify_by: list[str] | None = None,
    ) -> bool:
        """Would ``execute(..., calibration=cal)`` apply the calibration, or refuse it?

        Takes the same arguments as :meth:`execute` (minus ``calibration`` itself —
        the answer depends on the chart path, not on which calibration) and answers
        without running the analysis, so a caller can enable a control or pick a
        code path before committing to the work.

        **Use this instead of re-deriving the rule.** It is not the one-liner it
        looks like. Xbar/S stratify on ``by=[time]`` *only when charting the
        response*, so the same ``by`` is refused for the response and accepted for a
        residual; X/mR stratify whenever ``by`` is non-empty **or absent**, and
        ``by=[]`` is the collapsed case that phasing then rules out. A hand-written
        copy of this predicate is wrong on the first of those rows.

        Returns
        -------
        bool
            True if a calibration would be applied; False if
            :class:`~processbehavior.exceptions.CalibrationNotSupportedError`
            would be raised.

        Raises
        ------
        ValidationError, ChartNotAvailableError
            If the request is not executable at all. The question "can this be
            calibrated?" has no answer for a chart that cannot be produced, so it
            fails the same way :meth:`execute` would rather than returning False.

        Examples
        --------
        >>> study.supports_calibration(chart='X', by=[])
        True
        >>> study.supports_calibration(chart='X', by=['Machine'])
        False

        See Also
        --------
        execute : Runs the analysis; raises when a calibration cannot be applied.
        """
        if stratify_by is not None:
            if by is not None:
                raise ValidationError(
                    "Pass either 'by' or 'stratify_by', not both — they set the same "
                    'parameter.'
                )
            by = stratify_by

        from .analysis import Analysis
        from .formulation_spec import ChartRequest

        base_chart = self._resolve_execute_chart(chart)
        by_validated, value_col, is_residual = self._validate_execute_request(
            base_chart=base_chart,
            by=by,
            value=value,
            recentered=recentered,
            companion=companion,
            bins=bins,
            phased=phased,
            n_sigma=n_sigma,
            n_mode=n_mode,
        )
        request = ChartRequest(
            chart=base_chart,
            by=tuple(by_validated) if by_validated is not None else None,
            value_col=value_col,
            residual=value.upper() if is_residual else None,
            residual_chart_type=base_chart if is_residual else None,
            recentered=recentered,
            companion=companion,
            bins=bins,
            phased=phased,
            n_sigma=n_sigma,
            n_mode=n_mode,
        )
        analysis = Analysis(self._spec, request, analysis_dataset=self._ads)
        return analysis.calibration_rejection_reason() is None

    def _resolve_calibration(self, calibration: Calibration | str | None) -> Calibration | None:
        """Resolve a calibration argument to a Calibration object (or None).

        A string is treated as a label and looked up in ``self.calibrations``;
        a missing label raises a self-diagnostic ValidationError listing what is
        attached. A Calibration object is returned as-is.
        """
        from .calibration import Calibration as _Calibration

        if calibration is None:
            return None
        if isinstance(calibration, _Calibration):
            return calibration
        if isinstance(calibration, str):
            try:
                return self.calibrations[calibration]
            except KeyError:
                available = sorted(self.calibrations)
                hint = f'Attached calibrations: {available}.' if available else (
                    'No calibrations are attached; use study.with_calibration(Calibration(...)) first.'
                )
                raise ValidationError(f"No calibration labeled {calibration!r}. {hint}") from None
        raise ValidationError(
            f'calibration must be a Calibration, a label string, or None; got {type(calibration).__name__}.'
        )

    def with_calibration(self, calibration: Calibration) -> Study:
        """Return a new Study with ``calibration`` attached under its label.

        Immutable: the current Study is unchanged. Re-attaching the same label
        replaces the prior entry. Select an attached calibration at execute time
        with ``study.execute(calibration='<label>')``.
        """
        from .calibration import Calibration as _Calibration

        if not isinstance(calibration, _Calibration):
            raise ValidationError(
                f'with_calibration expects a Calibration; got {type(calibration).__name__}.'
            )
        updated = dict(self.calibrations)
        updated[calibration.label] = calibration
        return dataclasses.replace(self, calibrations=updated)

    def _resolve_execute_chart(self, chart: str | None) -> str:
        """Resolve the requested chart name to a base chart type.

        None → `recommended_chart` (or raise if none available).
        Validates basic shape and parses through `_parse_chart_request`.
        """
        if chart is None and self.recommended_chart is None:
            raise ValidationError(
                'No charts available: no valid observations after data cleaning. '
                'Check your data for missing or invalid response values.'
            )
        if chart is not None and (not isinstance(chart, str) or chart == ''):
            raise ValidationError('Chart name must be a non-empty string')
        chart_request = chart or self.recommended_chart
        return self._parse_chart_request(chart_request)

    def _validate_execute_request(  # noqa: C901
        self,
        *,
        base_chart: str,
        by: list[str] | None,
        value: str | None,
        recentered: bool,
        companion: bool,
        bins: int | None,
        phased: bool,
        n_sigma: float,
        n_mode: str,
    ) -> tuple[list[str] | None, str | None, bool]:
        """Run all execute() parameter validation; resolve by/value/is_residual.

        Returns
        -------
        by_validated : list[str] | None
            The `by=` argument after validation. May be replaced with `[]`
            for Histogram (full-distribution default).
        value_col : str | None
            The column name to chart (response or residual).
        is_residual : bool
            True when `value` requests a residual.

        Raises
        ------
        ValidationError, ChartNotAvailableError
            Any user-facing validation failure.
        """
        # by= validation (may raise)
        by_validated = self._validate_by_parameter(by, base_chart)

        if recentered:
            self._validate_recentered(value)
        if phased:
            self._validate_phased(base_chart, by_validated)

        if companion and base_chart == 'Histogram':
            raise ValidationError('companion is not applicable to Histogram charts.')
        if bins is not None and base_chart != 'Histogram':
            raise ValidationError(f"bins is only applicable to Histogram charts, not '{base_chart}'.")

        self._validate_n_sigma_n_mode(n_sigma, n_mode, base_chart)

        # Histogram with by=None defaults to the full distribution.
        if base_chart == 'Histogram' and by is None:
            by_validated = []

        # Primary chart-vs-ADS validation.
        if base_chart not in self.valid_charts:
            available_list = list(self.valid_charts)
            raise ChartNotAvailableError(
                f"Chart type '{base_chart}' is not valid for ADS {self.analytical_design_state.sds}.\n"
                f'Valid charts: {", ".join(available_list)}\n'
                f'Recommended: {self.recommended_chart}\n'
                f"Use study.why_not('{base_chart}') for explanation.",
                chart=base_chart,
                available=available_list,
            )

        # Resolve value column (response or residual; R6 has its own compute path).
        if value is not None and value.upper() == 'R6':
            value_col = self._compute_r6(by_validated, recentered)
        else:
            value_col = self._resolve_value_column(value, recentered)

        is_residual = value is not None and value.upper().startswith('R')

        # Validate residual availability (Histogram exempt — plots any numeric column).
        if (
            is_residual
            and base_chart != 'Histogram'
            and value_col not in self._ads.analysis_dataset.columns
        ):
            available_residuals = [
                c for c in self._ads.analysis_dataset.columns
                if c.upper().startswith('R') and c[1:2].isdigit()
            ]
            raise ChartNotAvailableError(
                f"Residual column '{value_col}' not found in the dataset "
                f'for ADS {self.analytical_design_state.sds}.\n'
                f'Available residual columns: {", ".join(available_residuals) if available_residuals else "None"}\n'
                f'Use study.residuals to see available options.',
                chart=value_col,
                available=available_residuals,
            )

        # Validate (chart, value) combo. The rule itself lives in
        # _residual_pair_problem so why_not() answers from the same predicate — see
        # its docstring for the three regimes.
        if is_residual:
            problem = self._residual_pair_problem(base_chart, value)
            if problem:
                base_residual = _base_residual_code(value)
                valid_for_value = [c for c, v in self.residual_charts if v == base_residual]
                raise ChartNotAvailableError(
                    f"{problem}\n"
                    f"Use study.why_not('{base_chart}', value='{value}') for details.",
                    chart=base_chart,
                    available=valid_for_value,
                )

        return by_validated, value_col, is_residual

    def _response_pair_problem(self, base_chart: str) -> str | None:
        """Why ``base_chart`` can't chart *the response* — or ``None`` if it can.

        The advisory half of the rule ``analysis._raise_no_replicated_subgroups``
        enforces at compute time. Xbar and S summarise a subgroup, so charting
        the response needs cells with more than one observation in them; on
        ADS 2 there are none and ``execute()`` refuses.

        This is deliberately *not* the same question as :attr:`valid_charts`,
        which answers at the chart-family level and stays true on ADS 2 —
        ``execute(chart='Xbar', value='R6', by=[...])`` and
        ``execute(chart='Xbar', by=[time])`` are both legal there, because a
        residual and a time-aggregated subgroup do have replication. Conflating
        the two questions is what let ``why_not('Xbar')`` answer "IS available"
        about a call that then raised.
        """
        if base_chart not in ('Xbar', 'S'):
            return None
        if self._plan.has_replication != 'none':
            return None

        factors = self._spec.rsg_vars_list
        pool_hint = f"by=['{factors[0]}']" if factors else 'by=[<factor>]'
        return (
            f"'{base_chart}' cannot chart the response here: every factor x time cell holds "
            f'a single observation, so there is no within-subgroup variation to summarise '
            f'(ADS {self.analytical_design_state.sds}, no replication).\n'
            f'Available instead:\n'
            f"  • chart='X' — the response as individuals\n"
            f'  • a coarser subgroup that pools those single observations — e.g. '
            f'{pool_hint}\n'
            f"  • a residual — e.g. value='R6', {pool_hint}"
        )

    def _residual_pair_problem(self, base_chart: str, value: str) -> str | None:
        """Why ``(base_chart, value)`` can't be charted — or ``None`` if it can.

        Single source of truth for the chart x residual rule. ``_validate_execute_request``
        raises on it; ``why_not`` explains it. They cannot disagree because there is one rule.

        Three regimes:

        - ``Histogram`` — accepts any residual (it plots a distribution, not a control chart).
        - ``mR`` — accepts any residual, standalone or as a companion. A moving range is
          computed from consecutive values of whatever series is charted, so it is defined
          for a residual exactly as it is for the response.
        - ``Xbar`` / ``S`` / ``X`` — governed by :attr:`residual_charts`, which varies by ADS.

        Neither ``Histogram`` nor ``mR`` appears in :attr:`residual_charts`, so that
        attribute is *not* the whole rule — a caller testing ``(chart, value) in
        residual_charts`` alone will wrongly reject both.
        """
        base_residual = _base_residual_code(value)

        if base_chart in ('Histogram', 'mR'):
            return None

        if (base_chart, base_residual) not in self.residual_charts:
            valid = [c for c, v in self.residual_charts if v == base_residual]
            return (
                f"'{base_chart}' with value='{value}' is not valid for "
                f'ADS {self.analytical_design_state.sds}.\n'
                f'Valid charts for {base_residual}: {", ".join(valid) if valid else "None"}'
            )
        return None

    def _validate_recentered(self, value: str | None) -> None:
        """Validate recentered=True requirements."""
        needs_residuals = self._spec.has_grouping and self._spec.has_time
        if not needs_residuals:
            raise ValidationError(
                'recentered=True requires VAS decomposition (factors + time). '
                'Recentered residuals (RCR) reconstruct values relative to '
                'factor and time means, which require both to be specified.'
            )
        recenterable = set(RESIDUAL_CODES)
        if value is None:
            raise ValidationError('recentered=True requires a residual value (R1-R6), got value=None')
        if value.upper() not in recenterable:
            raise ValidationError(f"recentered=True requires a residual value (R1-R6), got '{value}'")

    def _compute_r6(self, by: list[str] | None, recentered: bool) -> str:
        """Compute R6 (factor main effect residual) on the fly.

        R6 = α_i + R2 where α_i = mean(R5 | factor level(s)).
        Accepts one or more factors via ``by``.
        """
        factors = self._spec.rsg_vars_list
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
            groupby_key = by_factors if len(by_factors) > 1 else by_factors[0]
        elif len(factors) == 1:
            groupby_key = factors[0]
        else:
            raise ValidationError(
                f'R6 requires by=[factor(s)] to specify which factor(s).\n'
                f'Available factors: {factors}\n'
                f"Example: study.execute(chart='Xbar', value='R6', by=['{factors[0]}'])"
            )

        df = self._ads.analysis_dataset.copy()
        alpha = df.groupby(groupby_key)['R5'].transform('mean')

        df['R6'] = alpha + df['R2']
        if recentered:
            df['RCR6'] = df['Ybar'] + alpha + df['R2']
        self._ads.analysis_dataset = df
        return 'RCR6' if recentered else 'R6'

    def _validate_phased(self, base_chart: str, by_validated: list[str] | None) -> None:
        """Validate phased=True requirements."""
        if base_chart not in ('X', 'mR'):
            raise ValidationError(f"phased=True is only valid for X or mR charts, got '{base_chart}'.")
        if by_validated is None:
            raise ValidationError('phased=True requires an explicit by= parameter.')
        if not self._spec.rsg_vars_list:
            raise ValidationError(
                'phased=True requires factors to define phases (rsg_key). '
                'This study has no factors; phased limits do not apply.'
            )
        if len(by_validated) > 0:
            all_factors = set(self._spec.rsg_vars_list)
            by_set = set(by_validated)
            if not (all_factors - by_set):
                raise ValidationError(
                    'phased=True requires collapsed factors to define phases. '
                    f'by={list(by_validated)} includes all factors; '
                    'no factors remain to create phase boundaries. '
                    'Remove factors from by= or set phased=False.'
                )

    def _parse_chart_request(self, chart: str) -> str:
        """
        Parse chart string to validate base chart type.

        Parameters
        ----------
        chart : str
            Chart request string. Only accepts base charts: 'Xbar', 'S', 'X', 'mR'

        Returns
        -------
        str
            Validated base chart type

        Raises
        ------
        ValidationError
            For invalid chart types or old residual chart syntax
        """
        if not chart or not isinstance(chart, str):
            raise ValidationError('Chart name must be a non-empty string')

        # Normalize case (e.g. "x" -> "X", "XBAR" -> "Xbar")
        chart = normalize_chart_name(chart)

        # Base chart - valid
        if chart in VALID_BASE_CHARTS:
            return chart

        # R1-R6 without base chart
        if re.match(r'^R\d+$', chart) or re.match(r'^RCR\d+$', chart):
            raise ValidationError(
                f"'{chart}' is a residual identifier, not a chart type.\n"
                f"Use: study.execute(chart='Xbar', value='{chart}')\n"
                f"Or: study.execute(chart='X', value='{chart}')"
            )

        # Alias without base chart.
        #
        # Must precede the old-syntax branch below. Every alias but 'noise' contains an
        # underscore, so testing '_' first sent them all to _raise_old_syntax_error,
        # which finds no base chart in the trailing word ('within_cell' -> 'cell') and
        # falls through to the generic "Invalid chart name". Four of the five original
        # aliases could never produce this message.
        if chart in RESIDUAL_ALIASES:
            residual_id = RESIDUAL_ALIASES[chart]
            raise ValidationError(
                f"'{chart}' is a residual alias for {residual_id} "
                f'({RESIDUAL_LABELS[residual_id]}), not a chart type.\n'
                f"Use: study.execute(chart='Xbar', value='{residual_id}')\n"
                f"Or: study.execute(chart='X', value='{residual_id}')"
            )

        # Detect old residual chart syntax and provide migration guidance
        if '_' in chart:
            # Old syntax: R5_Xbar, noise_Xbar, rc_R5_Xbar, RCR5_Xbar
            self._raise_old_syntax_error(chart)

        raise ValidationError(f"Unknown chart '{chart}'. Valid chart types: {', '.join(sorted(VALID_BASE_CHARTS))}")

    def _raise_old_syntax_error(self, chart: str) -> None:
        """Raise helpful error for old residual chart syntax."""
        # Strip rc_ prefix if present
        working = chart
        recentered = False
        if working.startswith('rc_'):
            recentered = True
            working = working[3:]

        # Try to parse residual and base chart
        if '_' in working:
            residual_part, base_chart = working.rsplit('_', 1)

            # Check for RCR prefix
            if residual_part.startswith('RCR'):
                residual_id = f'R{residual_part[3:]}'
                recentered = True
            elif residual_part.startswith('R') and residual_part[1:].isdigit():
                residual_id = residual_part
            elif residual_part in RESIDUAL_ALIASES:
                residual_id = RESIDUAL_ALIASES[residual_part]
            else:
                residual_id = residual_part

            if base_chart in VALID_BASE_CHARTS:
                recentered_hint = ', recentered=True' if recentered else ''
                raise ValidationError(
                    f"Old residual chart syntax '{chart}' is no longer supported.\n"
                    f"Use: study.execute(chart='{base_chart}', value='{residual_id}'{recentered_hint})"
                )

        raise ValidationError(
            f"Invalid chart name '{chart}'. Valid chart types: {', '.join(sorted(VALID_BASE_CHARTS))}"
        )

    def _validate_by_parameter(  # noqa: C901
        self, by: list[str] | None, base_chart: str
    ) -> list[str] | None:
        """
        Validate and normalize the `by` parameter.

        Parameters
        ----------
        by : list[str] | None
            User-specified by parameter
        base_chart : str
            Base chart type being requested (Xbar, S, X, mR)

        Returns
        -------
        list[str] | None
            Normalized by parameter

        Raises
        ------
        ValidationError
            If by contains factors not in rsg_vars
            If by=None for X/mR with factors (must be explicit)
            If by contains time variable for X/mR charts
        """
        factors = self._spec.rsg_vars_list
        time_var = self._spec.time_var
        is_time_series_chart = base_chart in ('X', 'mR')

        # No factors case: by=None is fine, by=[] is also fine
        if not factors:
            if by is not None and by != []:
                raise ValidationError(
                    f'Invalid by={by}. No factors defined in this study. Use by=None or by=[] for single stream.'
                )
            return []  # Normalize to empty list for no-factor case

        # Has factors case
        if by is None:
            if is_time_series_chart:
                # X/mR with factors requires explicit by
                factor_str = ', '.join(f"'{f}'" for f in factors)
                raise ValidationError(
                    f"X/mR charts with factors require explicit 'by' parameter.\n"
                    f'Specify how to stratify:\n'
                    f'  by=[{factor_str}] for {self._get_factor_combinations()} charts (one per factor combination)\n'
                    f"  by=['{factors[0]}'] for fewer charts (stratify by single factor)\n"
                    f'  by=[] for single overall chart'
                )
            # Xbar/S: by=None means full rsg_key (current behavior)
            return None

        # Normalize string to list
        if isinstance(by, str):
            by = [by]

        # Validate by is subset of valid dimensions
        # - X/mR: factors only (time is x-axis)
        # - Xbar/S: factors + time (cell_key dimensions)
        by_set = set(by)
        factor_set = set(factors)
        invalid = by_set - factor_set

        if invalid and time_var and time_var in invalid:  # noqa: SIM102
            # Check if user tried to use time
            if is_time_series_chart:
                raise ValidationError(
                    f"Cannot use time variable '{time_var}' in by for {base_chart} charts. "
                    f'Time is the x-axis for {base_chart} charts, not a stratification dimension. '
                    f'Valid by dimensions: {sorted(factors)}'
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
                case_match = next((v for v in valid_dims if v.lower() == inv.lower()), None)
                if case_match:
                    suggestions.append(
                        f"'{inv}' is not a valid by variable.\nDid you mean '{case_match}'? (names are case-sensitive)"
                    )
                    if first_invalid is None:
                        first_invalid = inv
                        first_suggestion = case_match
                else:
                    # Try fuzzy matching
                    close = difflib.get_close_matches(inv, valid_dims, n=1, cutoff=0.6)
                    if close:
                        suggestions.append(f"'{inv}' is not a valid by variable.\nDid you mean '{close[0]}'?")
                        if first_invalid is None:
                            first_invalid = inv
                            first_suggestion = close[0]
                    else:
                        suggestions.append(f"'{inv}' is not a valid by variable.")
                        if first_invalid is None:
                            first_invalid = inv

            msg = '\n'.join(suggestions)
            msg += f'\nValid: {", ".join(valid_dims)}'
            raise FactorNotFoundError(msg, factor=first_invalid, suggestion=first_suggestion, available=valid_dims)

        # Deduplicate while preserving order (first occurrence wins)
        seen = set()
        deduped = []
        for v in by:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        if len(deduped) < len(by):
            import warnings

            warnings.warn(
                f'Duplicate values in by={list(by)} were removed. Using by={deduped}.',
                UserWarning,
                stacklevel=4,
            )
            by = deduped

        return list(by)

    @staticmethod
    def _validate_n_sigma_n_mode(n_sigma: float, n_mode: str, base_chart: str) -> None:
        """Validate n_sigma and n_mode parameters for execute()."""
        from .exceptions import ValidationError

        if (n_sigma != 3.0 or n_mode != 'actual') and base_chart not in ('Xbar', 'S'):
            raise ValidationError(f"n_sigma and n_mode are only supported for Xbar/S charts, got '{base_chart}'.")

        if base_chart in ('Xbar', 'S'):
            if not (isinstance(n_sigma, (int, float)) and math.isfinite(n_sigma) and n_sigma > 0):
                raise ValidationError(f'n_sigma must be a finite number > 0; got {n_sigma!r}')
            if n_mode not in ('actual', 'average'):
                raise ValidationError(f"n_mode must be 'actual' or 'average'; got {n_mode!r}")

    def _get_factor_combinations(self) -> int:
        """Get count of unique factor combinations in the dataset."""
        if not self._spec.rsg_vars:
            return 1
        return self._ads.analysis_dataset[self._spec.rsg_var_name].nunique()

    def _resolve_value_column(self, value: str | None, recentered: bool) -> str:
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
        ValidationError
            If value specifies an unavailable residual
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
            raise ValidationError(
                f"Invalid value '{value}'. "
                f"Valid options: 'response', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6' "
                f"(or 'RCR1'-'RCR5' for recentered, 'R6' with recentered=True)."
            )

        # Validate column exists in analysis dataset
        if col_name not in self._ads.analysis_dataset.columns:
            available = [
                c for c in self._ads.analysis_dataset.columns if c.startswith('R') and len(c) == 2 and c[1].isdigit()
            ]
            raise ValidationError(
                f"Residual column '{col_name}' not available for ADS {self.analytical_design_state.sds}. "
                f'Available: {available}'
            )

        return col_name

    # =========================================================================
    # Display Methods
    # =========================================================================

    def __repr__(self) -> str:
        """
        Concise study summary.

        Shows formulation, design-state lineage, and available charts in
        minimal format. Use ``study.design()`` for the full lineage
        report and ``study.support`` for the chart availability DataFrame.
        """
        # 1-line formulation summary
        factors_str = ', '.join(self.factors) if self.factors else 'None'
        time_str = self.time or 'None'

        ads_sds = self.analytical_design_state.sds
        ods_sds = self.observed_design_state.sds
        sds_display = f'ads={ads_sds}' if ads_sds == ods_sds else f'ods={ods_sds}, ads={ads_sds}'
        lines = [
            f"Study(response='{self.response}', factors=[{factors_str}], time='{time_str}', {sds_display})",
        ]
        if self.unit_of_analysis:
            lines.append(f'  Unit of analysis: {self.unit_of_analysis}')
        valid_str = ', '.join(self.valid_charts) if self.valid_charts else 'None'
        rec_str = self.recommended_chart or 'None'
        lines.append(f'  Valid: {valid_str} | Recommended: {rec_str}')
        if self.residuals:
            lines.append(f'  Residuals: {", ".join(self.residuals)}')
        lines.append('  → study.execute() or study.support for details')

        return '\n'.join(lines)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        factors_str = ', '.join(self.factors) if self.factors else 'None'
        time_str = self.time or 'None'

        residuals = self.residuals
        residual_html = f'<br><strong>Residuals:</strong> {", ".join(residuals)}' if residuals else ''
        uoa_html = f'<br><strong>Unit of analysis:</strong> {self.unit_of_analysis}' if self.unit_of_analysis else ''

        ads_sds = self.analytical_design_state.sds
        ods_sds = self.observed_design_state.sds
        sds_display = f'sds={ads_sds}' if ads_sds == ods_sds else f'ods={ods_sds}, ads={ads_sds}'

        style = 'font-family: monospace; padding: 8px; border: 1px solid #ccc; background: #f9f9f9'
        valid_charts_str = ', '.join(self.valid_charts) if self.valid_charts else 'None'
        rec_str = self.recommended_chart or 'None'
        html = f"""
        <div style="{style}">
            <code>Study(response='{self.response}', factors=[{factors_str}], time='{time_str}', {sds_display})</code>
            {uoa_html}
            <br><strong>Valid:</strong> {valid_charts_str} | <strong>Recommended:</strong> {rec_str}
            {residual_html}
            <br><em>→ study.execute() or study.support for details</em>
        </div>
        """
        return html
