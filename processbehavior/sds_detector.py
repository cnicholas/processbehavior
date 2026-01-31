"""
Sampling Design State (SDS) detection for process behavior analysis.

This module implements the complete SDS classification system from the
Variance Analysis System (VAS) framework by Wheeler and Bishop.

The SDS determines:
- What type of data structure we have
- What residual calculations are appropriate
- What variance estimation methods to use
- What analyses are supported

Follows the Pythonic Hadley philosophy:
- Clear, self-documenting logic
- Helpful error messages
- Explicit over implicit
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Literal

import pandas as pd

# Formal vocabulary for SDS classification reasons (per Wheeler/Bishop Table 1)
SDSReasonType = Literal[
    # Complete/Semi-Complete (SDS 1, 2, 3) - no empty cells
    "full_replication",            # SDS 1: all N_kt >= 2
    "no_replication",              # SDS 2: all N_kt = 1
    "partial_replication",         # SDS 3: mixed N_kt (some 1, some >=2)
    # Incomplete (SDS 4, 5, 6) - has empty cells (N_kt = 0)
    "incomplete_with_singletons",  # SDS 4: has 0s, has 1s, has >=2s
    "incomplete_no_singletons",    # SDS 5: has 0s, no 1s, has >=2s
    "incomplete_no_replication",   # SDS 6: has 0s, max = 1
]

# R2 calculation method - structure-driven, not SDS-driven
R2Method = Literal["exact", "ma2", "hybrid"]


@dataclass(frozen=True)
class StructureStats:
    """
    Observed structure of the data - drives R2 method selection and availability checks.

    This is computed once in AnalysisDataSet and threaded through to avoid
    recomputing structure statistics in multiple places.

    Attributes
    ----------
    has_grouping : bool
        Whether factor variables (rsg_vars) are present
    has_order : bool
        Whether ordering key (obs_id) exists
    n_cell_min : int
        Minimum observations per (rsg_key, time) cell
    n_cell_max : int
        Maximum observations per (rsg_key, time) cell
    K_obs : int
        Number of unique rsg_key values (for R5 availability check)
    """
    has_grouping: bool
    has_order: bool
    n_cell_min: int
    n_cell_max: int
    K_obs: int

if TYPE_CHECKING:
    from .analysis_specification import DataPrepConfig

logger = logging.getLogger(__name__)


# ============================================================================
# SDS Analysis Plan - The "Recipe" for Each Sampling Design State
# ============================================================================

@dataclass
class SDSAnalysisPlan:
    """
    Complete specification of what analysis can be performed for a given SDS.

    This is the authoritative "recipe" that the system follows when it detects
    a particular Sampling Design State. It defines capabilities and limitations
    for each SDS, enabling validation against Bishop's methodology.

    Attributes
    ----------
    sds : int
        Sampling Design State (0-6)
    name : str
        Short descriptive name
    description : str
        Detailed description of the data structure
    has_factors : bool
        Whether grouping/factor variables are present
    has_time : bool
        Whether time variable is present
    has_replication : str
        Replication pattern: 'full', 'partial', or 'none'
    valid_charts : list[str]
        Chart types that can be used with this SDS
    recommended_chart : str
        Default/recommended chart type
    invalid_charts : list[str]
        Chart types that cannot be used (with reasons)
    vas_residuals_supported : bool
        Whether VAS residual calculations (R1-R5) are supported
    residuals_available : list[str]
        Which residuals can be calculated
    residual_calculation_method : str
        How R2 is calculated ('exact', 'moving_average', 'hybrid', or 'none')
    main_effects_supported : bool
        Whether main effect calculations are supported
    interaction_effects_supported : bool
        Whether interaction effect calculations are supported
    supports_stratification : bool
        Whether data can be split into stratified charts
    typical_use_cases : list[str]
        Common real-world scenarios
    limitations : list[str]
        What cannot be done with this SDS
    bishop_reference : str
        Reference to Wheeler/Bishop methodology

    Examples
    --------
    >>> plan = SDSRegistry.get_analysis_plan(sds=1)
    >>> print(plan.name)
    'Full Factorial with Complete Replication'
    >>> print(plan.vas_residuals_supported)
    True
    >>> print(plan.valid_charts)
    ['Xbar', 'S', 'Imr']
    """
    sds: int
    name: str
    description: str

    # Data structure characteristics
    has_factors: bool
    has_time: bool
    has_replication: str  # 'full', 'partial', 'none'

    # Valid chart types
    valid_charts: list[str]
    recommended_chart: str
    invalid_charts: list[str]

    # VAS residual capabilities
    vas_residuals_supported: bool
    residuals_available: list[str]
    residual_calculation_method: str  # 'exact', 'moving_average', 'hybrid', 'none'

    # Effects and interactions
    main_effects_supported: bool
    interaction_effects_supported: bool

    # Stratification
    supports_stratification: bool

    # Use cases and notes
    typical_use_cases: list[str]
    limitations: list[str]
    bishop_reference: str

    # Actual data characteristics (populated during detection)
    min_cell_size: int = 0  # Minimum observations per cell from actual data

    # Class constant: questions each chart type answers
    CHART_QUESTIONS: ClassVar[dict[str, str]] = {
        # Primary charts
        'Xbar': 'Are subgroup means stable over time?',
        'S': 'Is within-subgroup variation stable?',
        'Imr': 'Is individual variation stable over time?',
        'R': 'Is range variation stable over time?',
        # Residual charts
        'R2_S': 'Is within-subgroup variation stable?',
        'R2_Imr': 'Is within-subgroup variation stable?',
        'R3_Imr': 'Is there factor×time interaction?',
        'R3_Xbar': 'Is there factor×time interaction affecting the mean?',
        'R3_S': 'Is there factor×time interaction affecting variation?',
        'R4_Imr': 'Does time have a significant effect?',
        'R4_Xbar': 'Does time have a significant effect on the mean?',
        'R4_S': 'Does time have a significant effect on variation?',
        'R5_Imr': 'Does the factor have a significant effect?',
        'R5_Xbar': 'Do factors have a significant effect on the mean?',
        'R5_S': 'Do factors have a significant effect on variation?',
    }

    @property
    def residual_charts(self) -> list[str]:
        """
        Residual chart types available for this SDS.

        Returns the appropriate residual charts based on VAS support
        and data structure:

        - R2: S chart when cells have n>=2, otherwise Imr
        - R3: Xbar/S when cells have n>=2, otherwise Imr (same subgrouping as R2)
        - R4: Xbar/S when has_factors (aggregate across factors by time)
        - R5: Xbar/S when has_time (aggregate across time by factor)

        Per Wheeler/Bishop Sections 20.6.1-4:
        - R2 uses (k,t) cell subgrouping - S when n>=2
        - R3 uses (k,t) cell subgrouping - Xbar/S when n>=2
        - R4_Xbar/S use time-based subgrouping (N_.t = Σ_k N_kt)
        - R5_Xbar/S use factor-based subgrouping (N_k. = Σ_t N_kt)
        """
        if not self.vas_residuals_supported:
            return []

        # R2: S chart when cells have replication (min_cell_size >= 2)
        r2_chart = 'R2_S' if self.min_cell_size >= 2 else 'R2_Imr'

        # R3: Same subgrouping as R2 - Xbar/S when min_cell_size >= 2
        # Per Wheeler Section 20.6.2: "Xbar and S charts can be used for SDS 1, 3, 4, 5"
        r3_charts = ['R3_Xbar', 'R3_S'] if self.min_cell_size >= 2 else ['R3_Imr']

        # R4: Xbar/S when aggregating across factors gives n>=2 per time subgroup
        # Requires has_factors=True (multiple factors to aggregate)
        r4_charts = ['R4_Xbar', 'R4_S'] if self.has_factors else ['R4_Imr']

        # R5: Xbar/S when aggregating across time gives n>=2 per factor subgroup
        # Requires has_time=True (multiple time points to aggregate)
        r5_charts = ['R5_Xbar', 'R5_S'] if self.has_time else ['R5_Imr']

        return [r2_chart] + r3_charts + r4_charts + r5_charts

    def __str__(self) -> str:
        """Pretty print the analysis plan."""
        lines = [
            f"SDS {self.sds}: {self.name}",
            "=" * 70,
            f"Description: {self.description}",
            "",
            "Data Structure:",
            f"  • Factors: {'Yes' if self.has_factors else 'No'}",
            f"  • Time: {'Yes' if self.has_time else 'No'}",
            f"  • Replication: {self.has_replication.capitalize()}",
            "",
            "Chart Capabilities:",
            f"  • Valid charts: {', '.join(self.valid_charts)}",
            f"  • Recommended: {self.recommended_chart}",
        ]

        if self.invalid_charts:
            lines.append(f"  • Invalid charts: {', '.join(self.invalid_charts)}")

        lines.extend([
            "",
            "Variance Analysis System (VAS):",
            f"  • VAS supported: {'Yes' if self.vas_residuals_supported else 'No'}",
        ])

        if self.vas_residuals_supported:
            lines.append(f"  • Residuals: {', '.join(self.residuals_available)}")
            lines.append(f"  • R2 calculation: {self.residual_calculation_method}")

        lines.extend([
            "",
            "Effects Analysis:",
            f"  • Main effects: {'Yes' if self.main_effects_supported else 'No'}",
            f"  • Interactions: {'Yes' if self.interaction_effects_supported else 'No'}",
            f"  • Stratification: {'Yes' if self.supports_stratification else 'No'}",
        ])

        if self.typical_use_cases:
            lines.append("")
            lines.append("Typical Use Cases:")
            for use_case in self.typical_use_cases:
                lines.append(f"  • {use_case}")

        if self.limitations:
            lines.append("")
            lines.append("Limitations:")
            for limitation in self.limitations:
                lines.append(f"  • {limitation}")

        lines.extend([
            "",
            f"Reference: {self.bishop_reference}",
        ])

        return "\n".join(lines)


@dataclass(frozen=True)
class SDSResult:
    """
    Result of SDS detection.

    Attributes
    ----------
    sds : int
        Sampling Design State (1-6) per Wheeler/Bishop Table 1
    min_cell_size : int
        Minimum observations per cell (for chart selection)
    reason : SDSReasonType | None
        Why this SDS was detected, based on N_kt distribution.

        Complete/Semi-Complete (no empty cells):
        - "full_replication": SDS 1 (all N_kt >= 2)
        - "no_replication": SDS 2 (all N_kt = 1)
        - "partial_replication": SDS 3 (mixed N_kt)

        Incomplete (has empty cells, requires plan):
        - "incomplete_with_singletons": SDS 4 (has 0s, 1s, and >=2s)
        - "incomplete_no_singletons": SDS 5 (has 0s and >=2s, no 1s)
        - "incomplete_no_replication": SDS 6 (has 0s, max = 1)
    n_empty_cells : int
        Count of cells with Nₖₜ=0 after plan reindex (if plan provided).
        For observed-only detection, counts cells where all responses are NA.
    """
    sds: int
    min_cell_size: int
    reason: SDSReasonType | None = None
    n_empty_cells: int = 0


class SDSRegistry:
    """
    Registry of Sampling Design State (SDS 1-6) definitions and rules.

    Implements Wheeler/Bishop Table 1 classification based on N_kt distribution.

    Provides:
    - SDS detection from data structure
    - Analysis plans and capabilities for each SDS
    - Validation of SDS/analysis compatibility
    - VAS residual calculation rules

    The SDS classification system (per Wheeler/Bishop Table 1):

    **Complete/Semi-Complete** (no empty cells in grid):

    **SDS 1**: Complete - Full replication
        - Min N_kt ≥ 2 (all cells have multiple observations)
        - Best for variance estimation

    **SDS 2**: Semi-Complete - No replication
        - Min N_kt = 1 and Max N_kt = 1 (all cells are singletons)
        - Requires moving average for R2

    **SDS 3**: Semi-Complete - Partial replication
        - Min N_kt = 1 and Max N_kt ≥ 2 (mixed)
        - Hybrid variance estimation

    **Incomplete** (has empty cells, requires sampling plan to detect):

    **SDS 4**: Incomplete with singletons
        - Has empty cells (N_kt = 0), singletons (N_kt = 1), and replicated (N_kt ≥ 2)
        - Mixed everything

    **SDS 5**: Incomplete without singletons
        - Has empty cells (N_kt = 0) and replicated (N_kt ≥ 2), but NO singletons
        - Can estimate variance from replicated cells

    **SDS 6**: Incomplete without replication
        - Has empty cells (N_kt = 0) and Max N_kt = 1
        - Cannot estimate within-cell variance

    Examples
    --------
    Detect SDS from prepared data:

    >>> registry = SDSRegistry()
    >>> result = registry.detect_sds(df, spec)
    >>> print(f"Detected SDS {result.sds}")
    Detected SDS 1

    With sampling plan (enables SDS 4-6 detection):

    >>> plan = {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}
    >>> result = registry.detect_sds(df, spec, plan=plan)
    """

    def detect_sds_from_structure(
        self,
        df: pd.DataFrame,
        spec: DataPrepConfig,
        response_col: str,
        plan: dict[str, list] | None = None,
        T_planned: int | None = None
    ) -> SDSResult:
        """
        Detect SDS from raw data structure (before response NA rows are dropped).

        This matches Tom Bishop's Minitab approach: cells with all-NA responses
        still count as "attempted" cells, revealing the true intended structure.

        Parameters
        ----------
        df : DataFrame
            Raw data (response NA rows NOT yet dropped)
        spec : DataPrepConfig
            Data preparation configuration
        response_col : str
            Name of response column
        plan : dict, optional
            Sampling plan defining intended structure. Should include time column
            values if time-based structure is expected.
        T_planned : int, optional
            Planned number of time points. Only used when plan is provided but
            doesn't include the time column. In that case, time values 1..T_planned
            are crossed with plan factors to create expected cells.
            If plan includes time column values, this parameter is ignored.

        Returns
        -------
        SDSResult
            SDS classification based on raw structure
        """
        # Build minimal structure view
        structure_view, kt_cols = self._build_structure_view(df, spec, response_col)

        # Compute N_kt = count of VALID responses per cell
        # groupby naturally yields 0 for cells where ALL responses are NA
        if kt_cols:
            nkt_observed = structure_view.groupby(kt_cols, observed=True)[response_col].apply(
                lambda s: s.notna().sum()
            )
        else:
            # No factors or time - single cell
            # NOTE: This uses regular Index (not MultiIndex). _classify_by_nkt must tolerate both.
            nkt_observed = pd.Series([structure_view[response_col].notna().sum()], index=[()])

        # Validate we have some data
        if nkt_observed.sum() == 0:
            raise ValueError("No valid response values found after filtering")

        if plan is not None:
            # WITH PLAN: Compare to canonicalized plan
            canonicalized_plan = self._canonicalize_plan_values(plan, kt_cols)

            # Handle time column:
            # 1. If T_planned is given, use 1..T_planned
            # 2. If plan includes time column, use that
            # 3. Otherwise, use observed time values from structure_view
            t = spec.time_var
            if t and t in kt_cols:
                if t not in canonicalized_plan or not canonicalized_plan[t]:
                    if T_planned is not None:
                        # Use 1..T_planned
                        canonicalized_plan[t] = list(range(1, T_planned + 1))
                    else:
                        # Use observed time values from the data
                        observed_times = structure_view[t].dropna().unique().tolist()
                        canonicalized_plan[t] = sorted(observed_times)
                        logger.debug(
                            f"Plan doesn't include time column '{t}'; "
                            f"using observed time values: {canonicalized_plan[t]}"
                        )

            # Check if we have all factor columns (not including time which we just filled)
            factor_cols = [c for c in kt_cols if c != t]
            missing_factor_cols = [c for c in factor_cols if c not in canonicalized_plan or not canonicalized_plan[c]]

            if missing_factor_cols:
                logger.warning(
                    f"Plan missing required factor columns or has empty values for {missing_factor_cols}; "
                    f"falling back to observed structure only."
                )
                nkt_counts = nkt_observed
            elif kt_cols:
                # Build expected cells using from_product (deterministic, avoids tuple list)
                planned_index = pd.MultiIndex.from_product(
                    [canonicalized_plan[c] for c in kt_cols],
                    names=kt_cols
                )
                nkt_counts = nkt_observed.reindex(planned_index, fill_value=0)
            else:
                # No kt columns - single cell
                nkt_counts = nkt_observed
        else:
            # WITHOUT PLAN: Use observed structure directly
            # Zeros naturally appear for cells where all responses are NA
            nkt_counts = nkt_observed

        # has_empty_cells is derived directly from N_kt distribution
        n_empty_cells = int((nkt_counts == 0).sum())
        has_empty_cells = n_empty_cells > 0

        if has_empty_cells:
            logger.debug(
                f"Structure detection: {n_empty_cells} cells with N_kt=0"
            )

        # Calculate min_cell_size for chart selection (only cells with valid data)
        valid_nkt = nkt_counts[nkt_counts > 0]
        min_cell_size = int(valid_nkt.min()) if len(valid_nkt) > 0 else 0

        # Classify using standard logic
        sds, reason = self._classify_by_nkt(nkt_counts, has_empty_cells)

        # Include n_empty_cells in result for debugging/diagnostics
        return SDSResult(sds=sds, min_cell_size=min_cell_size, reason=reason, n_empty_cells=n_empty_cells)

    def detect_sds(
        self,
        df: pd.DataFrame,
        spec: DataPrepConfig,
        plan: dict[str, list] | None = None,
        T_planned: int | None = None,
        response_col: str | None = None
    ) -> SDSResult:
        """
        Detect Sampling Design State from data structure.

        This is a convenience wrapper around detect_sds_from_structure().

        Parameters
        ----------
        df : DataFrame
            Data (ideally raw with response NA rows preserved for accurate detection)
        spec : DataPrepConfig
            Data preparation configuration
        plan : dict, optional
            Sampling plan specifying expected factor levels. Keys are column
            names, values are lists of expected levels. When provided, enables
            SDS 4-6 detection by comparing observed structure to planned
            structure (detecting empty cells where N_kt = 0).

            Mode 1 (plan=None): Classify based on observed data → SDS 1-3
            Mode 2 (plan={...}): Compare to plan → enables SDS 4-6
        T_planned : int, optional
            Expected number of time points from sampling plan. When provided,
            coverage calculation uses this instead of observed time count.
            This enables detection of incomplete temporal coverage.
        response_col : str, optional
            Response column name. If not provided, uses spec.response_var.

        Returns
        -------
        SDSResult
            Result containing sds (1-6), min_cell_size, reason, and n_empty_cells.

        Notes
        -----
        **Classification Logic (per Wheeler/Bishop Table 1)**

        Complete/Semi-Complete (no empty cells):
        - SDS 1: Min N_kt ≥ 2 (all cells replicated)
        - SDS 2: Min = Max = 1 (all cells singleton)
        - SDS 3: Min = 1, Max ≥ 2 (mixed)

        Incomplete (has empty cells, requires plan):
        - SDS 4: Has 0s, has 1s, has ≥2s (mixed everything)
        - SDS 5: Has 0s, NO 1s, has ≥2s (no singletons)
        - SDS 6: Has 0s, Max = 1 (no replication)

        Examples
        --------
        >>> # Full replication: all cells have n≥2
        >>> df = pd.DataFrame({
        ...     'rsg': ['A', 'A', 'B', 'B'],
        ...     'time': [1, 1, 1, 1],
        ...     'y': [10, 11, 9, 10]
        ... })
        >>> result = detector.detect_sds(df, spec)
        >>> result.sds
        1
        >>> result.reason
        'full_replication'

        >>> # With sampling plan (enables SDS 4-6 detection)
        >>> plan = {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}
        >>> result = detector.detect_sds(df, spec, plan=plan)
        """
        resp = response_col if response_col is not None else spec.response_var

        # Warn if data has no NA responses - likely passed prepared data
        # Use _normalize_missing_tokens to detect NA (consistent with SDS detection)
        # Guard against KeyError if column missing (will fail later with better error)
        if resp in df.columns:
            normalized_resp = self._normalize_missing_tokens(df[resp])
            if normalized_resp.isna().sum() == 0:
                logger.debug(
                    "detect_sds called with data that has no NA responses; "
                    "if this is prepared data, SDS may miss attempted-but-invalid cells."
                )

        return self.detect_sds_from_structure(df, spec, resp, plan, T_planned)

    def get_sds_characteristics(self, sds: int) -> dict:
        """
        Get detailed characteristics of an SDS.

        Returns a dictionary describing the SDS properties, including:
        - Human-readable description
        - Replication type
        - R2 calculation method
        - Analysis capabilities
        - Interaction analysis support

        Parameters
        ----------
        sds : int
            SDS number (0-6)

        Returns
        -------
        dict
            Dictionary with SDS characteristics

        Examples
        --------
        >>> info = detector.get_sds_characteristics(1)
        >>> info['description']
        'Full replication (all cells n≥2)'
        >>> info['r2_method']
        'within_cell'
        >>> info['capabilities']
        ['full_vas', 'all_residuals', 'interactions', 'main_effects']
        """
        characteristics = {
            0: {
                'description': 'No grouping or time structure',
                'replication_type': 'none',
                'r2_method': 'not_applicable',
                'capabilities': ['basic_statistics_only'],
                'interaction_analysis': False,
                'variance_decomposition': False
            },
            1: {
                'description': 'Full replication (all cells n≥2)',
                'replication_type': 'full',
                'r2_method': 'within_cell',
                'capabilities': ['full_vas', 'all_residuals', 'interactions', 'main_effects'],
                'interaction_analysis': True,
                'variance_decomposition': True
            },
            2: {
                'description': 'No replication (all cells n=1)',
                'replication_type': 'none',
                'r2_method': 'moving_average',
                'capabilities': ['all_residuals', 'limited_interactions', 'main_effects'],
                'interaction_analysis': 'limited',
                'variance_decomposition': True
            },
            3: {
                'description': 'Partial replication (mixed n=1 and n≥2)',
                'replication_type': 'partial',
                'r2_method': 'hybrid',
                'capabilities': ['all_residuals', 'partial_interactions', 'main_effects'],
                'interaction_analysis': 'partial',
                'variance_decomposition': True
            },
            4: {
                'description': 'Incomplete grid with singletons (has 0s, 1s, and ≥2s)',
                'replication_type': 'partial',
                'r2_method': 'hybrid',
                'capabilities': ['partial_vas', 'main_effects', 'stratification'],
                'interaction_analysis': False,
                'variance_decomposition': True
            },
            5: {
                'description': 'Incomplete grid without singletons (has 0s and ≥2s, no 1s)',
                'replication_type': 'partial',
                'r2_method': 'exact',  # All observed cells have replication
                'capabilities': ['full_vas', 'main_effects', 'stratification'],
                'interaction_analysis': False,  # Incomplete grid limits this
                'variance_decomposition': True
            },
            6: {
                'description': 'Unstructured/irregular grid',
                'replication_type': 'irregular',
                'r2_method': 'adaptive',
                'capabilities': ['regime_detection', 'adaptive_limits', 'sparse_analysis'],
                'interaction_analysis': False,
                'variance_decomposition': True  # VAS supported via moving average
            }
        }

        result = characteristics.get(sds, characteristics[0]).copy()
        result['sds'] = sds

        return result

    def get_r2_method(self, stats: StructureStats) -> R2Method:
        """
        Determine R2 calculation method from observed structure.

        This is the ONLY place where structure determines R2 method.
        Do NOT use SDS labels here - use observed counts directly.

        The R2 residual (within-cell variation) is the only VAS residual
        whose calculation varies by structure. All other residuals (R1, R3,
        R4, R5) are pure algebraic transformations once means are defined.

        Parameters
        ----------
        stats : StructureStats
            Observed structure statistics computed from data

        Returns
        -------
        R2Method
            'exact' if all cells have replication (n_cell_min >= 2)
            'ma2' if all cells are singletons (n_cell_max == 1)
            'hybrid' for mixed replication (exact where n >= 2, MA2 where n = 1)

        Notes
        -----
        Deterministic rule based on Wheeler/Bishop methodology:
        - Eq 59 (exact): R2 = Y - Ȳ_kt, requires replication
        - Eq 66 (MA2): R2 = (Y_j - Y_{j-1}) / 2, for singletons
        - Hybrid: exact where replicated, MA2 where singleton
        """
        if stats.n_cell_min >= 2:
            return "exact"
        elif stats.n_cell_max == 1:
            return "ma2"
        else:
            return "hybrid"

    def validate_sds_for_analysis(
        self,
        sds: int,
        analysis_type: str
    ) -> bool:
        """
        Validate that SDS is appropriate for requested analysis.

        Checks for known incompatibilities and logs helpful warnings.
        Raises ValueError for fatal incompatibilities.

        Parameters
        ----------
        sds : int
            Detected SDS (1-6)
        analysis_type : str
            Requested analysis type ('Xbar', 'S', 'Imr', 'R')

        Returns
        -------
        bool
            True if analysis can proceed

        Raises
        ------
        ValueError
            If SDS/analysis combination is invalid, with suggestion

        Examples
        --------
        >>> # SDS 4 with Xbar - valid (incomplete grid with mixed replication)
        >>> detector.validate_sds_for_analysis(sds=4, analysis_type='Xbar')
        True

        >>> # SDS 6 with Xbar - warning (no replication)
        >>> detector.validate_sds_for_analysis(sds=6, analysis_type='Xbar')
        # Logs warning about incomplete grid
        True
        """
        # SDS 4 or 5: Incomplete grid
        if sds in [4, 5]:
            logger.info(
                f"SDS {sds} detected: Incomplete grid.\n"
                f"Some factor×time combinations are missing from the data."
            )

        # SDS 6: Incomplete without replication
        if sds == 6:
            logger.warning(
                "SDS 6 detected: Incomplete grid without replication.\n"
                "Analysis results may be limited - no within-cell variance estimation.\n"
                "Consider using 'Imr' analysis for this data structure."
            )

        return True

    def should_calculate_vas_residuals(
        self,
        sds: int,
        analysis_type: str
    ) -> bool:
        """
        Determine if VAS residual decomposition (R1-R5) should be calculated.

        VAS residuals decompose total variation into components:
        - R1: Total deviation from grand mean
        - R2: Within-cell (unexplained) variation
        - R3: Interaction effects (factor × time)
        - R4: Time effects + unexplained
        - R5: Factor effects + unexplained

        **Calculate VAS when:**
        1. User requests Xbar or S chart (cell-level analysis)
        2. AND we have factorial structure (SDS 1-5)

        **Don't calculate VAS when:**
        1. User requests IMR or R chart (individual-level analysis)
        2. OR SDS 6 (incomplete without replication - very limited structure)

        Parameters
        ----------
        sds : int
            Detected SDS (1-6)
        analysis_type : str
            Analysis type ('Xbar', 'S', 'Imr', 'R')

        Returns
        -------
        bool
            True if VAS residuals should be calculated

        Examples
        --------
        >>> # Xbar with SDS 1 - needs VAS
        >>> detector.should_calculate_vas_residuals(sds=1, analysis_type='Xbar')
        True

        >>> # IMR with SDS 1 - doesn't need VAS (stratified charts)
        >>> detector.should_calculate_vas_residuals(sds=1, analysis_type='Imr')
        False
        """
        # SDS 6: Incomplete without replication - very limited analysis possible
        if sds == 6:
            logger.debug("No VAS: SDS 6 (incomplete without replication)")
            return False

        # IMR/R use moving ranges, not factorial decomposition
        # Grouping just creates separate stratified charts
        if analysis_type in ['Imr', 'R']:
            logger.debug(
                f"No VAS: {analysis_type} analysis uses moving ranges, "
                f"not factorial decomposition. "
                f"Grouping creates stratified charts (separate chart per group)."
            )
            return False

        # Xbar/S with factorial structure (SDS 1-5)
        if sds in [1, 2, 3, 4, 5]:
            logger.debug(
                f"Calculate VAS: SDS {sds} with {analysis_type} analysis "
                f"supports decomposition"
            )
            return True

        # Shouldn't reach here, but be conservative
        logger.debug(f"No VAS: Unexpected case (SDS={sds})")
        return False

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _normalize_missing_tokens(self, series: pd.Series) -> pd.Series:
        """
        Normalize missing value tokens to pd.NA without full type coercion.

        This only handles:
        - Standard missing tokens: *, NA, N/A, nan, null, None, empty string
        - Whitespace-only strings

        It does NOT:
        - Coerce "10.5" to 10.5 (stays as string)
        - Strip whitespace from non-missing values
        - Detect/convert numeric strings

        Use this for structure detection where we only need to identify NA responses,
        not coerce types (that happens later in prepare_dataset()).

        Parameters
        ----------
        series : pd.Series
            Series to normalize

        Returns
        -------
        pd.Series
            Series with missing tokens converted to pd.NA
        """
        # Convert known missing tokens to NA (case-insensitive)
        missing_tokens = {'*', 'na', 'n/a', 'nan', 'null', 'none'}

        def normalize(x):
            if pd.isna(x):
                return pd.NA
            if isinstance(x, str):
                stripped = x.strip()
                if stripped == '':
                    return pd.NA
                if stripped.lower() in missing_tokens:
                    return pd.NA
            return x

        return series.apply(normalize)

    def _build_structure_view(
        self,
        df: pd.DataFrame,
        spec: DataPrepConfig,
        response_col: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Build minimal structure view for SDS detection.

        This creates a lightweight projection of the data with:
        - Only kt_cols + response_col
        - Canonicalized kt columns (same type conversion as prepare_dataset)
        - Normalized response column (missing tokens → NA)
        - Filtered rows where kt columns have NA (can't determine cell membership)
        - Response NA rows PRESERVED (reveals all-NA cells)

        Parameters
        ----------
        df : DataFrame
            Raw data (response NA rows NOT yet dropped)
        spec : DataPrepConfig
            Data preparation configuration
        response_col : str
            Name of response column

        Returns
        -------
        tuple of (structure_view, kt_cols)
            structure_view: Canonicalized, filtered to valid kt rows, response NA preserved
            kt_cols: List of column names for grouping
        """
        from .data_preparation import DataPreparation

        prep = DataPreparation()

        # Determine kt_cols
        k_vars = list(spec.rsg_vars) if spec.rsg_vars else []
        t = spec.time_var
        # Include time_var only if not already in rsg_vars (avoid duplicates)
        kt_cols = k_vars + [t] if (t and t not in k_vars) else k_vars

        # Project to needed columns only (deduplicate response_col if it's in kt_cols)
        cols_needed = list(dict.fromkeys(kt_cols + [response_col]))  # preserves order, removes dupes
        out = df[cols_needed].copy()

        # Canonicalize kt columns (same type conversion as prepare_dataset)
        for col in kt_cols:
            out[col], _ = prep._detect_and_convert_type(out[col], col)

        # Normalize response column for consistent NA detection
        # NOTE: This only normalizes missing tokens (*, NA, etc.) to pd.NA
        # It does NOT do full type coercion (e.g., "10.5" stays "10.5", not 10.5)
        # Full coercion happens later in prepare_dataset() for analysis
        out[response_col] = self._normalize_missing_tokens(out[response_col])

        # Filter rows with NA in any kt column (can't determine cell membership)
        # Skip if kt_cols is empty (no factors, no time)
        if kt_cols:
            out = out[out[kt_cols].notna().all(axis=1)]

        # NOTE: Response NA rows are PRESERVED - this reveals all-NA cells

        return out, kt_cols

    def _canonicalize_plan_values(
        self,
        plan: dict[str, list],
        kt_cols: list[str],
    ) -> dict[str, list]:
        """
        Canonicalize plan values using the same type conversion as data.

        This prevents false "missing cells" when plan has "1" but data became 1,
        or plan has "A " but data became "A".

        Parameters
        ----------
        plan : dict
            Sampling plan with {column: [levels]} structure
        kt_cols : list[str]
            List of kt column names to canonicalize

        Returns
        -------
        dict[str, list]
            Canonicalized plan with same types as data
        """
        from .data_preparation import DataPreparation

        prep = DataPreparation()
        canonicalized = {}

        for col in kt_cols:
            if col in plan:
                # Create a temporary Series and canonicalize it
                temp = pd.Series(plan[col])
                converted, _ = prep._detect_and_convert_type(temp, col)
                canonicalized[col] = converted.tolist()
            else:
                canonicalized[col] = []

        return canonicalized

    def _classify_by_nkt(
        self,
        nkt_counts: pd.Series,
        has_empty_cells: bool
    ) -> tuple[int, SDSReasonType]:
        """
        Classify SDS directly from N_kt distribution per Wheeler/Bishop Table 1.

        This is the core classification logic - pure and simple.

        Parameters
        ----------
        nkt_counts : pd.Series
            Observation counts for each (factor × time) cell that EXISTS in data
        has_empty_cells : bool
            Whether any expected cells are missing (N_kt = 0), determined by
            comparing to sampling plan

        Returns
        -------
        tuple[int, SDSReasonType]
            (sds, reason) - SDS number and reason string

        Notes
        -----
        Table 1 Classification:

        Complete/Semi-Complete (no empty cells):
        - SDS 1: Min N_kt ≥ 2 (all replicated)
        - SDS 2: Min = Max = 1 (all singletons)
        - SDS 3: Min = 1, Max ≥ 2 (mixed)

        Incomplete (has empty cells):
        - SDS 4: Has 1s AND has ≥2s (mixed with singletons)
        - SDS 5: NO 1s, has ≥2s (no singletons)
        - SDS 6: Max = 1 (no replication)
        """
        min_n = nkt_counts.min()
        max_n = nkt_counts.max()
        has_singletons = (nkt_counts == 1).any()

        n_cells = len(nkt_counts)
        cells_with_n1 = (nkt_counts == 1).sum()
        cells_with_n2_plus = (nkt_counts >= 2).sum()

        logger.debug(
            f"N_kt classification: {cells_with_n1} singletons, "
            f"{cells_with_n2_plus} replicated, has_empty_cells={has_empty_cells}"
        )

        if not has_empty_cells:
            # Complete or Semi-Complete (SDS 1, 2, 3)
            if min_n >= 2:
                logger.debug(f"SDS 1: Complete - all cells replicated (min={min_n})")
                return (1, "full_replication")
            elif max_n == 1:
                logger.debug("SDS 2: Semi-Complete - all cells singleton")
                return (2, "no_replication")
            else:
                pct_replicated = 100 * cells_with_n2_plus / n_cells
                logger.debug(
                    f"SDS 3: Semi-Complete - mixed ({pct_replicated:.0f}% replicated)"
                )
                return (3, "partial_replication")
        else:
            # Incomplete (SDS 4, 5, 6) - has empty cells
            if max_n == 1:
                logger.debug("SDS 6: Incomplete - no replication (max=1)")
                return (6, "incomplete_no_replication")
            elif has_singletons:
                logger.debug(
                    f"SDS 4: Incomplete with singletons "
                    f"({cells_with_n1} singletons, {cells_with_n2_plus} replicated)"
                )
                return (4, "incomplete_with_singletons")
            else:
                logger.debug(
                    f"SDS 5: Incomplete without singletons "
                    f"({cells_with_n2_plus} replicated cells)"
                )
                return (5, "incomplete_no_singletons")

    def _calculate_coverage_ratio(
        self,
        df: pd.DataFrame,
        spec: DataPrepConfig,
        plan: dict[str, list],
        T_planned: int | None = None
    ) -> float:
        """
        Calculate coverage ratio: observed cells / expected cells.

        When a plan is provided, we can determine how complete the observed
        data grid is compared to what was intended.

        Expected cells = K (factor combinations) × T (time points)
        Observed cells = Unique (factor, time) combinations in data

        Parameters
        ----------
        df : DataFrame
            The analysis dataset
        spec : DataPrepConfig
            Analysis specification
        plan : dict
            Sampling plan with {column: [levels]} structure.
        T_planned : int, optional
            Expected number of time points. If provided, used instead of
            observed time count for expected cell calculation.

        Returns
        -------
        float
            Ratio of observed cells to expected cells (0.0 to 1.0).
            Only counts observed cells that match planned factor combinations,
            so extra levels in data don't inflate coverage.

        Notes
        -----
        This implementation avoids cartesian product enumeration for scalability:
        - expected_K computed via math.prod (no enumeration)
        - Filtering uses per-column isin (not rsg string matching)
        """
        import math

        import numpy as np

        # Expected K without enumeration (scalable for large factor spaces)
        expected_K = math.prod(len(plan[var]) for var in spec.rsg_vars)

        # Filter to planned rows using per-column isin (no cartesian product)
        mask = np.ones(len(df), dtype=bool)
        for var in spec.rsg_vars:
            mask &= df[var].isin(plan[var]).to_numpy()
        df_planned = df.loc[mask]

        # Group by actual factor columns (not derived rsg string)
        # This is more explicit and avoids any encoding issues
        group_cols = list(spec.rsg_vars)
        if spec.has_time:
            group_cols.append(spec.time_var)
            observed_T = df[spec.time_var].nunique(dropna=True)
            expected_T = T_planned if T_planned is not None else observed_T
            expected_R = expected_K * expected_T
        else:
            expected_R = expected_K

        observed_R = df_planned.groupby(group_cols, dropna=True, observed=True).ngroups

        # Safe edge case handling
        if expected_R == 0:
            return 1.0 if observed_R == 0 else 0.0

        ratio = observed_R / expected_R
        logger.debug(f"Coverage: {observed_R}/{expected_R} = {ratio:.1%}")
        return ratio

    # =========================================================================
    # SDS Analysis Plan Methods - Validation & Diagnostic Tools
    # =========================================================================

    @staticmethod
    def get_analysis_plan(sds: int, min_cell_size: int = 0) -> SDSAnalysisPlan:
        """
        Get comprehensive analysis plan for a Sampling Design State.

        This is the authoritative specification of what the system will do
        when it encounters a particular SDS. Use this to:
        - Validate implementation against Bishop's methodology
        - Understand system capabilities and limitations
        - Generate documentation
        - Debug unexpected behavior

        Parameters
        ----------
        sds : int
            Sampling Design State (0-6)
        min_cell_size : int, optional
            Actual minimum cell size from the data. If provided, enables
            data-driven R2 chart selection (R2_S when min_cell_size >= 2).
            Default is 0.

        Returns
        -------
        SDSAnalysisPlan
            Complete specification of capabilities and limitations

        Raises
        ------
        ValueError
            If sds is not in range 0-6

        Examples
        --------
        >>> plan = SDSRegistry.get_analysis_plan(sds=1)
        >>> print(plan.name)
        'Full Factorial with Complete Replication'
        >>> print(plan.vas_residuals_supported)
        True
        >>> print(plan.valid_charts)
        ['Xbar', 'S', 'Imr']

        >>> # Check what your data structure supports
        >>> detector = SDSRegistry()
        >>> sds, min_n = detector.detect_sds(df, spec)
        >>> plan = detector.get_analysis_plan(sds, min_cell_size=min_n)
        >>> print(f"Your data supports: {', '.join(plan.valid_charts)}")
        """
        # Note: SDS 0 was consolidated into SDS 4. Response-only data (no factors,
        # no time) is now treated as SDS 4 with implicit time ordering via obs_id.
        # See detect_sds() for rationale.
        plans = {
            1: SDSAnalysisPlan(
                sds=1,
                name="Full Factorial with Complete Replication",
                description="All factor × time cells have n ≥ 2 observations (best case for analysis)",
                has_factors=True,
                has_time=True,
                has_replication='full',
                valid_charts=['Xbar', 'S', 'R', 'Imr', 'Histogram'],
                recommended_chart='Xbar',
                invalid_charts=[],
                vas_residuals_supported=True,
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='exact',
                main_effects_supported=True,
                interaction_effects_supported=True,
                supports_stratification=True,
                typical_use_cases=[
                    'Designed experiments with replication',
                    'Process capability studies',
                    'Multi-factor ANOVA-style analyses',
                    'Complete factorial designs'
                ],
                limitations=[],
                bishop_reference="Wheeler/Bishop Methodology: Complete Data (SDS 1)"
            ),

            2: SDSAnalysisPlan(
                sds=2,
                name="Full Factorial with No Replication",
                description="All factor × time cells have exactly n = 1 observation",
                has_factors=True,
                has_time=True,
                has_replication='none',
                valid_charts=['Xbar', 'S', 'Imr', 'R', 'Histogram'],
                recommended_chart='Xbar',
                invalid_charts=[],
                vas_residuals_supported=True,
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='moving_average',
                main_effects_supported=True,
                interaction_effects_supported=True,
                supports_stratification=True,
                typical_use_cases=[
                    'Production data (one measurement per factor/time combination)',
                    'Unreplicated factorial experiments',
                    'Historical data analysis',
                    'Screening experiments'
                ],
                limitations=[
                    'R2 estimated via moving average (approximate, not exact)',
                    'Interaction confounded with pure error'
                ],
                bishop_reference="Wheeler/Bishop Methodology: No Replication (SDS 2)"
            ),

            3: SDSAnalysisPlan(
                sds=3,
                name="Partial Replication",
                description="Mixed cells: some have n ≥ 2, others have n = 1",
                has_factors=True,
                has_time=True,
                has_replication='partial',
                valid_charts=['Xbar', 'S', 'R', 'Imr', 'Histogram'],
                recommended_chart='Xbar',
                invalid_charts=[],
                vas_residuals_supported=True,
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='hybrid',
                main_effects_supported=True,
                interaction_effects_supported=True,
                supports_stratification=True,
                typical_use_cases=[
                    'Unbalanced designs',
                    'Real-world data with missing observations',
                    'Opportunistic replication in some cells',
                    'Pilot studies with targeted replication'
                ],
                limitations=[
                    'R2 uses hybrid calculation (exact where possible, approximate elsewhere)',
                    'Variance estimates less precise than SDS 1',
                    'May have unequal subgroup sizes'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Partial Replication (SDS 3)"
            ),

            4: SDSAnalysisPlan(
                sds=4,
                name="Incomplete Grid with Singletons",
                description="Incomplete factor × time grid with mixed replication",
                has_factors=True,
                has_time=True,
                has_replication='partial',  # Mixed: some n=1, some n≥2
                valid_charts=['Xbar', 'S', 'R', 'Imr', 'Histogram'],
                recommended_chart='Xbar',
                invalid_charts=[],
                vas_residuals_supported=True,
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='hybrid',  # Exact where n≥2, MA where n=1
                main_effects_supported=True,
                interaction_effects_supported=False,  # Incomplete grid limits this
                supports_stratification=True,
                typical_use_cases=[
                    'Incomplete sampling with mixed replication',
                    'Production data with gaps and variable sample sizes',
                    'Opportunistic data collection',
                    'Partial factorial experiments'
                ],
                limitations=[
                    'Incomplete grid - some factor×time combinations missing',
                    'Mixed variance estimation (hybrid R2)',
                    'Cannot analyze all interactions due to missing cells'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Incomplete with Singletons (SDS 4)"
            ),

            5: SDSAnalysisPlan(
                sds=5,
                name="Incomplete Grid without Singletons",
                description="Incomplete factor × time grid with full replication in observed cells",
                has_factors=True,
                has_time=True,
                has_replication='full',  # All observed cells have n≥2
                valid_charts=['Xbar', 'S', 'R', 'Imr', 'Histogram'],
                recommended_chart='Xbar',
                invalid_charts=[],
                vas_residuals_supported=True,
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='exact',  # All observed cells are replicated
                main_effects_supported=True,
                interaction_effects_supported=False,  # Incomplete grid limits this
                supports_stratification=True,
                typical_use_cases=[
                    'Incomplete sampling with consistent replication',
                    'Designed experiments with missing runs',
                    'Production data with gaps but replicated measurements',
                    'Partial factorial with replication'
                ],
                limitations=[
                    'Incomplete grid - some factor×time combinations missing',
                    'Cannot analyze all interactions due to missing cells',
                    'Effect estimates may be biased by missing data pattern'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Incomplete without Singletons (SDS 5)"
            ),

            6: SDSAnalysisPlan(
                sds=6,
                name="Incomplete Grid Without Replication",
                description="Incomplete factor × time grid with no replication (all n=1)",
                has_factors=True,
                has_time=True,
                has_replication='none',  # Cannot estimate within-cell variance
                valid_charts=['Imr', 'R', 'Histogram'],
                recommended_chart='Imr',
                invalid_charts=['Xbar (no within-cell variance)', 'S (no within-cell variance)'],
                vas_residuals_supported=True,  # VAS works via moving average (like SDS 2)
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='moving_average',  # Approximate variance estimation
                main_effects_supported=False,  # Incomplete grid limits this
                interaction_effects_supported=False,
                supports_stratification=True,
                typical_use_cases=[
                    'Opportunistic data collection without replication',
                    'Sparse monitoring with single measurements',
                    'Incomplete sampling plans',
                    'Ad-hoc measurements with irregular sampling'
                ],
                limitations=[
                    'Cannot estimate within-cell variance directly (no replication)',
                    'R2 estimated via moving average (approximate)',
                    'Cannot analyze interactions',
                    'More limited than complete grid designs'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Irregular/No Replication (SDS 6)"
            ),
        }

        if sds not in plans:
            raise ValueError(
                f"Invalid SDS: {sds}. Must be 1-6. "
                f"(Note: SDS 0 was consolidated into SDS 4) "
                f"Available: {list(plans.keys())}"
            )

        plan = plans[sds]
        # Set min_cell_size from actual data
        return replace(plan, min_cell_size=min_cell_size)

    @staticmethod
    def print_all_analysis_plans() -> None:
        """
        Print comprehensive analysis plans for all SDS (1-6).

        This generates a complete reference guide showing what the system
        will do for each Sampling Design State. Useful for:
        - Validating implementation against Bishop's methodology
        - Documentation generation
        - Training and education
        - Understanding system capabilities

        Examples
        --------
        >>> SDSRegistry.print_all_analysis_plans()
        # Prints complete analysis plan for SDS 0-6
        """
        print("=" * 70)
        print("SAMPLING DESIGN STATE (SDS) ANALYSIS PLANS")
        print("Complete Specification of System Capabilities")
        print("=" * 70)
        print()

        for sds in range(1, 7):  # SDS 1-6 (SDS 0 was consolidated into SDS 4)
            plan = SDSRegistry.get_analysis_plan(sds)
            print(plan)
            print()
            print()

    @staticmethod
    def get_capability_matrix() -> pd.DataFrame:
        """
        Generate a comparison matrix of all SDS capabilities.

        Returns a DataFrame showing what each SDS supports, useful for
        quick reference and validation.

        Returns
        -------
        pd.DataFrame
            Matrix with SDS as index and capabilities as columns

        Examples
        --------
        >>> matrix = SDSRegistry.get_capability_matrix()
        >>> print(matrix)
        >>> matrix.to_excel('sds_capabilities.xlsx')
        """
        data = []
        for sds in range(1, 7):  # SDS 1-6 (SDS 0 was consolidated into SDS 4)
            plan = SDSRegistry.get_analysis_plan(sds)
            data.append({
                'SDS': sds,
                'Name': plan.name,
                'Factors': '✓' if plan.has_factors else '✗',
                'Time': '✓' if plan.has_time else '✗',
                'Replication': plan.has_replication,
                'Valid Charts': ', '.join(plan.valid_charts),
                'Recommended': plan.recommended_chart,
                'VAS': '✓' if plan.vas_residuals_supported else '✗',
                'R2 Method': plan.residual_calculation_method,
                'Main Effects': '✓' if plan.main_effects_supported else '✗',
                'Interactions': '✓' if plan.interaction_effects_supported else '✗',
                'Stratification': '✓' if plan.supports_stratification else '✗',
            })

        return pd.DataFrame(data).set_index('SDS')
