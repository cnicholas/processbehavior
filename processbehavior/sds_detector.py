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

# Formal vocabulary for SDS classification reasons
SDSReasonType = Literal[
    "full_replication",
    "no_replication",
    "partial_replication",
    "single_condition",
    "implicit_single_condition",  # Response-only data: obs order = implicit time
    "nested",
    "incomplete_with_replication",
    "incomplete_no_replication",
]

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
        Sampling Design State (0-6)
    min_cell_size : int
        Minimum observations per cell (for chart selection)
    reason : SDSReasonType | None
        Why this SDS was detected. Useful for disambiguation when
        multiple conditions can lead to the same SDS (e.g., SDS 5
        can be "nested" or "incomplete_with_replication").

        Possible values (defined in SDSReasonType):
        - "full_replication": SDS 1
        - "no_replication": SDS 2
        - "partial_replication": SDS 3
        - "single_condition": SDS 4 (explicit single factor level)
        - "implicit_single_condition": SDS 4 (response-only, obs order = time)
        - "nested": SDS 5 (hierarchical factor structure)
        - "incomplete_with_replication": SDS 5 (incomplete grid, some n≥2)
        - "incomplete_no_replication": SDS 6
    """
    sds: int
    min_cell_size: int
    reason: SDSReasonType | None = None


class SDSRegistry:
    """
    Registry of Sampling Design State (SDS 1-6) definitions and rules.

    Provides:
    - SDS detection from data structure
    - Analysis plans and capabilities for each SDS
    - Validation of SDS/analysis compatibility
    - VAS residual calculation rules

    The SDS classification system describes the structure of your data:

    **SDS 1**: Full replication
        - Every (factor × time) cell has n ≥ 2
        - Best for variance estimation
        - Supports full interaction analysis

    **SDS 2**: No replication
        - Every (factor × time) cell has n = 1
        - Requires moving average for variance
        - Limited interaction analysis

    **SDS 3**: Partial replication
        - Mix of n=1 and n≥2 cells
        - Most common in practice
        - Requires hybrid variance estimation

    **SDS 4**: Single condition over time
        - One factor level (explicit or implicit), tracked over time
        - Includes "response-only" data where observation order = implicit time
        - Time series structure, appropriate for IMR charts

    **SDS 5**: Nested design
        - Hierarchical factor structure
        - Incomplete temporal coverage
        - Requires variance components

    **SDS 6**: Unstructured/irregular
        - Cannot form regular grid
        - May have regime changes
        - Complex structure

    Examples
    --------
    Detect SDS from prepared data:

    >>> registry = SDSRegistry()
    >>> sds = registry.detect_sds(df, spec)
    >>> print(f"Detected SDS {sds}")
    Detected SDS 1

    Get detailed characteristics:

    >>> info = registry.get_sds_characteristics(sds)
    >>> print(info['description'])
    'Full replication (all cells n≥2)'
    >>> print(info['r2_method'])
    'within_cell'

    Validate SDS for analysis:

    >>> registry.validate_sds_for_analysis(sds=2, analysis_type='Xbar')
    # Logs warning about no replication
    """

    def detect_sds(
        self,
        df: pd.DataFrame,
        spec: DataPrepConfig,
        plan: dict[str, list] | None = None,
        T_planned: int | None = None
    ) -> SDSResult:
        """
        Detect Sampling Design State from data structure.

        Examines the (factor × time) grid structure to determine which
        of the 6 SDS categories (1-6) best describes the data.

        Parameters
        ----------
        df : DataFrame
            Prepared analysis dataset
        spec : DataPrepConfig
            Data preparation configuration
        plan : dict, optional
            Sampling plan specifying expected factor levels. Keys are column
            names, values are lists of expected levels. When provided, enables
            SDS 4-6 detection by comparing observed structure to planned
            structure (e.g., detecting missing factor levels).

            Mode 1 (plan=None): Infer structure from observed data → SDS 1-4
            Mode 2 (plan={...}): Compare to plan → enables SDS 5-6
        T_planned : int, optional
            Expected number of time points from sampling plan. When provided,
            coverage calculation uses this instead of observed time count.
            This enables detection of incomplete temporal coverage.

        Notes
        -----
        **SDS 4 for Response-Only Data**

        When no grouping factors are specified, the data is classified as
        SDS 4 (Single Condition Over Time) with implicit single condition.
        This is intentional: observation order (``obs_id``) provides implicit
        temporal structure, and Wheeler's IMR chart assumes temporal ordering.

        Returns
        -------
        SDSResult
            Result containing sds (1-6), min_cell_size, and reason.

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
        # Store plan for coverage ratio calculation (enables SDS 5/6 detection)
        self._plan = plan

        # No grouping factors defined → treat as SDS 4 (single condition over time)
        # Rationale: observation order provides implicit temporal structure.
        # Wheeler's IMR chart assumes temporal ordering - moving ranges between
        # consecutive observations only make sense in sequence. Even "response-only"
        # data is analyzed as a time series where obs_id serves as implicit time.
        # See: formulate() docstring for full explanation.
        if not spec.has_grouping:
            logger.debug("SDS 4: No grouping factors - implicit single condition over time")
            return SDSResult(sds=4, min_cell_size=1, reason="implicit_single_condition")

        # From here: have grouping factors
        #
        # DESIGN DECISION (Issue #60):
        # SDS classification and analysis subgrouping serve different purposes:
        #
        # 1. SDS CLASSIFICATION (Wheeler/Bishop methodology):
        #    - Based on N_kt = observations per (factor k × time t) cell
        #    - Describes the DATA STRUCTURE
        #    - SDS 1: All N_kt >= 2 (Complete)
        #    - SDS 2: All N_kt = 1 (Semi-Complete)
        #    - SDS 3: Mixed N_kt (Semi-Complete)
        #
        # 2. ANALYSIS SUBGROUPING (practical charts):
        #    - Based on factor-only grouping
        #    - Determines rational subgroups for Xbar-S charts
        #    - Enables stratified Imr/R charts per subgroup
        #    - min_cell_size used for chart selection (R2_S vs R2_Imr)

        # --- SDS Classification: Group by (factor × time) for N_kt ---
        # Use actual factor columns (rsg_vars) rather than derived rsg string
        factor_cols = list(spec.rsg_vars)
        if spec.has_time:
            nkt_counts = (
                df.groupby(factor_cols + [spec.time_var], dropna=False, observed=True)
                .size()
            )
            min_nkt = nkt_counts.min()
            max_nkt = nkt_counts.max()
        else:
            # No time variable - N_kt reduces to N_k
            nkt_counts = (
                df.groupby(factor_cols, dropna=False, observed=True)
                .size()
            )
            min_nkt = nkt_counts.min()
            max_nkt = nkt_counts.max()

        # --- Analysis Subgrouping: Group by factor only for chart selection ---
        subgroup_sizes = (
            df.groupby(factor_cols, dropna=False, observed=True)
            .size()
        )
        min_cell_size = subgroup_sizes.min()  # For R2_S vs R2_Imr decision

        n_groups = len(subgroup_sizes)  # Number of unique factor combinations

        logger.debug(
            f"SDS Detection: N_kt range [{min_nkt}, {max_nkt}], "
            f"subgroup size range [{min_cell_size}, {subgroup_sizes.max()}], "
            f"{n_groups} factor groups"
        )

        # Use N_kt values for SDS classification
        min_n = min_nkt
        max_n = max_nkt
        cell_sizes = nkt_counts  # For downstream classification logic

        # Check for nested design (SDS 5) - only if multiple factor variables
        if len(spec.rsg_vars) >= 2:
            # For nested design check, compute proper full grid size
            n_cells = len(cell_sizes)
            if spec.has_time:
                T_obs = df[spec.time_var].nunique()
                full_grid_size = n_groups * T_obs
            else:
                full_grid_size = n_groups
            is_nested = self._check_nested_design(df, spec, n_cells, full_grid_size)
            if is_nested:
                return SDSResult(sds=5, min_cell_size=min_cell_size, reason="nested")

        # SDS 4: Single group (only one factor level)
        if n_groups == 1:
            logger.debug(f"SDS 4: Single group ({n_groups} factor level)")
            return SDSResult(sds=4, min_cell_size=min_cell_size, reason="single_condition")

        # Calculate coverage_ratio based on plan (if provided)
        if plan is not None:
            coverage_ratio = self._calculate_coverage_ratio(df, spec, plan, T_planned)
            logger.debug(f"Plan coverage ratio: {coverage_ratio:.2%}")
        else:
            coverage_ratio = 1.0  # No plan = assume observed is complete

        # SDS 1, 2, 3, 5, or 6: Based on N_kt replication pattern (Wheeler/Bishop)
        # Note: min_cell_size (factor-only) returned for chart selection
        sds, reason = self._classify_by_replication(
            cell_sizes, min_n, max_n, coverage_ratio=coverage_ratio
        )
        return SDSResult(sds=sds, min_cell_size=min_cell_size, reason=reason)

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
                'description': 'Single condition over time (K=1)',
                'replication_type': 'single_stream',
                'r2_method': 'moving_range',
                'capabilities': ['time_series', 'imr_chart', 'trend_analysis'],
                'interaction_analysis': False,
                'variance_decomposition': True  # VAS supported via time-based analysis
            },
            5: {
                'description': 'Nested design with asynchronous coverage',
                'replication_type': 'nested',
                'r2_method': 'nested_variance_components',
                'capabilities': ['variance_components', 'nested_effects', 'hierarchical_analysis'],
                'interaction_analysis': 'hierarchical',
                'variance_decomposition': True  # VAS supported (hierarchical)
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
            Detected SDS (0-6)
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
        >>> # SDS 4 with Xbar - warning (should use Imr)
        >>> detector.validate_sds_for_analysis(sds=4, analysis_type='Xbar')
        # Logs warning recommending Imr for single condition
        True

        >>> # SDS 2 with Xbar - warning but allowed
        >>> detector.validate_sds_for_analysis(sds=2, analysis_type='Xbar')
        # Logs warning about no replication
        True
        """

        # SDS 2: No within-cell variance
        if sds == 2 and analysis_type in ['Xbar', 'S']:
            logger.warning(
                f"SDS 2 detected: No replication (all cells n=1).\n"
                f"{analysis_type} analysis will use moving average for variance estimation.\n"
                f"Consider using 'Imr' analysis instead for better performance."
            )

        # SDS 4: Single stream
        if sds == 4 and analysis_type not in ['Imr', 'R']:
            logger.warning(
                f"SDS 4 detected: Single condition over time.\n"
                f"{analysis_type} analysis may not be appropriate.\n"
                f"Consider using 'Imr' analysis."
            )

        # SDS 6: Irregular
        if sds == 6:
            logger.warning(
                "SDS 6 detected: Unstructured/irregular grid.\n"
                "Analysis results may be unreliable due to incomplete data coverage.\n"
                "Check for missing data or irregular sampling patterns."
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
        2. AND we have proper (k,t) factorial structure (SDS 1, 2, 3)

        **Don't calculate VAS when:**
        1. User requests IMR or R chart (individual-level analysis)
        2. OR no proper structure (SDS 0, 4, 6)

        **Key Insight:**
        For Xbar-S: Grouping defines CELLS for variance decomposition
        For IMR/R: Grouping defines STRATA for separate charts (stratification)

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
        # Quick rejections - clearly don't need VAS
        if sds in [0, 4, 6]:
            logger.debug(f"No VAS: SDS {sds} (no proper factorial structure)")
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

        # Xbar/S with proper structure (SDS 1, 2, 3)
        if sds in [1, 2, 3]:
            logger.debug(
                f"Calculate VAS: SDS {sds} with {analysis_type} analysis "
                f"supports full decomposition"
            )
            return True

        # SDS 5 (nested) - special case
        if sds == 5:
            logger.warning(
                "SDS 5 (nested design) detected with Xbar-S analysis.\n"
                "VAS decomposition for nested structures requires special handling.\n"
                "Proceeding with standard VAS - results may need interpretation."
            )
            return True

        # Shouldn't reach here, but be conservative
        logger.debug(f"No VAS: Unexpected case (SDS={sds})")
        return False

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _check_nested_design(
        self,
        df: pd.DataFrame,
        spec: DataPrepConfig,
        n_cells: int,
        full_grid_size: int
    ) -> bool:
        """
        Check if data has nested design structure (SDS 5).

        In nested designs, one factor is nested within another.
        Example: heads nested within lanes (each head belongs to one lane only)

        Returns True if nested with incomplete coverage, False otherwise.
        """
        factor1 = spec.rsg_vars[0]
        factor2 = spec.rsg_vars[1]

        # Count how many levels of factor1 each level of factor2 appears with
        nesting_check = (
            df.groupby(factor2, observed=True)[factor1]
            .nunique()
        )

        is_nested = (nesting_check == 1).all()

        if is_nested:
            # Check for incomplete temporal coverage
            coverage_ratio = n_cells / full_grid_size

            if coverage_ratio < 0.90:  # Less than 90% coverage
                logger.debug(
                    f"SDS 5: Nested design detected - {factor2} nested in {factor1}, "
                    f"{coverage_ratio:.1%} grid coverage"
                )
                return True
            else:
                logger.debug(
                    f"Nested structure detected but high coverage ({coverage_ratio:.1%}) - "
                    f"treating as crossed design"
                )

        return False

    def _classify_by_replication(
        self,
        cell_sizes: pd.Series,
        min_n: int,
        max_n: int,
        coverage_ratio: float
    ) -> tuple[int, str]:
        """
        Classify as SDS 1, 2, 3, 5, or 6 based on cell replication and coverage.

        Incomplete grid detection (coverage < 95%):
        - With replication (any n≥2) → SDS 5
        - No replication (all n=1) → SDS 6

        Complete grid detection (coverage ≥ 95%):
        - All n≥2 → SDS 1 (full replication)
        - All n=1 → SDS 2 (no replication)
        - Mixed → SDS 3 (partial replication)

        Returns
        -------
        tuple[int, str]
            (sds, reason) - SDS number and reason string
        """
        n_cells = len(cell_sizes)
        cells_with_n1 = (cell_sizes == 1).sum()
        cells_with_n2_plus = (cell_sizes >= 2).sum()

        logger.debug(
            f"SDS Detection: {cells_with_n1} cells with n=1, "
            f"{cells_with_n2_plus} cells with n≥2"
        )

        # Incomplete grid detection (coverage < 95%)
        # Key insight: SDS 5 vs 6 is about variance estimation capability
        if coverage_ratio < 0.95:
            if min_n >= 2:
                # Full replication + incomplete → SDS 5
                logger.debug(
                    f"SDS 5: Incomplete grid with full replication "
                    f"({coverage_ratio:.1%} coverage, all cells n≥2)"
                )
                return (5, "incomplete_with_replication")
            elif max_n == 1:
                # No replication + incomplete → SDS 6
                logger.debug(
                    f"SDS 6: Incomplete grid without replication "
                    f"({coverage_ratio:.1%} coverage, all n=1)"
                )
                return (6, "incomplete_no_replication")
            else:
                # Mixed replication + incomplete → SDS 5
                # Rationale: Can estimate variance from replicated cells
                logger.debug(
                    f"SDS 5: Incomplete grid with mixed replication "
                    f"({coverage_ratio:.1%} coverage, {cells_with_n2_plus}/{n_cells} cells with n≥2)"
                )
                return (5, "incomplete_with_replication")

        # SDS 1: Full replication
        if min_n >= 2:
            logger.debug(
                f"SDS 1: Full replication "
                f"(all cells have n≥2, range: [{min_n}, {max_n}])"
            )
            return (1, "full_replication")

        # SDS 2: No replication (coverage already checked above)
        if max_n == 1:
            logger.debug(
                "SDS 2: No replication (all cells have n=1)"
            )
            return (2, "no_replication")

        # SDS 3: Partial replication
        if cells_with_n1 > 0 and cells_with_n2_plus > 0:
            pct_replicated = 100 * cells_with_n2_plus / n_cells
            logger.debug(
                f"SDS 3: Partial replication - "
                f"{cells_with_n2_plus}/{n_cells} cells replicated ({pct_replicated:.1f}%), "
                f"n range: [{min_n}, {max_n}]"
            )
            return (3, "partial_replication")

        # Fallback (shouldn't reach here - all replication patterns should be covered above)
        logger.warning(
            f"SDS Detection: Unexpected replication pattern - defaulting to SDS 3 (partial). "
            f"min_n={min_n}, max_n={max_n}"
        )
        return (3, "partial_replication")

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
                valid_charts=['Xbar', 'S', 'R', 'Imr'],
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
                valid_charts=['Xbar', 'Imr', 'R'],
                recommended_chart='Xbar',
                invalid_charts=['S (requires n≥2 per subgroup)'],
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
                    'Cannot use S or R charts (require n≥2)',
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
                valid_charts=['Xbar', 'S', 'R', 'Imr'],
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
                name="Single Condition Over Time",
                description="Single factor level (K=1) tracked over time",
                has_factors=True,  # One factor level
                has_time=True,     # Time series structure
                has_replication='varies',  # Depends on subgroup size
                valid_charts=['Imr', 'Xbar', 'S', 'R'],
                recommended_chart='Imr',
                invalid_charts=[],
                vas_residuals_supported=True,  # VAS works via time-based analysis
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='varies',  # 'standard' if n≥2, 'moving_range' if n=1
                main_effects_supported=False,  # Only one factor level
                interaction_effects_supported=False,
                supports_stratification=False,  # Single stream
                typical_use_cases=[
                    'Single process stream over time',
                    'One machine/operator monitored continuously',
                    'Baseline process monitoring',
                    'Individual measurements over time (IMR)'
                ],
                limitations=[
                    'Cannot compare factor levels (only one exists)',
                    'Cannot analyze factor effects',
                    'Limited to time-based trend analysis'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Single Condition (SDS 4)"
            ),

            5: SDSAnalysisPlan(
                sds=5,
                name="Nested/Incomplete with Replication",
                description="Nested factor structure OR incomplete grid with replication",
                has_factors=True,
                has_time=True,
                has_replication='partial',  # Has some n≥2 cells (can estimate variance)
                valid_charts=['Xbar', 'S', 'R', 'Imr'],
                recommended_chart='Xbar',
                invalid_charts=[],
                vas_residuals_supported=True,
                residuals_available=['R2_S', 'R3_Xbar', 'R3_S', 'R4_Xbar', 'R4_S', 'R5_Xbar', 'R5_S'],
                residual_calculation_method='hybrid',
                main_effects_supported=True,
                interaction_effects_supported=False,  # Incomplete grid limits this
                supports_stratification=True,
                typical_use_cases=[
                    'Hierarchical/nested factor structures',
                    'Incomplete sampling with some replication',
                    'Heads nested within lanes',
                    'Operators nested within shifts'
                ],
                limitations=[
                    'Incomplete grid may limit effect estimation',
                    'Cannot analyze all interactions',
                    'Variance components may be confounded'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Nested/Incomplete (SDS 5)"
            ),

            6: SDSAnalysisPlan(
                sds=6,
                name="Incomplete Grid Without Replication",
                description="Incomplete factor × time grid with no replication (all n=1)",
                has_factors=True,
                has_time=True,
                has_replication='none',  # Cannot estimate within-cell variance
                valid_charts=['Imr', 'R'],
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
