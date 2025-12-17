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
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

if TYPE_CHECKING:
    from .analysis_specification import AnalysisSpecification, DataPrepConfig

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


class SDSRegistry:
    """
    Registry of Sampling Design State (SDS 0-6) definitions and rules.

    Provides:
    - SDS detection from data structure
    - Analysis plans and capabilities for each SDS
    - Validation of SDS/analysis compatibility
    - VAS residual calculation rules

    The SDS classification system describes the structure of your data:

    **SDS 0**: No structure
        - No grouping or time variables
        - Limited analysis capabilities

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
        - One factor level, multiple time points
        - Time series structure
        - Appropriate for IMR charts

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
        plan: dict[str, list] | None = None
    ) -> tuple[int, int]:
        """
        Detect Sampling Design State from data structure.

        Examines the (factor × time) grid structure to determine which
        of the 7 SDS categories (0-6) best describes the data.

        Parameters
        ----------
        df : DataFrame
            Prepared analysis dataset
        spec : AnalysisSpecification
            Analysis specification
        plan : dict, optional
            Sampling plan specifying expected factor levels. Keys are column
            names, values are lists of expected levels. When provided, enables
            SDS 4-6 detection by comparing observed structure to planned
            structure (e.g., detecting missing factor levels).

            Mode 1 (plan=None): Infer structure from observed data → SDS 0-3
            Mode 2 (plan={...}): Compare to plan → enables SDS 4-6

            Time handling: The plan specifies factor levels only, not time.
            Observed unique time values are used as the "planned" time set.
            This detects missing factor combos within observed time blocks.

        Returns
        -------
        tuple[int, int]
            (sds, min_cell_size) - SDS number (0-6) and minimum cell size

        Examples
        --------
        >>> # Full replication: all cells have n≥2
        >>> df = pd.DataFrame({
        ...     'rsg': ['A', 'A', 'B', 'B'],
        ...     'time': [1, 1, 1, 1],
        ...     'y': [10, 11, 9, 10]
        ... })
        >>> sds, min_n = detector.detect_sds(df, spec)
        >>> sds
        1
        >>> min_n
        2

        >>> # With sampling plan (enables SDS 4-6 detection)
        >>> plan = {'Lane': [1, 2, 3, 4], 'Phase': [1, 2, 3]}
        >>> sds, min_n = detector.detect_sds(df, spec, plan=plan)
        """
        # Store plan for potential use in classification
        # (Currently used for future SDS 4-6 enhanced detection)
        self._plan = plan
        # SDS 0: No structure (no grouping factors defined)
        if not spec.has_grouping:
            logger.debug("SDS 0: No grouping factors defined")
            return (0, 0)

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
        if spec.has_time:
            nkt_counts = (
                df.groupby([spec.rsg_var_name, spec.time_var], dropna=False, observed=True)
                .size()
            )
            min_nkt = nkt_counts.min()
            max_nkt = nkt_counts.max()
        else:
            # No time variable - N_kt reduces to N_k
            nkt_counts = (
                df.groupby([spec.rsg_var_name], dropna=False, observed=True)
                .size()
            )
            min_nkt = nkt_counts.min()
            max_nkt = nkt_counts.max()

        # --- Analysis Subgrouping: Group by factor only for chart selection ---
        subgroup_sizes = (
            df.groupby([spec.rsg_var_name], dropna=False, observed=True)
            .size()
        )
        min_cell_size = subgroup_sizes.min()  # For R2_S vs R2_Imr decision

        n_groups = df[spec.rsg_var_name].nunique()

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
            # For nested design check, we still consider all factor combinations
            n_cells = len(cell_sizes)
            sds5 = self._check_nested_design(df, spec, n_cells, n_groups)
            if sds5 is not None:
                return (sds5, min_cell_size)

        # SDS 4: Single group (only one factor level)
        if n_groups == 1:
            logger.debug(f"SDS 4: Single group ({n_groups} factor level)")
            return (4, min_cell_size)

        # Calculate coverage_ratio based on plan (if provided)
        if plan is not None:
            coverage_ratio = self._calculate_coverage_ratio(df, spec, plan)
            logger.debug(f"Plan coverage ratio: {coverage_ratio:.2%}")
        else:
            coverage_ratio = 1.0  # No plan = assume observed is complete

        # SDS 1, 2, or 3: Based on N_kt replication pattern (Wheeler/Bishop)
        # Note: min_cell_size (factor-only) returned for chart selection
        sds = self._classify_by_replication(
            cell_sizes, min_n, max_n, coverage_ratio=coverage_ratio
        )
        return (sds, min_cell_size)

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
                'variance_decomposition': False
            },
            5: {
                'description': 'Nested design with asynchronous coverage',
                'replication_type': 'nested',
                'r2_method': 'nested_variance_components',
                'capabilities': ['variance_components', 'nested_effects', 'hierarchical_analysis'],
                'interaction_analysis': 'hierarchical',
                'variance_decomposition': 'hierarchical'
            },
            6: {
                'description': 'Unstructured/irregular grid',
                'replication_type': 'irregular',
                'r2_method': 'adaptive',
                'capabilities': ['regime_detection', 'adaptive_limits', 'sparse_analysis'],
                'interaction_analysis': False,
                'variance_decomposition': 'limited'
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
        >>> # SDS 0 with Xbar - fatal error
        >>> detector.validate_sds_for_analysis(sds=0, analysis_type='Xbar')
        Traceback (most recent call last):
            ...
        ValueError: Cannot perform Xbar analysis without grouping structure...

        >>> # SDS 2 with Xbar - warning but allowed
        >>> detector.validate_sds_for_analysis(sds=2, analysis_type='Xbar')
        # Logs warning about no replication
        True
        """
        # SDS 0: Very limited capabilities
        if sds == 0 and analysis_type in ['Xbar', 'S']:
            raise ValueError(
                f"Cannot perform {analysis_type} analysis without grouping structure.\n"
                f"Detected SDS 0 (no grouping or time variables).\n"
                f"Fix: Use 'Imr' analysis or specify grouping variables"
            )

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
            Detected SDS (0-6)
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
        spec: AnalysisSpecification,
        n_cells: int,
        full_grid_size: int
    ) -> int | None:
        """
        Check if data has nested design structure (SDS 5).

        In nested designs, one factor is nested within another.
        Example: heads nested within lanes (each head belongs to one lane only)

        Returns SDS 5 if nested and incomplete coverage, None otherwise.
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
                return 5
            else:
                logger.debug(
                    f"Nested structure detected but high coverage ({coverage_ratio:.1%}) - "
                    f"treating as crossed design"
                )

        return None

    def _classify_by_replication(
        self,
        cell_sizes: pd.Series,
        min_n: int,
        max_n: int,
        coverage_ratio: float
    ) -> int:
        """
        Classify as SDS 1, 2, 3, 5, or 6 based on cell replication and coverage.

        Incomplete grid detection (coverage < 95%):
        - With replication (any n≥2) → SDS 5
        - No replication (all n=1) → SDS 6

        Complete grid detection (coverage ≥ 95%):
        - All n≥2 → SDS 1 (full replication)
        - All n=1 → SDS 2 (no replication)
        - Mixed → SDS 3 (partial replication)
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
                return 5
            elif max_n == 1:
                # No replication + incomplete → SDS 6
                logger.debug(
                    f"SDS 6: Incomplete grid without replication "
                    f"({coverage_ratio:.1%} coverage, all n=1)"
                )
                return 6
            else:
                # Mixed replication + incomplete → SDS 5
                # Rationale: Can estimate variance from replicated cells
                logger.debug(
                    f"SDS 5: Incomplete grid with mixed replication "
                    f"({coverage_ratio:.1%} coverage, {cells_with_n2_plus}/{n_cells} cells with n≥2)"
                )
                return 5

        # SDS 1: Full replication
        if min_n >= 2:
            logger.debug(
                f"SDS 1: Full replication "
                f"(all cells have n≥2, range: [{min_n}, {max_n}])"
            )
            return 1

        # SDS 2: No replication (coverage already checked above)
        if max_n == 1:
            logger.debug(
                f"SDS 2: No replication (all cells have n=1)"
            )
            return 2

        # SDS 3: Partial replication
        if cells_with_n1 > 0 and cells_with_n2_plus > 0:
            pct_replicated = 100 * cells_with_n2_plus / n_cells
            logger.debug(
                f"SDS 3: Partial replication - "
                f"{cells_with_n2_plus}/{n_cells} cells replicated ({pct_replicated:.1f}%), "
                f"n range: [{min_n}, {max_n}]"
            )
            return 3

        # Fallback (shouldn't reach here)
        logger.warning(
            f"SDS Detection: Unexpected replication pattern - defaulting to SDS 0. "
            f"min_n={min_n}, max_n={max_n}"
        )
        return 0

    def _calculate_coverage_ratio(
        self,
        df: pd.DataFrame,
        spec: DataPrepConfig,
        plan: dict[str, list]
    ) -> float:
        """
        Calculate coverage ratio: observed cells / expected cells.

        When a plan is provided, we can determine how complete the observed
        data grid is compared to what was intended.

        Expected cells = Cartesian product of all plan levels × time points (if any)
        Observed cells = Unique (factor, time) combinations in data

        Time Handling
        -------------
        The plan only specifies factor levels, not time levels. For time
        expectation, we use observed unique time values as the "planned" set.
        This means:
        - Coverage detects missing factor combos within observed time blocks
        - A factor combo missing at time=3 but present at time=1,2 counts as missing
        - Time points not in the data are not considered "missing"

        Example: If plan={'Lane': [1,2,3], 'Phase': [A,B]} and data has time=[1,2]:
        - Expected cells = 6 factor combos × 2 time points = 12
        - If Lane=3 is missing entirely: observed = 4 combos × 2 = 8, coverage = 67%

        Note: This method compares COUNTS for efficiency. For identifying
        specific missing/extra combinations (e.g., DesignReport.missing_combos),
        use encode_rsg() from data_preparation to generate expected keys:

            from processbehavior.data_preparation import encode_rsg
            expected_keys = {encode_rsg(combo, spec.rsg_var_delim)
                            for combo in product(*plan.values())}
            observed_keys = set(df[spec.rsg_var_name].unique())
            missing = expected_keys - observed_keys

        Parameters
        ----------
        df : DataFrame
            The analysis dataset
        spec : DataPrepConfig
            Analysis specification
        plan : dict
            Sampling plan with {column: [levels]} structure.
            Specifies factor levels only; time is inferred from observed data.

        Returns
        -------
        float
            Ratio of observed cells to expected cells (0.0 to 1.0+)
            Can exceed 1.0 if observed has extra levels not in plan.
        """
        from itertools import product

        # Get expected factor combinations from plan
        factor_levels = list(plan.values())
        expected_factor_combos = set(product(*factor_levels))

        if spec.has_time:
            # Time expectation: use observed unique time values as planned time set.
            # This detects missing factor combos within observed time blocks.
            # (Plan only specifies factor levels, not time levels.)
            time_values = df[spec.time_var].dropna().unique()
            expected_cells = len(expected_factor_combos) * len(time_values)

            # Observed cells = unique (rsg, time) combos
            observed_cells = df.groupby(
                [spec.rsg_var_name, spec.time_var], dropna=False, observed=True
            ).ngroups
        else:
            # No time: expected = factor combos, observed = unique rsg values
            expected_cells = len(expected_factor_combos)
            observed_cells = df[spec.rsg_var_name].nunique()

        if expected_cells == 0:
            return 1.0  # Avoid division by zero

        ratio = observed_cells / expected_cells
        logger.debug(
            f"Coverage: {observed_cells} observed / {expected_cells} expected = {ratio:.1%}"
        )
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
        plans = {
            0: SDSAnalysisPlan(
                sds=0,
                name="Simple Series",
                description="Individual measurements with no rational subgrouping or time structure",
                has_factors=False,
                has_time=False,
                has_replication='none',
                valid_charts=['Imr', 'R'],
                recommended_chart='Imr',
                invalid_charts=['Xbar (requires rational subgroups)', 'S (requires rational subgroups)'],
                vas_residuals_supported=False,
                residuals_available=[],
                residual_calculation_method='none',
                main_effects_supported=False,
                interaction_effects_supported=False,
                supports_stratification=False,
                typical_use_cases=[
                    'Simple process monitoring (temperature, pH, daily output)',
                    'Individual measurements over time',
                    'Quality characteristic tracking with no grouping'
                ],
                limitations=[
                    'Cannot decompose variance (no factors or time structure)',
                    'Cannot detect interaction effects',
                    'Limited to individuals control chart (IMR)',
                    'No rational subgrouping available'
                ],
                bishop_reference="Wheeler 'Understanding Variation' Chapter 3: Individuals Charts"
            ),

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
                name="Factors Only (No Time)",
                description="Grouping factors present but no time variable",
                has_factors=True,
                has_time=False,
                has_replication='full',
                valid_charts=['Xbar', 'S', 'R', 'Imr'],
                recommended_chart='Xbar',
                invalid_charts=[],
                vas_residuals_supported=True,
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='hybrid',
                main_effects_supported=True,
                interaction_effects_supported=False,
                supports_stratification=True,
                typical_use_cases=[
                    'Cross-sectional studies',
                    'Between-group comparisons',
                    'Baseline capability studies',
                    'Multi-stream process monitoring'
                ],
                limitations=[
                    'Cannot analyze time trends',
                    'Cannot detect factor × time interactions',
                    'Limited to factor main effects'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Factors Only (SDS 4)"
            ),

            5: SDSAnalysisPlan(
                sds=5,
                name="Time Only (No Factors)",
                description="Time variable present but no grouping factors",
                has_factors=False,
                has_time=True,
                has_replication='partial',
                valid_charts=['Xbar', 'S', 'R', 'Imr'],
                recommended_chart='Xbar',
                invalid_charts=[],
                vas_residuals_supported=True,
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='hybrid',
                main_effects_supported=False,
                interaction_effects_supported=False,
                supports_stratification=False,
                typical_use_cases=[
                    'Single process over time with subgroups',
                    'Repeated measurements at time points',
                    'Time series with natural grouping (hourly batches)',
                    'Rational subgrouping by time period only'
                ],
                limitations=[
                    'Cannot analyze factor effects',
                    'Cannot detect interactions',
                    'Limited to time-based grouping'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Time Only (SDS 5)"
            ),

            6: SDSAnalysisPlan(
                sds=6,
                name="Incomplete/Irregular Grid",
                description="Sparse factor × time grid with many missing cells",
                has_factors=True,
                has_time=True,
                has_replication='none',
                valid_charts=['Imr', 'R'],
                recommended_chart='Imr',
                invalid_charts=['Xbar (requires complete grid)', 'S (requires complete grid)'],
                vas_residuals_supported=True,
                residuals_available=['R1', 'R2', 'R3', 'R4', 'R5'],
                residual_calculation_method='moving_average',
                main_effects_supported=False,
                interaction_effects_supported=False,
                supports_stratification=True,
                typical_use_cases=[
                    'Opportunistic data collection',
                    'Real-world incomplete data',
                    'Sparse monitoring programs',
                    'Ad-hoc measurements with irregular sampling'
                ],
                limitations=[
                    'Cannot calculate reliable main effects',
                    'Cannot analyze interactions',
                    'Limited to stratified IMR charts per factor level',
                    'Most limited analytical capabilities'
                ],
                bishop_reference="Wheeler/Bishop Methodology: Irregular Data (SDS 6)"
            ),
        }

        if sds not in plans:
            raise ValueError(
                f"Invalid SDS: {sds}. Must be 0-6. "
                f"Available: {list(plans.keys())}"
            )

        plan = plans[sds]
        # Set min_cell_size from actual data
        return replace(plan, min_cell_size=min_cell_size)

    @staticmethod
    def print_all_analysis_plans() -> None:
        """
        Print comprehensive analysis plans for all SDS (0-6).

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

        for sds in range(7):
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
        for sds in range(7):
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
