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
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from analysis_dataset import AnalysisSpecification

logger = logging.getLogger(__name__)


class SamplingDesignDetector:
    """
    Detects and characterizes Sampling Design State (SDS 0-6).

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

    >>> detector = SamplingDesignDetector()
    >>> sds = detector.detect_sds(df, spec)
    >>> print(f"Detected SDS {sds}")
    Detected SDS 1

    Get detailed characteristics:

    >>> info = detector.get_sds_characteristics(sds)
    >>> print(info['description'])
    'Full replication (all cells n≥2)'
    >>> print(info['r2_method'])
    'within_cell'

    Validate SDS for analysis:

    >>> detector.validate_sds_for_analysis(sds=2, analysis_type='Xbar')
    # Logs warning about no replication
    """

    def detect_sds(
        self,
        df: pd.DataFrame,
        spec: 'AnalysisSpecification'
    ) -> int:
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

        Returns
        -------
        int
            SDS number (0-6)

        Examples
        --------
        >>> # Full replication: all cells have n≥2
        >>> df = pd.DataFrame({
        ...     'rsg': ['A', 'A', 'B', 'B'],
        ...     'time': [1, 1, 1, 1],
        ...     'y': [10, 11, 9, 10]
        ... })
        >>> sds = detector.detect_sds(df, spec)
        >>> sds
        1
        """
        # SDS 0: No structure
        if not (spec.has_grouping and spec.has_time):
            logger.info("SDS 0: No grouping or time structure")
            return 0

        # From here: have both grouping and time
        grouping_vars = [spec.rsg_var_name, spec.time_var]

        # Count observations per (k,t) cell
        cell_sizes = (
            df.groupby(grouping_vars, dropna=False)[spec.response_var]
            .count()
        )

        n_cells = len(cell_sizes)
        min_n = cell_sizes.min()
        max_n = cell_sizes.max()

        # Calculate grid dimensions
        n_groups = df[spec.rsg_var_name].nunique()
        n_times = df[spec.time_var].nunique()
        full_grid_size = n_groups * n_times

        logger.debug(
            f"SDS Detection: {n_groups} groups × {n_times} times "
            f"= {full_grid_size} possible cells"
        )
        logger.debug(
            f"SDS Detection: {n_cells} cells observed, "
            f"n range: [{min_n}, {max_n}]"
        )

        # Check for nested design (SDS 5)
        if len(spec.rsg_vars) >= 2:
            sds5 = self._check_nested_design(df, spec, n_cells, full_grid_size)
            if sds5 is not None:
                return sds5

        # SDS 4: Single condition over time
        if n_groups == 1 and n_times > 1:
            logger.info(f"SDS 4: Single condition over time ({n_times} time points)")
            return 4

        # SDS 6: Incomplete grid
        coverage_ratio = n_cells / full_grid_size
        if coverage_ratio < 0.75:
            logger.info(
                f"SDS 6: Unstructured/incomplete grid - "
                f"{n_cells}/{full_grid_size} cells present ({coverage_ratio:.1%})"
            )
            return 6

        # SDS 1, 2, or 3: Based on cell sizes
        return self._classify_by_replication(
            cell_sizes, min_n, max_n, coverage_ratio
        )

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
        if sds == 0:
            if analysis_type in ['Xbar', 'S']:
                raise ValueError(
                    f"Cannot perform {analysis_type} analysis without grouping structure.\n"
                    f"Detected SDS 0 (no grouping or time variables).\n"
                    f"Fix: Use 'Imr' analysis or specify grouping variables"
                )

        # SDS 2: No within-cell variance
        if sds == 2:
            if analysis_type in ['Xbar', 'S']:
                logger.warning(
                    f"SDS 2 detected: No replication (all cells n=1).\n"
                    f"{analysis_type} analysis will use moving average for variance estimation.\n"
                    f"Consider using 'Imr' analysis instead for better performance."
                )

        # SDS 4: Single stream
        if sds == 4:
            if analysis_type not in ['Imr', 'R']:
                logger.warning(
                    f"SDS 4 detected: Single condition over time.\n"
                    f"{analysis_type} analysis may not be appropriate.\n"
                    f"Consider using 'Imr' analysis."
                )

        # SDS 6: Irregular
        if sds == 6:
            logger.warning(
                f"SDS 6 detected: Unstructured/irregular grid.\n"
                f"Analysis results may be unreliable due to incomplete data coverage.\n"
                f"Check for missing data or irregular sampling patterns."
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
        spec: 'AnalysisSpecification',
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
            df.groupby(factor2)[factor1]
            .nunique()
        )

        is_nested = (nesting_check == 1).all()

        if is_nested:
            # Check for incomplete temporal coverage
            coverage_ratio = n_cells / full_grid_size

            if coverage_ratio < 0.90:  # Less than 90% coverage
                logger.info(
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
        Classify as SDS 1, 2, or 3 based on cell replication pattern.

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

        # SDS 1: Full replication
        if min_n >= 2:
            logger.info(
                f"SDS 1: Full replication "
                f"(all cells have n≥2, range: [{min_n}, {max_n}])"
            )
            return 1

        # SDS 2: No replication
        if max_n == 1:
            if coverage_ratio >= 0.95:  # Complete grid
                logger.info(
                    f"SDS 2: No replication "
                    f"(all cells have n=1, {coverage_ratio:.1%} complete)"
                )
                return 2
            else:
                # Incomplete grid with no replication → SDS 6
                logger.info(
                    f"SDS 6: Incomplete grid with no replication "
                    f"({coverage_ratio:.1%} coverage)"
                )
                return 6

        # SDS 3: Partial replication
        if cells_with_n1 > 0 and cells_with_n2_plus > 0:
            pct_replicated = 100 * cells_with_n2_plus / n_cells
            logger.info(
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
