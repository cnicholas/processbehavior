
from __future__ import annotations

import logging

import pandas as pd

from .analysis_specification import AnalysisSpecification
from .data_preparation import DataPreparation
from .effects_calculator import EffectsCalculator
from .residual_calculator import ResidualCalculator
from .sds_detector import SDSRegistry, StructureStats

# Configure module logger
logger = logging.getLogger(__name__)

class AnalysisDataSet:
    """
    Orchestrates statistical process control analysis using Wheeler/Bishop methodology.

    This class coordinates the workflow:
    1. Data preparation and validation
    2. VAS residual calculation (R1-R5) based on provided SDS
    3. Effects and interactions analysis

    SDS (Sampling Design State) is required and must be detected at the entry
    point (ProcessBehavior) before creating an AnalysisDataSet. This ensures
    SDS is detected exactly once per workflow.

    Uses composition pattern - delegates to focused classes for each concern.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        analysis_specification: AnalysisSpecification,
        sds: int
    ):
        """
        Initialize analysis with data and specification.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data
        analysis_specification : AnalysisSpecification
            Configuration for the analysis
        sds : int
            Sampling Design State (0-6). This is required because SDS is the
            driver of the analysis system - it determines which calculations
            are performed. SDS should be detected once at the entry point
            (ProcessBehavior) and passed through the system.
        """
        # Store inputs
        self.raw_dataset = df
        self.spec = analysis_specification
        self._sds = sds

        # Initialize output containers
        self.statistics = {}
        self.residuals = {}
        self.interactions = {}
        self.effects = {}
        self.Rbar = 0

        # Structure stats (computed once in _initialize)
        self._structure_stats: StructureStats | None = None
        self._n_per_cell: pd.Series | None = None

        # Composition - each component has one job (Single Responsibility Principle)
        self.prep = DataPreparation()
        self.sds_detector = SDSRegistry()
        self.residual_calc = ResidualCalculator()
        self.effects_calc = EffectsCalculator()

        # Run the analysis workflow
        self._initialize()

    def _initialize(self):
        """
        Execute the analysis workflow.

        Clear orchestration that reads like a recipe:
        1. Validate and prepare data
        2. Apply SDS (passed from entry point)
        3. Calculate VAS residuals (if appropriate for SDS)
        4. Calculate effects and interactions (if appropriate)
        """
        # Step 1: Validate and prepare data
        logger.debug("Preparing dataset")
        self.prep.validate_columns(self.raw_dataset, self.spec)
        self.analysis_dataset = self.prep.prepare_dataset(self.raw_dataset, self.spec)
        self.analysis_dataset = self.prep.build_keys(self.analysis_dataset, self.spec)

        # Step 2: Use the provided SDS (required - detected at entry point)
        logger.debug(f"Using SDS: {self._sds}")
        self.sampling_design_state = self._sds
        self.sds_characteristics = self.sds_detector.get_sds_characteristics(
            self.sampling_design_state
        )

        # Log analysis summary and SDS
        logger.debug(self.analysis_summary)
        logger.debug(
            f"Detected: SDS {self.sampling_design_state} - "
            f"{self.sds_characteristics['description']}"
        )

        # Step 4: Compute structure stats once (single source of truth)
        # This enables R2 method selection and availability checks
        self._compute_structure_stats()

        # Step 5: Calculate VAS residuals when we have both grouping AND time
        # (need Ybar_k for factor effects and Ybar_t for time effects)
        needs_residuals = self.spec.has_grouping and self.spec.has_time

        # Also calculate residuals if a residual chart is requested
        residual_requested = getattr(self.spec, 'residual', None) is not None
        if residual_requested:
            needs_residuals = True
            logger.debug("Residual chart requested - forcing VAS residual calculation")

        if needs_residuals:
            logger.debug("Calculating VAS residuals (R1-R5)")

            # Get R2 method based on observed structure (not SDS)
            r2_method = self.sds_detector.get_r2_method(self._structure_stats)
            logger.debug(f"Using R2 method: {r2_method}")

            self.analysis_dataset = self.residual_calc.calculate_vas_residuals(
                self.analysis_dataset, self.spec, r2_method,
                n_per_cell=self._n_per_cell
            )

            # Calculate centered residuals (legacy support)
            self._calculate_centered_residuals()

            # Step 6: Calculate effects and interactions
            logger.debug("Calculating effects and interactions")
            self.effects = self.effects_calc.calculate_all_effects(
                self.analysis_dataset, self.spec
            )
            self.interactions = self.effects_calc.calculate_interactions(
                self.analysis_dataset, self.spec, self.sampling_design_state
            )
        else:
            logger.debug(
                f"Skipping VAS residuals: requires both grouping and time. "
                f"has_grouping={self.spec.has_grouping}, has_time={self.spec.has_time}"
            )

    # =========================================================================
    # Properties (for backward compatibility and convenience)
    # =========================================================================

    @property
    def has_vas_residuals(self) -> bool:
        """Check if VAS residuals were calculated."""
        return 'R1' in self.analysis_dataset.columns

    @property
    def structure_stats(self) -> StructureStats | None:
        """Get structure statistics (computed once during initialization)."""
        return self._structure_stats

    @property
    def n_per_cell(self) -> pd.Series | None:
        """Get observation counts per cell_key (computed once during initialization)."""
        return self._n_per_cell

    def _compute_structure_stats(self) -> None:
        """
        Compute and cache structure statistics once.

        This is the single source of truth for:
        - n_per_cell: observation counts per cell_key
        - structure_stats: StructureStats with min/max cell sizes

        These drive R2 method selection and availability checks.
        """
        df = self.analysis_dataset
        y = self.spec.response_var

        # Compute n_per_cell once
        self._n_per_cell = df.groupby("cell_key", observed=True)[y].transform("size")

        # Handle edge case: empty data or all NaN
        if len(self._n_per_cell) == 0 or self._n_per_cell.isna().all():
            n_cell_min = 0
            n_cell_max = 0
        else:
            n_cell_min = int(self._n_per_cell.min())
            n_cell_max = int(self._n_per_cell.max())

        # Compute structure stats
        self._structure_stats = StructureStats(
            has_grouping=bool(self.spec.rsg_vars),
            has_order="obs_id" in df.columns,
            n_cell_min=n_cell_min,
            n_cell_max=n_cell_max,
            K_obs=df["rsg_key"].nunique() if "rsg_key" in df.columns else 0
        )

        logger.debug(
            f"Structure stats: n_cell_min={self._structure_stats.n_cell_min}, "
            f"n_cell_max={self._structure_stats.n_cell_max}, "
            f"K_obs={self._structure_stats.K_obs}"
        )

    @property
    def analysis_summary(self) -> dict:
        """
        Get comprehensive summary of the analysis dataset.

        Returns
        -------
        dict
            Dictionary with analysis metadata including:
            - sds: Detected sampling design state
            - sds_info: Full SDS characteristics
            - has_vas: Whether VAS residuals calculated
            - n_observations: Total observations
            - analysis_type: Type of analysis being performed
        """
        summary = {
            'sds': self.sampling_design_state,
            'sds_info': self.sds_characteristics,
            'has_vas_residuals': self.has_vas_residuals,
            'n_observations': len(self.analysis_dataset),
            'analysis_type': self.spec.analysis_type
        }
        return summary

    # =========================================================================
    # Centered Residuals (legacy support - kept for backward compatibility)
    # =========================================================================

    def _calculate_centered_residuals(self):
        """
        Calculate centered residuals (Rbar and RCR values).

        These are legacy calculations that center residuals by their means.
        Kept for backward compatibility with existing code and tests.

        Calculates:
        - Rbar_kt: Mean of R1 per cell (factor × time)
        - Rbar_k: Mean of R1 per factor level
        - Rbar_t: Mean of R1 per time point
        - RCR1-RCR5: Reconstructed Y values from centered residuals
        """
        if not self.spec.has_grouping:
            return

        df = self.analysis_dataset

        # Calculate centered residual means
        rsg_time_groups = df.groupby([self.spec.rsg_var_name, self.spec.time_var], observed=True)
        df["Rbar_kt"] = rsg_time_groups["R1"].transform("mean")
        df['Rbar_k'] = df.groupby([self.spec.rsg_var_name], observed=True)["R1"].transform('mean')
        df['Rbar_t'] = df.groupby([self.spec.time_var], observed=True)["R1"].transform("mean")

        # Calculate RCR (Reconstructed Centered Residuals)
        # These verify that Y can be reconstructed from components
        df['RCR1'] = df['Ybar'] + df['R1']  # Y = Ybar + R1
        df['RCR2'] = df['Ybar_kt'] + df['R2']  # Y = Ybar_kt + R2
        # Y = (Ybar_k + Ybar_t - Ybar) + R3
        df['RCR3'] = (df['Ybar_k'] + df['Ybar_t'] - df['Ybar']) + df['R3']
        # Y = (Ybar + Ybar_kt - Ybar_t) + R4
        df['RCR4'] = (df['Ybar'] + df['Ybar_kt'] - df['Ybar_t']) + df['R4']
        # Y = (Ybar + Ybar_kt - Ybar_k) + R5
        df['RCR5'] = (df['Ybar'] + df['Ybar_kt'] - df['Ybar_k']) + df['R5']

