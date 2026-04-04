
from __future__ import annotations

import logging

import pandas as pd

from .data_preparation import DataPreparation
from .effects_calculator import EffectsCalculator
from .formulation_spec import FormulationSpec
from .residual_calculator import calculate_vas_residuals
from .sds_detector import SDSRegistry, SDSResult, StructureStats

# Configure module logger
logger = logging.getLogger(__name__)

class AnalysisDataSet:
    """
    Orchestrates statistical process control analysis using Bishop's VAS methodology.

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
        spec: FormulationSpec,
        observed_sds: int
    ):
        """
        Initialize analysis with data and specification.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data
        spec : FormulationSpec
            Structural configuration for the analysis
        observed_sds : int
            Observed Design State detected on raw data (ODS). This is the
            SDS detected at the entry point (ProcessBehavior) on raw data
            before NA filtering. Used for diagnostic logging.
            The Analytical Design State (ADS) is computed internally on
            tidy data and drives analysis decisions.
        """
        # Store inputs
        self.spec = spec
        self.observed_design_state: int = observed_sds

        # Initialize output containers
        self.interactions = {}
        self.effects = {}

        # Structure stats (computed once in _initialize)
        self._structure_stats: StructureStats | None = None
        self._n_per_cell: pd.Series | None = None
        self._ybar_kt: pd.Series | None = None

        # ADS result (computed in _initialize on tidy data)
        self._ads_result: SDSResult | None = None

        # Composition - each component has one job (Single Responsibility Principle)
        # SDS detection was done on raw data by ProcessBehavior. Here we only use
        # SDSRegistry for characteristic lookups and R2 method selection on tidy
        # structure. When the tidy-state concept is formalized, its classification
        # would live here alongside SDSRegistry.
        self._prep = DataPreparation()
        self._sds_registry = SDSRegistry()
        self._effects_calc = EffectsCalculator()

        # Run the analysis workflow
        self._initialize(df)

    def _initialize(self, raw_df: pd.DataFrame):
        """
        Execute the analysis workflow.

        Clear orchestration that reads like a recipe:
        1. Validate and prepare data
        2. Apply SDS (passed from entry point)
        3. Compute structure stats on tidy data
        4. Calculate VAS residuals (if appropriate for SDS)
        5. Calculate effects and interactions (if appropriate)
        """
        # Step 1: Validate and prepare data
        logger.debug("Preparing dataset")
        self._prep.validate_columns(raw_df, self.spec)
        self.analysis_dataset = self._prep.prepare_dataset(raw_df, self.spec)
        self.analysis_dataset = self._prep.build_keys(self.analysis_dataset, self.spec)

        # Step 2: Use the provided ODS (detected at entry point on raw data)
        logger.debug(f"Using ODS: {self.observed_design_state}")
        self.raw_sds_characteristics = self._sds_registry.get_sds_characteristics(
            self.observed_design_state
        )
        logger.debug(
            f"ODS {self.observed_design_state} - "
            f"{self.raw_sds_characteristics['description']}"
        )

        # Step 3: Compute structure stats once on tidy data (single source of truth)
        # This enables R2 method selection and availability checks.
        # The tidy structure may differ from the raw SDS — e.g., SDS 3 (partial
        # replication) on raw data can become structurally equivalent to SDS 1
        # (full replication) after NA rows are dropped.
        self._compute_structure_stats()

        # Step 3b: Compute Analytical Design State (ADS) on tidy data
        self._compute_ads()

        # Step 4: Calculate VAS residuals when we have both grouping AND time
        # (need Ybar_k for factor effects and Ybar_t for time effects)
        needs_residuals = self.spec.has_grouping and self.spec.has_time

        if needs_residuals:
            logger.debug("Calculating VAS residuals (R1-R5)")

            # Get R2 method based on observed tidy structure (not raw SDS)
            r2_method = self._sds_registry.get_r2_method(self._structure_stats)
            logger.debug(f"Using R2 method: {r2_method}")

            self.analysis_dataset = calculate_vas_residuals(
                self.analysis_dataset, self.spec, r2_method,
                n_per_cell=self._n_per_cell,
                ybar_kt=self._ybar_kt,
            )

            # Calculate centered residuals
            self._calculate_centered_residuals()

            # Step 5: Calculate effects and interactions
            # ADS drives method selection (not ODS)
            logger.debug("Calculating effects and interactions")
            self.effects = self._effects_calc.calculate_all_effects(
                self.analysis_dataset, self.spec
            )
            self.interactions = self._effects_calc.calculate_interactions(
                self.analysis_dataset, self.spec, self._ads_result.sds,
                effects=self.effects
            )
        else:
            logger.debug(
                f"Skipping VAS residuals: requires both grouping and time. "
                f"has_grouping={self.spec.has_grouping}, has_time={self.spec.has_time}"
            )

        # Log final analysis summary (after all calculations are complete)
        logger.debug(self.analysis_summary)

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

    @property
    def tidy_structure_summary(self) -> dict | None:
        """
        Summary of the tidy data structure after NA drops and preparation.

        The tidy structure drives R2 method selection and may differ from
        what the raw SDS (detected on uncleansed data) would predict.
        For example, SDS 3 (partial replication) on raw data can become
        structurally equivalent to full replication after tidying.

        Returns
        -------
        dict or None
            Dictionary with tidy structure details:
            - n_cell_min: minimum observations per cell
            - n_cell_max: maximum observations per cell
            - K_obs: number of unique factor levels observed
            - r2_method: 'exact', 'ma2', or 'hybrid'
            Returns None if structure stats haven't been computed.
        """
        if self._structure_stats is None:
            return None
        return {
            'n_cell_min': self._structure_stats.n_cell_min,
            'n_cell_max': self._structure_stats.n_cell_max,
            'K_obs': self._structure_stats.K_obs,
            'r2_method': self._sds_registry.get_r2_method(self._structure_stats),
        }

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

        # Compute n_per_cell and cell means once
        self._n_per_cell = df.groupby("cell_key", observed=True)[y].transform("size")
        self._ybar_kt = df.groupby("cell_key", observed=True)[y].transform("mean")

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

        r2_method = self._sds_registry.get_r2_method(self._structure_stats)

        logger.debug(
            f"Tidy structure: n_cell_min={n_cell_min}, "
            f"n_cell_max={n_cell_max}, "
            f"K_obs={self._structure_stats.K_obs}, "
            f"effective R2 method: {r2_method}"
        )

    def _compute_ads(self) -> None:
        """
        Compute Analytical Design State (ADS) on tidy data.

        The ADS reflects the structure of data that is fit for analysis,
        after NA filtering and tidying. It may differ from the ODS when
        tidying removes empty or singleton cells.
        """
        df = self.analysis_dataset
        y = self.spec.response_var

        # Compute N_kt from tidy data grouped by cell_key
        if "cell_key" in df.columns and len(df) > 0:
            tidy_nkt = df.groupby("cell_key", observed=True)[y].size()
        else:
            tidy_nkt = pd.Series(dtype=int)

        # Guard: empty tidy data → ADS 0
        if len(tidy_nkt) == 0:
            self._ads_result = SDSResult(sds=0, reason=None, min_cell_size=0, n_empty_cells=0)
            logger.debug("ADS 0: no valid observations after tidying")
            return

        has_empty_cells = (tidy_nkt == 0).any()
        n_empty = int((tidy_nkt == 0).sum())
        valid_nkt = tidy_nkt[tidy_nkt > 0]
        min_cell_size = int(valid_nkt.min()) if len(valid_nkt) > 0 else 0

        sds, reason = self._sds_registry._classify_by_nkt(tidy_nkt, has_empty_cells)
        self._ads_result = SDSResult(
            sds=sds, min_cell_size=min_cell_size,
            reason=reason, n_empty_cells=n_empty
        )

        if self._ads_result.sds != self.observed_design_state:
            logger.info(
                f"Design state drift: ODS {self.observed_design_state} → "
                f"ADS {self._ads_result.sds} ({self._ads_result.reason})"
            )
        else:
            logger.debug(f"ADS {self._ads_result.sds} ({self._ads_result.reason})")

    @property
    def analytical_design_state(self) -> SDSResult:
        """
        Analytical Design State (ADS) computed on tidy data.

        The ADS drives analysis decisions: valid charts, residual
        availability, R2 method, and interaction method selection.
        """
        return self._ads_result

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
            'observed_sds': self.observed_design_state,
            'analytical_sds': self._ads_result.sds if self._ads_result else self.observed_design_state,
            'sds_info': self.raw_sds_characteristics,
            'has_vas_residuals': self.has_vas_residuals,
            'n_observations': len(self.analysis_dataset),
        }
        return summary

    # =========================================================================
    # Centered Residuals
    # =========================================================================

    def _calculate_centered_residuals(self):
        """
        Calculate centered residuals (Rbar and RCR values).

        These calculations center residuals by their means and reconstruct
        Y from variance components to verify decomposition correctness.

        Note: This method intentionally mutates self.analysis_dataset in-place
        (adding columns directly to the DataFrame). This differs from
        calculate_vas_residuals() which returns a new DataFrame.
        The mutation is safe here because this is called once
        during __init__ as part of the analysis pipeline, and the columns
        added (Rbar_kt, Rbar_k, Rbar_t, RCR1-RCR5) are outputs of the
        analysis that become part of the dataset.

        Calculates:
        - Rbar_kt: Mean of R1 per cell (factor x time)
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
