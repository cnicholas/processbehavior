
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .analysis_specification import AnalysisSpecification
from .data_preparation import DataPreparation
from .effects_calculator import EffectsCalculator
from .residual_calculator import ResidualCalculator
from .sds_detector import SamplingDesignDetector

# Configure module logger
logger = logging.getLogger(__name__)

class AnalysisDataSet:
    """
    Orchestrates statistical process control analysis using Wheeler/Bishop methodology.

    This class coordinates the workflow:
    1. Data preparation and validation
    2. Sampling Design State (SDS) detection
    3. VAS residual calculation (R1-R5)
    4. Effects and interactions analysis
    5. Control chart frame building

    Uses composition pattern - delegates to focused classes for each concern.
    """

    def __init__(self, df: pd.DataFrame, analysis_specification: AnalysisSpecification):
        """
        Initialize analysis with data and specification.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data
        analysis_specification : AnalysisSpecification
            Configuration for the analysis
        """
        # Store inputs
        self.raw_dataset = df
        self.spec = analysis_specification

        # Initialize output containers (for backward compatibility)
        self.obs_df = None
        self.cell_df = None
        self.k_df = None
        self.t_df = None
        self.statistics = {}
        self.residuals = {}
        self.interactions = {}
        self.effects = {}
        self.Rbar = 0

        # Composition - each component has one job (Single Responsibility Principle)
        self.prep = DataPreparation()
        self.sds_detector = SamplingDesignDetector()
        self.residual_calc = ResidualCalculator()
        self.effects_calc = EffectsCalculator()

        # Run the analysis workflow
        self._initialize()

    def _initialize(self):
        """
        Execute the analysis workflow.

        Clear orchestration that reads like a recipe:
        1. Validate and prepare data
        2. Build frames for charting
        3. Detect sampling design state
        4. Calculate VAS residuals (if appropriate)
        5. Calculate effects and interactions (if appropriate)
        """
        # Step 1: Validate and prepare data
        logger.info("Preparing dataset")
        self.prep.validate_columns(self.raw_dataset, self.spec)
        self.analysis_dataset = self.prep.prepare_dataset(self.raw_dataset, self.spec)
        self.analysis_dataset = self.prep.build_keys(self.analysis_dataset, self.spec)

        # Step 2: Build frames for charting
        self._build_frames()

        # Step 3: Detect SDS
        logger.info("Detecting sampling design state")
        self.sampling_design_state = self.sds_detector.detect_sds(
            self.analysis_dataset, self.spec
        )
        self.sds_characteristics = self.sds_detector.get_sds_characteristics(
            self.sampling_design_state
        )

        # Log analysis summary and SDS
        logger.info(self.analysis_summary)
        logger.info(
            f"Detected: SDS {self.sampling_design_state} - "
            f"{self.sds_characteristics['description']}"
        )

        # Step 4: Validate compatibility (fail fast if incompatible)
        self.sds_detector.validate_sds_for_analysis(
            self.sampling_design_state, self.spec.analysis_type
        )

        # Step 5: Calculate VAS residuals only when appropriate
        # Force VAS calculation if residual charting is requested
        needs_residuals = self.sds_detector.should_calculate_vas_residuals(
            self.sampling_design_state, self.spec.analysis_type
        )

        # Also calculate residuals if a residual chart is requested
        residual_requested = getattr(self.spec, 'residual', None) is not None
        if residual_requested:
            needs_residuals = True
            logger.info("Residual chart requested - forcing VAS residual calculation")

        if needs_residuals:
            logger.info("Calculating VAS residuals (R1-R5)")
            self.analysis_dataset = self.residual_calc.calculate_residuals(
                self.analysis_dataset, self.spec, self.sampling_design_state
            )

            # Calculate centered residuals (legacy support)
            self._calculate_centered_residuals()

            # Step 6: Calculate effects and interactions
            logger.info("Calculating effects and interactions")
            self.effects = self.effects_calc.calculate_all_effects(
                self.analysis_dataset, self.spec
            )
            self.interactions = self.effects_calc.calculate_interactions(
                self.analysis_dataset, self.spec, self.sampling_design_state
            )
        else:
            logger.debug(
                f"Skipping VAS residuals for analysis_type={self.spec.analysis_type}, "
                f"SDS={self.sampling_design_state}"
            )

    # =========================================================================
    # Properties (for backward compatibility and convenience)
    # =========================================================================

    @property
    def has_vas_residuals(self) -> bool:
        """Check if VAS residuals were calculated."""
        return 'R1' in self.analysis_dataset.columns

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

    # =========================================================================
    # Frame Building (kept as-is for backward compatibility)
    # =========================================================================

    def _build_frames(self) -> None:
        """
        Materialize canonical frames by grain:
        - obs_df : one row per observation
        - cell_df: one row per (k_vars + time)
        - k_df   : one row per k_vars combination
        - t_df   : one row per time point
        """
        df = self.__ensure_keys(self.analysis_dataset)

        # Build each frame using extracted helper methods
        self.obs_df = self._build_obs_df(df)
        self.k_df = self._build_k_df(df)
        self.t_df = self._build_t_df(df)
        self.cell_df = self._build_cell_df(df)

    def _build_obs_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build observation-level frame (one row per observation)."""
        spec = self.spec
        y = spec.response_var
        k_vars = list(spec.rsg_vars or [])
        t = spec.time_var

        base_cols = [
            c for c in [*k_vars, t, spec.rsg_var_name, 'obs_id', 'n']
            if c in df.columns
        ]
        means = [c for c in ['Ybar','Ybar_k','Ybar_t','Ybar_kt'] if c in df.columns]
        residuals = [c for c in ['R1','R2','R3','R4','R5'] if c in df.columns]
        rcrs = [c for c in ['RCR1','RCR2','RCR3','RCR4','RCR5'] if c in df.columns]
        centered = [c for c in ['Rbar_k','Rbar_t','Rbar_kt'] if c in df.columns]
        inter_row = [
            c for c in ['pdc_by_pt','interaction_cell','factor_interaction_effects']
            if c in df.columns
        ]

        obs_keep = [
            c for c in [y, *base_cols, *means, *residuals, *rcrs, *centered, *inter_row]
            if c in df.columns
        ]
        return (df[obs_keep]
                .sort_values('obs_id', kind='stable')
                .reset_index(drop=True))

    def _build_k_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build factor-level frame (one row per k_vars combination)."""
        spec = self.spec
        k_vars = list(spec.rsg_vars or [])

        if spec.has_grouping and k_vars:
            # counts by factor combo
            k_counts = (
                df.groupby(k_vars, sort=False, observed=True)
                .size()
                .rename('n_k')
                .reset_index()
            )
            k_df = k_counts

            # add factor-level means if present
            if 'Ybar_k' in df.columns:
                k_first = self.__safe_first(df, k_vars, 'Ybar_k')
                k_df = k_df.merge(k_first, on=k_vars, how='left', validate='one_to_one')

            # join per-factor Main_Effect tables
            for factor in k_vars:
                me = self.effects.get(factor)
                required_cols = {factor, 'Main_Effect'}
                if isinstance(me, pd.DataFrame) and required_cols <= set(me.columns):
                    me_renamed = me[[factor, 'Main_Effect']].rename(
                        columns={'Main_Effect': f'{factor}_Main_Effect'}
                    )
                    k_df = k_df.merge(
                        me_renamed,
                        on=factor, how='left', validate='many_to_one'
                    )

            # single-factor convenience alias
            if len(k_vars) == 1 and f"{k_vars[0]}_Main_Effect" in k_df.columns:
                k_df['Main_Effect_k'] = k_df[f"{k_vars[0]}_Main_Effect"]

            return k_df.sort_values(k_vars, kind='stable').reset_index(drop=True)
        else:
            cols = k_vars + ['n_k']
            if 'Ybar_k' in df.columns:
                cols += ['Ybar_k']
            return pd.DataFrame(columns=cols)

    def _build_t_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build time-level frame (one row per time point)."""
        spec = self.spec
        t = spec.time_var

        if spec.has_time and t:
            t_counts = (
                df.groupby([t], sort=False, observed=True)
                .size()
                .rename('n_t')
                .reset_index()
            )
            t_df = t_counts

            if 'Ybar_t' in df.columns:
                t_first = self.__safe_first(df, [t], 'Ybar_t')
                t_df = t_df.merge(t_first, on=[t], how='left', validate='one_to_one')

            # time main effect
            pt_me = self.effects.get('pt_me')
            if isinstance(pt_me, pd.DataFrame):
                # normalize shape: either index=t or column=t
                if t not in pt_me.columns and pt_me.index.name == t:
                    pt_me = pt_me.reset_index()
                if {'PT_ME'} <= set(pt_me.columns) and t in pt_me.columns:
                    t_df = t_df.merge(pt_me[[t, 'PT_ME']], on=t, how='left', validate='many_to_one')

            return t_df.sort_values([t], kind='stable').reset_index(drop=True)
        else:
            cols = ([t] if t else []) + ['n_t']
            if 'Ybar_t' in df.columns:
                cols += ['Ybar_t']
            cols += ['PT_ME']
            return pd.DataFrame(columns=cols)

    def _build_cell_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build cell-level frame (one row per k_vars × time combination)."""
        spec = self.spec
        k_vars = list(spec.rsg_vars or [])
        t = spec.time_var

        if spec.has_grouping and spec.has_time and k_vars and t:
            keys = k_vars + [t]

            # n per cell
            cdf = (
                df.groupby(keys, sort=False, observed=True)
                .size()
                .rename('n_cell')
                .reset_index()
            )

            # firsts of broadcast means (only if present)
            for col in ['Ybar_kt', 'Ybar_k', 'Ybar_t']:
                if col in df.columns:
                    c_first = self.__safe_first(df, keys, col)
                    cdf = cdf.merge(c_first, on=keys, how='left', validate='one_to_one')

            # interaction per cell: prefer explicit column else reconstruct
            if 'interaction_cell' in df.columns:
                ic = self.__safe_first(df, keys, 'interaction_cell')
                cdf = cdf.merge(ic, on=keys, how='left', validate='one_to_one')
            else:
                required_cols = ['Ybar_kt', 'Ybar_k', 'Ybar_t']
                has_cols = all(c in cdf.columns for c in required_cols)
                if has_cols and 'Ybar' in self.statistics:
                    ybar = float(self.statistics['Ybar'])
                    cdf['interaction_cell'] = (
                        cdf['Ybar_kt'] - cdf['Ybar_k'] - cdf['Ybar_t'] + ybar
                    )

            # centered residual cell means if present
            for col in ['Rbar_kt']:
                if col in df.columns:
                    c_first = self.__safe_first(df, keys, col)
                    cdf = cdf.merge(c_first, on=keys, how='left', validate='one_to_one')

            return cdf.sort_values(keys, kind='stable').reset_index(drop=True)
        else:
            # empty shell with predictable columns
            cols = k_vars + ([t] if t else [])
            cols += ['n_cell', 'Ybar_kt', 'Ybar_k', 'Ybar_t', 'interaction_cell', 'Rbar_kt']
            return pd.DataFrame(columns=cols)


    # --- helpers (put inside the class) ------------------------------------------
    def __ensure_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make sure key columns exist; create a deterministic obs_id if missing."""
        spec = self.spec
        out = df.copy()

        # ensure obs_id (stable) for row-grain sorting/debug
        if 'obs_id' not in out.columns:
            sort_cols = []
            if spec.has_grouping:
                sort_cols += [spec.rsg_var_name]
            if spec.has_time:
                sort_cols += [spec.time_var]
            if sort_cols:
                out = out.sort_values(sort_cols, kind='stable')
            out = out.reset_index(drop=True)
            out['obs_id'] = np.arange(len(out), dtype=int)

        return out

    def __safe_first(self, df: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame:
        """Return one row per keys with the first value of col, if col exists; else empty."""
        if col not in df.columns:
            return pd.DataFrame(columns=keys + [col])
        return (
            df.groupby(keys, sort=False, observed=True)[col]
            .first()
            .reset_index()
        )
    # -----------------------------------------------------------------------------
