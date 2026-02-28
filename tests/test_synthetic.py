"""Tests for synthetic data generators.

This module tests the make_sds*() functions to ensure they generate valid data
with the correct structure for each Sampling Design State.
"""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets.synthetic import make_sds


class TestSyntheticStructure:
    """Verify generated data has correct structure."""

    @pytest.mark.parametrize("sds", [1, 2, 3, 6])
    def test_factor2_populated(self, sds):
        """Factor 2 should have actual values, not 'NA' for SDS 1, 2, 3, 6."""
        df = make_sds(sds, K1=3, K2=2, T=4, seed=42)
        assert (df['factor 2'] != 'NA').all(), f"SDS {sds} should populate factor 2"
        assert df['factor 2'].nunique() == 2, f"SDS {sds} should have K2=2 factor 2 levels"

    def test_sds1_cell_structure(self):
        """SDS1: K1 × K2 × T cells, each with n >= 2."""
        df = make_sds(1, K1=3, K2=2, T=4, n_min=2, n_max=3, seed=42)
        cells = df.groupby(['factor 1', 'factor 2', 'time']).size()
        assert len(cells) == 3 * 2 * 4, "SDS1 should have K1 × K2 × T cells"
        assert cells.min() >= 2, "SDS1 should have n >= 2 per cell"

    def test_sds2_cell_structure(self):
        """SDS2: K1 × K2 × T cells, each with exactly n=1."""
        df = make_sds(2, K1=3, K2=2, T=4, seed=42)
        cells = df.groupby(['factor 1', 'factor 2', 'time']).size()
        assert len(cells) == 3 * 2 * 4, "SDS2 should have K1 × K2 × T cells"
        assert (cells == 1).all(), "SDS2 should have n=1 per cell"

    def test_sds3_mixed_replication(self):
        """SDS3: Mix of n=1 and n >= 2 cells."""
        df = make_sds(3, K1=3, K2=2, T=8, p_replicated=0.5, seed=42)
        cells = df.groupby(['factor 1', 'factor 2', 'time']).size()
        has_singles = (cells == 1).any()
        has_multiples = (cells >= 2).any()
        assert has_singles and has_multiples, "SDS3 should have mixed replication"

    def test_sds4_no_factor2(self):
        """SDS4: Single condition, factor 2 = 'NA'."""
        df = make_sds(4, T=20, seed=42)
        assert df['factor 2'].eq('NA').all(), "SDS4 should have factor 2 = 'NA'"

    def test_sds5_nested_structure(self):
        """SDS5: Factor 2 nested in factor 1."""
        df = make_sds(5, L=2, H_per_L=3, T=4, seed=42)
        # Each factor 2 value should appear with only one factor 1 value
        factor2_to_factor1 = df.groupby('factor 2')['factor 1'].nunique()
        assert factor2_to_factor1.max() == 1, "SDS5 factor 2 should be nested in factor 1"

    def test_sds6_incomplete_grid(self):
        """SDS6: Incomplete (factor × time) grid due to sparse sampling."""
        df = make_sds(6, T=40, K1=3, K2=2, p_sampled=0.5, seed=42)
        full_grid = 3 * 2 * 40  # K1 × K2 × T
        actual_cells = df.groupby(['factor 1', 'factor 2', 'time']).ngroups
        assert actual_cells < full_grid * 0.95, "SDS6 should have incomplete grid"


class TestSDSDetection:
    """Verify SDS detection with ProcessBehavior API."""

    def test_sds1_detected(self):
        """SDS1 should be detected as SDS 1."""
        df = make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=3, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1, pdf.cols.factor_2],
            time=pdf.cols.time
        )
        assert study.observed_design_state.sds == 1, f"Expected SDS 1, got SDS {study.observed_design_state.sds}"

    def test_sds2_detected(self):
        """SDS2 should be detected as SDS 2."""
        df = make_sds(2, K1=3, K2=2, T=6, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1, pdf.cols.factor_2],
            time=pdf.cols.time
        )
        assert study.observed_design_state.sds == 2, f"Expected SDS 2, got SDS {study.observed_design_state.sds}"

    def test_sds4_data_classifies_by_replication(self):
        """make_sds(4) generates K=1 data, classifies by N_kt pattern.

        Per Table 1: K=1 with all N_kt=1 → SDS 2 (no replication).
        SDS 4 in Table 1 means "incomplete with singletons", not K=1.
        """
        df = make_sds(4, T=20, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        # K=1, all N_kt=1 → SDS 2 (no replication)
        obs_sds = study.observed_design_state.sds
        assert obs_sds == 2, f"Expected SDS 2 (no replication), got SDS {obs_sds}"

    def test_sds5_data_classifies_by_nkt_pattern(self):
        """make_sds(5) generates nested data, classifies by N_kt pattern.

        Without a plan, nested data classifies based on observed N_kt.
        The SDS depends on whether the observed cells show replication.
        """
        df = make_sds(5, L=2, H_per_L=3, T=6, p_active=0.7, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1, pdf.cols.factor_2],
            time=pdf.cols.time
        )
        # Without a plan, classifies based on observed N_kt pattern
        # Nested structure with n=1 per cell typically → SDS 2
        obs_sds = study.observed_design_state.sds
        assert obs_sds in [1, 2, 3], f"Expected SDS 1/2/3, got SDS {obs_sds}"

    def test_sds6_detected_with_plan(self):
        """SDS6 requires plan to detect incomplete grid."""
        df = make_sds(6, T=40, K1=3, K2=2, p_sampled=0.5, seed=42)
        pdf = ProcessBehavior(df)

        # Get actual factor levels from the data
        factor1_levels = df['factor 1'].unique().tolist()
        factor2_levels = df['factor 2'].unique().tolist()

        # Plan specifies expected levels - data is incomplete relative to this
        # Note: Use plan alone (not with factors) per API requirements
        study = pdf.formulate(
            response=pdf.cols.y,
            time=pdf.cols.time,
            plan={
                'factors': {
                    pdf.cols.factor_1: factor1_levels,
                    pdf.cols.factor_2: factor2_levels
                },
                'T': 40,
                'N': 2
            }
        )
        assert study.observed_design_state.sds == 6, f"Expected SDS 6, got SDS {study.observed_design_state.sds}"


class TestFullPipeline:
    """Verify synthetic data works through full analysis."""

    def test_sds1_two_factor_analysis(self):
        """SDS1 data should complete full Xbar-S analysis with residuals."""
        df = make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=3, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1, pdf.cols.factor_2],
            time=pdf.cols.time
        )
        result = study.execute()
        assert result is not None, "Analysis result should not be None"
        assert result.has_residuals, "SDS1 analysis should have residuals"

    def test_sds2_analysis(self):
        """SDS2 data should complete analysis."""
        df = make_sds(2, K1=3, K2=2, T=6, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1, pdf.cols.factor_2],
            time=pdf.cols.time
        )
        # SDS2 has n=1 per cell, so use factor-level aggregation for Xbar
        result = study.execute(chart='Xbar', by=['factor 1', 'factor 2'])
        assert result is not None, "Analysis result should not be None"


class TestConvenienceFunction:
    """Test make_sds() convenience function."""

    @pytest.mark.parametrize("sds", [1, 2, 3, 4, 5, 6])
    def test_make_sds_all_types(self, sds):
        """make_sds() should generate valid data for all SDS types (1-6)."""
        df = make_sds(sds, K1=3, K2=2, T=8, seed=42)
        assert len(df) > 0, f"SDS {sds} should generate non-empty data"
        assert 'y' in df.columns, f"SDS {sds} should have 'y' column"

    def test_make_sds_invalid_type(self):
        """make_sds() should raise ValueError for invalid SDS type."""
        with pytest.raises(ValueError, match="not implemented"):
            make_sds(99, seed=42)

    def test_make_sds_kwargs_passthrough(self):
        """make_sds() should pass kwargs to specific generator."""
        df = make_sds(3, K1=3, K2=2, T=8, p_replicated=0.8, seed=42)
        cells = df.groupby(['factor 1', 'factor 2', 'time']).size()
        # With high p_replicated, most cells should have replication
        replicated_ratio = (cells > 1).sum() / len(cells)
        assert replicated_ratio > 0.5, "High p_replicated should result in more replicated cells"


class TestFactorNames:
    """Test custom factor naming."""

    def test_custom_factor1_names(self):
        """Custom factor 1 names should be used."""
        names = ['Alpha', 'Beta', 'Gamma']
        df = make_sds(1, K1=3, K2=2, T=4, factor1_names=names, seed=42)
        assert set(df['factor 1'].unique()) == set(names)

    def test_custom_factor2_names(self):
        """Custom factor 2 names should be used."""
        names = ['Shift_A', 'Shift_B']
        df = make_sds(1, K1=3, K2=2, T=4, factor2_names=names, seed=42)
        assert set(df['factor 2'].unique()) == set(names)

    def test_factor_names_wrong_length_raises(self):
        """Wrong number of factor names should raise ValueError."""
        with pytest.raises(ValueError, match="must equal K1"):
            make_sds(1, K1=3, K2=2, T=4, factor1_names=['A', 'B'], seed=42)


class TestReproducibility:
    """Test that seeded generation is reproducible."""

    def test_same_seed_same_data(self):
        """Same seed should produce identical data."""
        df1 = make_sds(1, K1=3, K2=2, T=4, seed=42)
        df2 = make_sds(1, K1=3, K2=2, T=4, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_data(self):
        """Different seeds should produce different data."""
        df1 = make_sds(1, K1=3, K2=2, T=4, seed=42)
        df2 = make_sds(1, K1=3, K2=2, T=4, seed=123)
        assert not df1['y'].equals(df2['y'])
