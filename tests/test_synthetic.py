"""Tests for synthetic data generators.

This module tests the make_sds*() functions to ensure they generate valid data
with the correct structure for each Sampling Design State.
"""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets.synthetic import make_sds
from processbehavior.exceptions import ValidationError


class TestSyntheticStructure:
    """Verify generated data has correct structure."""

    @pytest.mark.parametrize('sds', [1, 2, 3, 6])
    def test_factor2_populated(self, sds):
        """Factor 2 should have actual values, not 'NA' for SDS 1, 2, 3, 6."""
        df = make_sds(sds, K1=3, K2=2, T=4, seed=42)
        assert (df['factor 2'] != 'NA').all(), f'SDS {sds} should populate factor 2'
        assert df['factor 2'].nunique() == 2, f'SDS {sds} should have K2=2 factor 2 levels'

    def test_sds1_cell_structure(self):
        """SDS1: K1 × K2 × T cells, each with n >= 2."""
        df = make_sds(1, K1=3, K2=2, T=4, n_min=2, n_max=3, seed=42)
        cells = df.groupby(['factor 1', 'factor 2', 'time']).size()
        assert len(cells) == 3 * 2 * 4, 'SDS1 should have K1 × K2 × T cells'
        assert cells.min() >= 2, 'SDS1 should have n >= 2 per cell'

    def test_sds2_cell_structure(self):
        """SDS2: K1 × K2 × T cells, each with exactly n=1."""
        df = make_sds(2, K1=3, K2=2, T=4, seed=42)
        cells = df.groupby(['factor 1', 'factor 2', 'time']).size()
        assert len(cells) == 3 * 2 * 4, 'SDS2 should have K1 × K2 × T cells'
        assert (cells == 1).all(), 'SDS2 should have n=1 per cell'

    def test_sds3_mixed_replication(self):
        """SDS3: Mix of n=1 and n >= 2 cells."""
        df = make_sds(3, K1=3, K2=2, T=8, p_replicated=0.5, seed=42)
        cells = df.groupby(['factor 1', 'factor 2', 'time']).size()
        has_singles = (cells == 1).any()
        has_multiples = (cells >= 2).any()
        assert has_singles and has_multiples, 'SDS3 should have mixed replication'

    def test_sds4_incomplete_replicated(self):
        """ODS 4: Incomplete grid, all occupied cells replicated (N>=2)."""
        df = make_sds(4, K1=3, K2=2, T=6, seed=42)
        # Empty cells appear as rows with y=NaN
        assert df['y'].isna().any(), 'ODS 4 must include at least one empty cell (NaN y)'
        occupied = df.dropna(subset=['y'])
        sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
        assert (sizes >= 2).all(), 'ODS 4 occupied cells must all have N>=2'

    def test_sds5_incomplete_singletons(self):
        """ODS 5: Incomplete grid, all occupied cells singleton (N=1)."""
        df = make_sds(5, K1=3, K2=2, T=6, seed=42)
        assert df['y'].isna().any(), 'ODS 5 must include at least one empty cell'
        occupied = df.dropna(subset=['y'])
        sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
        assert (sizes == 1).all(), 'ODS 5 occupied cells must all be singletons'

    def test_sds6_incomplete_mixed(self):
        """ODS 6: Incomplete grid, mix of singleton and replicated cells."""
        df = make_sds(6, K1=3, K2=2, T=8, seed=42)
        assert df['y'].isna().any(), 'ODS 6 must include at least one empty cell'
        occupied = df.dropna(subset=['y'])
        sizes = occupied.groupby(['factor 1', 'factor 2', 'time']).size()
        assert (sizes == 1).any(), 'ODS 6 must have at least one singleton cell'
        assert (sizes >= 2).any(), 'ODS 6 must have at least one replicated cell'


class TestSDSDetection:
    """Verify SDS detection with ProcessBehavior API."""

    def test_sds1_detected(self):
        """SDS1 should be detected as SDS 1."""
        df = make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=3, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.y, factors=[pdf.cols.factor_1, pdf.cols.factor_2], time=pdf.cols.time)
        assert study.observed_design_state.sds == 1, f'Expected SDS 1, got SDS {study.observed_design_state.sds}'

    def test_sds2_detected(self):
        """SDS2 should be detected as SDS 2."""
        df = make_sds(2, K1=3, K2=2, T=6, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.y, factors=[pdf.cols.factor_1, pdf.cols.factor_2], time=pdf.cols.time)
        assert study.observed_design_state.sds == 2, f'Expected SDS 2, got SDS {study.observed_design_state.sds}'

    def test_sds4_classifies_as_ods4(self):
        """make_sds(4) data classifies as ODS 4 (incomplete, all replicated)."""
        df = make_sds(4, K1=3, K2=2, T=6, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.y, factors=[pdf.cols.factor_1, pdf.cols.factor_2], time=pdf.cols.time)
        assert study.observed_design_state.sds == 4, f'Expected ODS 4, got ODS {study.observed_design_state.sds}'
        # Lineage: after tidy → ADS 1 (full replication on surviving grid)
        assert study.analytical_design_state.sds == 1, f'Expected ADS 1, got ADS {study.analytical_design_state.sds}'

    def test_sds5_classifies_as_ods5(self):
        """make_sds(5) data classifies as ODS 5 (incomplete, all singletons)."""
        df = make_sds(5, K1=3, K2=2, T=6, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.y, factors=[pdf.cols.factor_1, pdf.cols.factor_2], time=pdf.cols.time)
        assert study.observed_design_state.sds == 5, f'Expected ODS 5, got ODS {study.observed_design_state.sds}'
        assert study.analytical_design_state.sds == 2, f'Expected ADS 2, got ADS {study.analytical_design_state.sds}'

    def test_sds6_classifies_as_ods6(self):
        """make_sds(6) data classifies as ODS 6 (incomplete, mixed N)."""
        df = make_sds(6, K1=3, K2=2, T=8, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.y, factors=[pdf.cols.factor_1, pdf.cols.factor_2], time=pdf.cols.time)
        assert study.observed_design_state.sds == 6, f'Expected ODS 6, got ODS {study.observed_design_state.sds}'
        assert study.analytical_design_state.sds == 3, f'Expected ADS 3, got ADS {study.analytical_design_state.sds}'


class TestFullPipeline:
    """Verify synthetic data works through full analysis."""

    def test_sds1_two_factor_analysis(self):
        """SDS1 data should complete full Xbar-S analysis with residuals."""
        df = make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=3, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.y, factors=[pdf.cols.factor_1, pdf.cols.factor_2], time=pdf.cols.time)
        result = study.execute()
        assert result is not None, 'Analysis result should not be None'
        assert result.has_residuals, 'SDS1 analysis should have residuals'

    def test_sds2_analysis(self):
        """SDS2 data should complete analysis."""
        df = make_sds(2, K1=3, K2=2, T=6, seed=42)
        pdf = ProcessBehavior(df)
        study = pdf.formulate(response=pdf.cols.y, factors=[pdf.cols.factor_1, pdf.cols.factor_2], time=pdf.cols.time)
        # SDS2 has n=1 per cell, so use factor-level aggregation for Xbar
        result = study.execute(chart='Xbar', by=['factor 1', 'factor 2'])
        assert result is not None, 'Analysis result should not be None'


class TestConvenienceFunction:
    """Test make_sds() convenience function."""

    @pytest.mark.parametrize('sds', [1, 2, 3, 4, 5, 6])
    def test_make_sds_all_types(self, sds):
        """make_sds() should generate valid data for all SDS types (1-6)."""
        df = make_sds(sds, K1=3, K2=2, T=8, seed=42)
        assert len(df) > 0, f'SDS {sds} should generate non-empty data'
        assert 'y' in df.columns, f"SDS {sds} should have 'y' column"

    def test_make_sds_invalid_type(self):
        """make_sds() should raise ValueError for invalid SDS type."""
        with pytest.raises(ValidationError, match='not implemented'):
            make_sds(99, seed=42)

    def test_make_sds_kwargs_passthrough(self):
        """make_sds() should pass kwargs to specific generator."""
        df = make_sds(3, K1=3, K2=2, T=8, p_replicated=0.8, seed=42)
        cells = df.groupby(['factor 1', 'factor 2', 'time']).size()
        # With high p_replicated, most cells should have replication
        replicated_ratio = (cells > 1).sum() / len(cells)
        assert replicated_ratio > 0.5, 'High p_replicated should result in more replicated cells'


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
        with pytest.raises(ValidationError, match='must equal K1'):
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
