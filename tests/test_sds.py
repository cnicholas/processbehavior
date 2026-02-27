"""
Tests for Sampling Design State (SDS) detection and VAS calculation decisions.

These tests verify:
1. Correct SDS detection for different data structures
2. VAS residual calculation decisions based on SDS and chart type
3. Nested structure handling (SDS5)
"""

import numpy as np
import pandas as pd

from processbehavior import ProcessBehavior
from processbehavior import analysis_dataset as ad
from processbehavior.data_preparation import DataPreparation
from processbehavior.datasets import synthetic
from processbehavior.formulation_spec import FormulationSpec
from processbehavior.sds_detector import SDSRegistry


def _make_spec(spec_dict: dict) -> FormulationSpec:
    """Convert old-style spec dict to FormulationSpec."""
    rsg_vars = spec_dict.get('rsg_vars')
    return FormulationSpec(
        response_var=spec_dict['response_var'],
        rsg_vars=tuple(rsg_vars) if rsg_vars else None,
        time_var=spec_dict.get('time_var'),
        round_to=spec_dict.get('round_to', 3),
        rsg_var_name=spec_dict.get('rsg_var_name', 'rsg'),
        rsg_var_delim=spec_dict.get('rsg_var_delim', '_'),
    )


def detect_sds_for_test(df: pd.DataFrame, spec: dict) -> int:
    """
    Helper to detect SDS for tests that need to create AnalysisDataSet directly.

    Uses FormulationSpec (without analysis_type) so data isn't filtered during SDS detection.
    This matches the real workflow where SDS is detected before analysis_type is known.

    Returns only the SDS integer, not the (sds, min_cell_size) tuple.
    """
    config = _make_spec(spec)
    prep = DataPreparation()
    prep.validate_columns(df, config)
    prepared_df = prep.prepare_dataset(df, config)
    detector = SDSRegistry()
    result = detector.detect_sds(prepared_df, config)
    return result.sds


class TestSDSDetection:
    """Test SDS detection for different data structures."""

    def test_sds1_full_replication(self):
        """Test SDS 1: Multiple observations per cell (k,t)."""
        df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)

        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1, pdf.cols.factor_2],
            time=pdf.cols.time
        )

        assert study.sds == 1, f"Expected SDS=1, got {study.sds}"
        assert 'Xbar' in study.valid_charts
        assert 'S' in study.valid_charts

    def test_sds2_no_replication_time_as_factor(self):
        """Test SDS 2: Single observation per cell when time is a factor."""
        # Use K2=1 to have exactly n=1 per (factor 1, time) cell
        df = synthetic.make_sds(2, K1=3, K2=1, T=10, seed=42)

        # With time as factor: 30 factor×time cells with n=1 each = SDS 2
        spec = {
            'analysis_type': 'XmR',
            'rsg_vars': ['factor 1', 'time'],  # time is now a factor!
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg',
            'round_to': 3
        }

        sds = detect_sds_for_test(df, spec)
        assert sds == 2, f"Time as factor: Expected SDS=2, got {sds}"

    def test_sds2_detection_with_nkt_grouping(self):
        """Test that SDS 2 data is correctly detected as SDS 2 per Wheeler/Bishop.

        Per Wheeler/Bishop methodology, SDS is based on N_kt (factor × time cells):
        - make_sds(2) generates data with all N_kt = 1 (no replication)
        - This correctly classifies as SDS 2 (Semi-Complete, no replication)

        Note: Analysis subgrouping uses factor-only (n=10 per factor), but
        SDS classification uses N_kt.
        """
        # Use K2=1 to ensure n=1 per (factor 1, time) cell for proper SDS 2 detection
        df = synthetic.make_sds(2, K1=3, K2=1, T=10, seed=42)

        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg',
            'round_to': 3
        }

        sds = detect_sds_for_test(df, spec)
        assert sds == 2, f"Wheeler/Bishop: all N_kt=1 → Expected SDS=2, got {sds}"

    def test_sds5_nesting_structure(self):
        """Verify that SDS5 properly implements nested structure."""
        df = synthetic.make_sds(5, L=3, H_per_L=4, T=6, seed=42)

        # Test 1: Each head should appear with exactly one line
        head_line_map = df.groupby('factor 2')['factor 1'].unique()
        for head, lines in head_line_map.items():
            assert len(lines) == 1, f"{head} appears with multiple lines: {lines}"

        # Test 2: Should have L*H_per_L unique heads total
        n_unique_heads = df['factor 2'].nunique()
        expected_heads = 3 * 4  # L * H_per_L
        assert n_unique_heads == expected_heads, \
            f"Expected {expected_heads} unique heads, got {n_unique_heads}"

        # Test 3: Each line should have H_per_L heads
        heads_per_line = df.groupby('factor 1')['factor 2'].nunique()
        assert (heads_per_line == 4).all(), \
            f"Not all lines have 4 heads: {heads_per_line.to_dict()}"

        # Test 4: Head names should reflect nesting
        sample_heads = df['factor 2'].unique()[:3]
        for head in sample_heads:
            assert 'Line' in head, f"Head name should contain 'Line': {head}"

    def test_sds6_stratified_xmr(self):
        """Test that stratified XmR works with irregular SDS6 data."""
        df = synthetic.make_sds(6, T=80, K1=3, K2=2, p_sampled=0.6, seed=42)

        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1, pdf.cols.factor_2],
            time=pdf.cols.time
        )

        # XmR with factors requires explicit 'by' parameter
        # SRP: XmR only returns XmR (use companion=True for bundled XmR+R)
        result = study.execute(chart='XmR', by=['factor 1', 'factor 2'])

        assert result is not None
        # SRP: Should have single XmR chart
        assert len(result.charts) == 1
        assert 'XmR' in result.charts


class TestVASCalculationDecisions:
    """Test VAS residual calculation decision matrix."""

    def test_sds1_xbar_calculates_vas(self):
        """SDS 1 + Xbar should calculate VAS residuals."""
        df = synthetic.make_sds(1, K1=2, K2=2, T=3, n_min=2, n_max=4, seed=42)

        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1'],
            'response_var': 'y',
            'time_var': 'time',
            'rsg_var_name': 'rsg'
        }

        sds = detect_sds_for_test(df, spec)
        aspec = _make_spec(spec)
        ads = ad.AnalysisDataSet(df, aspec, sds=sds)

        assert 'R1' in ads.analysis_dataset.columns, "SDS1 + Xbar should have VAS residuals"
        assert 'R2' in ads.analysis_dataset.columns

    def test_sds1_xmr_has_vas(self):
        """SDS 1 + XmR should have VAS residuals (ADS is chart-agnostic).

        Note: VAS residuals are always computed when we have grouping AND time.
        The analysis_type no longer gates VAS calculation - ADS is chart-agnostic.
        """
        df = synthetic.make_sds(1, K1=2, K2=2, T=3, n_min=2, n_max=4, seed=42)

        spec = {
            'analysis_type': 'XmR',
            'rsg_vars': ['factor 1'],
            'response_var': 'y',
            'time_var': 'time',
            'rsg_var_name': 'rsg'
        }

        sds = detect_sds_for_test(df, spec)
        aspec = _make_spec(spec)
        ads = ad.AnalysisDataSet(df, aspec, sds=sds)

        # VAS is computed for any SDS with grouping and time
        assert 'R1' in ads.analysis_dataset.columns, \
            "SDS1 with grouping+time should have VAS residuals"

    def test_vas_decision_matrix_chart_agnostic(self):
        """Test VAS calculation is now chart-agnostic.

        VAS residuals are computed based on data structure (grouping + time),
        NOT based on analysis_type. This test verifies the new behavior.
        """
        # New decision matrix: VAS is computed when we have grouping AND time
        # analysis_type no longer matters for VAS calculation
        decision_matrix = {
            # SDS 1: Has grouping + time → always VAS
            (1, 'Xbar'): True,
            (1, 'S'): True,
            (1, 'XmR'): True,  # Changed from False
            (1, 'R'): True,    # Changed from False

            # SDS 2: Has grouping + time → always VAS
            (2, 'Xbar'): True,
            (2, 'XmR'): True,  # Changed from False
        }

        generators = {
            1: lambda: synthetic.make_sds(1, K1=2, K2=2, T=3, n_min=2, n_max=4, seed=42),
            2: lambda: synthetic.make_sds(2, K1=2, K2=2, T=3, seed=42),
        }

        for (sds_expected, analysis_type), expected_vas in decision_matrix.items():
            df = generators[sds_expected]()

            spec = {
                'analysis_type': analysis_type,
                'rsg_vars': ['factor 1'],
                'time_var': 'time',
                'response_var': 'y',
                'rsg_var_name': 'rsg'
            }

            detected_sds = detect_sds_for_test(df, spec)
            aspec = _make_spec(spec)
            ads = ad.AnalysisDataSet(df, aspec, sds=detected_sds)

            actual_vas = 'R1' in ads.analysis_dataset.columns

            assert actual_vas == expected_vas, \
                f"SDS {sds_expected} + {analysis_type}: expected VAS={expected_vas}, got {actual_vas}"


class TestResidualCalculations:
    """Test VAS residual calculation correctness."""

    def test_r2_within_cell_residual_sds1(self):
        """Test R2 is within-cell residual for SDS1."""
        df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)

        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg',
            'round_to': 3
        }

        sds = detect_sds_for_test(df, spec)
        aspec = _make_spec(spec)
        ads = ad.AnalysisDataSet(df=df, spec=aspec, sds=sds)

        # R2 should be y - Ybar_kt (within-cell residual)
        # Verify by checking correlation with expected calculation
        ds = ads.analysis_dataset
        expected_r2 = ds['y'] - ds['Ybar_kt']

        # Check correlation is very high (should be ~1.0)
        correlation = np.corrcoef(ds['R2'], expected_r2)[0, 1]
        assert correlation > 0.9999, f"R2 calculation correlation: {correlation}"

    def test_residual_identities(self):
        """Test that residual relationships hold: R3 = (Ybar_kt - Ybar_k - Ybar_t + Ybar) + R2."""
        df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)

        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg'
        }

        sds = detect_sds_for_test(df, spec)
        aspec = _make_spec(spec)
        ads = ad.AnalysisDataSet(df=df, spec=aspec, sds=sds)

        ds = ads.analysis_dataset
        TOL = 1e-10

        # R3 identity
        r3_rhs = (ds['Ybar_kt'] - ds['Ybar_k'] - ds['Ybar_t'] + ds['Ybar']) + ds['R2']
        assert (ds['R3'] - r3_rhs).abs().max() <= TOL

        # R4 identity
        r4_rhs = (ds['Ybar_t'] - ds['Ybar']) + ds['R2']
        assert (ds['R4'] - r4_rhs).abs().max() <= TOL

        # R5 identity
        r5_rhs = (ds['Ybar_k'] - ds['Ybar']) + ds['R2']
        assert (ds['R5'] - r5_rhs).abs().max() <= TOL

    def test_rcr_centered_residuals(self):
        """Test that RCR formulas are correct for each residual type."""
        df = synthetic.make_sds(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)

        spec = {
            'analysis_type': 'Xbar',
            'rsg_vars': ['factor 1'],
            'time_var': 'time',
            'response_var': 'y',
            'rsg_var_name': 'rsg'
        }

        sds = detect_sds_for_test(df, spec)
        aspec = _make_spec(spec)
        ads = ad.AnalysisDataSet(df=df, spec=aspec, sds=sds)

        ds = ads.analysis_dataset
        TOL = 1e-10

        # RCR formulas from analysis_dataset.py:
        # RCR1 = Ybar + R1
        # RCR2 = Ybar_kt + R2
        # RCR3 = (Ybar_k + Ybar_t - Ybar) + R3
        # RCR4 = (Ybar + Ybar_kt - Ybar_t) + R4
        # RCR5 = (Ybar + Ybar_kt - Ybar_k) + R5

        expected_rcr1 = ds['Ybar'] + ds['R1']
        assert (ds['RCR1'] - expected_rcr1).abs().max() <= TOL, "RCR1 formula incorrect"

        expected_rcr2 = ds['Ybar_kt'] + ds['R2']
        assert (ds['RCR2'] - expected_rcr2).abs().max() <= TOL, "RCR2 formula incorrect"

        expected_rcr3 = (ds['Ybar_k'] + ds['Ybar_t'] - ds['Ybar']) + ds['R3']
        assert (ds['RCR3'] - expected_rcr3).abs().max() <= TOL, "RCR3 formula incorrect"

        expected_rcr4 = (ds['Ybar'] + ds['Ybar_kt'] - ds['Ybar_t']) + ds['R4']
        assert (ds['RCR4'] - expected_rcr4).abs().max() <= TOL, "RCR4 formula incorrect"

        expected_rcr5 = (ds['Ybar'] + ds['Ybar_kt'] - ds['Ybar_k']) + ds['R5']
        assert (ds['RCR5'] - expected_rcr5).abs().max() <= TOL, "RCR5 formula incorrect"

        # End-to-end reconstruction identity
        # The fixture is SDS 1 with n_min=2, guaranteeing the exact R2 method
        # (all cells have n >= 2). Under the exact method:
        #   R1 = Y - Ybar      → RCR1 = Ybar + R1 = Y
        #   R2 = Y - Ybar_kt   → RCR2 = Ybar_kt + R2 = Y
        assert (ds['RCR1'] - ds['y']).abs().max() <= TOL, (
            "RCR1 must reconstruct Y exactly (RCR1 = Ybar + R1 = Y)"
        )
        assert (ds['RCR2'] - ds['y']).abs().max() <= TOL, (
            "RCR2 must reconstruct Y exactly under exact R2 method "
            "(SDS 1, n_min=2, all cells replicated)"
        )
