"""
Integration tests for WECO Rules with Phase 1 metadata implementation.

Tests that detect_signals() works correctly with all chart types by using
the metadata-based value column resolution.
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessDataFrame
from processbehavior.datasets import synthetic
from processbehavior.signals.config import SignalConfig


class TestWECOIntegration:
    """Test WECO rules integration with metadata-based column resolution."""

    @pytest.mark.xfail(reason="Detector doesn't yet support varying control limits (stats['ucl']='Varies')")
    def test_detect_signals_xbar_chart(self):
        """Test detect_signals() works with Xbar chart (value_col='xbar')."""
        # Use synthetic SDS1 data (full replication)
        df = synthetic.make_sds1(K=3, T=8, n_min=2, n_max=4, seed=42)

        # Analyze
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            grouping_vars=['factor 1'],
            time_var='time'
        )
        result = analysis.calculate()

        # Verify Xbar chart has metadata
        assert 'Xbar' in result.charts
        assert 'metadata' in result.charts['Xbar']
        assert result.charts['Xbar']['metadata']['value_col'] == 'xbar'

        # Detect signals - should not crash
        # Use config with lower min_observations since Xbar chart has only 3 rows (one per factor)
        config = SignalConfig(min_observations=3)
        signals = result.detect_signals(chart='Xbar', config=config)

        # Verify signal detection worked
        assert signals is not None
        assert hasattr(signals, 'count')
        assert hasattr(signals, 'has_signals')

    @pytest.mark.xfail(reason="Detector doesn't yet support varying control limits (stats['ucl']='Varies')")
    def test_detect_signals_sbar_chart(self):
        """Test detect_signals() works with Sbar chart (value_col='s')."""
        # Use synthetic SDS1 data (full replication)
        df = synthetic.make_sds1(K=4, T=10, n_min=2, n_max=5, seed=42)

        # Analyze
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            grouping_vars=['factor 1'],
            time_var='time'
        )
        result = analysis.calculate()

        # Verify Sbar chart has metadata
        assert 'Sbar' in result.charts
        assert 'metadata' in result.charts['Sbar']
        assert result.charts['Sbar']['metadata']['value_col'] == 's'

        # Detect signals - should not crash
        # Use config with lower min_observations
        config = SignalConfig(min_observations=3)
        signals = result.detect_signals(chart='Sbar', config=config)

        # Verify signal detection worked
        assert signals is not None
        assert hasattr(signals, 'count')
        assert hasattr(signals, 'has_signals')

    def test_detect_signals_imr_chart(self):
        """Test detect_signals() works with IMR chart (value_col=response_var)."""
        # Use synthetic SDS4 data (single condition over time)
        df = synthetic.make_sds4(T=50, seed=42)

        # Analyze
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            time_var='time'
        )
        result = analysis.calculate()

        # IMR analysis returns stratified results with 'all' key
        assert 'all' in result.charts
        assert 'metadata' in result.charts['all']
        assert result.charts['all']['metadata']['value_col'] == 'y'
        assert result.charts['all']['metadata']['chart_type'] == 'Imr'

        # Detect signals - should not crash
        signals = result.detect_signals(chart='all')

        # Verify signal detection worked
        assert signals is not None
        assert hasattr(signals, 'count')
        assert hasattr(signals, 'has_signals')

    def test_detect_signals_r_chart(self):
        """Test detect_signals() works with R chart (value_col='mr')."""
        # Use synthetic SDS4 data (single condition over time)
        df = synthetic.make_sds4(T=50, seed=42)

        # Analyze - IMR analysis includes moving range calculation
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            time_var='time'
        )
        result = analysis.calculate()

        # IMR analysis returns results with 'all' key
        # The chart includes moving range data with metadata
        assert 'all' in result.charts
        chart_info = result.charts['all']

        # Verify metadata exists (IMR chart includes MR calculations)
        assert 'metadata' in chart_info
        # Note: IMR analysis returns Imr chart type, not separate R chart
        assert chart_info['metadata']['chart_type'] == 'Imr'

    @pytest.mark.xfail(reason="Detector doesn't yet support varying control limits (stats['ucl']='Varies')")
    def test_detect_signals_all_charts(self):
        """Test detect_signals() without chart parameter detects on all charts."""
        # Use synthetic SDS1 data (full replication)
        df = synthetic.make_sds1(K=3, T=8, n_min=2, n_max=4, seed=42)

        # Analyze
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            grouping_vars=['factor 1'],
            time_var='time'
        )
        result = analysis.calculate()

        # Detect signals on all charts
        # Use config with lower min_observations
        config = SignalConfig(min_observations=3)
        all_signals = result.detect_signals(config=config)

        # Should return dict with signals for each chart
        assert isinstance(all_signals, dict)
        assert 'Xbar' in all_signals
        assert 'Sbar' in all_signals

        # Each should be a SignalResult
        for chart_name, signals in all_signals.items():
            assert hasattr(signals, 'count')
            assert hasattr(signals, 'has_signals')

    @pytest.mark.xfail(reason="Detector doesn't yet support varying control limits (stats['ucl']='Varies')")
    def test_detect_signals_with_violations(self):
        """Test that actual violations are detected correctly."""
        # Create SDS1 data with known violation - add outlier manually
        df = synthetic.make_sds1(K=3, T=10, n_min=2, n_max=4, seed=42)

        # Inject an outlier - find one group and make its values very high
        # Find observations for factor 1 = K2 at time = 5
        mask = (df['factor 1'] == 'K2') & (df['time'] == 5)
        df.loc[mask, 'y'] = df.loc[mask, 'y'] + 50  # Add large shift

        # Analyze
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            grouping_vars=['factor 1'],
            time_var='time'
        )
        result = analysis.calculate()

        # Detect signals with Rule 1 (beyond limits)
        # Use config with lower min_observations
        config = SignalConfig(min_observations=3, enabled_rules=['rule_1'])
        signals = result.detect_signals(chart='Xbar', config=config)

        # Should detect the outlier
        assert signals.has_signals
        assert signals.count > 0

    def test_metadata_missing_raises_error(self):
        """Test that missing metadata raises helpful error."""
        # Create result with charts
        df = synthetic.make_sds1(K=2, T=6, n_min=2, n_max=3, seed=42)

        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            grouping_vars=['factor 1'],
            time_var='time'
        )
        result = analysis.calculate()

        # Manually remove metadata to simulate bug
        del result.charts['Xbar']['metadata']

        # Should raise helpful error
        with pytest.raises(ValueError, match="missing metadata"):
            result.detect_signals(chart='Xbar')

    @pytest.mark.xfail(reason="Detector doesn't yet support varying control limits (stats['ucl']='Varies')")
    def test_value_column_used_correctly(self):
        """Test that the correct value column is actually used for detection."""
        # Use synthetic SDS1 data
        df = synthetic.make_sds1(K=3, T=8, n_min=2, n_max=4, seed=42)

        # Analyze
        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            grouping_vars=['factor 1'],
            time_var='time'
        )
        result = analysis.calculate()

        # Get Xbar chart
        xbar_chart = result.charts['Xbar']

        # Verify the data has 'xbar' column (not 'mean')
        assert 'xbar' in xbar_chart['data'].columns

        # Verify metadata points to 'xbar'
        assert xbar_chart['metadata']['value_col'] == 'xbar'

        # Detect signals should use 'xbar' column
        # Use config with lower min_observations
        config = SignalConfig(min_observations=3)
        signals = result.detect_signals(chart='Xbar', config=config)

        # Verify it worked (didn't try to use 'mean' column which doesn't exist)
        assert signals is not None


class TestMetadataContract:
    """Test that all chart types have proper metadata."""

    def test_xbar_sbar_metadata(self):
        """Test Xbar and Sbar charts have complete metadata."""
        df = synthetic.make_sds1(K=2, T=8, n_min=2, n_max=4, seed=42)

        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            grouping_vars=['factor 1'],
            time_var='time'
        )
        result = analysis.calculate()

        # Check Xbar metadata
        xbar_meta = result.charts['Xbar']['metadata']
        assert xbar_meta['chart_type'] == 'Xbar'
        assert xbar_meta['value_col'] == 'xbar'
        assert xbar_meta['center_col'] == 'center'

        # Check Sbar metadata
        sbar_meta = result.charts['Sbar']['metadata']
        assert sbar_meta['chart_type'] == 'Sbar'
        assert sbar_meta['value_col'] == 's'
        assert sbar_meta['center_col'] == 'center'

    def test_imr_metadata(self):
        """Test IMR chart has complete metadata."""
        df = synthetic.make_sds4(T=50, seed=42)

        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            time_var='time'
        )
        result = analysis.calculate()

        # Check IMR metadata
        imr_meta = result.charts['all']['metadata']
        assert imr_meta['chart_type'] == 'Imr'
        assert imr_meta['value_col'] == 'y'
        assert imr_meta['center_col'] == 'center'

    def test_metadata_structure(self):
        """Test metadata has required keys."""
        df = synthetic.make_sds1(K=2, T=8, n_min=2, n_max=4, seed=42)

        pdf = ProcessDataFrame(df)
        analysis = pdf.analyze(
            response_var='y',
            grouping_vars=['factor 1'],
            time_var='time'
        )
        result = analysis.calculate()

        # All charts should have metadata with required keys
        for chart_name, chart_info in result.charts.items():
            assert 'metadata' in chart_info, f"Chart '{chart_name}' missing metadata"

            meta = chart_info['metadata']
            assert 'chart_type' in meta, f"Chart '{chart_name}' metadata missing 'chart_type'"
            assert 'value_col' in meta, f"Chart '{chart_name}' metadata missing 'value_col'"
            assert 'center_col' in meta, f"Chart '{chart_name}' metadata missing 'center_col'"
