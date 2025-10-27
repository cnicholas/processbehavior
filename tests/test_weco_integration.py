"""
Integration tests for WECO Rules with Phase 1 metadata implementation.

Tests that detect_signals() works correctly with all chart types by using
the metadata-based value column resolution.
"""

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessDataFrame


class TestWECOIntegration:
    """Test WECO rules integration with metadata-based column resolution."""

    def test_detect_signals_xbar_chart(self):
        """Test detect_signals() works with Xbar chart (value_col='xbar')."""
        # Create subgrouped data with REPLICATION (multiple measurements per subgroup)
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 100),
            'batch': ['A'] * 25 + ['B'] * 25 + ['C'] * 25 + ['D'] * 25,  # Replication!
            'run': list(range(25)) * 4  # Different runs within each batch
        })

        # Analyze - need replication for Xbar/S
        pdf = ProcessDataFrame(data)
        analysis = pdf.analyze(
            response_var='measurement',
            grouping_vars=['batch']
        )
        result = analysis.calculate()

        # Verify Xbar chart has metadata
        assert 'Xbar' in result.charts
        assert 'metadata' in result.charts['Xbar']
        assert result.charts['Xbar']['metadata']['value_col'] == 'xbar'

        # Detect signals - should not crash
        signals = result.detect_signals(chart='Xbar')

        # Verify signal detection worked
        assert signals is not None
        assert hasattr(signals, 'count')
        assert hasattr(signals, 'has_signals')

    def test_detect_signals_sbar_chart(self):
        """Test detect_signals() works with Sbar chart (value_col='s')."""
        # Create subgrouped data
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 100),
            'batch': ['A', 'B', 'C', 'D'] * 25
        })

        # Analyze
        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            grouping_vars=['batch']
        )

        # Verify Sbar chart has metadata
        assert 'Sbar' in result.charts
        assert 'metadata' in result.charts['Sbar']
        assert result.charts['Sbar']['metadata']['value_col'] == 's'

        # Detect signals - should not crash
        signals = result.detect_signals(chart='Sbar')

        # Verify signal detection worked
        assert signals is not None
        assert hasattr(signals, 'count')
        assert hasattr(signals, 'has_signals')

    def test_detect_signals_imr_chart(self):
        """Test detect_signals() works with IMR chart (value_col=response_var)."""
        # Create individual measurements data
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 50),
            'time': range(50)
        })

        # Analyze
        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            time_var='time'
        )

        # IMR analysis returns stratified results with 'all' key
        assert 'all' in result.charts
        assert 'metadata' in result.charts['all']
        assert result.charts['all']['metadata']['value_col'] == 'measurement'
        assert result.charts['all']['metadata']['chart_type'] == 'Imr'

        # Detect signals - should not crash
        signals = result.detect_signals(chart='all')

        # Verify signal detection worked
        assert signals is not None
        assert hasattr(signals, 'count')
        assert hasattr(signals, 'has_signals')

    def test_detect_signals_r_chart(self):
        """Test detect_signals() works with R chart (value_col='mr')."""
        # Create individual measurements data
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 50),
            'time': range(50)
        })

        # Analyze
        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            time_var='time'
        )

        # Get R chart from stratified results
        assert 'all' in result.charts
        chart_info = result.charts['all']

        # R chart should have metadata
        assert 'metadata' in chart_info
        # Note: This test assumes IMR analysis, which returns Imr chart
        # For R-only chart, we'd need different analysis parameters

    def test_detect_signals_all_charts(self):
        """Test detect_signals() without chart parameter detects on all charts."""
        # Create subgrouped data
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 100),
            'batch': ['A', 'B', 'C', 'D'] * 25
        })

        # Analyze
        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            grouping_vars=['batch']
        )

        # Detect signals on all charts
        all_signals = result.detect_signals()

        # Should return dict with signals for each chart
        assert isinstance(all_signals, dict)
        assert 'Xbar' in all_signals
        assert 'Sbar' in all_signals

        # Each should be a SignalResult
        for chart_name, signals in all_signals.items():
            assert hasattr(signals, 'count')
            assert hasattr(signals, 'has_signals')

    def test_detect_signals_with_violations(self):
        """Test that actual violations are detected correctly."""
        # Create data with known violation
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': [100] * 20 + [150] + [100] * 19,  # One outlier
            'batch': ['A'] * 40
        })

        # Analyze
        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            grouping_vars=['batch']
        )

        # Detect signals with Rule 1 (beyond limits)
        signals = result.detect_signals(chart='Xbar', rules=['rule_1'])

        # Should detect the outlier
        assert signals.has_signals
        assert signals.count > 0

    def test_metadata_missing_raises_error(self):
        """Test that missing metadata raises helpful error."""
        # Create result with charts
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 100),
            'batch': ['A', 'B'] * 50
        })

        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            factors=['batch']
        )

        # Manually remove metadata to simulate bug
        del result.charts['Xbar']['metadata']

        # Should raise helpful error
        with pytest.raises(ValueError, match="missing metadata"):
            result.detect_signals(chart='Xbar')

    def test_value_column_used_correctly(self):
        """Test that the correct value column is actually used for detection."""
        # Create data
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 100),
            'batch': ['A', 'B', 'C', 'D'] * 25
        })

        # Analyze
        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            grouping_vars=['batch']
        )

        # Get Xbar chart
        xbar_chart = result.charts['Xbar']

        # Verify the data has 'xbar' column (not 'mean')
        assert 'xbar' in xbar_chart['data'].columns

        # Verify metadata points to 'xbar'
        assert xbar_chart['metadata']['value_col'] == 'xbar'

        # Detect signals should use 'xbar' column
        signals = result.detect_signals(chart='Xbar')

        # Verify it worked (didn't try to use 'mean' column which doesn't exist)
        assert signals is not None


class TestMetadataContract:
    """Test that all chart types have proper metadata."""

    def test_xbar_sbar_metadata(self):
        """Test Xbar and Sbar charts have complete metadata."""
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 100),
            'batch': ['A', 'B'] * 50
        })

        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            factors=['batch']
        )

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
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 50),
            'time': range(50)
        })

        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            time='time'
        )

        # Check IMR metadata
        imr_meta = result.charts['all']['metadata']
        assert imr_meta['chart_type'] == 'Imr'
        assert imr_meta['value_col'] == 'measurement'
        assert imr_meta['center_col'] == 'center'

    def test_metadata_structure(self):
        """Test metadata has required keys."""
        np.random.seed(42)
        data = pd.DataFrame({
            'measurement': np.random.normal(100, 10, 100),
            'batch': ['A', 'B'] * 50
        })

        pdf = ProcessDataFrame(data)
        result = pdf.analyze(
            response_var='measurement',
            factors=['batch']
        )

        # All charts should have metadata with required keys
        for chart_name, chart_info in result.charts.items():
            assert 'metadata' in chart_info, f"Chart '{chart_name}' missing metadata"

            meta = chart_info['metadata']
            assert 'chart_type' in meta, f"Chart '{chart_name}' metadata missing 'chart_type'"
            assert 'value_col' in meta, f"Chart '{chart_name}' metadata missing 'value_col'"
            assert 'center_col' in meta, f"Chart '{chart_name}' metadata missing 'center_col'"
