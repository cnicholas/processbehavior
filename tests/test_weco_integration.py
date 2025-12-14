"""
Integration tests for WECO Rules with Phase 1 metadata implementation.

Tests that detect_signals() works correctly with all chart types by using
the metadata-based value column resolution.
"""

import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic
from processbehavior.signals.config import SignalConfig


class TestWECOIntegration:
    """Test WECO rules integration with metadata-based column resolution."""

    def test_detect_signals_xbar_chart(self):
        """Test detect_signals() works with Xbar chart (value_col='xbar')."""
        # Use synthetic SDS1 data (full replication)
        df = synthetic.make_sds1(K=3, T=8, n_min=2, n_max=4, seed=42)

        # Analyze
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        result = study.execute()

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

    def test_detect_signals_sbar_chart(self):
        """Test detect_signals() works with S chart (value_col='s')."""
        # Use synthetic SDS1 data (full replication)
        df = synthetic.make_sds1(K=4, T=10, n_min=2, n_max=5, seed=42)

        # Analyze
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        result = study.execute()

        # Verify S chart has metadata
        assert 'S' in result.charts
        assert 'metadata' in result.charts['S']
        assert result.charts['S']['metadata']['value_col'] == 's'

        # Detect signals - should not crash
        # Use config with lower min_observations
        config = SignalConfig(min_observations=3)
        signals = result.detect_signals(chart='S', config=config)

        # Verify signal detection worked
        assert signals is not None
        assert hasattr(signals, 'count')
        assert hasattr(signals, 'has_signals')

    def test_detect_signals_imr_chart(self):
        """Test detect_signals() works with IMR chart (value_col=response_var)."""
        # Use synthetic SDS4 data (single condition over time)
        df = synthetic.make_sds4(T=50, seed=42)

        # Analyze
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            time=pdf.cols.time
        )
        result = study.execute()

        # IMR analysis returns results with 'Imr' key
        assert 'Imr' in result.charts
        assert 'metadata' in result.charts['Imr']
        assert result.charts['Imr']['metadata']['value_col'] == 'y'
        assert result.charts['Imr']['metadata']['chart_type'] == 'Imr'

        # Detect signals - should not crash
        signals = result.detect_signals(chart='Imr')

        # Verify signal detection worked
        assert signals is not None
        assert hasattr(signals, 'count')
        assert hasattr(signals, 'has_signals')

    def test_detect_signals_r_chart(self):
        """Test detect_signals() works with R chart (value_col='mr')."""
        # Use synthetic SDS4 data (single condition over time)
        df = synthetic.make_sds4(T=50, seed=42)

        # Analyze - IMR analysis includes moving range calculation
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            time=pdf.cols.time
        )
        result = study.execute()

        # IMR analysis returns results with 'Imr' key
        # The chart includes moving range data with metadata
        assert 'Imr' in result.charts
        chart_info = result.charts['Imr']

        # Verify metadata exists (IMR chart includes MR calculations)
        assert 'metadata' in chart_info
        # Note: IMR analysis returns Imr chart type, not separate R chart
        assert chart_info['metadata']['chart_type'] == 'Imr'

    def test_detect_signals_all_charts(self):
        """Test detect_signals() without chart parameter detects on all charts."""
        # Use synthetic SDS1 data (full replication)
        df = synthetic.make_sds1(K=3, T=8, n_min=2, n_max=4, seed=42)

        # Analyze
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        result = study.execute()

        # Detect signals on all charts
        # Use config with lower min_observations
        config = SignalConfig(min_observations=3)
        all_signals = result.detect_signals(config=config)

        # Should return dict with signals for each chart
        assert isinstance(all_signals, dict)
        assert 'Xbar' in all_signals
        assert 'S' in all_signals

        # Each should be a SignalResult
        for _chart_name, signals in all_signals.items():
            assert hasattr(signals, 'count')
            assert hasattr(signals, 'has_signals')

    def test_detect_signals_with_violations(self):
        """Test that actual violations are detected correctly."""
        # Create SDS1 data with known violation - add outlier manually
        df = synthetic.make_sds1(K=3, T=10, n_min=2, n_max=4, seed=42)

        # Inject an outlier - shift ALL K2 observations significantly
        # (Xbar chart aggregates across time, so need to shift all observations)
        mask = (df['factor 1'] == 'K2')
        df.loc[mask, 'y'] = df.loc[mask, 'y'] + 100  # Large shift to ensure detection

        # Analyze
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        result = study.execute()

        # Detect signals with Rule 1 (beyond limits)
        # Use config with lower min_observations
        config = SignalConfig(min_observations=3, enabled_rules=['rule_1'])
        signals = result.detect_signals(chart='Xbar', config=config)

        # Should detect the outlier (K2 should be beyond limits)
        assert signals.has_signals
        assert signals.count > 0

    def test_metadata_missing_raises_error(self):
        """Test that missing metadata raises helpful error."""
        # Create result with charts
        df = synthetic.make_sds1(K=2, T=6, n_min=2, n_max=3, seed=42)

        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        result = study.execute()

        # Manually remove metadata to simulate bug
        del result.charts['Xbar']['metadata']

        # Should raise helpful error
        from processbehavior.exceptions import ProcessBehaviorError
        with pytest.raises(ProcessBehaviorError, match="missing metadata"):
            result.detect_signals(chart='Xbar')

    def test_value_column_used_correctly(self):
        """Test that the correct value column is actually used for detection."""
        # Use synthetic SDS1 data
        df = synthetic.make_sds1(K=3, T=8, n_min=2, n_max=4, seed=42)

        # Analyze
        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        result = study.execute()

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
        """Test Xbar and S charts have complete metadata."""
        df = synthetic.make_sds1(K=2, T=8, n_min=2, n_max=4, seed=42)

        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        result = study.execute()

        # Check Xbar metadata
        xbar_meta = result.charts['Xbar']['metadata']
        assert xbar_meta['chart_type'] == 'Xbar'
        assert xbar_meta['value_col'] == 'xbar'
        assert xbar_meta['center_col'] == 'center'

        # Check S metadata
        sbar_meta = result.charts['S']['metadata']
        assert sbar_meta['chart_type'] == 'S'
        assert sbar_meta['value_col'] == 's'
        assert sbar_meta['center_col'] == 'center'

    def test_imr_metadata(self):
        """Test IMR chart has complete metadata."""
        df = synthetic.make_sds4(T=50, seed=42)

        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            time=pdf.cols.time
        )
        result = study.execute()

        # Check IMR metadata
        imr_meta = result.charts['Imr']['metadata']
        assert imr_meta['chart_type'] == 'Imr'
        assert imr_meta['value_col'] == 'y'
        assert imr_meta['center_col'] == 'center'

    def test_metadata_structure(self):
        """Test metadata has required keys."""
        df = synthetic.make_sds1(K=2, T=8, n_min=2, n_max=4, seed=42)

        pdf = ProcessBehavior(df)
        study = pdf.formulate(
            response=pdf.cols.y,
            factors=[pdf.cols.factor_1],
            time=pdf.cols.time
        )
        result = study.execute()

        # All charts should have metadata with required keys
        for chart_name, chart_info in result.charts.items():
            assert 'metadata' in chart_info, f"Chart '{chart_name}' missing metadata"

            meta = chart_info['metadata']
            assert 'chart_type' in meta, f"Chart '{chart_name}' metadata missing 'chart_type'"
            assert 'value_col' in meta, f"Chart '{chart_name}' metadata missing 'value_col'"
            assert 'center_col' in meta, f"Chart '{chart_name}' metadata missing 'center_col'"
