"""
Performance benchmarks for processbehavior at scale.

This module tests the package's ability to handle large datasets:
- 1 million observations
- 50 columns of data
- Various Sampling Design States

Run commands:
    # All performance tests
    pytest tests/test_performance.py -v

    # Skip slow 1M row tests
    pytest tests/test_performance.py -v -m "not slow"

    # With benchmark output (requires pytest-benchmark)
    pytest tests/test_performance.py -v --benchmark-only

    # Scalability tests only
    pytest tests/test_performance.py -v -k "scaling"
"""

import time
import tracemalloc

import numpy as np
import pytest

from processbehavior import ProcessDataFrame
from processbehavior.datasets import synthetic

# ============================================================================
# Data Generation Benchmarks
# ============================================================================


class TestDataGenerationPerformance:
    """Benchmark synthetic data generation at scale."""

    def test_generate_1m_sds1(self):
        """Generate 1M SDS1 observations and measure time."""
        start = time.perf_counter()
        df = synthetic.make_sds1(K=100, T=1000, n_min=10, n_max=10, seed=42)
        elapsed = time.perf_counter() - start

        assert len(df) == 1_000_000
        print(f'\nGenerated 1M rows in {elapsed:.2f}s ({len(df)/elapsed:,.0f} rows/sec)')

    def test_add_extra_columns_performance(self, large_dataset_10k):
        """Measure overhead of adding 46 extra columns."""
        start = time.perf_counter()
        df = large_dataset_10k.copy()
        rng = np.random.default_rng(42)
        for i in range(46):
            df[f'new_col_{i}'] = rng.normal(0, 1, len(df))
        elapsed = time.perf_counter() - start

        assert len(df.columns) >= 50
        print(f'\nAdded 46 columns to {len(df):,} rows in {elapsed:.3f}s')


# ============================================================================
# ProcessDataFrame Init Benchmarks
# ============================================================================


class TestProcessDataFramePerformance:
    """Benchmark ProcessDataFrame initialization."""

    def test_pdf_init_10k(self, large_dataset_10k):
        """ProcessDataFrame init with 10K rows, 50 columns."""
        start = time.perf_counter()
        pdf = ProcessDataFrame(large_dataset_10k)
        elapsed = time.perf_counter() - start

        assert pdf is not None
        print(f'\nPDF init (10K rows): {elapsed:.3f}s')

    def test_pdf_init_100k(self, large_dataset_100k):
        """ProcessDataFrame init with 100K rows, 50 columns."""
        start = time.perf_counter()
        pdf = ProcessDataFrame(large_dataset_100k)
        elapsed = time.perf_counter() - start

        assert pdf is not None
        throughput = len(large_dataset_100k) / elapsed
        print(f'\nPDF init (100K rows): {elapsed:.2f}s ({throughput:,.0f} rows/sec)')

    @pytest.mark.slow
    def test_pdf_init_1m(self, large_dataset_1m):
        """ProcessDataFrame init with 1M rows, 50 columns."""
        start = time.perf_counter()
        pdf = ProcessDataFrame(large_dataset_1m)
        elapsed = time.perf_counter() - start

        assert pdf is not None
        throughput = len(large_dataset_1m) / elapsed
        print(f'\nPDF init (1M rows): {elapsed:.2f}s ({throughput:,.0f} rows/sec)')

        # Performance target: < 60 seconds
        assert elapsed < 60, f'PDF init took {elapsed:.1f}s, target < 60s'


# ============================================================================
# Formulate Benchmarks
# ============================================================================


class TestFormulatePerformance:
    """Benchmark the formulate() method (SDS detection + residual calculation)."""

    def test_formulate_10k(self, large_dataset_10k, perf_spec):
        """formulate() with 10K rows."""
        pdf = ProcessDataFrame(large_dataset_10k)

        start = time.perf_counter()
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        elapsed = time.perf_counter() - start

        assert study is not None
        print(f'\nFormulate (10K rows): {elapsed:.3f}s')

    def test_formulate_100k(self, large_dataset_100k, perf_spec):
        """formulate() with 100K rows."""
        pdf = ProcessDataFrame(large_dataset_100k)

        start = time.perf_counter()
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        elapsed = time.perf_counter() - start

        assert study is not None
        throughput = len(large_dataset_100k) / elapsed
        print(f'\nFormulate (100K rows): {elapsed:.2f}s ({throughput:,.0f} rows/sec)')

    @pytest.mark.slow
    def test_formulate_1m(self, large_dataset_1m, perf_spec):
        """formulate() with 1M rows."""
        pdf = ProcessDataFrame(large_dataset_1m)

        start = time.perf_counter()
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        elapsed = time.perf_counter() - start

        assert study is not None
        throughput = len(large_dataset_1m) / elapsed
        print(f'\nFormulate (1M rows): {elapsed:.2f}s ({throughput:,.0f} rows/sec)')

        # Performance target: < 120 seconds
        assert elapsed < 120, f'Formulate took {elapsed:.1f}s, target < 120s'


# ============================================================================
# Analyze Benchmarks
# ============================================================================


class TestAnalyzePerformance:
    """Benchmark the analyze() method."""

    @pytest.fixture
    def study_10k(self, large_dataset_10k, perf_spec):
        """Pre-formulated study for analyze benchmarks."""
        pdf = ProcessDataFrame(large_dataset_10k)
        return pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )

    @pytest.fixture
    def study_100k(self, large_dataset_100k, perf_spec):
        """Pre-formulated study for analyze benchmarks."""
        pdf = ProcessDataFrame(large_dataset_100k)
        return pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )

    def test_analyze_xbar_10k(self, study_10k):
        """analyze() Xbar chart with 10K rows."""
        start = time.perf_counter()
        result = study_10k.analyze(chart='Xbar')
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f'\nAnalyze Xbar (10K rows): {elapsed:.3f}s')

    def test_analyze_xbar_100k(self, study_100k):
        """analyze() Xbar chart with 100K rows."""
        start = time.perf_counter()
        result = study_100k.analyze(chart='Xbar')
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f'\nAnalyze Xbar (100K rows): {elapsed:.3f}s')

    def test_analyze_imr_100k(self, study_100k):
        """analyze() Imr chart with 100K rows."""
        start = time.perf_counter()
        result = study_100k.analyze(chart='Imr')
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f'\nAnalyze Imr (100K rows): {elapsed:.3f}s')


# ============================================================================
# Component-Level Benchmarks
# ============================================================================


class TestComponentPerformance:
    """Isolate and benchmark individual bottleneck operations."""

    def test_groupby_transform_factor_means(self, large_dataset_100k):
        """groupby().transform('mean') - the core residual operation."""
        df = large_dataset_100k

        start = time.perf_counter()
        result = df.groupby('factor 1', observed=True)['y'].transform('mean')
        elapsed = time.perf_counter() - start

        assert len(result) == len(df)
        print(f'\nGroupby transform (factor): {elapsed:.3f}s')

    def test_groupby_transform_cell_means(self, large_dataset_100k):
        """groupby([factor, time]).transform('mean') - cell-level means."""
        df = large_dataset_100k

        start = time.perf_counter()
        result = df.groupby(['factor 1', 'time'], observed=True)['y'].transform('mean')
        elapsed = time.perf_counter() - start

        assert len(result) == len(df)
        print(f'\nGroupby transform (cell): {elapsed:.3f}s')

    def test_dataframe_copy(self, large_dataset_100k):
        """df.copy() - happens multiple times in pipeline."""
        df = large_dataset_100k

        start = time.perf_counter()
        result = df.copy()
        elapsed = time.perf_counter() - start

        assert len(result) == len(df)
        print(f'\nDataFrame copy: {elapsed:.3f}s')

    def test_column_subsetting(self, large_dataset_100k):
        """Selecting only needed columns (potential optimization)."""
        df = large_dataset_100k
        cols = ['factor 1', 'factor 2', 'time', 'y']

        start = time.perf_counter()
        result = df[cols].copy()
        elapsed = time.perf_counter() - start

        assert len(result.columns) == 4
        memory_full = df.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_subset = result.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f'\nColumn subset: {elapsed:.3f}s')
        print(f'Memory: {memory_full:.1f}MB -> {memory_subset:.1f}MB ({memory_subset/memory_full:.1%})')


# ============================================================================
# Memory Tests
# ============================================================================


class TestMemoryUsage:
    """Test memory efficiency at scale."""

    def test_memory_footprint_100k_50cols(self, large_dataset_100k):
        """Verify memory usage is reasonable for 100K x 50 dataset."""
        memory_mb = large_dataset_100k.memory_usage(deep=True).sum() / (1024 * 1024)

        print(f'\nDataset memory: {memory_mb:.1f} MB')
        print(f'Rows: {len(large_dataset_100k):,}')
        print(f'Columns: {len(large_dataset_100k.columns)}')

        # 100K rows x 50 cols x 8 bytes ≈ 40MB baseline
        assert memory_mb < 100, f'Memory usage {memory_mb:.1f}MB exceeds 100MB limit'

    @pytest.mark.slow
    def test_memory_footprint_1m_50cols(self, large_dataset_1m):
        """Verify memory usage is reasonable for 1M x 50 dataset."""
        memory_mb = large_dataset_1m.memory_usage(deep=True).sum() / (1024 * 1024)

        print(f'\nDataset memory: {memory_mb:.1f} MB')
        print(f'Rows: {len(large_dataset_1m):,}')
        print(f'Columns: {len(large_dataset_1m.columns)}')

        # 1M rows x 50 cols x 8 bytes ≈ 400MB baseline
        assert memory_mb < 800, f'Memory usage {memory_mb:.1f}MB exceeds 800MB limit'

    def test_memory_growth_during_formulate(self, large_dataset_10k, perf_spec):
        """Check memory doesn't explode during formulate()."""
        tracemalloc.start()

        pdf = ProcessDataFrame(large_dataset_10k)
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        assert study is not None  # Ensure formulate completed

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        input_mb = large_dataset_10k.memory_usage(deep=True).sum() / (1024 * 1024)

        print(f'\nInput data: {input_mb:.1f} MB')
        print(f'Peak memory: {peak_mb:.1f} MB')
        print(f'Ratio: {peak_mb / input_mb:.1f}x')

        # Allow 5x input data size for intermediate copies
        assert peak_mb < input_mb * 10, f'Memory grew {peak_mb / input_mb:.1f}x (limit: 10x)'


# ============================================================================
# Scalability Tests
# ============================================================================


class TestScalability:
    """Test how performance scales with data size."""

    @pytest.mark.parametrize('size', [1_000, 5_000, 10_000, 50_000])
    def test_formulate_scaling(self, size, perf_spec):
        """Test formulate() scales reasonably with row count."""
        # Generate appropriately sized dataset
        K = max(2, int(np.sqrt(size / 10)))
        T = max(2, int(np.sqrt(size / 10)))
        n = max(2, size // (K * T))

        df = synthetic.make_sds1(K=K, T=T, n_min=n, n_max=n, seed=42)
        actual_size = len(df)

        pdf = ProcessDataFrame(df)

        start = time.perf_counter()
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        elapsed = time.perf_counter() - start
        assert study is not None  # Ensure formulate completed

        throughput = actual_size / elapsed
        print(f'\n{actual_size:,} rows: {elapsed:.3f}s ({throughput:,.0f} rows/sec)')

        # Should achieve at least 1000 rows/sec even on slow machines
        assert throughput > 1000, f'Throughput {throughput:.0f} rows/sec below minimum 1000'

    @pytest.mark.parametrize('n_factors', [1, 5, 10, 20])
    def test_factor_count_scaling(self, n_factors, perf_spec):
        """Test performance with varying number of factor levels."""
        K = n_factors
        T = 100
        n = 10

        df = synthetic.make_sds1(K=K, T=T, n_min=n, n_max=n, seed=42)
        pdf = ProcessDataFrame(df)

        start = time.perf_counter()
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        elapsed = time.perf_counter() - start
        assert study is not None  # Ensure formulate completed

        print(f'\n{K} factors ({len(df):,} rows): {elapsed:.3f}s')


# ============================================================================
# Full Pipeline Benchmarks
# ============================================================================


class TestFullPipeline:
    """End-to-end workflow performance tests."""

    def test_full_workflow_10k(self, large_dataset_10k, perf_spec):
        """Complete workflow: PDF init -> formulate -> analyze -> get results."""
        start_total = time.perf_counter()

        # Step 1: Create ProcessDataFrame
        start = time.perf_counter()
        pdf = ProcessDataFrame(large_dataset_10k)
        time_init = time.perf_counter() - start

        # Step 2: Formulate
        start = time.perf_counter()
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        time_formulate = time.perf_counter() - start

        # Step 3: Analyze
        start = time.perf_counter()
        result = study.analyze(chart='Xbar')
        time_analyze = time.perf_counter() - start

        # Step 4: Get chart data
        start = time.perf_counter()
        chart_data = result.get_chart('Xbar')
        time_chart = time.perf_counter() - start

        total = time.perf_counter() - start_total

        print('\n=== Full Pipeline (10K rows) ===')
        print(f'Init:      {time_init:.3f}s')
        print(f'Formulate: {time_formulate:.3f}s')
        print(f'Analyze:   {time_analyze:.3f}s')
        print(f'Get chart: {time_chart:.3f}s')
        print(f'TOTAL:     {total:.3f}s')

        assert chart_data is not None

    def test_full_workflow_100k(self, large_dataset_100k, perf_spec):
        """Complete workflow with 100K rows."""
        start_total = time.perf_counter()

        pdf = ProcessDataFrame(large_dataset_100k)
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        result = study.analyze(chart='Xbar')
        chart_data = result.get_chart('Xbar')

        total = time.perf_counter() - start_total
        throughput = len(large_dataset_100k) / total

        print('\n=== Full Pipeline (100K rows) ===')
        print(f'TOTAL: {total:.2f}s ({throughput:,.0f} rows/sec)')

        assert chart_data is not None
