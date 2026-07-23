"""
Performance benchmarks for processbehavior at scale.

Covers both residual regimes: SDS-1 (replicated → exact R2, O(N)) and the SDS-2 worst case
(no replication → MA2 sort, O(N log N)), across 10K/100K/1M rows × 50 columns, for init /
formulate / execute / memory. Assertions are drift-aware without being hardware-flaky:
loose sanity ceilings + hardware-independent *scaling shape* (per-row time stays flat as N
grows ⇒ linear). Absolute per-machine drift is reported against benchmarks/baseline.json and
only fails the build under PB_PERF_STRICT=1.

Run commands:
    pytest tests/test_performance.py -v -s              # all (incl. @slow 1M); -s to see prints
    pytest tests/test_performance.py -v -m "not slow"   # skip 1M
    pytest tests/test_performance.py -v -m benchmark -s  # the consolidated report + drift table
    python scripts/benchmark.py [--quick] [--update-baseline]  # standalone report (incl. 1M)
"""

import os
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

# scripts/ (namespace pkg) holds the shared benchmark harness; conftest also adds this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic
from scripts import benchmark as bench

# ============================================================================
# Data Generation Benchmarks
# ============================================================================


class TestDataGenerationPerformance:
    """Benchmark synthetic data generation at scale."""

    def test_generate_1m_sds1(self):
        """Generate 1M SDS1 observations and measure time."""
        start = time.perf_counter()
        # K1=100 × K2=2 × T=500 × n=10 = 1,000,000
        df = synthetic.make_design(1, K1=100, K2=2, T=500, n_min=10, n_max=10, seed=42)
        elapsed = time.perf_counter() - start

        assert len(df) == 1_000_000
        print(f'\nGenerated 1M rows in {elapsed:.2f}s ({len(df) / elapsed:,.0f} rows/sec)')

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
# ProcessBehavior Init Benchmarks
# ============================================================================


class TestProcessBehaviorPerformance:
    """Benchmark ProcessBehavior initialization."""

    def test_pdf_init_10k(self, large_dataset_10k):
        """ProcessBehavior init with 10K rows, 50 columns."""
        start = time.perf_counter()
        pdf = ProcessBehavior(large_dataset_10k)
        elapsed = time.perf_counter() - start

        assert pdf is not None
        print(f'\nPDF init (10K rows): {elapsed:.3f}s')

    def test_pdf_init_100k(self, large_dataset_100k):
        """ProcessBehavior init with 100K rows, 50 columns."""
        start = time.perf_counter()
        pdf = ProcessBehavior(large_dataset_100k)
        elapsed = time.perf_counter() - start

        assert pdf is not None
        throughput = len(large_dataset_100k) / elapsed
        print(f'\nPDF init (100K rows): {elapsed:.2f}s ({throughput:,.0f} rows/sec)')

    @pytest.mark.slow
    def test_pdf_init_1m(self, large_dataset_1m):
        """ProcessBehavior init with 1M rows, 50 columns."""
        start = time.perf_counter()
        pdf = ProcessBehavior(large_dataset_1m)
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
        pdf = ProcessBehavior(large_dataset_10k)

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
        pdf = ProcessBehavior(large_dataset_100k)

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
        pdf = ProcessBehavior(large_dataset_1m)

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
        pdf = ProcessBehavior(large_dataset_10k)
        return pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )

    @pytest.fixture
    def study_100k(self, large_dataset_100k, perf_spec):
        """Pre-formulated study for analyze benchmarks."""
        pdf = ProcessBehavior(large_dataset_100k)
        return pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )

    def test_analyze_xbar_10k(self, study_10k):
        """analyze() Xbar chart with 10K rows."""
        start = time.perf_counter()
        result = study_10k.execute(chart='Xbar')
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f'\nAnalyze Xbar (10K rows): {elapsed:.3f}s')

    def test_analyze_xbar_100k(self, study_100k):
        """analyze() Xbar chart with 100K rows."""
        start = time.perf_counter()
        result = study_100k.execute(chart='Xbar')
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f'\nAnalyze Xbar (100K rows): {elapsed:.3f}s')

    def test_analyze_xmr_100k(self, study_100k):
        """analyze() XmR chart with 100K rows."""
        start = time.perf_counter()
        # XmR with factors now requires explicit 'by' parameter
        result = study_100k.execute(chart='X', by=['factor 1'])
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f'\nAnalyze XmR (100K rows): {elapsed:.3f}s')


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
        print(f'Memory: {memory_full:.1f}MB -> {memory_subset:.1f}MB ({memory_subset / memory_full:.1%})')


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

        pdf = ProcessBehavior(large_dataset_10k)
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

        # Intermediate copies during formulate. Measured ~3.4-3.8x across sizes; cap at
        # 6x for headroom. (Previously 10x in code but "5x" in docs — reconciled here.)
        assert peak_mb < input_mb * 6, f'Memory grew {peak_mb / input_mb:.1f}x (limit: 6x)'


# ============================================================================
# Scalability Tests
# ============================================================================


class TestScalability:
    """Test how performance scales with data size."""

    @pytest.mark.parametrize('size', [1_000, 5_000, 10_000, 50_000])
    def test_formulate_scaling(self, size, perf_spec):
        """Test formulate() scales reasonably with row count."""
        # Generate appropriately sized dataset
        # K1 × K2 × T × n = size
        K1 = max(2, int(np.sqrt(size / 20)))
        K2 = 2
        T = max(2, int(np.sqrt(size / 20)))
        n = max(2, size // (K1 * K2 * T))

        df = synthetic.make_design(1, K1=K1, K2=K2, T=T, n_min=n, n_max=n, seed=42)
        actual_size = len(df)

        pdf = ProcessBehavior(df)

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

        # Loose sanity floor (won't flake across hardware); the real linearity guard is
        # TestScalingShape, which checks per-row time stays flat as N grows.
        assert throughput > 10_000, f'Throughput {throughput:.0f} rows/sec below floor 10k'

    @pytest.mark.parametrize('n_factors', [1, 5, 10, 20])
    def test_factor_count_scaling(self, n_factors, perf_spec):
        """Test performance with varying number of factor levels."""
        K1 = n_factors
        K2 = 2
        T = 50
        n = 10

        df = synthetic.make_design(1, K1=K1, K2=K2, T=T, n_min=n, n_max=n, seed=42)
        pdf = ProcessBehavior(df)

        start = time.perf_counter()
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        elapsed = time.perf_counter() - start
        assert study is not None  # Ensure formulate completed

        print(f'\n{K1} factors ({len(df):,} rows): {elapsed:.3f}s')


# ============================================================================
# Full Pipeline Benchmarks
# ============================================================================


class TestFullPipeline:
    """End-to-end workflow performance tests."""

    def test_full_workflow_10k(self, large_dataset_10k, perf_spec):
        """Complete workflow: PDF init -> formulate -> analyze -> get results."""
        start_total = time.perf_counter()

        # Step 1: Create ProcessBehavior
        start = time.perf_counter()
        pdf = ProcessBehavior(large_dataset_10k)
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
        result = study.execute(chart='Xbar')
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

        pdf = ProcessBehavior(large_dataset_100k)
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        result = study.execute(chart='Xbar')
        chart_data = result.get_chart('Xbar')

        total = time.perf_counter() - start_total
        throughput = len(large_dataset_100k) / total

        print('\n=== Full Pipeline (100K rows) ===')
        print(f'TOTAL: {total:.2f}s ({throughput:,.0f} rows/sec)')

        assert chart_data is not None


# ============================================================================
# Worst-Case Residual Path (SDS-2 / MA2 sort) — the historical coverage gap
# ============================================================================


@pytest.mark.slow
@pytest.mark.benchmark
class TestWorstCaseMA2:
    """Exercise the unreplicated (MA2, O(N log N)) residual path.

    The single-factor perf_spec collapses make_design(2) to SDS-1 (the (factor1×time) cell
    still holds both factor-2 levels), so the rest of this file only ever hits exact-R2.
    Using BOTH factors makes the cells singletons → MA2 sort — ~8x slower than exact-R2.
    """

    def test_formulate_ma2_confirms_path_100k(self):
        """SDS-2 formulate uses the MA2 path and stays well within a sanity ceiling."""
        df = bench.make_dataset(2, 100_000)
        pdf = ProcessBehavior(df)

        start = time.perf_counter()
        study = pdf.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
        elapsed = time.perf_counter() - start

        # Confirm we actually exercised the no-replication (MA2) path, not exact-R2.
        assert study.analytical_design_state.sds == 2, 'expected SDS-2 (no replication) → MA2'
        throughput = len(df) / elapsed
        print(f'\nMA2 formulate (100K rows): {elapsed:.2f}s ({throughput:,.0f} rows/sec)')
        assert elapsed < 30, f'MA2 formulate took {elapsed:.1f}s, sanity ceiling 30s'

    @pytest.mark.slow
    def test_formulate_ma2_1m(self):
        """SDS-2 (MA2 sort) formulate at 1M rows — the true worst case."""
        df = bench.make_dataset(2, 1_000_000)
        pdf = ProcessBehavior(df)

        start = time.perf_counter()
        study = pdf.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
        elapsed = time.perf_counter() - start

        assert study.analytical_design_state.sds == 2
        throughput = len(df) / elapsed
        print(f'\nMA2 formulate (1M rows): {elapsed:.1f}s ({throughput:,.0f} rows/sec)')
        assert elapsed < 300, f'MA2 formulate took {elapsed:.1f}s, sanity ceiling 300s'


class TestExecuteAtScale:
    """execute() at 1M rows (the rest of the file only measured execute up to 100K)."""

    @pytest.mark.slow
    def test_execute_xbar_1m(self, large_dataset_1m, perf_spec):
        pdf = ProcessBehavior(large_dataset_1m)
        study = pdf.formulate(
            response=perf_spec['response_var'],
            factors=perf_spec['rsg_vars'],
            time=perf_spec['time_var'],
        )
        start = time.perf_counter()
        result = study.execute(chart='Xbar')
        elapsed = time.perf_counter() - start

        assert result is not None
        throughput = len(large_dataset_1m) / elapsed
        print(f'\nExecute Xbar (1M rows): {elapsed:.2f}s ({throughput:,.0f} rows/sec)')
        assert elapsed < 60, f'Execute took {elapsed:.1f}s, sanity ceiling 60s'


# ============================================================================
# Scaling shape (hardware-independent complexity guard) + drift report
# ============================================================================


def _per_row_us(results, sds, op, size):
    """Microseconds per row for one (design, op, size) from a benchmark_results list."""
    for r in results:
        if r['sds'] == sds and r['op'] == op and r['size'] == size and r.get('seconds'):
            return r['seconds'] / r['rows'] * 1e6
    raise AssertionError(f'no result for sds{sds} {op} {size}')


@pytest.mark.slow
@pytest.mark.benchmark
class TestScalingShape:
    """Per-row time must stay roughly flat as N grows — catches super-linear regressions
    independent of hardware (an O(N^2) slip would blow the ratio up regardless of CPU)."""

    # exact-R2 is O(N) → flat; MA2 is O(N log N) → a mild rise. Generous cap absorbs both
    # plus GC/cache noise while still catching a true complexity change.
    MAX_RATIO = 3.0

    def test_exact_r2_scales_linearly(self, benchmark_results):
        small = _per_row_us(benchmark_results, 1, 'formulate', 10_000)
        large = _per_row_us(benchmark_results, 1, 'formulate', 100_000)
        print(f'\nSDS-1 formulate per-row: {small:.2f}us (10K) -> {large:.2f}us (100K), ratio {large / small:.2f}')
        assert large <= small * self.MAX_RATIO, f'exact-R2 per-row grew {large / small:.1f}x (>{self.MAX_RATIO}x)'

    def test_ma2_scales_near_linearly(self, benchmark_results):
        small = _per_row_us(benchmark_results, 2, 'formulate', 10_000)
        large = _per_row_us(benchmark_results, 2, 'formulate', 100_000)
        print(f'\nSDS-2 MA2 per-row: {small:.2f}us (10K) -> {large:.2f}us (100K), ratio {large / small:.2f}')
        assert large <= small * self.MAX_RATIO, f'MA2 per-row grew {large / small:.1f}x (>{self.MAX_RATIO}x)'


@pytest.mark.slow
@pytest.mark.benchmark
class TestBenchmarkReport:
    """Consolidated report + drift check against the committed same-machine baseline.

    Informational by default (prints + writes benchmarks/last_run.json); set PB_PERF_STRICT=1
    to make a regression beyond the tolerance fail the build (opt-in — CI never flakes on it).
    """

    def test_report_and_drift(self, benchmark_results):
        print(bench.format_report(benchmark_results))

        doc = bench.to_document(benchmark_results)
        bench.LAST_RUN.parent.mkdir(exist_ok=True)
        bench.LAST_RUN.write_text(__import__('json').dumps(doc, indent=2))

        baseline = bench.load_baseline()
        drift = bench.compare_to_baseline(benchmark_results, baseline)
        print('\nDrift vs baseline:\n' + '\n'.join(drift))

        if os.environ.get('PB_PERF_STRICT') == '1':
            regressions = [d for d in drift if 'REGRESSION' in d]
            assert not regressions, 'performance regression(s) vs baseline:\n' + '\n'.join(regressions)
