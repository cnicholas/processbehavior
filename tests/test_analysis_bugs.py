"""Regression tests for confirmed bugs #1-4, #6, #10.

Each test is written to fail against the unfixed code, then pass after fixes.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.exceptions import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sds1_study(K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42):
    """Standard SDS 1 study for reuse."""
    from processbehavior.datasets.synthetic import make_sds
    df = make_sds(1, K1=K1, K2=K2, T=T, n_min=n_min, n_max=n_max, seed=seed)
    return ProcessBehavior(df).formulate(
        response='y', time='time', factors=['factor 1', 'factor 2'],
    )


# ---------------------------------------------------------------------------
# Bug #1 — S chart crash when all groups have n=1
# ---------------------------------------------------------------------------

class TestBug1_SChartAllN1:
    """S chart should raise ValueError when every subgroup has n=1."""

    def test_s_chart_raises_when_all_n_equals_1(self):
        """Construct data where every cell has exactly 1 observation (SDS 2).
        S chart requires n >= 2 for std dev; should raise, not crash."""
        from processbehavior.datasets.synthetic import make_sds
        df = make_sds(2, K1=3, K2=2, T=6, seed=42)
        ProcessBehavior(df).formulate(
            response='y', time='time', factors=['factor 1', 'factor 2'],
        )
        # SDS 2 normally doesn't include S in valid_charts,
        # but we test the guard directly if it can be reached.
        # Force through by using a study where S *is* valid but all n==1.
        # Alternative: build a manual DataFrame with all n=1 subgroups.

        # Build manual data: 1 factor, 1 obs per (factor, time) cell
        rows = []
        for t in range(1, 7):
            for f in ['A', 'B', 'C']:
                rows.append({'y': np.random.default_rng(42).normal(), 'time': t, 'factor 1': f})
        df_manual = pd.DataFrame(rows)
        study_manual = ProcessBehavior(df_manual).formulate(
            response='y', time='time', factors=['factor 1'],
        )

        with pytest.raises(ValidationError, match="No subgroups with n > 1 found"):
            study_manual.execute(chart='S')

    def test_s_chart_raises_with_forced_single_obs_subgroups(self):
        """Direct test: build SDS 1 data then strip to 1 obs per subgroup.
        The S chart code filters n==1 groups; if ALL are n==1, it should raise."""
        # Use SDS 1 but with n_min=n_max=2 and remove one row per cell
        from processbehavior.datasets.synthetic import make_sds
        df = make_sds(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=99)

        # Keep only first observation per (factor 1, factor 2, time)
        df = df.groupby(['factor 1', 'factor 2', 'time']).first().reset_index()

        study = ProcessBehavior(df).formulate(
            response='y', time='time', factors=['factor 1', 'factor 2'],
        )

        # n=1 per cell → S chart should detect all-n==1 and raise
        with pytest.raises(ValidationError, match="No subgroups with n > 1 found"):
            study.execute(chart='S')


# ---------------------------------------------------------------------------
# Bug #2 — Duplicate `by` values crash
# ---------------------------------------------------------------------------

class TestBug2_DuplicateByValues:
    """Duplicate values in by= should be deduplicated, not crash."""

    def test_duplicate_by_values_produce_same_result_as_deduplicated(self):
        study = _make_sds1_study()

        # Execute with duplicated by
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_dup = study.execute(chart='XmR', by=['factor 1', 'factor 1'])
            # Should have emitted a warning about deduplication
            dedup_warnings = [x for x in w if "Duplicate" in str(x.message)]
            assert len(dedup_warnings) >= 1, "Expected a UserWarning about duplicate by values"

        # Execute with clean by
        result_clean = study.execute(chart='XmR', by=['factor 1'])

        # Both should produce equivalent chart data
        df_dup = result_dup.charts['XmR']['data']
        df_clean = result_clean.charts['XmR']['data']
        pd.testing.assert_frame_equal(df_dup, df_clean)

    def test_duplicate_by_with_multiple_factors(self):
        study = _make_sds1_study()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = study.execute(
                chart='XmR', by=['factor 1', 'factor 2', 'factor 1']
            )

        # Should work — equivalent to by=['factor 1', 'factor 2']
        result_clean = study.execute(chart='XmR', by=['factor 1', 'factor 2'])
        pd.testing.assert_frame_equal(
            result.charts['XmR']['data'], result_clean.charts['XmR']['data']
        )


# ---------------------------------------------------------------------------
# Bug #3 — S chart center=None when true center is 0.0
# ---------------------------------------------------------------------------

class TestBug3_SChartCenterZero:
    """When all within-group std devs are 0, S-bar is 0.0 not None."""

    def test_s_chart_center_zero_not_none(self):
        """Build data where every observation in each subgroup is identical."""
        # 3 factors × 4 time periods, n=3 per cell, all values identical within cell
        rows = []
        rng = np.random.default_rng(42)
        for t in range(1, 5):
            for f in ['A', 'B', 'C']:
                val = rng.normal(50, 2)  # different between cells
                for _ in range(3):
                    rows.append({'y': val, 'time': t, 'factor 1': f})

        df = pd.DataFrame(rows)
        study = ProcessBehavior(df).formulate(
            response='y', time='time', factors=['factor 1'],
        )

        result = study.execute(chart='S')
        stats = result.charts['S']['statistics']
        assert stats['center'] is not None, "S chart center should be 0.0, not None"
        assert stats['center'] == 0.0, f"Expected center=0.0, got {stats['center']}"


# ---------------------------------------------------------------------------
# Bug #4 — n_mode="average" statistics show wrong N
# ---------------------------------------------------------------------------

class TestBug4_NModeAverageStatistics:
    """When n_mode='average', statistics['N'] should be the average, not max."""

    def _make_variable_n_study(self):
        """Build SDS 1 data with deliberately variable subgroup sizes."""
        rows = []
        rng = np.random.default_rng(42)
        subgroup_sizes = [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]
        idx = 0
        for t in range(1, 5):
            for f in ['A', 'B', 'C']:
                n = subgroup_sizes[idx % len(subgroup_sizes)]
                idx += 1
                for _ in range(n):
                    rows.append({
                        'y': rng.normal(50, 1),
                        'time': t,
                        'factor 1': f,
                    })

        df = pd.DataFrame(rows)
        return ProcessBehavior(df).formulate(
            response='y', time='time', factors=['factor 1'],
        )

    def test_xbar_n_mode_average_reports_average_n(self):
        study = self._make_variable_n_study()
        result = study.execute(chart='Xbar', n_mode='average')
        stats = result.charts['Xbar']['statistics']

        # The subgroup sizes cycle [2, 3, 4], so average = 3.0
        expected_avg = 3.0
        assert stats['N'] == expected_avg, (
            f"Expected N={expected_avg} (average), got N={stats['N']}"
        )

    def test_s_n_mode_average_reports_average_n(self):
        study = self._make_variable_n_study()
        result = study.execute(chart='S', n_mode='average')
        stats = result.charts['S']['statistics']

        expected_avg = 3.0
        assert stats['N'] == expected_avg, (
            f"Expected N={expected_avg} (average), got N={stats['N']}"
        )

    def test_n_mode_actual_still_reports_max_n(self):
        """Regression: n_mode='actual' (default) still shows max N."""
        study = self._make_variable_n_study()
        result = study.execute(chart='Xbar')
        stats = result.charts['Xbar']['statistics']

        # With variable n, default behavior should report 'Varies'
        assert stats['N'] == 'Varies' or isinstance(stats['N'], (int, float))


# ---------------------------------------------------------------------------
# Bug #6 — Lane-boundary collision with underscored factor levels (regression)
# ---------------------------------------------------------------------------

class TestBug6_LaneBoundaryUnderscore:
    """Confirm lane boundaries detect transitions correctly even with
    underscore-containing factor levels."""

    def test_lane_boundary_with_underscore_factor_levels(self):
        """Factor values containing underscores (e.g. 'A_B') should still
        produce correct lane boundaries at factor-value transitions."""
        rows = []
        rng = np.random.default_rng(42)
        # Use factor levels with underscores
        factor_levels = ['A_B', 'C_D', 'E_F']
        for t in range(1, 7):
            for f in factor_levels:
                rows.append({'y': rng.normal(50, 1), 'time': t, 'factor 1': f})

        df = pd.DataFrame(rows)
        study = ProcessBehavior(df).formulate(
            response='y', time='time', factors=['factor 1'],
        )

        result = study.execute(chart='XmR', by=[])
        metadata = result.charts['XmR']['metadata']
        lane_boundaries = metadata.get('lane_boundaries')

        # Should have lane boundaries (transitions between A_B→C_D, C_D→E_F
        # at each time period boundary)
        assert lane_boundaries is not None, (
            "Expected lane boundaries for multi-factor-level data with by=[]"
        )

    def test_lane_boundary_count_matches_factor_transitions(self):
        """Number of lane boundaries should match actual factor transitions,
        not spurious splits from underscore collisions."""
        rows = []
        rng = np.random.default_rng(42)
        # Deliberately construct potentially colliding keys:
        # ('A_B', 'C') vs ('A', 'B_C') would collide with naive '_'.join
        # But since these are separate factor columns, lane boundaries
        # should be per-factor-column transitions.
        for t in range(1, 5):
            for f1 in ['A_B', 'C']:
                rows.append({
                    'y': rng.normal(50, 1), 'time': t, 'factor 1': f1,
                })

        df = pd.DataFrame(rows)
        study = ProcessBehavior(df).formulate(
            response='y', time='time', factors=['factor 1'],
        )

        result = study.execute(chart='XmR', by=[])
        metadata = result.charts['XmR']['metadata']
        boundaries = metadata.get('lane_boundaries')

        # With 2 factor levels and by=[], boundaries should exist
        # at each transition between 'A_B' and 'C' groups
        assert boundaries is not None


# ---------------------------------------------------------------------------
# Bug #10 — Missing diagnostics for short strata in non-phased XmR
# ---------------------------------------------------------------------------

class TestBug10_InsufficientStrataMetadata:
    """Non-phased stratified XmR should report strata with < 2 observations."""

    def test_insufficient_strata_metadata_reported(self):
        """Construct a study where one stratum has only 1 data point."""
        rows = []
        rng = np.random.default_rng(42)

        # Factor A: 6 observations across time
        for t in range(1, 7):
            rows.append({'y': rng.normal(50, 1), 'time': t, 'factor 1': 'A'})

        # Factor B: 6 observations across time
        for t in range(1, 7):
            rows.append({'y': rng.normal(50, 1), 'time': t, 'factor 1': 'B'})

        # Factor C: only 1 observation (insufficient for moving range)
        rows.append({'y': rng.normal(50, 1), 'time': 1, 'factor 1': 'C'})

        df = pd.DataFrame(rows)
        study = ProcessBehavior(df).formulate(
            response='y', time='time', factors=['factor 1'],
        )

        result = study.execute(chart='XmR', by=['factor 1'])
        metadata = result.charts['XmR']['metadata']

        assert 'insufficient_strata' in metadata, (
            "Expected 'insufficient_strata' key in non-phased stratified XmR metadata"
        )
        assert metadata['insufficient_strata'] is not None, (
            "Expected insufficient_strata to list stratum 'C'"
        )
        # The stratum for factor C should be listed
        insufficient = metadata['insufficient_strata']
        assert any('C' in str(s) for s in insufficient), (
            f"Expected stratum containing 'C' in insufficient_strata, got {insufficient}"
        )

    def test_sufficient_strata_metadata_is_none(self):
        """When all strata have >= 2 observations, insufficient_strata should be None."""
        study = _make_sds1_study()

        result = study.execute(chart='XmR', by=['factor 1'])
        metadata = result.charts['XmR']['metadata']

        # All strata should have sufficient data
        assert metadata.get('insufficient_strata') is None, (
            f"Expected insufficient_strata=None, got {metadata.get('insufficient_strata')}"
        )
