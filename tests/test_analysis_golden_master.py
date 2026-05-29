"""Golden-master snapshot tests for analysis.py refactoring safety.

Captures exact chart outputs (DataFrames, statistics, metadata) before refactoring,
then asserts equivalence after each phase. This proves that refactoring preserves
behavior — "tests pass" is necessary but not sufficient for internal refactors.

Serialization:
    - DataFrames: Parquet (preserves dtypes exactly — categoricals, int64, float64)
    - Statistics/metadata dicts: JSON with normalized types

Usage:
    # Normal mode: assert snapshots match
    pytest tests/test_analysis_golden_master.py

    # Regenerate snapshots (local only — CI blocks this)
    REGEN_GOLDEN_MASTERS=1 pytest tests/test_analysis_golden_master.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic

pytestmark = pytest.mark.golden

# ============================================================================
# Configuration
# ============================================================================

FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'golden_masters'

# Single source of truth for numeric comparison tolerances
GOLDEN_MASTER_RTOL = 1e-12
GOLDEN_MASTER_ATOL = 1e-12

# Environment variable controls
REGEN = os.environ.get('REGEN_GOLDEN_MASTERS', '').strip() == '1'
IS_CI = any(
    os.environ.get(v, '').lower() in ('true', '1') for v in ('CI', 'GITHUB_ACTIONS', 'TRAVIS', 'CIRCLECI', 'GITLAB_CI')
)


# ============================================================================
# CI Safety Guard
# ============================================================================

if REGEN and IS_CI:
    raise RuntimeError(
        'REGEN_GOLDEN_MASTERS=1 is set in CI. '
        'Golden-master regeneration must only happen locally with human review. '
        'Remove the REGEN_GOLDEN_MASTERS env var from your CI configuration.'
    )


# ============================================================================
# Type Normalization
# ============================================================================


def _normalize_for_comparison(obj):
    """Normalize Python/numpy/pandas types for stable JSON serialization and comparison.

    Converts:
    - np.int64/int32 -> int
    - np.float64/float32 -> float (rounded to tolerance)
    - np.bool_ -> bool
    - pd.Timestamp -> ISO 8601 string
    - tuple -> list (JSON has no tuples)
    - dict keys sorted recursively
    - NaN -> None (JSON-safe)
    """
    if isinstance(obj, dict):
        return {str(k): _normalize_for_comparison(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_comparison(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if isinstance(obj, np.ndarray):
        return _normalize_for_comparison(obj.tolist())
    return obj


# ============================================================================
# Snapshot I/O
# ============================================================================


def _snapshot_path(scenario_name: str, chart_key: str, suffix: str) -> Path:
    """Build path for a snapshot file."""
    return FIXTURE_DIR / scenario_name / f'{chart_key}.{suffix}'


def _save_snapshot(scenario_name: str, chart_key: str, data: pd.DataFrame, statistics: dict, metadata: dict):
    """Save a chart snapshot (DataFrame as parquet, dicts as JSON)."""
    base_dir = FIXTURE_DIR / scenario_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # DataFrame -> parquet
    data.to_parquet(base_dir / f'{chart_key}_data.parquet')

    # Statistics -> JSON
    with open(base_dir / f'{chart_key}_statistics.json', 'w') as f:
        json.dump(_normalize_for_comparison(statistics), f, indent=2, default=str)

    # Metadata -> JSON
    with open(base_dir / f'{chart_key}_metadata.json', 'w') as f:
        json.dump(_normalize_for_comparison(metadata), f, indent=2, default=str)


def _load_snapshot(scenario_name: str, chart_key: str):
    """Load a chart snapshot. Returns (data, statistics, metadata) or None if missing."""
    base_dir = FIXTURE_DIR / scenario_name
    data_path = base_dir / f'{chart_key}_data.parquet'
    stats_path = base_dir / f'{chart_key}_statistics.json'
    meta_path = base_dir / f'{chart_key}_metadata.json'

    if not data_path.exists():
        return None

    data = pd.read_parquet(data_path)
    with open(stats_path) as f:
        statistics = json.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)

    return data, statistics, metadata


def _save_strata(scenario_name: str, chart_key: str, strata: list):
    """Save strata list as JSON."""
    base_dir = FIXTURE_DIR / scenario_name
    base_dir.mkdir(parents=True, exist_ok=True)
    with open(base_dir / f'{chart_key}_strata.json', 'w') as f:
        json.dump(_normalize_for_comparison(strata), f, indent=2, default=str)


def _load_strata(scenario_name: str, chart_key: str):
    """Load strata list from JSON."""
    path = FIXTURE_DIR / scenario_name / f'{chart_key}_strata.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Assertion Helpers
# ============================================================================


def _assert_chart_matches_snapshot(scenario_name: str, chart_key: str, chart_result: dict):
    """Assert a single chart result matches its golden-master snapshot.

    If REGEN_GOLDEN_MASTERS=1, writes new snapshot instead.
    """
    data = chart_result['data']
    statistics = chart_result['statistics']
    metadata = chart_result.get('metadata', {})
    strata = chart_result.get('strata')

    if REGEN:
        _save_snapshot(scenario_name, chart_key, data, statistics, metadata)
        if strata is not None:
            _save_strata(scenario_name, chart_key, strata)
        return

    snapshot = _load_snapshot(scenario_name, chart_key)
    assert snapshot is not None, (
        f'No golden-master snapshot for {scenario_name}/{chart_key}. Run with REGEN_GOLDEN_MASTERS=1 to generate.'
    )

    expected_data, expected_stats, expected_meta = snapshot

    # DataFrame comparison (exact dtypes, column order, values within tolerance)
    pd.testing.assert_frame_equal(
        data.reset_index(drop=True),
        expected_data.reset_index(drop=True),
        rtol=GOLDEN_MASTER_RTOL,
        atol=GOLDEN_MASTER_ATOL,
        check_column_type=True,
    )

    # Statistics dict comparison (normalized)
    actual_stats = _normalize_for_comparison(statistics)
    assert actual_stats == expected_stats, (
        f'Statistics mismatch for {scenario_name}/{chart_key}:\n'
        f'  actual:   {actual_stats}\n'
        f'  expected: {expected_stats}'
    )

    # Metadata dict comparison (normalized)
    actual_meta = _normalize_for_comparison(metadata)
    assert actual_meta == expected_meta, (
        f'Metadata mismatch for {scenario_name}/{chart_key}:\n  actual:   {actual_meta}\n  expected: {expected_meta}'
    )

    # Strata comparison (if present)
    if strata is not None:
        expected_strata = _load_strata(scenario_name, chart_key)
        actual_strata = _normalize_for_comparison(strata)
        assert actual_strata == expected_strata, (
            f'Strata mismatch for {scenario_name}/{chart_key}:\n'
            f'  actual:   {actual_strata}\n'
            f'  expected: {expected_strata}'
        )


def _assert_result_matches_snapshot(scenario_name: str, result):
    """Assert all charts in a result match their golden-master snapshots."""
    for chart_key, chart_data in result.charts.items():
        _assert_chart_matches_snapshot(scenario_name, chart_key, chart_data)


# ============================================================================
# Dataset Factories
# ============================================================================
# Deterministic data generators for each scenario.
# These must be stable across runs (fixed seeds, no randomness).


def _make_unstratified_small():
    """Scenario 1: Unstratified, small N — single condition over time (15 obs).

    Uses ``make_design(2, K1=1, K2=1, T=15)`` to produce a single-stream
    time series suitable for unstratified X/mR shape testing.
    """
    return synthetic.make_design(2, K1=1, K2=1, T=15, seed=42)


def _make_stratified_balanced():
    """Scenario 2: Stratified with equal group sizes — balanced SDS 1."""
    return synthetic.make_design(1, K1=3, K2=2, T=4, n_min=3, n_max=3, seed=42)


def _make_stratified_uneven():
    """Scenario 3: Stratified with uneven group sizes — varying n."""
    return synthetic.make_design(1, K1=2, K2=2, T=4, n_min=2, n_max=5, seed=42)


def _make_missing_values():
    """Scenario 4: Data with missing values in response column."""
    df = synthetic.make_design(1, K1=2, K2=2, T=4, n_min=3, n_max=3, seed=42)
    # Introduce NaN in response at specific positions (deterministic)
    rng = np.random.default_rng(99)
    mask = rng.random(len(df)) < 0.1  # ~10% missing
    df.loc[mask, 'y'] = np.nan
    return df


def _make_companion_mode():
    """Scenario 5: Companion mode — Xbar+S and X+mR companion charts."""
    return synthetic.make_design(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)


def _make_residual_chart():
    """Scenario 6: Residual chart (R2 via XmR)."""
    return synthetic.make_design(1, K1=3, K2=2, T=4, n_min=3, n_max=3, seed=42)


def _make_single_obs_strata():
    """Scenario 7: Stratum with n=1 — edge case for mR computation."""
    # SDS 3 has mixed replication. Use specific params to get some n=1 cells.
    return synthetic.make_design(3, K1=3, K2=2, T=4, p_replicated=0.3, seed=42)


def _make_pathological_ordering():
    """Scenario 8: Pathological ordering — reverse-sorted input."""
    df = synthetic.make_design(1, K1=2, K2=2, T=4, n_min=2, n_max=2, seed=42)
    # Reverse the entire DataFrame
    return df.iloc[::-1].reset_index(drop=True)


# ============================================================================
# Scenario Execution Helpers
# ============================================================================


def _execute_unstratified_small():
    """Execute scenario 1: SDS 4 single-condition X (by=[] for single stream)."""
    df = _make_unstratified_small()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1'], time='time')
    return study.execute(chart='X', by=[], companion=True)


def _execute_stratified_balanced():
    """Execute scenario 2: balanced SDS 1, default Xbar chart."""
    df = _make_stratified_balanced()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    return study.execute(chart='Xbar', companion=True)


def _execute_stratified_uneven():
    """Execute scenario 3: uneven n, Xbar+S companion."""
    df = _make_stratified_uneven()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    return study.execute(chart='Xbar', companion=True)


def _execute_missing_values():
    """Execute scenario 4: missing values, X."""
    df = _make_missing_values()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    return study.execute(chart='X', by=[], companion=True)


def _execute_companion_xmr_r():
    """Execute scenario 5a: companion X+mR."""
    df = _make_companion_mode()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    return study.execute(chart='X', by=[], companion=True)


def _execute_companion_xbar_s():
    """Execute scenario 5b: companion Xbar+S."""
    df = _make_companion_mode()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    return study.execute(chart='Xbar', companion=True)


def _execute_residual_chart():
    """Execute scenario 6: R2 residual via S chart (R2_S available for SDS 1)."""
    df = _make_residual_chart()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    return study.execute(chart='S', value='R2')


def _execute_single_obs_strata():
    """Execute scenario 7: SDS 3 with n=1 strata, X by=[]."""
    df = _make_single_obs_strata()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    return study.execute(chart='X', by=[], companion=True)


def _execute_pathological_ordering():
    """Execute scenario 8: reverse-sorted input, X companion."""
    df = _make_pathological_ordering()
    pb = ProcessBehavior(df)
    study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    return study.execute(chart='X', by=[], companion=True)


# ============================================================================
# Snapshot Tests
# ============================================================================


class TestGoldenMasterSnapshots:
    """Assert chart outputs match golden-master snapshots."""

    def test_unstratified_small(self):
        result = _execute_unstratified_small()
        _assert_result_matches_snapshot('unstratified_small', result)

    def test_stratified_balanced(self):
        result = _execute_stratified_balanced()
        _assert_result_matches_snapshot('stratified_balanced', result)

    def test_stratified_uneven(self):
        result = _execute_stratified_uneven()
        _assert_result_matches_snapshot('stratified_uneven', result)

    def test_missing_values(self):
        result = _execute_missing_values()
        _assert_result_matches_snapshot('missing_values', result)

    def test_companion_xmr_r(self):
        result = _execute_companion_xmr_r()
        _assert_result_matches_snapshot('paired_xmr_r', result)

    def test_companion_xbar_s(self):
        result = _execute_companion_xbar_s()
        _assert_result_matches_snapshot('paired_xbar_s', result)

    def test_residual_chart(self):
        result = _execute_residual_chart()
        _assert_result_matches_snapshot('residual_chart', result)

    def test_single_obs_strata(self):
        result = _execute_single_obs_strata()
        _assert_result_matches_snapshot('single_obs_strata', result)

    def test_pathological_ordering(self):
        result = _execute_pathological_ordering()
        _assert_result_matches_snapshot('pathological_ordering', result)


# ============================================================================
# Companion-Mode Equivalence Tests
# ============================================================================


class TestCompanionEquivalence:
    """Assert companion output == independent output for XmR+R and Xbar+S.

    Checks all three result components: data, statistics, metadata.
    """

    def test_xmr_companion_vs_independent(self):
        """Companion X+mR should match independent X + independent mR."""
        df = _make_companion_mode()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')

        # Companion
        companion = study.execute(chart='X', by=[], companion=True)

        # Independent
        xmr_only = study.execute(chart='X', by=[])
        r_only = study.execute(chart='mR', by=[])

        # Compare X
        _assert_chart_equivalence(companion.charts['X'], xmr_only.charts['X'], 'X companion vs independent')

        # Compare mR
        _assert_chart_equivalence(companion.charts['mR'], r_only.charts['mR'], 'mR companion vs independent')

    def test_xbar_s_companion_vs_independent(self):
        """Companion Xbar+S should match independent Xbar + independent S."""
        df = _make_companion_mode()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')

        # Companion
        companion = study.execute(chart='Xbar', companion=True)

        # Independent
        xbar_only = study.execute(chart='Xbar')
        s_only = study.execute(chart='S')

        # Compare Xbar
        _assert_chart_equivalence(companion.charts['Xbar'], xbar_only.charts['Xbar'], 'Xbar companion vs independent')

        # Compare S
        _assert_chart_equivalence(companion.charts['S'], s_only.charts['S'], 'S companion vs independent')

    def test_xmr_stratified_companion_vs_independent(self):
        """Companion stratified X+mR should match independent stratified X + mR."""
        df = _make_stratified_balanced()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')

        # Companion stratified
        companion = study.execute(chart='X', by=['factor 1', 'factor 2'], companion=True)

        # Independent stratified
        xmr_only = study.execute(chart='X', by=['factor 1', 'factor 2'])
        r_only = study.execute(chart='mR', by=['factor 1', 'factor 2'])

        _assert_chart_equivalence(companion.charts['X'], xmr_only.charts['X'], 'X stratified companion vs independent')
        _assert_chart_equivalence(companion.charts['mR'], r_only.charts['mR'], 'mR stratified companion vs independent')


def _assert_chart_equivalence(actual: dict, expected: dict, label: str):
    """Assert two chart result dicts are equivalent across data, statistics, metadata."""
    # Data
    pd.testing.assert_frame_equal(
        actual['data'].reset_index(drop=True),
        expected['data'].reset_index(drop=True),
        rtol=GOLDEN_MASTER_RTOL,
        atol=GOLDEN_MASTER_ATOL,
        check_column_type=True,
        obj=f'{label} data',
    )

    # Statistics
    actual_stats = _normalize_for_comparison(actual['statistics'])
    expected_stats = _normalize_for_comparison(expected['statistics'])
    assert actual_stats == expected_stats, (
        f'{label} statistics mismatch:\n  actual:   {actual_stats}\n  expected: {expected_stats}'
    )

    # Metadata
    actual_meta = _normalize_for_comparison(actual.get('metadata', {}))
    expected_meta = _normalize_for_comparison(expected.get('metadata', {}))
    assert actual_meta == expected_meta, (
        f'{label} metadata mismatch:\n  actual:   {actual_meta}\n  expected: {expected_meta}'
    )

    # Strata (if present)
    if 'strata' in actual or 'strata' in expected:
        actual_strata = _normalize_for_comparison(actual.get('strata'))
        expected_strata = _normalize_for_comparison(expected.get('strata'))
        assert actual_strata == expected_strata, (
            f'{label} strata mismatch:\n  actual:   {actual_strata}\n  expected: {expected_strata}'
        )


# ============================================================================
# Shape / Invariant Property Checks
# ============================================================================


def _limits_are_scalar(stats: dict) -> bool:
    """Check if limits are scalar floats (not None / not varying)."""
    return isinstance(stats.get('lpl'), (int, float)) and isinstance(stats.get('upl'), (int, float))


class TestShapeInvariants:
    """Assert structural properties that hold for any dataset."""

    def test_xmr_row_count_preserved(self):
        """X: len(output) == len(input) per stratum."""
        df = _make_unstratified_small()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='X', by=[])

        xmr_data = result.charts['X']['data']
        assert len(xmr_data) == len(df), f'X row count mismatch: output={len(xmr_data)}, input={len(df)}'

    def test_r_row_count_drops_first(self):
        """mR: output drops first observation (NaN mR)."""
        df = _make_unstratified_small()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='mR', by=[])

        r_data = result.charts['mR']['data']
        assert len(r_data) == len(df) - 1, f'mR row count mismatch: output={len(r_data)}, expected={len(df) - 1}'

    def test_beyond_limits_values_valid(self):
        """Beyond-limits flags must be in {-1, 0, 1}."""
        df = _make_stratified_balanced()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
        result = study.execute(chart='X', by=[], companion=True)

        for chart_key in ('X', 'mR'):
            data = result.charts[chart_key]['data']
            if 'beyond_limits' in data.columns:
                unique_vals = set(data['beyond_limits'].unique())
                assert unique_vals <= {-1, 0, 1}, f'{chart_key} beyond_limits has invalid values: {unique_vals}'

    def test_center_line_constant_within_ungrouped(self):
        """Center line should be constant for ungrouped (single-stream) charts."""
        df = _make_unstratified_small()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='X', by=[])

        xmr_data = result.charts['X']['data']
        centers = xmr_data['center'].unique()
        assert len(centers) == 1, f'Center line should be constant, got {len(centers)} unique values'

    def test_lpl_le_center_le_upl_scalar(self):
        """LPL <= center <= UPL when limits are scalar floats."""
        df = _make_unstratified_small()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='X', by=[])

        stats = result.charts['X']['statistics']
        if _limits_are_scalar(stats):
            assert stats['lpl'] <= stats['center'] <= stats['upl'], (
                f'Limit ordering violated: LPL={stats["lpl"]}, center={stats["center"]}, UPL={stats["upl"]}'
            )

    def test_lpl_le_center_le_upl_stratified(self):
        """LPL <= center <= UPL for each stratum when limits are scalar."""
        df = _make_stratified_balanced()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
        result = study.execute(chart='X', by=['factor 1', 'factor 2'])

        stats = result.charts['X']['statistics']
        for stratum, s in stats.items():
            if _limits_are_scalar(s):
                assert s['lpl'] <= s['center'] <= s['upl'], (
                    f'Limit ordering violated for {stratum}: LPL={s["lpl"]}, center={s["center"]}, UPL={s["upl"]}'
                )

    def test_xbar_lpl_le_center_le_upl(self):
        """Xbar: LPL <= center <= UPL when limits are scalar."""
        df = _make_stratified_balanced()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
        result = study.execute(chart='Xbar')

        stats = result.charts['Xbar']['statistics']
        if _limits_are_scalar(stats):
            assert stats['lpl'] <= stats['center'] <= stats['upl'], (
                f'Xbar limit ordering violated: LPL={stats["lpl"]}, center={stats["center"]}, UPL={stats["upl"]}'
            )

    def test_r_chart_lpl_ge_zero(self):
        """mR chart: LPL is always >= 0 (ranges can't be negative)."""
        df = _make_unstratified_small()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1'], time='time')
        result = study.execute(chart='mR', by=[])

        stats = result.charts['mR']['statistics']
        if _limits_are_scalar(stats):
            assert stats['lpl'] >= 0, f'mR chart LPL should be >= 0, got {stats["lpl"]}'

    def test_pathological_ordering_matches_normal(self):
        """Reverse-sorted input should produce identical results for n=1 data.

        Uses SDS 2 (n=1 per cell) so within-cell obs ordering is deterministic.
        For n>1 cells, obs_id assigned BEFORE sort means reversing input changes
        within-cell order, which changes moving ranges — that's expected behavior.
        """
        # Normal ordering (SDS 2: n=1 per cell — order fully determined by sort_key)
        df_normal = synthetic.make_design(2, K1=2, K2=2, T=6, seed=42)
        pb_normal = ProcessBehavior(df_normal)
        study_normal = pb_normal.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
        result_normal = study_normal.execute(chart='X', by=[], companion=True)

        # Reverse ordering
        df_reverse = df_normal.iloc[::-1].reset_index(drop=True)
        pb_reverse = ProcessBehavior(df_reverse)
        study_reverse = pb_reverse.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
        result_reverse = study_reverse.execute(chart='X', by=[], companion=True)

        for chart_key in ('X', 'mR'):
            _assert_chart_equivalence(
                result_normal.charts[chart_key],
                result_reverse.charts[chart_key],
                f'{chart_key} normal vs reverse ordering',
            )

    def test_lane_boundary_positions_monotonic(self):
        """Lane boundary positions should be monotonically increasing."""
        df = _make_stratified_balanced()
        pb = ProcessBehavior(df)
        study = pb.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
        result = study.execute(chart='X', by=[])

        meta = result.charts['X'].get('metadata', {})
        boundaries = meta.get('lane_boundaries')
        if boundaries:
            # boundaries can be a list (ungrouped) or dict (stratified)
            if isinstance(boundaries, list):
                positions = [b['position'] for b in boundaries]
                assert positions == sorted(positions), f'Lane boundary positions not monotonic: {positions}'
            elif isinstance(boundaries, dict):
                for stratum, bs in boundaries.items():
                    positions = [b['position'] for b in bs]
                    assert positions == sorted(positions), (
                        f'Lane boundary positions not monotonic for {stratum}: {positions}'
                    )
