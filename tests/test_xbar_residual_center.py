"""Regression tests for the Bishop VAS unweighted Xbar center line.

The Xbar center line must be the mean of (factor x time) cell means on
`value_col` (Bishop VAS unweighted grand mean), NOT the observation-weighted
`df[value_col].mean()`. The two coincide on balanced designs but diverge on
unbalanced designs (e.g. SDS 3 with mixed cell N).

Prior to the fix, a dead-code branch in `_calculate_xbar` caused the center
to fall through to the observation-weighted mean. These tests pin the
Bishop-canonical behavior and guard against regression.
"""

from pathlib import Path

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.datasets import synthetic

VALIDATION_CSV = Path(__file__).parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'


@pytest.mark.skipif(
    not VALIDATION_CSV.exists(),
    reason='Bishop validation CSV not present',
)
def test_sds3_r3_pt_xbar_center_matches_bishop():
    """SDS 3 R3 Xbar by [PRODUCTION TIME] recentered must match Bishop's 237.83.

    Row 22 of validation/e2e_bishop_report.html — the only e2e divergence
    before the fix. Bishop reports CL = 237.83; the pre-fix code produced
    237.81 (observation-weighted on unbalanced cells).
    """
    df = pd.read_csv(VALIDATION_CSV)
    pb_obj = ProcessBehavior(df)
    study = pb_obj.formulate(
        response=pb_obj.cols.PM_SDS_3,
        factors=[pb_obj.cols.FACTOR_1, pb_obj.cols.FACTOR_2],
        time=pb_obj.cols.PRODUCTION_TIME,
    )
    result = study.execute(
        chart='Xbar',
        by=[pb_obj.cols.PRODUCTION_TIME],
        value='R3',
        recentered=True,
    )
    center = result.get_statistics('Xbar')['center']
    assert abs(center - 237.83) <= 0.01, f'Expected Bishop CL 237.83 +/- 0.01, got {center}'


def test_residual_xbar_center_is_mean_of_cell_means_on_unbalanced_data():
    """On unbalanced SDS 3, residual Xbar center equals mean-of-cell-means
    and differs from the observation-weighted mean.

    Asserts both: (a) the new methodology-correct behavior, and (b) that
    the dead-branch observation-weighted fallback is no longer in use.
    """
    df = synthetic.make_sds(3, K1=3, K2=2, T=4, p_replicated=0.5, n_when_replicated=4, seed=42)
    pb_obj = ProcessBehavior(df)
    study = pb_obj.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')

    # Sanity: cell sizes actually vary (precondition for the test to bite).
    ads = study._ads.analysis_dataset
    cell_counts = ads.groupby(['factor 1', 'factor 2', 'time'], observed=True).size()
    assert cell_counts.nunique() > 1, 'Test setup error: cell sizes should vary on unbalanced SDS 3'

    # Execute a residual Xbar chart by time.
    result = study.execute(chart='Xbar', by=['time'], value='R3', recentered=True)
    actual_center = result.get_statistics('Xbar')['center']

    # Bishop unweighted: mean of (factor x time) cell means on RCR3.
    cell_means_center = ads.groupby(['factor 1', 'factor 2', 'time'], observed=True)['RCR3'].mean().mean()

    # Old (buggy) behavior would have been observation-weighted.
    obs_weighted_center = ads['RCR3'].mean()

    # Tolerance accounts for chart center rounding (round_to=3 default).
    assert abs(actual_center - cell_means_center) <= 1e-3, (
        f'Center should be Bishop mean-of-cell-means {cell_means_center}, got {actual_center}'
    )
    # Divergence between weighted and unweighted means is the bug signature;
    # if these match exactly on unbalanced data the dead branch has regressed.
    assert abs(cell_means_center - obs_weighted_center) > 1e-4, (
        'Test setup error: cell-mean and obs-weighted means should differ on unbalanced data.'
    )


def test_response_xbar_center_matches_precomputed_ybar():
    """Response Xbar reuses the pre-computed 'Ybar' column (also Bishop unweighted).

    Confirms the response-path branch wires into residual_calculator's
    grand_mean computation.
    """
    df = synthetic.make_sds(3, K1=3, K2=2, T=4, p_replicated=0.5, n_when_replicated=4, seed=42)
    pb_obj = ProcessBehavior(df)
    study = pb_obj.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    result = study.execute(chart='Xbar')
    actual_center = result.get_statistics('Xbar')['center']

    ads = study._ads.analysis_dataset
    expected_center = ads['Ybar'].iloc[0]
    assert abs(actual_center - expected_center) <= 1e-3, (
        f'Response Xbar center should equal pre-computed Ybar {expected_center}, got {actual_center}'
    )


def test_r6_xbar_center_is_factor_grain_not_full_cell_grid():
    """R6 is a factor-effect residual: its natural grain is (rsg) only, not
    (rsg x time). The Xbar center for any R6 chart equals the mean of (F1, F2)
    cell means on RCR6, not the mean over the full cell grid.

    On unbalanced data these two values differ; pinning to (F1, F2) is what
    matches Bishop's reference for SDS 3 R6 effects charts (pages 20/24/26).
    """
    df = synthetic.make_sds(3, K1=3, K2=2, T=4, p_replicated=0.5, n_when_replicated=4, seed=42)
    pb_obj = ProcessBehavior(df)
    study = pb_obj.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    # Force RCR6 column to materialize
    result = study.execute(chart='Xbar', by=['factor 2'], value='R6', recentered=True)
    actual_center = result.get_statistics('Xbar')['center']

    ads = study._ads.analysis_dataset
    rsg_grain_center = ads.groupby(['factor 1', 'factor 2'], observed=True)['RCR6'].mean().mean()
    full_grid_center = ads.groupby(['factor 1', 'factor 2', 'time'], observed=True)['RCR6'].mean().mean()

    # Tolerance for chart round_to=3.
    assert abs(actual_center - rsg_grain_center) <= 1e-3, (
        f'R6 Xbar center should equal mean of (rsg) cell means {rsg_grain_center}, got {actual_center}'
    )
    # And differ from the full cell grid mean on unbalanced data — that's the
    # whole point of the per-residual grain rule.
    assert abs(rsg_grain_center - full_grid_center) > 1e-4, (
        'Test setup error: rsg-grain and full-grid centers should differ on '
        'unbalanced SDS 3 with replication. Adjust generator params.'
    )


def test_balanced_design_unaffected_by_fix():
    """On balanced SDS 1 cells, observation-weighted and unweighted means
    coincide — fix is a no-op."""
    df = synthetic.make_sds(1, K1=3, K2=2, T=4, n_min=3, n_max=3, seed=42)
    pb_obj = ProcessBehavior(df)
    study = pb_obj.formulate(response='y', factors=['factor 1', 'factor 2'], time='time')
    result = study.execute(chart='Xbar', by=['time'], value='R3', recentered=True)
    actual_center = result.get_statistics('Xbar')['center']

    ads = study._ads.analysis_dataset
    obs_weighted = ads['RCR3'].mean()
    cell_means = ads.groupby(['factor 1', 'factor 2', 'time'], observed=True)['RCR3'].mean().mean()

    # On balanced data the two are equal; center matches both.
    assert abs(obs_weighted - cell_means) < 1e-9
    assert abs(actual_center - cell_means) <= 1e-3
