"""The Xbar/S helpers extracted for issue #144 (its first PR), tested on their own.

The behaviour they carry is pinned end to end by the golden masters and by
``validation/xbar_s_sweep.py`` (identical fingerprints before and after the extraction). These
tests pin the pieces the phased Xbar/S chart will reuse: one label rule for lane dividers and
phases, a per-row sigma-hat in ``calculate_limits_vectorized``, and the per-group fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import detect_sds_for_test, make_spec

from processbehavior.analysis import Analysis, _lane_labels
from processbehavior.datasets.synthetic import make_design
from processbehavior.formulation_spec import ChartRequest
from processbehavior.spc_constants import b3, b4, calculate_limits_vectorized

SPEC = {'rsg_vars': ['factor 1', 'factor 2'], 'response_var': 'y', 'time_var': 'time', 'round_to': 3}


def _analysis(**request) -> Analysis:
    df = make_design(1, K1=2, K2=2, T=3, n_min=2, n_max=3, seed=1)
    sds = detect_sds_for_test(df, SPEC)
    return Analysis(spec=make_spec(SPEC), request=ChartRequest(chart='Xbar', **request), sds=sds, df=df)


class TestLaneLabels:
    def test_single_column_keeps_astype_rendering(self):
        df = pd.DataFrame({'d': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-02'])})
        # astype(str) renders a date without the time part; str(Timestamp) would not.
        assert list(_lane_labels(df, ['d'])) == ['2024-01-01', '2024-01-02', '2024-01-02']

    def test_several_columns_join_with_underscore(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'y', 'y']})
        labels = _lane_labels(df, ['a', 'b'])
        assert isinstance(labels, np.ndarray)
        assert list(labels) == ['1_x', '1_y', '2_y']


class TestVectorizedSigmaPerRow:
    def test_series_sigma_matches_the_formula_row_by_row(self):
        mean = pd.Series([10.0, 10.0, 10.0], index=[5, 6, 7])
        sigma = pd.Series([1.0, 2.0, 0.5], index=[5, 6, 7])
        n = pd.Series([4, 4, 1], index=[5, 6, 7])
        limits = calculate_limits_vectorized('Xbar', mean=mean, sigma=sigma, N=n)
        expected_half = 3 * sigma / np.sqrt(n)
        pd.testing.assert_series_equal(limits['lpl'], mean - expected_half, check_names=False)
        pd.testing.assert_series_equal(limits['upl'], mean + expected_half, check_names=False)
        assert list(limits.index) == [5, 6, 7]

    def test_scalar_sigma_equals_a_constant_series(self):
        n = pd.Series([2, 3, 5, 1])
        scalar = calculate_limits_vectorized('Xbar', mean=4.5, sigma=0.7, N=n)
        series = calculate_limits_vectorized('Xbar', mean=4.5, sigma=pd.Series([0.7] * 4), N=n)
        assert np.array_equal(scalar.to_numpy(), series.to_numpy())

    def test_series_sigma_supplies_the_index_when_nothing_else_does(self):
        sigma = pd.Series([1.0, 2.0], index=[30, 40])
        limits = calculate_limits_vectorized('Xbar', mean=0.0, sigma=sigma, N=np.array([4, 4]))
        assert list(limits.index) == [30, 40]


class TestFitXbarGroup:
    def test_none_when_no_subgroup_is_replicated(self):
        out = pd.DataFrame({'s': [np.nan, np.nan], 'xbar': [1.0, 2.0], 'n': [1, 1], 'N': [1, 1]})
        assert _analysis()._fit_xbar_group(out) is None

    def test_one_reading_subgroups_are_charted_but_not_fitted(self):
        out = pd.DataFrame(
            {'s': [np.nan, 1.0, 3.0, 2.0], 'xbar': [9.0, 10.0, 11.0, 12.0], 'n': [1, 3, 3, 0], 'N': [3] * 4}
        )
        group = _analysis()._fit_xbar_group(out)
        assert group is not None
        assert list(group.out['n']) == [1, 3, 3]  # n = 0 dropped, n = 1 kept
        assert list(group.fit) == [False, True, True]
        assert group.s_bar == pytest.approx(2.0)
        assert group.n_to_use == 'n'  # sizes vary on the Xbar chart (1, 3, 3) ...
        assert group.s_n_to_use == 'N'  # ... but not on the S chart (3, 3)
        assert group.n_max == 3
        assert group.n_avg is None
        assert list(group.out['N']) == [3, 3, 3]

    def test_average_n_mode_forces_the_constant_n_path(self):
        out = pd.DataFrame({'s': [1.0, 3.0, 2.0], 'xbar': [10.0, 11.0, 12.0], 'n': [2, 4, 3], 'N': [4] * 3})
        group = _analysis(n_mode='average')._fit_xbar_group(out)
        assert group is not None
        assert group.n_avg == pytest.approx(3.0)
        assert group.n_to_use == group.s_n_to_use == 'N'
        assert list(group.out['N']) == [3.0, 3.0, 3.0]
        assert list(group.out['n']) == [2, 4, 3]  # actual sizes survive on n


class TestApplySLimits:
    def test_b3_b4_per_row_around_s_bar(self):
        out = pd.DataFrame({'s': [0.5, 4.0, 1.2], 'n': [3, 5, 5], 'N': [5] * 3})
        result = _analysis()._apply_s_limits(out.copy(), s_bar=1.5, n_to_use='n')
        assert list(result['center']) == [1.5, 1.5, 1.5]
        assert list(result['lpl']) == [round(b3(3) * 1.5, 3), round(b3(5) * 1.5, 3), round(b3(5) * 1.5, 3)]
        assert list(result['upl']) == [round(b4(3) * 1.5, 3), round(b4(5) * 1.5, 3), round(b4(5) * 1.5, 3)]
        assert list(result['beyond_limits']) == [0, 1, 0]


class TestAggregateSubgroups:
    def test_cached_cell_mean_replaces_the_subgroup_mean(self):
        df = pd.DataFrame(
            {
                'k': ['A', 'A', 'A', 'B', 'B'],
                't': [1, 1, 2, 1, 1],
                'y': [1.0, 3.0, 5.0, 2.0, 4.0],
                'r2': [np.nan, 0.5, -0.5, 0.2, -0.2],
                'Ybar_kt': [2.0, 2.0, 5.0, 3.0, 3.0],
            }
        )
        analysis = _analysis()
        plain = analysis._aggregate_subgroups(df, ['k', 't'], 'y', 'y')
        assert list(plain.columns) == ['k', 't', 's', 'xbar', 'n', 'N']
        assert list(plain['xbar']) == [2.0, 5.0, 3.0]
        assert list(plain['n']) == [2, 1, 2]
        assert list(plain['N']) == [2, 2, 2]

        cached = analysis._aggregate_subgroups(df, ['k', 't'], 'y', 'y', ybar_col='Ybar_kt')
        assert list(cached['xbar']) == [2.0, 5.0, 3.0]
        assert 'Ybar_kt' not in cached.columns and 'mean' not in cached.columns

        residual = analysis._aggregate_subgroups(df, ['k', 't'], 'y', 'r2')
        # n and s count the limits column: the subgroup missing its R2 is one smaller
        assert list(residual['n']) == [1, 1, 2]
        assert 's_value' in residual.columns


class TestSubgroupLabelColumn:
    def test_default_grouping_joins_into_group(self):
        out = pd.DataFrame({'factor 1': [1, 2], 'time': [10, 10], 'xbar': [0.0, 0.0]})
        labelled, col = _analysis(by=None)._subgroup_label_column(out, ['factor 1', 'time'])
        assert col == 'group' and list(labelled['group']) == ['1_10', '2_10']

    def test_single_by_keeps_the_column_as_strings(self):
        out = pd.DataFrame({'factor 1': [1, 2], 'xbar': [0.0, 0.0]})
        labelled, col = _analysis(by=('factor 1',))._subgroup_label_column(out, ['factor 1'])
        assert col == 'factor 1' and list(labelled['factor 1']) == ['1', '2']

    def test_several_by_columns_join_into_subgroup(self):
        out = pd.DataFrame({'factor 1': [1, 2], 'time': [10, 10], 'xbar': [0.0, 0.0]})
        _, col = _analysis(by=('factor 1', 'time'))._subgroup_label_column(out, ['factor 1', 'time'])
        assert col == 'subgroup'

    def test_no_grouping_has_no_label_column(self):
        out = pd.DataFrame({'group': ['All'], 'xbar': [0.0]})
        _, col = _analysis(by=None)._subgroup_label_column(out, [])
        assert col is None
