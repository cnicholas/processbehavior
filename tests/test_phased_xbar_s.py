"""Phased (staged) Xbar/S charts: one centre line and one set of limits per phase (issue #144).

A phase is one combination of the grouping factors, charted the way the per-condition charts
of ``by=[time]`` chart one stratum, and the phases sit side by side on one axis. Dr. Bishop
settled the two design questions on 2026-10-18: each stage has its own S-bar and sigma-hat
("treat each stage separately as if it were a single analysis"), and with other factors pooled
into the subgroup the centre is "the average of the rational subgroup averages for that stage".
He also sent Wheeler's four-cavity thickness data with Minitab's per-cavity limits; that file is
``tests/fixtures/data/wheeler_cavity_thickness.csv`` (Hour 1-20, Cavity A-D, Cycle 1-5,
Thickness, plus VAS's per-cavity CL/LBL/UBL columns on its first four rows).
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import processbehavior as pb
from processbehavior.calibration import Calibration
from processbehavior.datasets.synthetic import make_design
from processbehavior.exceptions import CalibrationNotSupportedError, ValidationError
from processbehavior.spc_constants import b3, b4, c4

REFERENCE_CSV = Path(__file__).resolve().parent.parent / 'validation' / 'PBTESTDATABASE_T100.csv'
CAVITY_CSV = Path(__file__).resolve().parent / 'fixtures' / 'data' / 'wheeler_cavity_thickness.csv'
FACTORS = ['FACTOR 1', 'FACTOR 2']
TIME = 'PRODUCTION TIME'
XBAR_COLS = ['n', 'xbar', 'center', 'lpl', 'upl', 'beyond_limits']
S_COLS = ['n', 's', 'center', 'lpl', 'upl', 'beyond_limits']
VALUE_COLS = {'Xbar': XBAR_COLS, 'S': S_COLS}


@pytest.fixture(scope='module')
def reference_df():
    return pd.read_csv(REFERENCE_CSV, na_values=['*'])


@pytest.fixture(scope='module')
def reference_studies(reference_df):
    return {
        response: pb.formulate(reference_df, response=response, factors=FACTORS, time=TIME)
        for response in ('PM SDS 1', 'PM SDS 3')
    }


@pytest.fixture(scope='module')
def cavity_df():
    return pd.read_csv(CAVITY_CSV)


@pytest.fixture(scope='module')
def cavity_study(cavity_df):
    return pb.formulate(cavity_df, response='Thickness', time='Hour', factors=['Cavity'])


@pytest.fixture
def sds1_study():
    df = make_design(1, K1=3, K2=2, T=6, n_min=2, n_max=4, seed=42)
    return pb.formulate(df, response='y', time='time', factors=['factor 1', 'factor 2'])


def _staged(values_by_stage: dict[str, list[list[float]]]) -> pd.DataFrame:
    """A staged study: ``{stage: [readings at time 1, readings at time 2, ...]}``."""
    rows = []
    for stage, subgroups in values_by_stage.items():
        for t, readings in enumerate(subgroups, start=1):
            rows += [{'stage': stage, 'time': t, 'y': float(v)} for v in readings]
    return pd.DataFrame(rows)


def _focus_concat(result, chart: str) -> pd.DataFrame:
    """The per-condition charts of a stratified result, one after another, in strata order."""
    return pd.concat([result.focus(k).get_chart(chart) for k in result.strata], ignore_index=True)


def _frame(result, chart: str) -> pd.DataFrame:
    return result.get_chart(chart)[VALUE_COLS[chart]].reset_index(drop=True)


# =============================================================================
# Oracles: the phased chart is the per-condition charts laid side by side
# =============================================================================


class TestOracles:
    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    @pytest.mark.parametrize('response', ['PM SDS 1', 'PM SDS 3'])
    def test_phased_lanes_equal_stratified_focus(self, reference_studies, response, chart):
        study = reference_studies[response]
        phased = study.execute(chart=chart, phased=True)
        stratified = study.execute(chart=chart, by=[TIME])
        expected = _focus_concat(stratified, chart)

        pd.testing.assert_frame_equal(_frame(phased, chart), expected[VALUE_COLS[chart]].reset_index(drop=True))

        data = phased.get_chart(chart)
        phases = [k for k in stratified.strata for _ in range(len(stratified.focus(k).get_chart(chart)))]
        assert list(data['phase']) == phases
        assert list(data['group']) == [f'{k}_{t}' for k, t in zip(phases, expected[TIME], strict=True)]

    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    @pytest.mark.parametrize('response', ['PM SDS 1', 'PM SDS 3'])
    def test_phase_statistics_equal_stratified_statistics(self, reference_studies, response, chart):
        study = reference_studies[response]
        phases = study.execute(chart=chart, phased=True).phase_statistics(chart).set_index('phase')
        stratified = study.execute(chart=chart, by=[TIME])
        assert list(phases.index) == stratified.strata
        for stratum in stratified.strata:
            stats = stratified.statistics(chart, stratum=stratum)
            row = phases.loc[stratum]
            for key in ('center', 'N', 'lpl', 'upl'):
                if stats[key] is None:
                    assert np.isnan(row[key]), (stratum, key, row[key])
                else:
                    assert row[key] == pytest.approx(stats[key]), (stratum, key)

    @pytest.mark.parametrize('chart', ['Xbar', 'S'])
    def test_pooled_phase_equals_reformulated_study(self, reference_df, chart):
        """by=['FACTOR 1', time] with FACTOR 2 pooled: each phase is the chart of a study without FACTOR 2.

        Checked on the balanced reference data and on an unbalanced copy, where the mean of the
        pooled subgroup means (Bishop's rule) differs from the mean of the fine cell means.
        """
        df = reference_df
        thinned = df.drop(
            df[(df['FACTOR 1'] == 1) & (df['FACTOR 2'] == 1) & (df[TIME] <= 50)].groupby(TIME).head(2).index
        )
        for frame in (df, thinned):
            two_factor = pb.formulate(frame, response='PM SDS 1', factors=FACTORS, time=TIME)
            phased = two_factor.execute(chart=chart, by=['FACTOR 1', TIME], phased=True)
            one_factor = pb.formulate(frame, response='PM SDS 1', factors=['FACTOR 1'], time=TIME)
            expected = _focus_concat(one_factor.execute(chart=chart, by=[TIME]), chart)
            pd.testing.assert_frame_equal(_frame(phased, chart), expected[VALUE_COLS[chart]].reset_index(drop=True))

        # The non-phased pooled chart keeps its study-wide centre: the mean of the fine cell means.
        two_factor = pb.formulate(thinned, response='PM SDS 1', factors=FACTORS, time=TIME)
        pooled = two_factor.execute(chart='Xbar', by=['FACTOR 1', TIME]).get_statistics('Xbar')['center']
        fine = thinned.groupby(FACTORS + [TIME])['PM SDS 1'].mean().mean()
        plotted = thinned.groupby(['FACTOR 1', TIME])['PM SDS 1'].mean().mean()
        assert pooled == pytest.approx(round(fine, 3))
        assert abs(fine - plotted) > 1e-3  # the two rules really differ here

    def test_by_none_equals_by_empty_and_the_full_by_in_any_order(self, reference_studies):
        study = reference_studies['PM SDS 3']
        base = study.execute(chart='Xbar', phased=True, companion=True)
        for by in ([], FACTORS + [TIME], ['FACTOR 2', 'FACTOR 1', TIME]):
            other = study.execute(chart='Xbar', by=by, phased=True, companion=True)
            for chart in ('Xbar', 'S'):
                pd.testing.assert_frame_equal(_frame(base, chart), _frame(other, chart))
                assert list(base.get_chart(chart)['phase']) == list(other.get_chart(chart)['phase'])
                pd.testing.assert_frame_equal(base.phase_statistics(chart), other.phase_statistics(chart))
                assert base.charts[chart]['metadata']['phase_by'] == other.charts[chart]['metadata']['phase_by']

    def test_wheeler_cavity_oracle(self, cavity_df, cavity_study):
        """Tom's file carries VAS's per-cavity limits; the phased chart reproduces them to the digit."""
        result = cavity_study.execute(chart='Xbar', phased=True, companion=True)
        chart_cols = [
            'XBAR CHART CL',
            'XBAR CHART LBL',
            'XBAR CHART UBL',
            'S CHART CL',
            'S CHART LBL',
            'S CHART UBL',
            'RSG SAMPLE SIZE',
        ]
        charts = cavity_df.dropna(subset=['CAVITY ID'])
        assert (charts.groupby('CAVITY ID')[chart_cols].nunique() == 1).all().all()  # one value per cavity
        vas = charts.groupby('CAVITY ID')[chart_cols].first()
        xbar = result.phase_statistics('Xbar').set_index('phase')
        s = result.phase_statistics('S').set_index('phase')
        assert list(xbar.index) == ['A', 'B', 'C', 'D']
        for cavity in 'ABCD':
            assert xbar.loc[cavity, 'center'] == pytest.approx(vas.loc[cavity, 'XBAR CHART CL'], abs=5e-4)
            assert xbar.loc[cavity, 'lpl'] == pytest.approx(vas.loc[cavity, 'XBAR CHART LBL'], abs=5e-4)
            assert xbar.loc[cavity, 'upl'] == pytest.approx(vas.loc[cavity, 'XBAR CHART UBL'], abs=5e-4)
            assert s.loc[cavity, 'center'] == pytest.approx(vas.loc[cavity, 'S CHART CL'], abs=5e-4)
            assert s.loc[cavity, 'lpl'] == pytest.approx(vas.loc[cavity, 'S CHART LBL'], abs=5e-4)
            assert s.loc[cavity, 'upl'] == pytest.approx(vas.loc[cavity, 'S CHART UBL'], abs=5e-4)
            assert xbar.loc[cavity, 'N'] == vas.loc[cavity, 'RSG SAMPLE SIZE']
        assert xbar.loc['A', 'center'] == 14.65 and xbar.loc['D', 'upl'] == 9.831  # the Minitab screenshot

        stratified = cavity_study.execute(chart='Xbar', by=['Hour'], companion=True)
        for chart in ('Xbar', 'S'):
            expected = _focus_concat(stratified, chart)[VALUE_COLS[chart]].reset_index(drop=True)
            pd.testing.assert_frame_equal(_frame(result, chart), expected)


# =============================================================================
# Hand-computed
# =============================================================================


class TestHandComputed:
    def test_two_stage_example(self):
        stages = {
            'A': [[10, 12, 14], [11, 13, 12], [9, 10, 14], [12, 12, 15]],
            'B': [[20, 22, 21], [19, 23, 22], [21, 21, 24], [20, 20, 20]],
        }
        study = pb.formulate(_staged(stages), response='y', time='time', factors=['stage'], precision=9)
        result = study.execute(chart='Xbar', phased=True, companion=True)
        xbar, s = result.get_chart('Xbar'), result.get_chart('S')
        phases = result.phase_statistics('Xbar').set_index('phase')

        for stage, subgroups in stages.items():
            means = np.array([np.mean(g) for g in subgroups])
            stds = np.array([np.std(g, ddof=1) for g in subgroups])
            centre, s_bar = means.mean(), stds.mean()
            sigma_hat = s_bar / c4(3)
            half = 3 * sigma_hat / np.sqrt(3)
            rows = xbar[xbar['phase'] == stage]
            assert rows['xbar'].to_numpy() == pytest.approx(means, abs=1e-9)
            assert rows['center'].to_numpy() == pytest.approx(centre, abs=1e-9)
            assert rows['lpl'].to_numpy() == pytest.approx(centre - half, abs=1e-9)
            assert rows['upl'].to_numpy() == pytest.approx(centre + half, abs=1e-9)
            srows = s[s['phase'] == stage]
            assert srows['s'].to_numpy() == pytest.approx(stds, abs=1e-9)
            assert srows['center'].to_numpy() == pytest.approx(s_bar, abs=1e-9)
            assert srows['lpl'].to_numpy() == pytest.approx(b3(3) * s_bar, abs=1e-9)  # 0 at n = 3
            assert srows['upl'].to_numpy() == pytest.approx(b4(3) * s_bar, abs=1e-9)
            row = phases.loc[stage]
            assert row['subgroups'] == 4 and row['N'] == 3
            assert (row['center'], row['s_bar'], row['sigma_hat']) == pytest.approx(
                (centre, s_bar, sigma_hat), abs=1e-9
            )
            assert (row['lpl'], row['upl']) == pytest.approx((centre - half, centre + half), abs=1e-9)

    def test_unequal_n_uses_eq_11_5_within_the_phase(self):
        stages = {
            'A': [[10, 12, 14], [11, 13, 12, 16], [9, 10, 14, 11], [12, 12, 15]],
            'B': [[20, 22, 21], [19, 23, 22], [21, 21, 24], [20, 20, 25]],
        }
        study = pb.formulate(_staged(stages), response='y', time='time', factors=['stage'], precision=9)
        result = study.execute(chart='Xbar', phased=True)
        xbar = result.get_chart('Xbar')
        phases = result.phase_statistics('Xbar').set_index('phase')

        a = stages['A']
        stds = np.array([np.std(g, ddof=1) for g in a])
        sizes = np.array([len(g) for g in a])
        sigma_hat = np.mean(stds / np.array([c4(n) for n in sizes]))  # Eq 11-5 within the phase
        rows = xbar[xbar['phase'] == 'A']
        assert (rows['upl'] - rows['center']).to_numpy() == pytest.approx(3 * sigma_hat / np.sqrt(sizes), abs=1e-9)
        assert phases.loc['A', 'sigma_hat'] == pytest.approx(sigma_hat, abs=1e-9)
        assert all(np.isnan(phases.loc['A', k]) for k in ('N', 'lpl', 'upl'))  # the limits step within A
        assert phases.loc['B', 'N'] == 3 and not np.isnan(phases.loc['B', 'upl'])  # B has one pair
        assert result.get_statistics('Xbar') == {
            'N': None,
            'center': None,
            'lpl': None,
            'upl': None,
            'limits_vary': True,
        }

    def test_average_n_mode_is_applied_per_phase(self):
        stages = {
            'A': [[10, 12, 14], [11, 13, 12, 16], [9, 10, 14, 11], [12, 12, 15]],
            'B': [[20, 22], [19, 23, 22, 25], [21, 21, 24, 26], [20, 20]],
        }
        study = pb.formulate(_staged(stages), response='y', time='time', factors=['stage'], precision=9)
        result = study.execute(chart='Xbar', phased=True, companion=True, n_mode='average')
        xbar = result.get_chart('Xbar')
        phases = result.phase_statistics('Xbar').set_index('phase')
        for stage, subgroups in stages.items():
            n_avg = np.mean([len(g) for g in subgroups])
            s_bar = np.mean([np.std(g, ddof=1) for g in subgroups])
            half = 3 * (s_bar / c4(int(n_avg))) / np.sqrt(n_avg)  # Eq 11-20, as the unphased chart applies it
            rows = xbar[xbar['phase'] == stage]
            assert phases.loc[stage, 'N'] == pytest.approx(n_avg)
            assert rows['upl'].nunique() == 1
            assert (rows['upl'] - rows['center']).iloc[0] == pytest.approx(half, abs=1e-9)
        assert 'n_avg' not in result.charts['Xbar']['metadata']
        assert 'n_avg' not in result.charts['S']['metadata']

    def test_n_sigma_two_halves_the_width_ratio(self, sds1_study):
        three = sds1_study.execute(chart='Xbar', phased=True).get_chart('Xbar')
        two = sds1_study.execute(chart='Xbar', phased=True, n_sigma=2).get_chart('Xbar')
        assert ((two['upl'] - two['center']) / (three['upl'] - three['center'])).to_numpy() == pytest.approx(
            2 / 3, abs=1e-2
        )


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    def test_one_reading_subgroups_sit_on_xbar_and_not_on_s(self, reference_studies):
        result = reference_studies['PM SDS 3'].execute(chart='Xbar', phased=True, companion=True)
        xbar, s = result.get_chart('Xbar'), result.get_chart('S')
        singles = xbar[xbar['n'] == 1]
        assert len(singles) > 0
        assert (s['n'] >= 2).all()
        assert len(s) == len(xbar) - len(singles)
        phases = result.phase_statistics('Xbar').set_index('phase')
        for _, row in singles.iterrows():
            # N_r = 1: center ± n_sigma · sigma_hat (10-12 manual Eqs 11-16, 11-17)
            assert row['upl'] - row['center'] == pytest.approx(3 * phases.loc[row['phase'], 'sigma_hat'], abs=3e-3)

    def test_single_subgroup_phase_has_finite_limits(self):
        stages = {'A': [[10, 12, 14], [11, 13, 12], [9, 10, 14]], 'B': [[20, 22, 21]]}
        result = pb.formulate(_staged(stages), response='y', time='time', factors=['stage']).execute(
            chart='Xbar', phased=True, companion=True
        )
        b = result.get_chart('Xbar').query("phase == 'B'")
        assert len(b) == 1 and np.isfinite(b[['lpl', 'upl']].to_numpy()).all()
        assert result.charts['Xbar']['metadata']['single_point_phases'] == 1
        assert result.phase_statistics('Xbar').set_index('phase').loc['B', 'subgroups'] == 1

    def test_insufficient_phase_keeps_its_points_without_limits(self):
        stages = {'A': [[10, 12, 14], [11, 13, 12], [9, 10, 14]], 'B': [[20], [19], [21]]}
        study = pb.formulate(_staged(stages), response='y', time='time', factors=['stage'])
        result = study.execute(chart='Xbar', phased=True, companion=True)
        xbar, s = result.get_chart('Xbar'), result.get_chart('S')
        b = xbar[xbar['phase'] == 'B']
        assert len(b) == 3
        assert b[['center', 'lpl', 'upl']].isna().all().all()
        assert (b['beyond_limits'] == 0).all()
        assert 'B' not in set(s['phase'])
        for chart in ('Xbar', 'S'):
            assert result.charts[chart]['metadata']['insufficient_phases'] == ['B']
        row = result.phase_statistics('Xbar').set_index('phase').loc['B']
        assert row['subgroups'] == 3 and np.isnan(row['center']) and np.isnan(row['upl'])
        assert 'B' not in set(result.phase_statistics('S')['phase'])
        flagged = result.detect_signals('Xbar').flagged_observations
        assert flagged.isdisjoint(set(b.index))

    def test_no_replicated_phase_raises(self):
        stages = {'A': [[10], [11], [9]], 'B': [[20], [19], [21]]}
        study = pb.formulate(_staged(stages), response='y', time='time', factors=['stage'])
        with pytest.raises(ValidationError):
            study.execute(chart='Xbar', phased=True)


# =============================================================================
# Contracts
# =============================================================================


class TestContracts:
    def test_solo_charts_equal_the_companion_pair(self, reference_studies):
        study = reference_studies['PM SDS 3']
        pair = study.execute(chart='Xbar', phased=True, companion=True)
        for chart in ('Xbar', 'S'):
            solo = study.execute(chart=chart, phased=True)
            pd.testing.assert_frame_equal(solo.get_chart(chart), pair.get_chart(chart))
            assert solo.get_statistics(chart) == pair.get_statistics(chart)
            assert solo.charts[chart]['metadata'] == pair.charts[chart]['metadata']
            pd.testing.assert_frame_equal(solo.phase_statistics(chart), pair.phase_statistics(chart))

    def test_statistics_and_metadata_shape(self, sds1_study):
        result = sds1_study.execute(chart='Xbar', phased=True, companion=True, n_mode='average')
        for chart, value_col in (('Xbar', 'xbar'), ('S', 's')):
            assert result.get_statistics(chart) == {
                'N': None,
                'center': None,
                'lpl': None,
                'upl': None,
                'limits_vary': True,
            }
            meta = result.charts[chart]['metadata']
            assert set(meta) == {
                'chart_type',
                'value_col',
                'center_col',
                'n_sigma',
                'n_mode',
                'group_col',
                'phased',
                'run_rules_applicable',
                'phase_by',
                'lane_boundaries',
                'single_point_phases',
                'insufficient_phases',
            }
            assert meta['value_col'] == value_col and meta['group_col'] == 'group'
            assert meta['phased'] is True and meta['run_rules_applicable'] is False
            assert meta['phase_by'] == ['factor 1', 'factor 2']
            data = result.get_chart(chart)
            assert list(data.columns) == ['group', 'phase', 'n', value_col, 'center', 'lpl', 'upl', 'beyond_limits']
            assert list(result.phase_statistics(chart).columns) == [
                'phase',
                'subgroups',
                'N',
                'center',
                's_bar',
                'sigma_hat',
                'lpl',
                'upl',
            ]

    def test_lane_positions_are_the_first_rows_of_phases(self, reference_studies):
        result = reference_studies['PM SDS 3'].execute(chart='Xbar', phased=True, companion=True)
        for chart in ('Xbar', 'S'):
            data = result.get_chart(chart).reset_index(drop=True)
            starts = [i for i in range(1, len(data)) if data.loc[i, 'phase'] != data.loc[i - 1, 'phase']]
            boundaries = result.charts[chart]['metadata']['lane_boundaries']
            assert [b['position'] for b in boundaries] == starts
            assert [b['label'] for b in boundaries] == [data.loc[i, 'phase'] for i in starts]
        xbar_positions = [b['position'] for b in result.charts['Xbar']['metadata']['lane_boundaries']]
        s_positions = [b['position'] for b in result.charts['S']['metadata']['lane_boundaries']]
        assert xbar_positions != s_positions  # one-reading subgroups shift the S chart's dividers

    def test_chart_table_has_phase_after_subgroup(self, sds1_study):
        result = sds1_study.execute(chart='Xbar', phased=True, companion=True)
        for chart in ('Xbar', 'S'):
            table = result.chart_table(chart)
            assert list(table.columns) == ['subgroup', 'phase', 'n', 'value', 'center', 'lpl', 'upl', 'signal']

    def test_phase_statistics_refuses_unphased(self, sds1_study):
        with pytest.raises(ValidationError, match='not phased'):
            sds1_study.execute(chart='Xbar').phase_statistics('Xbar')
        with pytest.raises(ValidationError, match='does not apply'):
            sds1_study.execute(chart='Xbar', phased=True).phase_statistics('Xbar', stratum='1_1')

    def test_detect_signals_rule_1_against_each_phase(self, reference_studies):
        result = reference_studies['PM SDS 1'].execute(chart='Xbar', phased=True, companion=True)
        for chart in ('Xbar', 'S'):
            signals = result.detect_signals(chart)
            data = result.get_chart(chart)
            assert signals.rules_evaluated == ['rule_1']
            assert not signals.is_partial
            assert signals.flagged_observations == set(data.index[data['beyond_limits'] != 0])

    def test_stage_order_follows_the_labels(self):
        readings = [[10, 12, 14], [11, 13, 12]]
        by_name = _staged({'before': readings, 'after': [[20, 22, 21], [19, 23, 22]]})
        result = pb.formulate(by_name, response='y', time='time', factors=['stage']).execute(chart='Xbar', phased=True)
        assert list(result.phase_statistics('Xbar')['phase']) == ['after', 'before']  # strings sort

        sortable = by_name.assign(stage=by_name['stage'].map({'before': '1 before', 'after': '2 after'}))
        result = pb.formulate(sortable, response='y', time='time', factors=['stage']).execute(chart='Xbar', phased=True)
        assert list(result.phase_statistics('Xbar')['phase']) == ['1 before', '2 after']

        ordered = by_name.assign(stage=pd.Categorical(by_name['stage'], categories=['before', 'after'], ordered=True))
        result = pb.formulate(ordered, response='y', time='time', factors=['stage']).execute(chart='Xbar', phased=True)
        assert list(result.phase_statistics('Xbar')['phase']) == ['before', 'after']

    def test_unphased_output_is_unchanged(self, reference_studies):
        study = reference_studies['PM SDS 3']
        plain = study.execute(chart='Xbar', companion=True)
        phased = study.execute(chart='Xbar', phased=True, companion=True)
        for chart, value_col in (('Xbar', 'xbar'), ('S', 's')):
            a = plain.get_chart(chart).reset_index(drop=True)
            b = phased.get_chart(chart).reset_index(drop=True)
            assert list(a['group']) == list(b['group'])
            pd.testing.assert_series_equal(a[value_col], b[value_col])
            pd.testing.assert_series_equal(a['n'], b['n'])
            assert 'phase' not in a.columns
            assert 'phased' not in plain.charts[chart]['metadata']

    @pytest.mark.slow
    def test_a_thousand_phases_is_quick(self):
        rng = np.random.default_rng(0)
        rows = [
            {'stage': f'{k:04d}', 'time': t, 'y': rng.normal(k % 7, 1)}
            for k in range(1000)
            for t in range(3)
            for _ in range(2)
        ]
        study = pb.formulate(pd.DataFrame(rows), response='y', time='time', factors=['stage'])
        started = _time.perf_counter()
        result = study.execute(chart='Xbar', phased=True, companion=True)
        assert _time.perf_counter() - started < 5
        assert len(result.phase_statistics('Xbar')) == 1000


# =============================================================================
# Validation
# =============================================================================


class TestValidation:
    @pytest.mark.parametrize(
        ('kwargs', 'fragment'),
        [
            (dict(chart='Histogram'), 'only valid for'),
            (dict(chart='Xbar', by=[TIME]), 'already gives each factor combination its own chart'),
            (dict(chart='S', by=[TIME]), 'use by=None with phased=True'),
            (dict(chart='Xbar', by=['FACTOR 1']), 'requires the time variable in by='),
            (dict(chart='Xbar', by=FACTORS), 'requires the time variable in by='),
            (dict(chart='Xbar', by=[TIME, 'FACTOR 1']), 'time variable last in by='),
            (dict(chart='Xbar', value='R5'), 'charts the response'),
            (dict(chart='S', value='R1', recentered=True), 'charts the response'),
        ],
    )
    def test_refused_requests(self, reference_studies, kwargs, fragment):
        with pytest.raises(ValidationError, match=fragment):
            reference_studies['PM SDS 1'].execute(phased=True, **kwargs)

    def test_refused_without_factors(self, reference_df):
        study = pb.formulate(reference_df, response='PM SDS 1', time=TIME)
        with pytest.raises(ValidationError, match='requires factors to define phases'):
            study.execute(chart='Xbar', phased=True)

    @pytest.mark.parametrize('by', [None, [], FACTORS + [TIME], ['FACTOR 2', 'FACTOR 1', TIME], ['FACTOR 1', TIME]])
    def test_accepted_shapes(self, reference_studies, by):
        result = reference_studies['PM SDS 1'].execute(chart='Xbar', by=by, phased=True, companion=True)
        assert result.charts['Xbar']['metadata']['phased'] is True
        expected_phase_by = ['FACTOR 1'] if by == ['FACTOR 1', TIME] else FACTORS
        assert result.charts['Xbar']['metadata']['phase_by'] == expected_phase_by

    def test_calibration_is_refused_with_the_chart_in_the_context(self, reference_studies):
        study = reference_studies['PM SDS 1']
        cal = Calibration(label='ref', mean=10.0, sigma=1.0)
        for chart in ('Xbar', 'S'):
            with pytest.raises(CalibrationNotSupportedError) as exc:
                study.execute(chart=chart, phased=True, calibration=cal)
            assert exc.value.context == f'phased {chart}'
            assert study.supports_calibration(chart=chart, phased=True) is False


# =============================================================================
# Plotting
# =============================================================================


@pytest.mark.plotting
class TestPlotting:
    def test_stepped_lines_dividers_and_note(self, cavity_study):
        result = cavity_study.execute(chart='Xbar', phased=True, companion=True)
        for chart in ('Xbar', 'S'):
            fig = result.plot(chart=chart)
            stepped = [t.name for t in fig.data if t.name and 'varies' in t.name]
            assert any('CL' in name for name in stepped)
            assert any('UPL' in name for name in stepped)
            assert any('LPL' in name for name in stepped)
            texts = [a.text for a in fig.layout.annotations]
            assert 'Process limits computed per phase' in texts
            assert {'B', 'C', 'D'} <= set(texts)  # divider labels name the phases
            assert len(fig.layout.shapes) >= 3
            assert not any(t.name and 'Zone' in t.name for t in fig.data)
