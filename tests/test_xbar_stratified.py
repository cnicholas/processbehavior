"""Tests for stratified Xbar/S charts (by=[time_var] with factors)."""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior


@pytest.fixture
def sds1_study():
    """PM SDS 1 study: 2 factors, time, replication."""
    df = pd.read_csv('validation/TABVASTESTDATABASE.csv')
    pb = ProcessBehavior(df)
    return pb.formulate(
        response='PM SDS 1',
        factors=['FACTOR 1', 'FACTOR 2'],
        time='PRODUCTION TIME',
    )


class TestXbarStratified:
    """Stratified Xbar charts: by=[time_var] produces per-factor-combo charts."""

    def test_stratified_has_strata_key(self, sds1_study):
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        xbar = result.charts['Xbar']
        assert 'strata' in xbar
        assert xbar['metadata']['stratified'] is True

    def test_strata_count_matches_factor_combos(self, sds1_study):
        """4 levels of F1 × 2 levels of F2 = 8 combinations."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        xbar = result.charts['Xbar']
        assert len(xbar['strata']) == 8

    def test_reference_values_combo_1_2(self, sds1_study):
        """Validate against Tom's reference: combo 1_2."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        stats = result.charts['Xbar']['statistics']['1_2']
        assert stats['center'] == pytest.approx(238.615, abs=0.01)
        assert stats['lpl'] == pytest.approx(237.467, abs=0.01)
        assert stats['upl'] == pytest.approx(239.763, abs=0.01)

    def test_time_on_x_axis(self, sds1_study):
        """Each stratum has time points on x-axis."""
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        xbar = result.charts['Xbar']
        data = xbar['data']
        assert 'PRODUCTION TIME' in data.columns
        # Each stratum should have 100 time points
        for stratum in xbar['strata']:
            stratum_data = data[data['rsg'] == stratum]
            assert len(stratum_data) == 100

    def test_metadata_fields(self, sds1_study):
        result = sds1_study.execute(chart='Xbar', by=['PRODUCTION TIME'])
        meta = result.charts['Xbar']['metadata']
        assert meta['chart_type'] == 'Xbar'
        assert meta['stratified'] is True
        assert meta['stratify_col'] == 'rsg'
        assert meta['stratify_by'] == ['rsg']
        assert meta['value_col'] == 'xbar'


class TestSChartStratified:
    """Companion S chart must also be stratified."""

    def test_companion_s_stratified(self, sds1_study):
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True
        )
        s_chart = result.charts['S']
        assert 'strata' in s_chart
        assert s_chart['metadata']['stratified'] is True

    def test_companion_s_matching_strata(self, sds1_study):
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True
        )
        xbar_strata = result.charts['Xbar']['strata']
        s_strata = result.charts['S']['strata']
        assert xbar_strata == s_strata

    def test_s_reference_values_combo_1_2(self, sds1_study):
        """Validate S chart statistics for combo 1_2."""
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], companion=True
        )
        stats = result.charts['S']['statistics']['1_2']
        assert stats['center'] == pytest.approx(0.80, abs=0.01)
        assert stats['lpl'] == pytest.approx(0.0, abs=0.01)
        assert stats['upl'] == pytest.approx(1.68, abs=0.01)


class TestDefaultBehaviorUnchanged:
    """Existing by=None and by=[factor] paths must not regress."""

    def test_by_none_no_strata(self, sds1_study):
        """Default by=None: Kt-level single chart, no stratification."""
        result = sds1_study.execute(chart='Xbar')
        xbar = result.charts['Xbar']
        assert 'strata' not in xbar
        assert xbar['metadata'].get('stratified') is None

    def test_by_factor_no_strata(self, sds1_study):
        """by=[FACTOR 1]: factor levels on x-axis, no stratification."""
        result = sds1_study.execute(chart='Xbar', by=['FACTOR 1'])
        xbar = result.charts['Xbar']
        assert 'strata' not in xbar
        assert xbar['metadata'].get('stratified') is None


class TestResidualNotStratified:
    """Residuals have factor effects removed — stratification is meaningless."""

    def test_r4_by_time_not_stratified(self, sds1_study):
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], value='R4'
        )
        xbar = result.charts['Xbar']
        assert 'strata' not in xbar
        assert xbar['metadata'].get('stratified') is None

    def test_r5_by_time_not_stratified(self, sds1_study):
        result = sds1_study.execute(
            chart='Xbar', by=['PRODUCTION TIME'], value='R5'
        )
        xbar = result.charts['Xbar']
        assert 'strata' not in xbar
        assert xbar['metadata'].get('stratified') is None
