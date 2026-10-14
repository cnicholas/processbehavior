"""Tests for recentered=True mR inflation fix.

When recentered=True, moving ranges must be computed from non-recentered
residuals (R*) to avoid structural jumps between cells inflating mR.
The plotted values remain recentered (RCR*).
"""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior
from processbehavior.analysis import Analysis


@pytest.fixture
def sds2_study():
    """PM SDS 2 study from validation dataset."""
    df = pd.read_csv('validation/PBTESTDATABASE_T100.csv')
    pb = ProcessBehavior(df)
    return pb.formulate(
        response='PM SDS 2',
        factors=['FACTOR 1', 'FACTOR 2'],
        time='PRODUCTION TIME',
    )


class TestResolveMrSourceColumn:
    """Unit tests for _resolve_mr_source_column."""

    def test_rcr_maps_to_r(self):
        assert Analysis._resolve_mr_source_column('RCR3') == 'R3'

    def test_rcr6_maps_to_r6(self):
        assert Analysis._resolve_mr_source_column('RCR6') == 'R6'

    def test_non_recentered_passthrough(self):
        assert Analysis._resolve_mr_source_column('R3') == 'R3'

    def test_response_var_passthrough(self):
        assert Analysis._resolve_mr_source_column('y') == 'y'


class TestRecenteredXmRLimits:
    """Recentered XmR limits should have same width as non-recentered."""

    @pytest.mark.parametrize('residual', ['R2', 'R3'])
    def test_xmr_limit_width_matches(self, sds2_study, residual):
        """Limit half-width must be identical with and without recentering."""
        result_plain = sds2_study.execute(
            chart='X',
            value=residual,
            by=[],
        )
        result_rc = sds2_study.execute(
            chart='X',
            value=residual,
            by=[],
            recentered=True,
        )

        stats_plain = result_plain.charts['X']['statistics']
        stats_rc = result_rc.charts['X']['statistics']

        width_plain = stats_plain['upl'] - stats_plain['lpl']
        width_rc = stats_rc['upl'] - stats_rc['lpl']

        assert width_plain == pytest.approx(width_rc, abs=0.01), (
            f'Recentered {residual} XmR limit width {width_rc:.4f} != non-recentered {width_plain:.4f}'
        )

    def test_xmr_r3_reference_values(self, sds2_study):
        """SDS 2, PM2, R3, by=[], XmR recentered: match Tom's VAS run of 10/3/2026."""
        result = sds2_study.execute(
            chart='X',
            value='R3',
            by=[],
            recentered=True,
        )
        stats = result.charts['X']['statistics']

        # Tom's 10-1 VAS (PM SDS 2 deck, slide 28): CL=237.79, LBL=233.04, UBL=242.53
        assert stats['center'] == pytest.approx(237.79, abs=0.01)
        assert stats['lpl'] == pytest.approx(233.04, abs=0.01)
        assert stats['upl'] == pytest.approx(242.53, abs=0.01)


class TestRecenteredRChart:
    """Companion R chart should be unaffected by recentering."""

    @pytest.mark.parametrize('residual', ['R2', 'R3'])
    def test_r_chart_limits_match(self, sds2_study, residual):
        """R chart CL/UPL must be identical with and without recentering."""
        result_plain = sds2_study.execute(
            chart='X',
            value=residual,
            by=[],
            companion=True,
        )
        result_rc = sds2_study.execute(
            chart='X',
            value=residual,
            by=[],
            companion=True,
            recentered=True,
        )

        r_stats_plain = result_plain.charts['mR']['statistics']
        r_stats_rc = result_rc.charts['mR']['statistics']

        assert r_stats_plain['center'] == pytest.approx(r_stats_rc['center'], abs=0.01)
        assert r_stats_plain['upl'] == pytest.approx(r_stats_rc['upl'], abs=0.01)
        assert r_stats_plain['lpl'] == pytest.approx(r_stats_rc['lpl'], abs=0.01)

    def test_r_chart_r3_reference_values(self, sds2_study):
        """mR chart of R3 matches Tom's VAS run of 10/3/2026 (PM SDS 2 deck, slide 29): CL=1.78, UBL=5.83."""
        result = sds2_study.execute(
            chart='X',
            value='R3',
            by=[],
            companion=True,
        )
        r_stats = result.charts['mR']['statistics']

        assert r_stats['center'] == pytest.approx(1.78, abs=0.01)
        assert r_stats['upl'] == pytest.approx(5.83, abs=0.01)
        assert r_stats['lpl'] == pytest.approx(0.0, abs=0.01)


class TestRecenteredStratified:
    """Stratified (default by=None) path should also use non-recentered mR."""

    def test_stratified_xmr_width_matches(self, sds2_study):
        """Stratified by=['FACTOR 1']: limit width identical with recentering."""
        result_plain = sds2_study.execute(
            chart='X',
            value='R3',
            by=['FACTOR 1'],
        )
        result_rc = sds2_study.execute(
            chart='X',
            value='R3',
            by=['FACTOR 1'],
            recentered=True,
        )

        xmr_plain = result_plain.charts['X']
        xmr_rc = result_rc.charts['X']

        # Both should be stratified
        assert xmr_plain['metadata']['stratified'] is True
        assert xmr_rc['metadata']['stratified'] is True

        # Check per-stratum limit widths match
        for stratum in xmr_plain['strata']:
            sp = xmr_plain['statistics'][stratum]
            sr = xmr_rc['statistics'][stratum]
            width_plain = sp['upl'] - sp['lpl']
            width_rc = sr['upl'] - sr['lpl']
            assert width_plain == pytest.approx(width_rc, abs=0.01), (
                f'Stratum {stratum}: recentered width {width_rc:.4f} != non-recentered {width_plain:.4f}'
            )


class TestRecenteredR6:
    """R6 should have both R6 and RCR6 when recentered=True."""

    def test_r6_xbar_limit_width_matches(self, sds2_study):
        """R6 recentered Xbar limits should have same width as non-recentered."""
        result_plain = sds2_study.execute(
            chart='Xbar',
            value='R6',
            by=['FACTOR 1'],
        )
        result_rc = sds2_study.execute(
            chart='Xbar',
            value='R6',
            by=['FACTOR 1'],
            recentered=True,
        )

        plain_data = result_plain.charts['Xbar']['data']
        rc_data = result_rc.charts['Xbar']['data']

        width_plain = (plain_data['upl'] - plain_data['lpl']).iloc[0]
        width_rc = (rc_data['upl'] - rc_data['lpl']).iloc[0]

        assert width_plain == pytest.approx(width_rc, abs=0.01), (
            f'R6 recentered Xbar width {width_rc:.4f} != non-recentered {width_plain:.4f}'
        )


class TestNonRecenteredUnchanged:
    """Non-recentered paths must produce identical results (regression guard)."""

    def test_non_recentered_xmr_unchanged(self, sds2_study):
        """Default (recentered=False) XmR should be byte-for-byte stable."""
        result = sds2_study.execute(chart='X', value='R3', by=[])
        stats = result.charts['X']['statistics']

        # Half-width of Tom's 10-1 VAS limits (PM SDS 2 deck, slide 28): (242.53 - 233.04) / 2
        half_width = (stats['upl'] - stats['lpl']) / 2
        assert half_width == pytest.approx(4.745, abs=0.01)
