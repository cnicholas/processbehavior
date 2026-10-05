"""Individuals and moving-range chart limits match Bishop's VAS software.

The expected values are read from Bishop's VAS output decks of 29 September 2026 (Medicare and
PM SDS 2) and, for the R2 chart, 3 October 2026, at the precision the charts print. They pin the
XmR constants E2 = 3/1.128 and D4 = 1 + 3(0.8525)/1.128 from the manual (Eq 12.4, 12.5, 12.10,
12.11): with the rounded 2.66 and 3.268 the Medicare limits miss by 0.6 and 1.0, and the PM SDS 2
moving-range limit prints 2.84.
"""

from pathlib import Path

import pandas as pd
import pytest

import processbehavior as pb

VALIDATION = Path(__file__).resolve().parent.parent / 'validation'
MEDICARE_CSV = VALIDATION / 'aco_per_capita_expenditure.csv'
T100_CSV = VALIDATION / 'PBTESTDATABASE_T100.csv'


# precision=6 so the comparison rounds once, at the precision VAS prints.
@pytest.fixture(scope='module')
def medicare():
    df = pd.read_csv(MEDICARE_CSV)
    return pb.formulate(df, response='PER CAPITA EXPENDITURE', factors=['ACO'], time='YEAR', precision=6)


@pytest.fixture(scope='module')
def pm_sds_2():
    df = pd.read_csv(T100_CSV, na_values=['*'])
    return pb.formulate(df, response='PM SDS 2', factors=['FACTOR 1', 'FACTOR 2'], time='PRODUCTION TIME', precision=6)


class TestMedicareFirstChart:
    """VAS MEDICARE ANALYSIS, slides 1-2: individuals chart ±3838.32, moving range 1443.21 / 4715.38."""

    def test_individuals_half_width(self, medicare):
        s = medicare.execute(chart='X', by=[]).get_statistics('X')
        assert s['upl'] - s['center'] == pytest.approx(3838.32, abs=0.005)
        assert s['center'] - s['lpl'] == pytest.approx(3838.32, abs=0.005)

    def test_moving_range_limits(self, medicare):
        s = medicare.execute(chart='mR', by=[]).get_statistics('mR')
        assert s['center'] == pytest.approx(1443.21, abs=0.005)
        assert s['upl'] == pytest.approx(4715.38, abs=0.005)


class TestMedicarePerOrganisationCharts:
    """VAS MEDICARE ANALYSIS, slides 3-6: the per-organisation individuals and moving-range charts.

    VAS prints six significant figures and appears to place the printed limits about a centre line
    already rounded to that precision (ACO-002: 12637.4 + 2169.97 = 14807.37, printed 14807.4; the exact
    limit is 14807.346), so one-decimal values allow 0.1.
    """

    @pytest.mark.parametrize(
        'aco, lpl, upl, lpl_tol, upl_tol',
        [('ACO-001', 7503.91, 11991.3, 0.005, 0.05), ('ACO-002', 10467.4, 14807.4, 0.1, 0.1)],
    )
    def test_individuals_limits(self, medicare, aco, lpl, upl, lpl_tol, upl_tol):
        t = medicare.execute(chart='X', by=['ACO']).chart_table()
        row = t[t['subgroup'] == aco].iloc[0]
        assert row['lpl'] == pytest.approx(lpl, abs=lpl_tol)
        assert row['upl'] == pytest.approx(upl, abs=upl_tol)

    @pytest.mark.parametrize('aco, center, upl', [('ACO-001', 843.62, 2756.36), ('ACO-002', 815.91, 2665.81)])
    def test_moving_range_limits(self, medicare, aco, center, upl):
        t = medicare.execute(chart='mR', by=['ACO']).chart_table()
        row = t[t['subgroup'] == aco].iloc[0]
        assert row['center'] == pytest.approx(center, abs=0.005)
        assert row['upl'] == pytest.approx(upl, abs=0.005)


class TestPmSds2FirstChart:
    """VAS PM SDS 2 ANALYSIS, slides 1-2: 237.78 (235.47 / 240.09), moving range 0.87 / 2.83."""

    def test_individuals_limits(self, pm_sds_2):
        s = pm_sds_2.execute(chart='X', by=[]).get_statistics('X')
        assert round(s['center'], 2) == 237.78
        assert round(s['lpl'], 2) == 235.47
        assert round(s['upl'], 2) == 240.09

    def test_moving_range_limits(self, pm_sds_2):
        s = pm_sds_2.execute(chart='mR', by=[]).get_statistics('mR')
        assert round(s['center'], 2) == 0.87
        assert round(s['upl'], 2) == 2.83


class TestMedicareR2Chart:
    """VAS run of 10/3/2026 (10-1 manual R2), MEDICARE PCE slide 60: R2 chart -1.15 (-1045.19 / 1042.88).

    R2 here is the condition-and-period-adjusted series differenced along the ACO-then-year
    stream and divided by twice the R2 scale factor c(24, 96); the raw-difference R2 of
    earlier releases gave limits about three times wider.
    """

    def test_maximum_information_limits(self, medicare):
        mi = medicare.maximum_information()
        assert mi.r2_mean == pytest.approx(-1.15, abs=0.005)
        assert mi.lpl == pytest.approx(-1045.19, abs=0.005)
        assert mi.upl == pytest.approx(1042.88, abs=0.005)

    def test_individuals_chart_on_r2(self, medicare):
        s = medicare.execute(chart='X', by=[], value='R2').get_statistics('X')
        assert s['center'] == pytest.approx(-1.15, abs=0.005)
        assert s['lpl'] == pytest.approx(-1045.19, abs=0.005)
        assert s['upl'] == pytest.approx(1042.88, abs=0.005)


class TestMedicarePotentialCapability:
    """VAS run of 10/3/2026, MEDICARE PCE slide 59 (LSL 6000, USL 16000): PPL 5.36, PP 5.55, PPU 5.74.

    The potential indices are measured from the centre of the potential values, y_bar + mean(R2)
    ("PROCESS MEAN = 10831.3" on the slide; y_bar is 10832.4 and mean R2 is -1.15). Only PPU
    discriminates at two decimals: 5.7355 from that centre, 5.734 from y_bar.
    """

    def test_potential_indices(self, medicare):
        cap = medicare.capability(lsl=6000, usl=16000)
        assert cap.potential_center == pytest.approx(10831.27, abs=0.01)
        assert round(cap.cpk_lower, 2) == 5.36
        assert round(cap.cp, 2) == 5.55
        assert round(cap.cpk_upper, 2) == 5.74
