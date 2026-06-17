"""Tests for named Calibrations (standards-given control limits).

The calibration limit math is determined by the calibration's (mean, sigma)
plus the per-chart sampling-distribution form — so the *closed form* is ground
truth here, not Bishop's residual reference data. Each test asserts the limits
equal the analytic standards-given expression exactly.
"""

import dataclasses
import math

import pytest

import processbehavior as pb
from processbehavior import Calibration, make_design
from processbehavior.spc_constants import D2_N2, R_UPPER_LIMIT_MULTIPLIER, c4


@pytest.fixture
def study():
    df = make_design(1, seed=42)
    return pb.ProcessBehavior(df).formulate(
        response='y', time='time', factors=['factor 1', 'factor 2']
    )


@pytest.fixture
def cal():
    return Calibration(label='cal', mean=100.0, sigma=2.0)


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------


def test_calibration_constructs_and_is_frozen():
    c = Calibration('base', 10.0, 0.5)
    assert (c.label, c.mean, c.sigma) == ('base', 10.0, 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.sigma = 1.0  # frozen


def test_calibration_value_equality():
    assert Calibration('a', 1.0, 2.0) == Calibration('a', 1.0, 2.0)
    assert Calibration('a', 1.0, 2.0) != Calibration('a', 1.0, 3.0)


@pytest.mark.parametrize(
    'args',
    [
        ('', 1.0, 1.0),  # empty label
        ('  ', 1.0, 1.0),  # blank label
        ('x', float('nan'), 1.0),  # non-finite mean
        ('x', float('inf'), 1.0),
        ('x', 1.0, 0.0),  # sigma not > 0
        ('x', 1.0, -1.0),
        ('x', 1.0, float('nan')),
    ],
)
def test_calibration_rejects_invalid_fields(args):
    with pytest.raises(pb.ValidationError):
        Calibration(*args)


# ---------------------------------------------------------------------------
# Named set on Study
# ---------------------------------------------------------------------------


def test_with_calibration_is_immutable(study, cal):
    s2 = study.with_calibration(cal)
    assert dict(study.calibrations) == {}  # original untouched
    assert s2.calibrations['cal'] == cal


def test_execute_resolves_label_and_object(study, cal):
    s2 = study.with_calibration(cal)
    by_label = s2.execute(chart='X', by=[], calibration='cal').get_statistics('X')
    by_obj = study.execute(chart='X', by=[], calibration=cal).get_statistics('X')
    assert by_label == by_obj


def test_unknown_label_raises_self_diagnostic(study, cal):
    s2 = study.with_calibration(cal)
    with pytest.raises(pb.ValidationError, match="No calibration labeled 'nope'"):
        s2.execute(chart='X', by=[], calibration='nope')


# ---------------------------------------------------------------------------
# Location charts — sigma used directly, no constants
# ---------------------------------------------------------------------------


def test_x_individuals_limits_are_mean_plus_minus_3sigma(study, cal):
    stats = study.execute(chart='X', by=[], calibration=cal).get_statistics('X')
    assert stats['center'] == pytest.approx(100.0)
    assert stats['lpl'] == pytest.approx(100.0 - 3 * 2.0)  # 94
    assert stats['upl'] == pytest.approx(100.0 + 3 * 2.0)  # 106


def test_xbar_limits_carry_sqrt_n(study, cal):
    res = study.execute(chart='Xbar', calibration=cal)
    data = res.get_chart('Xbar')
    # center is the calibration mean on every subgroup
    assert (data['center'] == 100.0).all()
    # half-width = n_sigma * sigma / sqrt(N); recover the integer N and assert
    # the displayed (rounded) limit equals the closed form for that N exactly.
    for _, row in data.iterrows():
        half = row['upl'] - 100.0
        n = round((3 * 2.0 / half) ** 2)
        assert n >= 2
        assert row['upl'] == pytest.approx(round(100.0 + 3 * 2.0 / math.sqrt(n), 3))
        assert row['lpl'] == pytest.approx(round(100.0 - 3 * 2.0 / math.sqrt(n), 3))


def test_calibration_metadata_badge(study, cal):
    res = study.execute(chart='X', by=[], calibration=cal)
    meta = res.charts['X']['metadata']
    assert meta['limits_source'] == 'calibration'
    assert meta['calibration'] == {'label': 'cal', 'mean': 100.0, 'sigma': 2.0}


# ---------------------------------------------------------------------------
# Residual centering: plain residual -> 0, recentered -> mean
# ---------------------------------------------------------------------------


def test_plain_residual_centers_at_zero(study, cal):
    stats = study.execute(chart='X', value='R2', by=[], calibration=cal).get_statistics('X')
    assert stats['center'] == pytest.approx(0.0)
    assert stats['lpl'] == pytest.approx(-6.0)
    assert stats['upl'] == pytest.approx(6.0)


def test_recentered_residual_uses_calibration_mean(study, cal):
    stats = (
        study.execute(chart='X', value='R2', recentered=True, by=[], calibration=cal)
        .get_statistics('X')
    )
    assert stats['center'] == pytest.approx(100.0)
    assert stats['upl'] == pytest.approx(106.0)


# ---------------------------------------------------------------------------
# Dispersion charts — standards-given constants (documented exception)
# ---------------------------------------------------------------------------


def test_mr_companion_standards_given(study, cal):
    stats = study.execute(chart='mR', by=[], calibration=cal).get_statistics('mR')
    d2 = D2_N2
    assert stats['center'] == pytest.approx(d2 * 2.0, abs=5e-4)
    assert stats['lpl'] == pytest.approx(0.0)
    assert stats['upl'] == pytest.approx(R_UPPER_LIMIT_MULTIPLIER * d2 * 2.0, abs=5e-4)


def test_s_chart_standards_given(study, cal):
    data = study.execute(chart='S', calibration=cal).get_chart('S')
    sigma = 2.0
    # Recover the integer N whose c4(N)*sigma rounds to the displayed center,
    # then assert center/limits equal the standards-given closed form exactly:
    #   center = c4(N)*sigma; upl = (c4 + 3*sqrt(1-c4^2))*sigma; lpl clamps at 0.
    for _, row in data.iterrows():
        n = next(
            k for k in range(2, 100)
            if round(c4(k) * sigma, 3) == row['center']
        )
        spread = 3 * math.sqrt(1 - c4(n) ** 2) * sigma
        assert row['center'] == pytest.approx(round(c4(n) * sigma, 3))
        assert row['upl'] == pytest.approx(round(c4(n) * sigma + spread, 3))
        assert row['lpl'] == pytest.approx(round(max(0.0, c4(n) * sigma - spread), 3))


# ---------------------------------------------------------------------------
# n_sigma policy: composes on Xbar/S, rejected on X/mR
# ---------------------------------------------------------------------------


def test_n_sigma_composes_on_xbar(study, cal):
    res = study.execute(chart='Xbar', calibration=cal, n_sigma=2.0)
    data = res.get_chart('Xbar')
    for _, row in data.iterrows():
        half = row['upl'] - 100.0
        n = round((2.0 * 2.0 / half) ** 2)  # n_sigma=2 now
        assert n >= 2
        assert row['upl'] == pytest.approx(round(100.0 + 2.0 * 2.0 / math.sqrt(n), 3))


@pytest.mark.parametrize('chart', ['X', 'mR'])
def test_calibrated_individuals_reject_non_default_n_sigma(study, cal, chart):
    with pytest.raises(pb.ValidationError):
        study.execute(chart=chart, by=[], calibration=cal, n_sigma=2.5)


# ---------------------------------------------------------------------------
# Regression: calibration=None is byte-for-byte the data-derived path
# ---------------------------------------------------------------------------


def test_calibration_none_matches_uncalibrated(study, cal):
    base = study.execute(chart='X', by=[]).get_statistics('X')
    explicit_none = study.execute(chart='X', by=[], calibration=None).get_statistics('X')
    assert base == explicit_none
    # and the calibrated path actually differs (sanity: feature has an effect)
    calibrated = study.execute(chart='X', by=[], calibration=cal).get_statistics('X')
    assert calibrated != base


# ---------------------------------------------------------------------------
# Unsupported paths are rejected with a clear error (documented follow-up)
# ---------------------------------------------------------------------------


def test_phased_calibration_rejected(study, cal):
    with pytest.raises(pb.ValidationError, match='phased'):
        study.execute(chart='X', by=[], calibration=cal, phased=True)


def test_stratified_calibration_rejected(study, cal):
    with pytest.raises(pb.ValidationError, match='stratified'):
        study.execute(chart='X', by=['factor 1', 'factor 2'], calibration=cal)


# ---------------------------------------------------------------------------
# Helper-level exactness (closed-form ground truth, no rounding/aggregation)
# ---------------------------------------------------------------------------


def test_calibrated_limits_closed_forms():
    from processbehavior.spc_constants import calibrated_limits

    sigma, mean, k = 2.0, 100.0, 3.0

    # X individuals: mean ± 3·sigma, no constants.
    c, lims = calibrated_limits('XmR', mean=mean, sigma=sigma)
    assert (c, lims['lpl'], lims['upl']) == pytest.approx((100.0, 94.0, 106.0))

    # mR companion: center d2·sigma, 0 … D4·d2·sigma.
    c, lims = calibrated_limits('R', mean=mean, sigma=sigma)
    assert c == pytest.approx(D2_N2 * sigma)
    assert lims['lpl'] == 0
    assert lims['upl'] == pytest.approx(R_UPPER_LIMIT_MULTIPLIER * D2_N2 * sigma)

    # Xbar: mean ± k·sigma/√N (c4 cancels).
    for n in (2, 5, 9):
        c, lims = calibrated_limits('Xbar', mean=mean, sigma=sigma, N=n, n_sigma=k)
        assert c == mean
        assert lims['upl'] == pytest.approx(mean + k * sigma / math.sqrt(n))

    # S: center c4(N)·sigma, (c4 ∓ k√(1−c4²))·sigma = B5..B6·sigma.
    for n in (2, 5, 9):
        c, lims = calibrated_limits('S', mean=mean, sigma=sigma, N=n, n_sigma=k)
        assert c == pytest.approx(c4(n) * sigma)
        assert lims['upl'] == pytest.approx((c4(n) + k * math.sqrt(1 - c4(n) ** 2)) * sigma)

    # center_zero pins location center at 0 (plain residual).
    c, lims = calibrated_limits('XmR', mean=mean, sigma=sigma, center_zero=True)
    assert (c, lims['lpl'], lims['upl']) == pytest.approx((0.0, -6.0, 6.0))
