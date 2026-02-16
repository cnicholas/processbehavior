"""
Tests for process capability analysis (Wheeler/Bishop Chapter 16).

Covers:
- SpecLimits validation
- Pure function unit tests (compute_sigma_hat, compute_capability_indices, compute_pct_outside)
- Hand-calculated Pp/Ppk with known data
- Hand-calculated Cp/Cpk with known R2
- One-sided specs (USL only, LSL only)
- Both limits (full Pp, Ppk)
- SDS 1 integration (has R2)
- No-R2 integration (factors without time → Cp/Cpk unavailable)
- Edge cases: σ=0, N<2, NaN values
- Scale and shift invariance
- Z-scores = 3 × Ppk algebraic consistency
- kwargs API parity with SpecLimits
- as_dict rounding vs raw storage
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processbehavior import ProcessBehavior, SpecLimits, ValidationError
from processbehavior.capability import (
    CapabilityResult,
    assess_capability,
    compute_capability_indices,
    compute_pct_outside,
    compute_sigma_hat,
)
from processbehavior.spc_constants import c4

# ============================================================================
# Helpers
# ============================================================================


def _make_sds1_study():
    """SDS 1 study: factorial + time + replication → has R2."""
    from processbehavior.datasets.synthetic import make_sds

    df = make_sds(1, K1=2, K2=2, T=4, n=3, seed=42)
    pb = ProcessBehavior(df)
    return pb.formulate(
        response="y",
        factors=["factor 1", "factor 2"],
        time="time",
    )


def _make_no_r2_study():
    """Study without R2: factors but no time → no VAS residuals."""
    from processbehavior.datasets.synthetic import make_sds

    df = make_sds(1, K1=2, K2=2, T=4, n=3, seed=42)
    pb = ProcessBehavior(df)
    # Omit time → has_time=False → no VAS residuals
    return pb.formulate(
        response="y",
        factors=["factor 1", "factor 2"],
    )


# ============================================================================
# SpecLimits Validation
# ============================================================================


class TestSpecLimits:
    """Validation of SpecLimits frozen dataclass."""

    def test_two_sided(self):
        """Both USL and LSL provided — normal case."""
        specs = SpecLimits(usl=10, lsl=5)
        assert specs.usl == 10
        assert specs.lsl == 5
        assert specs.target is None
        assert specs.is_two_sided is True

    def test_usl_only(self):
        """USL-only is valid one-sided spec."""
        specs = SpecLimits(usl=10)
        assert specs.usl == 10
        assert specs.lsl is None
        assert specs.is_two_sided is False

    def test_lsl_only(self):
        """LSL-only is valid one-sided spec."""
        specs = SpecLimits(lsl=5)
        assert specs.lsl == 5
        assert specs.usl is None
        assert specs.is_two_sided is False

    def test_with_target(self):
        """Target between LSL and USL is valid."""
        specs = SpecLimits(usl=10, lsl=5, target=7.5)
        assert specs.target == 7.5

    def test_target_at_boundary(self):
        """Target exactly at LSL or USL is allowed (<=)."""
        SpecLimits(usl=10, lsl=5, target=5)
        SpecLimits(usl=10, lsl=5, target=10)

    def test_no_limits_raises(self):
        """No spec limits at all is an error."""
        with pytest.raises(ValidationError, match="At least one"):
            SpecLimits()

    def test_lsl_ge_usl_raises(self):
        """LSL >= USL is an error."""
        with pytest.raises(ValidationError, match="LSL.*must be less than USL"):
            SpecLimits(usl=5, lsl=10)

    def test_lsl_eq_usl_raises(self):
        """LSL == USL is an error."""
        with pytest.raises(ValidationError, match="LSL.*must be less than USL"):
            SpecLimits(usl=5, lsl=5)

    def test_target_outside_range_raises(self):
        """Target outside [LSL, USL] is an error (two-sided)."""
        with pytest.raises(ValidationError, match="Target.*must be between"):
            SpecLimits(usl=10, lsl=5, target=11)

        with pytest.raises(ValidationError, match="Target.*must be between"):
            SpecLimits(usl=10, lsl=5, target=4)

    def test_target_with_one_sided_permissive(self):
        """Target with one-sided spec — no error in v1."""
        SpecLimits(usl=10, target=12)  # target > usl is fine for one-sided
        SpecLimits(lsl=5, target=3)    # target < lsl is fine for one-sided

    def test_frozen(self):
        """SpecLimits is immutable."""
        specs = SpecLimits(usl=10, lsl=5)
        with pytest.raises(AttributeError):
            specs.usl = 20


# ============================================================================
# Pure Function Tests
# ============================================================================


class TestComputeSigmaHat:
    """Tests for compute_sigma_hat — sample std and unbiased estimate."""

    def test_known_values(self):
        """Hand-calculated: [2, 4, 4, 4, 5, 5, 7, 9] → S=2.138, σ̂=S/c4(8)."""
        values = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
        s, sigma_hat = compute_sigma_hat(values)

        expected_s = np.std(values, ddof=1)
        assert s == pytest.approx(expected_s)
        assert sigma_hat == pytest.approx(expected_s / c4(8))

    def test_ddof_1_explicit(self):
        """Verify ddof=1 is used, not ddof=0."""
        values = np.array([10.0, 20.0])
        s, _ = compute_sigma_hat(values)
        # ddof=1: sqrt((10-15)^2 + (20-15)^2) / 1) = sqrt(50) ≈ 7.071
        # ddof=0 would give sqrt(25) = 5.0
        assert s == pytest.approx(np.sqrt(50), rel=1e-10)

    def test_large_n(self):
        """c4(N) approaches 1.0 for large N — numerically stable."""
        rng = np.random.default_rng(42)
        values = rng.normal(100, 10, size=1000)
        s, sigma_hat = compute_sigma_hat(values)
        # c4(1000) ≈ 0.9998... so sigma_hat ≈ s
        assert sigma_hat == pytest.approx(s, rel=0.001)

    def test_identical_values(self):
        """All identical → S=0, sigma_hat=0."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        s, sigma_hat = compute_sigma_hat(values)
        assert s == 0.0
        assert sigma_hat == 0.0


class TestComputeCapabilityIndices:
    """Tests for compute_capability_indices — Pp/Ppk computation."""

    def test_two_sided_centered(self):
        """Process centered on target with known sigma."""
        specs = SpecLimits(usl=106, lsl=94, target=100)
        result = compute_capability_indices(y_bar=100.0, sigma_hat=2.0, specs=specs)

        assert result["pp"] == pytest.approx((106 - 94) / (6 * 2.0))  # 1.0
        assert result["ppk_lower"] == pytest.approx((100 - 94) / (3 * 2.0))  # 1.0
        assert result["ppk_upper"] == pytest.approx((106 - 100) / (3 * 2.0))  # 1.0
        assert result["ppk"] == pytest.approx(1.0)

    def test_two_sided_shifted(self):
        """Process shifted toward USL."""
        specs = SpecLimits(usl=106, lsl=94)
        result = compute_capability_indices(y_bar=102.0, sigma_hat=2.0, specs=specs)

        assert result["pp"] == pytest.approx(1.0)
        assert result["ppk_lower"] == pytest.approx((102 - 94) / 6)  # 1.333
        assert result["ppk_upper"] == pytest.approx((106 - 102) / 6)  # 0.667
        assert result["ppk"] == pytest.approx(0.667, rel=0.01)

    def test_usl_only(self):
        """USL-only → pp=None, ppk=ppk_upper, ppk_lower=None."""
        specs = SpecLimits(usl=106)
        result = compute_capability_indices(y_bar=100.0, sigma_hat=2.0, specs=specs)

        assert result["pp"] is None
        assert result["ppk_lower"] is None
        assert result["ppk_upper"] == pytest.approx((106 - 100) / 6)
        assert result["ppk"] == result["ppk_upper"]

    def test_lsl_only(self):
        """LSL-only → pp=None, ppk=ppk_lower, ppk_upper=None."""
        specs = SpecLimits(lsl=94)
        result = compute_capability_indices(y_bar=100.0, sigma_hat=2.0, specs=specs)

        assert result["pp"] is None
        assert result["ppk_upper"] is None
        assert result["ppk_lower"] == pytest.approx((100 - 94) / 6)
        assert result["ppk"] == result["ppk_lower"]

    def test_sigma_zero(self):
        """σ=0 → indices are inf, no crash."""
        specs = SpecLimits(usl=106, lsl=94)
        result = compute_capability_indices(y_bar=100.0, sigma_hat=0.0, specs=specs)

        assert result["pp"] == float("inf")
        assert result["ppk"] == float("inf")
        assert result["z_lower"] == float("inf")
        assert result["z_upper"] == float("inf")

    def test_z_scores(self):
        """Z-scores: z_lower = (Ȳ-LSL)/σ̂, z_upper = (USL-Ȳ)/σ̂."""
        specs = SpecLimits(usl=106, lsl=94)
        result = compute_capability_indices(y_bar=100.0, sigma_hat=2.0, specs=specs)

        assert result["z_lower"] == pytest.approx(3.0)
        assert result["z_upper"] == pytest.approx(3.0)

    def test_z_scores_equal_3_times_ppk(self):
        """Algebraic consistency: z = 3 × ppk."""
        specs = SpecLimits(usl=110, lsl=90)
        result = compute_capability_indices(y_bar=103.0, sigma_hat=2.5, specs=specs)

        assert result["z_lower"] == pytest.approx(3 * result["ppk_lower"])
        assert result["z_upper"] == pytest.approx(3 * result["ppk_upper"])


class TestComputePctOutside:
    """Tests for compute_pct_outside — empirical counts."""

    def test_both_limits(self):
        """Count-based: 2 below, 1 above, out of 10."""
        values = np.array([1, 2, 5, 5, 5, 5, 5, 5, 5, 9], dtype=float)
        specs = SpecLimits(usl=8, lsl=3)
        result = compute_pct_outside(values, specs)

        assert result["n_below_lsl"] == 2
        assert result["n_above_usl"] == 1
        assert result["n_outside"] == 3
        assert result["pct_below_lsl"] == pytest.approx(20.0)
        assert result["pct_above_usl"] == pytest.approx(10.0)
        assert result["pct_outside"] == pytest.approx(30.0)

    def test_none_outside(self):
        """All within limits."""
        values = np.array([5, 6, 7, 8, 9], dtype=float)
        specs = SpecLimits(usl=10, lsl=4)
        result = compute_pct_outside(values, specs)

        assert result["n_outside"] == 0
        assert result["pct_outside"] == pytest.approx(0.0)

    def test_usl_only(self):
        """USL-only: n_below_lsl and pct_below_lsl are None."""
        values = np.array([5, 6, 7, 11], dtype=float)
        specs = SpecLimits(usl=10)
        result = compute_pct_outside(values, specs)

        assert result["n_below_lsl"] is None
        assert result["pct_below_lsl"] is None
        assert result["n_above_usl"] == 1
        assert result["n_outside"] == 1

    def test_lsl_only(self):
        """LSL-only: n_above_usl and pct_above_usl are None."""
        values = np.array([1, 5, 6, 7], dtype=float)
        specs = SpecLimits(lsl=3)
        result = compute_pct_outside(values, specs)

        assert result["n_above_usl"] is None
        assert result["pct_above_usl"] is None
        assert result["n_below_lsl"] == 1
        assert result["n_outside"] == 1

    def test_at_boundary_not_outside(self):
        """Values exactly at spec limits are NOT outside (strict </>)."""
        values = np.array([5.0, 10.0, 7.5], dtype=float)
        specs = SpecLimits(usl=10, lsl=5)
        result = compute_pct_outside(values, specs)

        assert result["n_outside"] == 0


# ============================================================================
# CapabilityResult Presentation
# ============================================================================


class TestCapabilityResultPresentation:
    """Tests for as_dict rounding and __repr__."""

    @pytest.fixture
    def sample_result(self):
        """A capability result with known raw values."""
        return CapabilityResult(
            specs=SpecLimits(usl=10, lsl=5),
            n=100,
            y_bar=7.12345,
            s=1.23456,
            sigma_hat=1.24567,
            pp=0.66789,
            ppk_lower=1.42345,
            ppk_upper=0.38901,
            ppk=0.38901,
            sigma_hat_r2=0.98765,
            cp=0.84321,
            cpk_lower=1.79654,
            cpk_upper=0.49123,
            cpk=0.49123,
            potential_unavailable_reason=None,
            z_lower=4.27035,
            z_upper=1.16703,
            n_below_lsl=2,
            n_above_usl=5,
            n_outside=7,
            pct_below_lsl=2.0,
            pct_above_usl=5.0,
            pct_outside=7.0,
            round_to=3,
        )

    def test_raw_values_unrounded(self, sample_result):
        """Raw float fields are stored without rounding."""
        assert sample_result.y_bar == 7.12345
        assert sample_result.ppk == 0.38901

    def test_as_dict_rounds(self, sample_result):
        """as_dict() rounds to specified precision."""
        d = sample_result.as_dict(round_to=2)
        assert d["y_bar"] == 7.12
        assert d["ppk"] == 0.39
        assert d["pp"] == 0.67

    def test_as_dict_default_round_to(self, sample_result):
        """as_dict() uses self.round_to when no override."""
        d = sample_result.as_dict()
        assert d["y_bar"] == 7.123
        assert d["ppk"] == 0.389

    def test_as_dict_preserves_none(self, sample_result):
        """None values stay None after rounding."""
        result = CapabilityResult(
            specs=SpecLimits(usl=10),
            n=50, y_bar=7.0, s=1.0, sigma_hat=1.01,
            pp=None, ppk_lower=None, ppk_upper=1.5, ppk=1.5,
            sigma_hat_r2=None, cp=None, cpk_lower=None,
            cpk_upper=None, cpk=None,
            potential_unavailable_reason="no R2",
            z_lower=None, z_upper=4.5,
            n_below_lsl=None, n_above_usl=0, n_outside=0,
            pct_below_lsl=None, pct_above_usl=0.0, pct_outside=0.0,
        )
        d = result.as_dict()
        assert d["pp"] is None
        assert d["ppk_lower"] is None
        assert d["z_lower"] is None

    def test_as_dict_preserves_inf(self):
        """inf values pass through rounding unchanged."""
        result = CapabilityResult(
            specs=SpecLimits(usl=10, lsl=5),
            n=50, y_bar=7.5, s=0.0, sigma_hat=0.0,
            pp=float("inf"), ppk_lower=float("inf"),
            ppk_upper=float("inf"), ppk=float("inf"),
            sigma_hat_r2=None, cp=None, cpk_lower=None,
            cpk_upper=None, cpk=None,
            potential_unavailable_reason="no R2",
            z_lower=float("inf"), z_upper=float("inf"),
            n_below_lsl=0, n_above_usl=0, n_outside=0,
            pct_below_lsl=0.0, pct_above_usl=0.0, pct_outside=0.0,
        )
        d = result.as_dict()
        assert d["pp"] == float("inf")

    def test_repr_stability_note(self, sample_result):
        """__repr__ includes stability note when not evaluated."""
        r = repr(sample_result)
        assert "Stability not assessed" in r

    def test_repr_contains_key_values(self, sample_result):
        """__repr__ shows Pp, Ppk, Cp, Cpk."""
        r = repr(sample_result)
        assert "Pp=" in r
        assert "Ppk=" in r
        assert "Cp=" in r
        assert "Cpk=" in r

    def test_repr_potential_unavailable(self):
        """__repr__ shows unavailable reason when Cp/Cpk not computed."""
        result = CapabilityResult(
            specs=SpecLimits(usl=10, lsl=5),
            n=50, y_bar=7.5, s=1.0, sigma_hat=1.01,
            pp=0.83, ppk_lower=0.83, ppk_upper=0.83, ppk=0.83,
            sigma_hat_r2=None, cp=None, cpk_lower=None,
            cpk_upper=None, cpk=None,
            potential_unavailable_reason="R2 residuals not available",
            z_lower=2.48, z_upper=2.48,
            n_below_lsl=0, n_above_usl=0, n_outside=0,
            pct_below_lsl=0.0, pct_above_usl=0.0, pct_outside=0.0,
        )
        r = repr(result)
        assert "R2 residuals not available" in r
        assert "Cp=" not in r

    def test_frozen(self, sample_result):
        """CapabilityResult is immutable."""
        with pytest.raises(AttributeError):
            sample_result.ppk = 99


# ============================================================================
# Hand-Calculated End-to-End Tests
# ============================================================================


class TestHandCalculated:
    """Verify Pp/Ppk/Cp/Cpk against hand-calculated values."""

    def test_pp_ppk_known_data(self):
        """
        Known dataset:
            Y = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
            Ȳ = 19.0
            S = std(Y, ddof=1) = 6.0553
            σ̂ = S / c4(10) = 6.0553 / 0.97227 = 6.2282
            Specs: USL=30, LSL=8
            Pp = (30-8) / (6*6.2282) = 22 / 37.369 = 0.5887
            Ppk_lower = (19-8) / (3*6.2282) = 11 / 18.685 = 0.5887
            Ppk_upper = (30-19) / (3*6.2282) = 11 / 18.685 = 0.5887
            Ppk = 0.5887
        """
        values = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=float)
        specs = SpecLimits(usl=30, lsl=8)

        y_bar = float(np.mean(values))
        assert y_bar == pytest.approx(19.0)

        s, sigma_hat = compute_sigma_hat(values)
        assert s == pytest.approx(6.0553, rel=1e-3)
        assert sigma_hat == pytest.approx(s / c4(10))

        result = compute_capability_indices(y_bar, sigma_hat, specs)
        expected_pp = 22 / (6 * sigma_hat)
        assert result["pp"] == pytest.approx(expected_pp, rel=1e-4)
        # Centered process: ppk_lower ≈ ppk_upper
        assert result["ppk_lower"] == pytest.approx(result["ppk_upper"], rel=1e-4)
        assert result["ppk"] == pytest.approx(expected_pp, rel=1e-4)

    def test_ppk_shifted_process(self):
        """
        Process shifted toward USL.
            Y = [48, 49, 50, 51, 52], Ȳ = 50.0, S = 1.5811
            Specs: USL=55, LSL=44
            σ̂ = S / c4(5)
            Ppk_upper = (55-50) / (3σ̂) < Ppk_lower = (50-44) / (3σ̂)
            Ppk = Ppk_upper (closer to USL)
        """
        values = np.array([48, 49, 50, 51, 52], dtype=float)
        specs = SpecLimits(usl=55, lsl=44)

        y_bar = float(np.mean(values))
        s, sigma_hat = compute_sigma_hat(values)
        result = compute_capability_indices(y_bar, sigma_hat, specs)

        # Ppk should be the smaller one (closer to USL side)
        assert result["ppk_upper"] < result["ppk_lower"]
        assert result["ppk"] == pytest.approx(result["ppk_upper"])


# ============================================================================
# Integration Tests with Study
# ============================================================================


class TestIntegrationSDS1:
    """SDS 1 integration: factorial + time + replication → has R2."""

    @pytest.fixture
    def sds1_study(self):
        """Create a Study from SDS 1 data (has VAS residuals)."""
        return _make_sds1_study()

    def test_has_r2(self, sds1_study):
        """SDS 1 study should have VAS residuals."""
        assert sds1_study._ads.has_vas_residuals

    def test_capability_with_specs(self, sds1_study):
        """capability() returns valid CapabilityResult with Cp/Cpk populated."""
        cap = sds1_study.capability(usl=120, lsl=80)
        assert isinstance(cap, CapabilityResult)
        assert cap.pp is not None
        assert cap.ppk is not None
        assert cap.cp is not None
        assert cap.cpk is not None
        assert cap.sigma_hat_r2 is not None
        assert cap.potential_unavailable_reason is None

    def test_capability_with_spec_limits_object(self, sds1_study):
        """capability() accepts SpecLimits directly."""
        specs = SpecLimits(usl=120, lsl=80, target=100)
        cap = sds1_study.capability(specs)
        assert cap.specs.target == 100

    def test_different_specs_same_study(self, sds1_study):
        """Multiple capability assessments from the same study."""
        cap1 = sds1_study.capability(usl=120, lsl=80)
        cap2 = sds1_study.capability(usl=150, lsl=50)

        # Wider specs → higher capability indices
        assert cap2.ppk > cap1.ppk
        # Same mean and sigma
        assert cap1.y_bar == cap2.y_bar
        assert cap1.sigma_hat == cap2.sigma_hat

    def test_cp_ge_cpk(self, sds1_study):
        """Cp >= Cpk always (for two-sided specs)."""
        cap = sds1_study.capability(usl=120, lsl=80)
        if cap.cp is not None and cap.cpk is not None:
            assert cap.cp >= cap.cpk or cap.cp == pytest.approx(cap.cpk)

    def test_pp_ge_ppk(self, sds1_study):
        """Pp >= Ppk always (for two-sided specs, centered gives equality)."""
        cap = sds1_study.capability(usl=120, lsl=80)
        if cap.pp is not None and cap.ppk is not None:
            assert cap.pp >= cap.ppk or cap.pp == pytest.approx(cap.ppk)

    def test_y_bar_is_overall_mean(self, sds1_study):
        """Ȳ for potential capability is overall response mean, not mean of R2."""
        cap = sds1_study.capability(usl=120, lsl=80)
        ds = sds1_study.dataset
        expected_y_bar = ds[sds1_study.response].dropna().mean()
        assert cap.y_bar == pytest.approx(expected_y_bar)

    def test_precision_from_study(self, sds1_study):
        """round_to comes from Study's spec."""
        cap = sds1_study.capability(usl=120, lsl=80)
        assert cap.round_to == sds1_study.precision


class TestIntegrationNoR2:
    """Integration: factors without time → no VAS residuals → Cp/Cpk unavailable."""

    @pytest.fixture
    def no_r2_study(self):
        """Study without R2: factors but no time."""
        return _make_no_r2_study()

    def test_no_r2(self, no_r2_study):
        """Study without time should NOT have VAS residuals."""
        assert not no_r2_study._ads.has_vas_residuals

    def test_capability_no_potential(self, no_r2_study):
        """Cp/Cpk are None, potential_unavailable_reason is set."""
        cap = no_r2_study.capability(usl=120, lsl=80)
        assert cap.cp is None
        assert cap.cpk is None
        assert cap.sigma_hat_r2 is None
        assert cap.potential_unavailable_reason is not None
        assert "R2" in cap.potential_unavailable_reason

    def test_current_capability_still_works(self, no_r2_study):
        """Pp/Ppk are computed even without R2."""
        cap = no_r2_study.capability(usl=120, lsl=80)
        assert cap.pp is not None
        assert cap.ppk is not None
        assert cap.sigma_hat > 0


# ============================================================================
# One-Sided Specs
# ============================================================================


class TestOneSided:
    """One-sided spec limits produce correct None patterns."""

    @pytest.fixture
    def values_and_sigma(self):
        """Known values for testing."""
        values = np.array([10, 12, 14, 16, 18], dtype=float)
        y_bar = float(np.mean(values))
        _, sigma_hat = compute_sigma_hat(values)
        return values, y_bar, sigma_hat

    def test_usl_only_indices(self, values_and_sigma):
        """USL-only: pp=None, ppk=ppk_upper, ppk_lower=None."""
        values, y_bar, sigma_hat = values_and_sigma
        specs = SpecLimits(usl=25)
        result = compute_capability_indices(y_bar, sigma_hat, specs)

        assert result["pp"] is None
        assert result["ppk_lower"] is None
        assert result["ppk_upper"] is not None
        assert result["ppk"] == result["ppk_upper"]
        assert result["z_lower"] is None
        assert result["z_upper"] is not None

    def test_lsl_only_indices(self, values_and_sigma):
        """LSL-only: pp=None, ppk=ppk_lower, ppk_upper=None."""
        values, y_bar, sigma_hat = values_and_sigma
        specs = SpecLimits(lsl=5)
        result = compute_capability_indices(y_bar, sigma_hat, specs)

        assert result["pp"] is None
        assert result["ppk_upper"] is None
        assert result["ppk_lower"] is not None
        assert result["ppk"] == result["ppk_lower"]
        assert result["z_upper"] is None
        assert result["z_lower"] is not None

    def test_usl_only_pct_outside(self, values_and_sigma):
        """USL-only: n_below_lsl=None.  Values [10,12,14,16,18], USL=15 → 16,18 above."""
        values, _, _ = values_and_sigma
        specs = SpecLimits(usl=15)
        result = compute_pct_outside(values, specs)

        assert result["n_below_lsl"] is None
        assert result["pct_below_lsl"] is None
        assert result["n_above_usl"] == 2  # 16, 18 > 15
        assert result["n_outside"] == 2

    def test_lsl_only_pct_outside(self, values_and_sigma):
        """LSL-only: n_above_usl=None.  Values [10,12,14,16,18], LSL=13 → 10,12 below."""
        values, _, _ = values_and_sigma
        specs = SpecLimits(lsl=13)
        result = compute_pct_outside(values, specs)

        assert result["n_above_usl"] is None
        assert result["pct_above_usl"] is None
        assert result["n_below_lsl"] == 2  # 10, 12 < 13
        assert result["n_outside"] == 2


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge cases: σ=0, N<2, NaN values."""

    def test_all_identical_sigma_zero(self):
        """All identical values → σ̂=0 → indices=inf, no crash."""
        from processbehavior.datasets.synthetic import make_sds

        df = make_sds(1, K1=2, K2=2, T=4, n=3, seed=42)
        df["y"] = 100.0  # all identical

        pb = ProcessBehavior(df)
        study = pb.formulate(
            response="y",
            factors=["factor 1", "factor 2"],
            time="time",
        )
        cap = study.capability(usl=110, lsl=90)

        assert cap.sigma_hat == 0.0
        assert cap.pp == float("inf")
        assert cap.ppk == float("inf")
        assert cap.n_outside == 0

    def test_n_less_than_2_raises(self):
        """Fewer than 2 valid observations raises ValidationError."""
        df = pd.DataFrame({
            "y": [5.0],
            "Factor": ["A"],
            "Time": [1],
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(response="y", factors=["Factor"], time="Time")

        with pytest.raises(ValidationError, match="at least 2"):
            study.capability(usl=10, lsl=0)

    def test_nan_values_dropped(self):
        """NaN response values are dropped; N = count of valid."""
        df = pd.DataFrame({
            "y": [10.0, np.nan, 14.0, np.nan, 18.0, 20.0, 12.0, 16.0],
            "Factor": ["A"] * 4 + ["B"] * 4,
            "Time": [1, 2, 3, 4, 1, 2, 3, 4],
        })
        pb = ProcessBehavior(df)
        study = pb.formulate(response="y", factors=["Factor"], time="Time")
        cap = study.capability(usl=25, lsl=5)

        # NaN rows are dropped during data preparation
        assert cap.n >= 4  # at least the valid values
        assert cap.ppk is not None

    def test_r2_with_nan_dropped(self):
        """R2 residuals with NaN (MA2 first obs) are dropped before sigma_hat_r2."""
        study = _make_sds1_study()
        cap = study.capability(usl=150, lsl=50)

        # R2 should have NaN values dropped, but sigma_hat_r2 should still be valid
        if cap.sigma_hat_r2 is not None:
            assert np.isfinite(cap.sigma_hat_r2)
            assert cap.sigma_hat_r2 > 0


# ============================================================================
# Invariance Properties
# ============================================================================


class TestInvariance:
    """Scale and shift invariance of capability indices."""

    @pytest.fixture
    def base_data(self):
        """Base dataset for invariance tests."""
        rng = np.random.default_rng(42)
        return rng.normal(100, 5, size=50)

    def test_scale_invariance(self, base_data):
        """
        Scale Y and specs by k → same indices.
        If Y' = k*Y, USL' = k*USL, LSL' = k*LSL:
          σ̂' = k*σ̂, (USL'-LSL') = k*(USL-LSL)
          Pp' = k*(USL-LSL) / (6*k*σ̂) = Pp
        """
        specs = SpecLimits(usl=115, lsl=85)
        k = 3.7

        y_bar = float(np.mean(base_data))
        _, sigma_hat = compute_sigma_hat(base_data)
        original = compute_capability_indices(y_bar, sigma_hat, specs)

        scaled_data = base_data * k
        scaled_specs = SpecLimits(usl=115 * k, lsl=85 * k)
        y_bar_s = float(np.mean(scaled_data))
        _, sigma_hat_s = compute_sigma_hat(scaled_data)
        scaled = compute_capability_indices(y_bar_s, sigma_hat_s, scaled_specs)

        assert scaled["pp"] == pytest.approx(original["pp"], rel=1e-10)
        assert scaled["ppk"] == pytest.approx(original["ppk"], rel=1e-10)

    def test_shift_invariance(self, base_data):
        """
        Shift Y and specs by b → same indices.
        If Y' = Y+b, USL' = USL+b, LSL' = LSL+b:
          σ̂' = σ̂, Ȳ' = Ȳ+b
          Pp' = (USL+b - LSL-b) / 6σ̂ = (USL-LSL)/6σ̂ = Pp
          Ppk_lower' = (Ȳ+b - LSL-b) / 3σ̂ = (Ȳ-LSL)/3σ̂ = Ppk_lower
        """
        specs = SpecLimits(usl=115, lsl=85)
        b = 42.0

        y_bar = float(np.mean(base_data))
        _, sigma_hat = compute_sigma_hat(base_data)
        original = compute_capability_indices(y_bar, sigma_hat, specs)

        shifted_data = base_data + b
        shifted_specs = SpecLimits(usl=115 + b, lsl=85 + b)
        y_bar_sh = float(np.mean(shifted_data))
        _, sigma_hat_sh = compute_sigma_hat(shifted_data)
        shifted = compute_capability_indices(y_bar_sh, sigma_hat_sh, shifted_specs)

        assert shifted["pp"] == pytest.approx(original["pp"], rel=1e-10)
        assert shifted["ppk"] == pytest.approx(original["ppk"], rel=1e-10)
        assert shifted["ppk_lower"] == pytest.approx(original["ppk_lower"], rel=1e-10)
        assert shifted["ppk_upper"] == pytest.approx(original["ppk_upper"], rel=1e-10)


# ============================================================================
# Z-Score Consistency
# ============================================================================


class TestZScoreConsistency:
    """Z-scores = 3 × Ppk algebraic identity."""

    def test_z_equals_3_ppk_two_sided(self):
        """z_lower = 3*ppk_lower and z_upper = 3*ppk_upper."""
        specs = SpecLimits(usl=110, lsl=90)
        result = compute_capability_indices(y_bar=103.0, sigma_hat=2.5, specs=specs)

        assert result["z_lower"] == pytest.approx(3 * result["ppk_lower"], rel=1e-10)
        assert result["z_upper"] == pytest.approx(3 * result["ppk_upper"], rel=1e-10)

    def test_z_equals_3_ppk_usl_only(self):
        """z_upper = 3*ppk for USL-only."""
        specs = SpecLimits(usl=110)
        result = compute_capability_indices(y_bar=100.0, sigma_hat=2.0, specs=specs)

        assert result["z_upper"] == pytest.approx(3 * result["ppk"], rel=1e-10)

    def test_z_equals_3_ppk_lsl_only(self):
        """z_lower = 3*ppk for LSL-only."""
        specs = SpecLimits(lsl=90)
        result = compute_capability_indices(y_bar=100.0, sigma_hat=2.0, specs=specs)

        assert result["z_lower"] == pytest.approx(3 * result["ppk"], rel=1e-10)


# ============================================================================
# kwargs API Parity
# ============================================================================


class TestKwargsAPI:
    """study.capability(usl=..., lsl=...) matches SpecLimits path."""

    @pytest.fixture
    def study(self):
        """Simple study for API testing."""
        return _make_sds1_study()

    def test_kwargs_matches_speclimits(self, study):
        """Both calling conventions produce identical results."""
        cap_kwargs = study.capability(usl=120, lsl=80, target=100)
        cap_specs = study.capability(SpecLimits(usl=120, lsl=80, target=100))

        assert cap_kwargs.pp == cap_specs.pp
        assert cap_kwargs.ppk == cap_specs.ppk
        assert cap_kwargs.y_bar == cap_specs.y_bar
        assert cap_kwargs.sigma_hat == cap_specs.sigma_hat
        assert cap_kwargs.n == cap_specs.n

    def test_kwargs_usl_only(self, study):
        """kwargs path with USL only."""
        cap = study.capability(usl=120)
        assert cap.specs.usl == 120
        assert cap.specs.lsl is None

    def test_kwargs_no_limits_raises(self, study):
        """kwargs path with nothing raises ValidationError."""
        with pytest.raises(ValidationError, match="At least one"):
            study.capability()


# ============================================================================
# assess_capability Direct Usage
# ============================================================================


class TestAssessCapabilityDirect:
    """Power-user path: import assess_capability directly."""

    def test_accepts_study(self):
        """assess_capability accepts a Study object."""
        study = _make_sds1_study()

        specs = SpecLimits(usl=120, lsl=80)
        cap = assess_capability(study, specs)
        assert isinstance(cap, CapabilityResult)

    def test_accepts_ads(self):
        """assess_capability accepts an AnalysisDataSet directly."""
        study = _make_sds1_study()

        specs = SpecLimits(usl=120, lsl=80)
        cap = assess_capability(study._ads, specs)
        assert isinstance(cap, CapabilityResult)

    def test_custom_round_to(self):
        """round_to parameter flows through."""
        study = _make_sds1_study()

        specs = SpecLimits(usl=120, lsl=80)
        cap = assess_capability(study, specs, round_to=5)
        assert cap.round_to == 5
