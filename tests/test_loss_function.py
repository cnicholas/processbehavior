"""
Tests for Taguchi Loss Function Analysis (Bishop Ch. 15).

Uses validation dataset (TABVASTESTDATABASE.csv) as ground truth.
"""

import pytest

from processbehavior import LossResult, ProcessBehavior, ValidationError

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def pb():
    """Load validation dataset."""
    return ProcessBehavior.read_csv('validation/TABVASTESTDATABASE.csv')


@pytest.fixture
def study_sds1(pb):
    """PM SDS 1: 2 factors, time, full replication."""
    return pb.formulate(
        response='PM SDS 1',
        factors=['FACTOR 1', 'FACTOR 2'],
        time='PRODUCTION TIME',
    )


@pytest.fixture
def study_sds2(pb):
    """PM SDS 2: 2 factors, time, some singleton cells."""
    return pb.formulate(
        response='PM SDS 2',
        factors=['FACTOR 1', 'FACTOR 2'],
        time='PRODUCTION TIME',
    )


@pytest.fixture
def study_sds3(pb):
    """PM SDS 3: 2 factors, time, all singleton cells."""
    return pb.formulate(
        response='PM SDS 3',
        factors=['FACTOR 1', 'FACTOR 2'],
        time='PRODUCTION TIME',
    )


@pytest.fixture
def study_single_factor(pb):
    """Single factor study."""
    return pb.formulate(
        response='PM SDS 1',
        factors=['FACTOR 1'],
        time='PRODUCTION TIME',
    )


@pytest.fixture
def study_no_time(pb):
    """Study with factors but no time — no VAS residuals."""
    return pb.formulate(
        response='PM SDS 1',
        factors=['FACTOR 1'],
    )


# ============================================================================
# Validation: PM SDS 1 with T=237 (Tom's reference)
# ============================================================================

class TestValidationPMSDS1:
    """Match Tom's reference output: Fig 15-4 / 15-5."""

    def test_unstructured_percentages(self, study_sds1):
        """5-component Pareto matches Tom's reference."""
        result = study_sds1.loss_function(target=237.0)

        assert round(result.pct_interaction, 1) == 43.3
        assert round(result.pct_unexplained, 1) == 23.0
        assert round(result.pct_centering, 1) == 16.8
        assert round(result.pct_pdc, 1) == 14.2
        assert round(result.pct_time, 1) == 2.7

    def test_structured_pdc_decomposition(self, study_sds1):
        """PDC broken into F1, F2, PDF INT matches Tom's reference."""
        result = study_sds1.loss_function(target=237.0)

        total = result.total
        f1_pct = result.pdc_by_factor['FACTOR 1'] / total * 100
        f2_pct = result.pdc_by_factor['FACTOR 2'] / total * 100
        pdc_int_pct = result.pdc_factor_interaction / total * 100

        assert round(f1_pct, 1) == 8.9
        assert round(f2_pct, 1) == 3.7
        assert round(pdc_int_pct, 1) == 1.5

    def test_target_237(self, study_sds1):
        result = study_sds1.loss_function(target=237.0)
        assert result.target == 237.0
        assert result.target_is_default is False

    def test_sds_is_1(self, study_sds1):
        result = study_sds1.loss_function(target=237.0)
        assert result.sds == 1


# ============================================================================
# Centering
# ============================================================================

class TestCentering:
    def test_centering_default_target(self, study_sds1):
        """Default target = grand mean → centering = 0."""
        result = study_sds1.loss_function()
        assert result.target_is_default is True
        assert result.centering == 0.0
        assert result.pct_centering == 0.0
        assert result.target == result.y_bar

    def test_centering_with_target(self, study_sds1):
        """Non-zero centering when T ≠ Ȳ."""
        result = study_sds1.loss_function(target=237.0)
        assert result.centering > 0.0
        expected = (result.y_bar - 237.0) ** 2
        assert abs(result.centering - expected) < 1e-10


# ============================================================================
# Decomposition identities
# ============================================================================

class TestDecompositionIdentities:
    def test_five_components_sum_to_total(self, study_sds1):
        """5 components = EL within floating point tolerance."""
        result = study_sds1.loss_function(target=237.0)
        component_sum = (
            result.centering + result.unexplained + result.pdc
            + result.time + result.interaction
        )
        assert abs(component_sum - result.total) < 1e-10

    def test_pareto_sums_to_100(self, study_sds1):
        """Percentages sum to 100%."""
        result = study_sds1.loss_function(target=237.0)
        pct_sum = (
            result.pct_centering + result.pct_unexplained + result.pct_pdc
            + result.pct_time + result.pct_interaction
        )
        assert abs(pct_sum - 100.0) < 0.1

    def test_pdc_decomposition_identity(self, study_sds1):
        """per-factor + interaction = PDC total."""
        result = study_sds1.loss_function(target=237.0)
        recon = sum(result.pdc_by_factor.values()) + result.pdc_factor_interaction
        assert abs(recon - result.pdc) < 1e-10

    def test_decomposition_default_target(self, study_sds1):
        """Identity holds when centering = 0."""
        result = study_sds1.loss_function()
        component_sum = (
            result.centering + result.unexplained + result.pdc
            + result.time + result.interaction
        )
        assert abs(component_sum - result.total) < 1e-10


# ============================================================================
# SDS-dependent unexplained
# ============================================================================

class TestUnexplained:
    def test_sds1_uses_per_cell(self, study_sds1):
        """SDS 1 (replicated) uses per-cell S/c4 path."""
        result = study_sds1.loss_function()
        assert result.sds == 1
        assert result.unexplained > 0.0

    def test_sds2_percentages(self, study_sds2):
        """SDS 2 5-component Pareto — Eq 15.18 pooled R2 path."""
        result = study_sds2.loss_function(target=237.0)
        assert result.sds == 2

        assert round(result.pct_interaction, 1) == 42.5
        assert round(result.pct_unexplained, 1) == 23.6
        assert round(result.pct_centering, 1) == 16.1
        assert round(result.pct_pdc, 1) == 15.6
        assert round(result.pct_time, 1) == 2.2

    def test_sds3_percentages(self, study_sds3):
        """SDS 3 5-component Pareto — Eq 15.18 pooled R2 path."""
        result = study_sds3.loss_function(target=237.0)
        assert result.sds == 3

        assert round(result.pct_interaction, 1) == 42.4
        assert round(result.pct_unexplained, 1) == 25.0
        assert round(result.pct_centering, 1) == 16.4
        assert round(result.pct_pdc, 1) == 13.7
        assert round(result.pct_time, 1) == 2.5


# ============================================================================
# Single factor
# ============================================================================

class TestSingleFactor:
    def test_no_factor_interaction(self, study_single_factor):
        """Single factor: pdc_factor_interaction = 0."""
        result = study_single_factor.loss_function()
        assert result.pdc_factor_interaction == 0.0

    def test_single_factor_pdc_equals_factor_loss(self, study_single_factor):
        """Single factor: pdc_by_factor[factor] = pdc."""
        result = study_single_factor.loss_function()
        assert abs(result.pdc_by_factor['FACTOR 1'] - result.pdc) < 1e-10


# ============================================================================
# Guards
# ============================================================================

class TestGuards:
    def test_no_vas_raises(self, study_no_time):
        """Study without time (no VAS residuals) raises ValidationError."""
        with pytest.raises(ValidationError, match="VAS residuals"):
            study_no_time.loss_function()


# ============================================================================
# Presentation
# ============================================================================

class TestPresentation:
    def test_repr(self, study_sds1):
        result = study_sds1.loss_function(target=237.0)
        text = repr(result)
        assert "LossResult:" in text
        assert "Target=" in text
        assert "MEAN" in text
        assert "UNEXPLAINED" in text

    def test_as_dict(self, study_sds1):
        result = study_sds1.loss_function(target=237.0)
        d = result.as_dict()
        assert "centering" in d
        assert "pct_centering" in d
        assert "pdc_by_factor" in d
        assert isinstance(d["pdc_by_factor"], dict)

    def test_as_dict_round_to(self, study_sds1):
        result = study_sds1.loss_function(target=237.0)
        d = result.as_dict(round_to=1)
        # Values should be rounded to 1 decimal
        assert d["pct_interaction"] == round(result.pct_interaction, 1)


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_pareto_zero_total(self):
        """When total=0, all percentages should be 0."""
        from processbehavior.loss_function import _compute_pareto
        pct = _compute_pareto(0, 0, 0, 0, 0, 0)
        assert all(p == 0.0 for p in pct)

    def test_context_fields(self, study_sds1):
        """Verify context fields are populated."""
        result = study_sds1.loss_function(target=237.0)
        assert result.n > 0
        assert result.K > 0
        assert result.T_periods > 0
        assert isinstance(result, LossResult)
