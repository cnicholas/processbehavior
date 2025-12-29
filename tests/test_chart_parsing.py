"""Tests for chart name parsing in Study._parse_chart_request()."""

import pytest
import pandas as pd

from processbehavior import ProcessBehavior


@pytest.fixture
def study():
    """Create a study for testing parser (SDS with residual charts available)."""
    df = pd.DataFrame({
        'y': [1, 2, 3, 4, 5, 6] * 4,
        'time': [1, 1, 2, 2, 3, 3] * 4,
        'factor': ['A', 'B'] * 12
    })
    return ProcessBehavior(df).formulate(
        response='y',
        time='time',
        factors=['factor']
    )


class TestBaseCharts:
    """Test parsing of base chart types."""

    def test_xbar(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("Xbar")
        assert residual_id is None
        assert base_chart == "Xbar"
        assert recentered is False

    def test_s(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("S")
        assert residual_id is None
        assert base_chart == "S"
        assert recentered is False

    def test_imr(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("Imr")
        assert residual_id is None
        assert base_chart == "Imr"
        assert recentered is False

    def test_r(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("R")
        assert residual_id is None
        assert base_chart == "R"
        assert recentered is False


class TestNumericResidualCharts:
    """Test parsing of numeric residual chart names (R5_Xbar, RCR5_Xbar)."""

    def test_r5_xbar(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("R5_Xbar")
        assert residual_id == "R5"
        assert base_chart == "Xbar"
        assert recentered is False

    def test_r2_s(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("R2_S")
        assert residual_id == "R2"
        assert base_chart == "S"
        assert recentered is False

    def test_r4_imr(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("R4_Imr")
        assert residual_id == "R4"
        assert base_chart == "Imr"
        assert recentered is False

    def test_rcr5_xbar_recentered_from_string(self, study):
        """RCR prefix sets recentered=True."""
        residual_id, base_chart, recentered = study._parse_chart_request("RCR5_Xbar")
        assert residual_id == "R5"
        assert base_chart == "Xbar"
        assert recentered is True

    def test_rcr2_s_recentered_from_string(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("RCR2_S")
        assert residual_id == "R2"
        assert base_chart == "S"
        assert recentered is True

    def test_multi_digit_residual(self, study):
        """Multi-digit residual IDs should parse correctly."""
        residual_id, base_chart, recentered = study._parse_chart_request("R10_Xbar")
        assert residual_id == "R10"
        assert base_chart == "Xbar"
        assert recentered is False

    def test_recentered_kwarg(self, study):
        """Recentered kwarg sets recentered=True."""
        residual_id, base_chart, recentered = study._parse_chart_request(
            "R5_Xbar", recentered_kwarg=True
        )
        assert residual_id == "R5"
        assert base_chart == "Xbar"
        assert recentered is True


class TestAliasResidualCharts:
    """Test parsing of human-readable alias chart names."""

    def test_noise_xbar(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("noise_Xbar")
        assert residual_id == "R5"
        assert base_chart == "Xbar"
        assert recentered is False

    def test_within_cell_s(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("within_cell_S")
        assert residual_id == "R2"
        assert base_chart == "S"
        assert recentered is False

    def test_structure_removed_imr(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("structure_removed_Imr")
        assert residual_id == "R3"
        assert base_chart == "Imr"
        assert recentered is False

    def test_time_structure_removed_xbar(self, study):
        """Multi-underscore alias should parse correctly."""
        residual_id, base_chart, recentered = study._parse_chart_request("time_structure_removed_Xbar")
        assert residual_id == "R4"
        assert base_chart == "Xbar"
        assert recentered is False

    def test_mean_removed_s(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request("mean_removed_S")
        assert residual_id == "R1"
        assert base_chart == "S"
        assert recentered is False

    def test_rc_noise_xbar(self, study):
        """rc_ prefix sets recentered=True."""
        residual_id, base_chart, recentered = study._parse_chart_request("rc_noise_Xbar")
        assert residual_id == "R5"
        assert base_chart == "Xbar"
        assert recentered is True

    def test_rc_time_structure_removed_imr(self, study):
        """rc_ prefix with multi-underscore alias."""
        residual_id, base_chart, recentered = study._parse_chart_request("rc_time_structure_removed_Imr")
        assert residual_id == "R4"
        assert base_chart == "Imr"
        assert recentered is True

    def test_alias_with_recentered_kwarg(self, study):
        residual_id, base_chart, recentered = study._parse_chart_request(
            "noise_Xbar", recentered_kwarg=True
        )
        assert residual_id == "R5"
        assert base_chart == "Xbar"
        assert recentered is True


class TestInvalidInputs:
    """Test that invalid inputs raise ValueError with helpful messages."""

    def test_missing_base_chart_alias(self, study):
        """Bare alias should error with helpful message."""
        with pytest.raises(ValueError, match="Missing base chart type"):
            study._parse_chart_request("noise")

    def test_missing_base_chart_numeric(self, study):
        """Bare R5 should error with helpful message."""
        with pytest.raises(ValueError, match="Missing base chart type"):
            study._parse_chart_request("R5")

    def test_missing_base_chart_rcr(self, study):
        """Bare RCR5 should error with helpful message."""
        with pytest.raises(ValueError, match="Missing base chart type"):
            study._parse_chart_request("RCR5")

    def test_unknown_base_chart(self, study):
        """Unknown base chart should error."""
        with pytest.raises(ValueError, match="Unknown chart 'Foo'"):
            study._parse_chart_request("Foo")

    def test_unknown_chart_type_in_residual(self, study):
        """Unknown chart type after underscore should error."""
        with pytest.raises(ValueError, match="Unknown chart type 'Blah'"):
            study._parse_chart_request("R5_Blah")

    def test_unknown_residual_alias(self, study):
        """Unknown alias should error with list of valid aliases."""
        with pytest.raises(ValueError, match="Unknown residual 'unknown'"):
            study._parse_chart_request("unknown_Xbar")

    def test_double_recenter_rc_rcr(self, study):
        """Double recenter (rc_ + RCR) should error."""
        with pytest.raises(ValueError, match="Double recenter specification"):
            study._parse_chart_request("rc_RCR5_Xbar")

    def test_malformed_leading_underscore(self, study):
        """Leading underscore should error."""
        with pytest.raises(ValueError, match="missing residual identifier"):
            study._parse_chart_request("_Xbar")

    def test_malformed_rc_double_underscore(self, study):
        """rc__ should error."""
        with pytest.raises(ValueError, match="missing residual identifier"):
            study._parse_chart_request("rc__Xbar")

    def test_trailing_underscore(self, study):
        """Trailing underscore should error (empty base chart)."""
        with pytest.raises(ValueError, match="Unknown chart type ''"):
            study._parse_chart_request("Xbar_")

    def test_empty_string(self, study):
        """Empty string should error."""
        with pytest.raises(ValueError, match="non-empty string"):
            study._parse_chart_request("")

    def test_none_input(self, study):
        """None should error."""
        with pytest.raises(ValueError, match="non-empty string"):
            study._parse_chart_request(None)

    def test_rc_alone(self, study):
        """Just 'rc_' should error."""
        with pytest.raises(ValueError, match="missing residual identifier"):
            study._parse_chart_request("rc_")


class TestRecenteredSemantics:
    """Test recentered flag behavior."""

    def test_string_true_kwarg_false_uses_string(self, study):
        """String recentered wins (OR semantics)."""
        # RCR5_Xbar implies recentered=True, kwarg=False
        # Result should be True (string OR kwarg)
        residual_id, base_chart, recentered = study._parse_chart_request(
            "RCR5_Xbar", recentered_kwarg=False
        )
        assert recentered is True

    def test_string_false_kwarg_true_uses_kwarg(self, study):
        """Kwarg can set recentered when string doesn't."""
        residual_id, base_chart, recentered = study._parse_chart_request(
            "R5_Xbar", recentered_kwarg=True
        )
        assert recentered is True

    def test_both_true(self, study):
        """Both true results in true."""
        residual_id, base_chart, recentered = study._parse_chart_request(
            "RCR5_Xbar", recentered_kwarg=True
        )
        assert recentered is True

    def test_both_false(self, study):
        """Both false results in false."""
        residual_id, base_chart, recentered = study._parse_chart_request(
            "R5_Xbar", recentered_kwarg=False
        )
        assert recentered is False
