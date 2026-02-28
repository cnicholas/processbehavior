"""Tests for chart name parsing in Study._parse_chart_request().

The parser now only accepts base chart types (Xbar, S, XmR, R).
Residual charts are specified via the `value` parameter:
  study.execute(chart='Xbar', value='R5')  # instead of chart='R5_Xbar'
"""

import pandas as pd
import pytest

from processbehavior import ProcessBehavior, ValidationError


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
        base_chart = study._parse_chart_request("Xbar")
        assert base_chart == "Xbar"

    def test_s(self, study):
        base_chart = study._parse_chart_request("S")
        assert base_chart == "S"

    def test_xmr(self, study):
        base_chart = study._parse_chart_request("XmR")
        assert base_chart == "XmR"

    def test_r(self, study):
        base_chart = study._parse_chart_request("R")
        assert base_chart == "R"


class TestCaseInsensitiveChartNames:
    """Test that chart names are resolved case-insensitively."""

    @pytest.mark.parametrize("input_name,expected", [
        ("xbar", "Xbar"),
        ("XBAR", "Xbar"),
        ("Xbar", "Xbar"),
        ("xmr", "XmR"),
        ("Xmr", "XmR"),
        ("XMR", "XmR"),
        ("XmR", "XmR"),
        ("s", "S"),
        ("r", "R"),
        ("histogram", "Histogram"),
        ("HISTOGRAM", "Histogram"),
    ])
    def test_parse_chart_request_case_insensitive(self, study, input_name, expected):
        assert study._parse_chart_request(input_name) == expected


class TestOldSyntaxRaisesError:
    """Test that old residual chart syntax raises helpful errors."""

    def test_r5_xbar_old_syntax(self, study):
        """Old R5_Xbar syntax should error with migration guidance."""
        with pytest.raises(ValidationError, match="no longer supported"):
            study._parse_chart_request("R5_Xbar")

    def test_r2_s_old_syntax(self, study):
        """Old R2_S syntax should error with migration guidance."""
        with pytest.raises(ValidationError, match="no longer supported"):
            study._parse_chart_request("R2_S")

    def test_r4_xmr_old_syntax(self, study):
        """Old R4_XmR syntax should error with migration guidance."""
        with pytest.raises(ValidationError, match="no longer supported"):
            study._parse_chart_request("R4_XmR")

    def test_rcr5_xbar_old_syntax(self, study):
        """Old RCR5_Xbar syntax should error with migration guidance."""
        with pytest.raises(ValidationError, match="no longer supported"):
            study._parse_chart_request("RCR5_Xbar")

    def test_noise_xbar_old_syntax(self, study):
        """Old alias syntax should error with migration guidance."""
        with pytest.raises(ValidationError, match="no longer supported"):
            study._parse_chart_request("noise_Xbar")

    def test_rc_noise_xbar_old_syntax(self, study):
        """Old rc_ prefix syntax should error with migration guidance."""
        with pytest.raises(ValidationError, match="no longer supported"):
            study._parse_chart_request("rc_noise_Xbar")

    def test_error_message_includes_new_syntax(self, study):
        """Error message should show the new syntax to use."""
        with pytest.raises(ValidationError) as exc_info:
            study._parse_chart_request("R5_Xbar")
        assert "chart='Xbar'" in str(exc_info.value)
        assert "value='R5'" in str(exc_info.value)

    def test_error_message_includes_recentered_hint(self, study):
        """Error message should include recentered=True for RCR charts."""
        with pytest.raises(ValidationError) as exc_info:
            study._parse_chart_request("RCR5_Xbar")
        assert "recentered=True" in str(exc_info.value)


class TestResidualIdWithoutChart:
    """Test that bare residual identifiers raise helpful errors."""

    def test_r5_without_chart(self, study):
        """Bare R5 should error with guidance to use value parameter."""
        with pytest.raises(ValidationError, match="residual identifier"):
            study._parse_chart_request("R5")

    def test_rcr5_without_chart(self, study):
        """Bare RCR5 should error with guidance."""
        with pytest.raises(ValidationError, match="residual identifier"):
            study._parse_chart_request("RCR5")

    def test_r5_error_shows_value_syntax(self, study):
        """Error should show the new value= syntax."""
        with pytest.raises(ValidationError) as exc_info:
            study._parse_chart_request("R5")
        assert "value='R5'" in str(exc_info.value)


class TestAliasWithoutChart:
    """Test that bare aliases raise helpful errors."""

    def test_noise_without_chart(self, study):
        """Bare 'noise' alias should error with guidance."""
        with pytest.raises(ValidationError, match="residual alias"):
            study._parse_chart_request("noise")

    def test_within_cell_without_chart(self, study):
        """Bare 'within_cell' has underscore so triggers old syntax error path."""
        # within_cell has an underscore, so it looks like old syntax
        with pytest.raises(ValidationError, match="no longer supported|Invalid chart"):
            study._parse_chart_request("within_cell")


class TestInvalidInputs:
    """Test that invalid inputs raise ValidationError."""

    def test_unknown_base_chart(self, study):
        """Unknown base chart should error."""
        with pytest.raises(ValidationError, match="Unknown chart 'Foo'"):
            study._parse_chart_request("Foo")

    def test_empty_string(self, study):
        """Empty string should error."""
        with pytest.raises(ValidationError, match="non-empty string"):
            study._parse_chart_request("")

    def test_none_input(self, study):
        """None should error."""
        with pytest.raises(ValidationError, match="non-empty string"):
            study._parse_chart_request(None)


class TestNewSyntaxWorks:
    """Test that the new value= syntax works correctly."""

    def test_xbar_with_r5_value(self, study):
        """chart='Xbar', value='R5' should work."""
        result = study.execute(chart='Xbar', value='R5')
        assert result is not None
        # Chart key matches the chart parameter, not the combined format
        assert 'Xbar' in result.charts

    def test_s_with_r2_value(self, study):
        """chart='S', value='R2' should work."""
        result = study.execute(chart='S', value='R2')
        assert result is not None
        assert 'S' in result.charts

    def test_xbar_with_r5_recentered(self, study):
        """chart='Xbar', value='R5', recentered=True should work."""
        result = study.execute(chart='Xbar', value='R5', recentered=True)
        assert result is not None
        assert 'Xbar' in result.charts
