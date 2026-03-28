"""Tests for maximum_information analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from processbehavior import MaximumInformationResult, ProcessBehavior, ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALIDATION_CSV = "validation/TABVASTESTDATABASE.csv"


def _make_study(sds: int):
    """Build a Study from the validation dataset for the given PM SDS."""
    df = pd.read_csv(VALIDATION_CSV)
    col = f"PM SDS {sds}"
    if col not in df.columns:
        pytest.skip(f"{col} not in validation dataset")

    pb = ProcessBehavior(df)
    study = pb.formulate(
        response=col,
        factors=["FACTOR 1", "FACTOR 2"],
        time="PRODUCTION TIME",
    )
    return study


# ---------------------------------------------------------------------------
# Core functionality tests
# ---------------------------------------------------------------------------

class TestMaximumInformationSDS1:
    """SDS 1 — fully balanced design."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.study = _make_study(1)
        self.result = self.study.maximum_information()

    def test_returns_correct_type(self):
        assert isinstance(self.result, MaximumInformationResult)

    def test_all_fields_populated(self):
        r = self.result
        assert r.n > 0
        assert np.isfinite(r.r2_mean)
        assert np.isfinite(r.r2_mR)
        assert np.isfinite(r.sigma_hat)
        assert np.isfinite(r.upl)
        assert np.isfinite(r.lpl)
        assert isinstance(r.n_signals, int)

    def test_r2_mean_near_zero(self):
        """R2 residuals should be centered near zero."""
        assert abs(self.result.r2_mean) < 1.0

    def test_npl_symmetric(self):
        """NPLs should be roughly symmetric around the mean."""
        r = self.result
        upper_dist = r.upl - r.r2_mean
        lower_dist = r.r2_mean - r.lpl
        assert abs(upper_dist - lower_dist) < 1e-10

    def test_sigma_hat_positive(self):
        assert self.result.sigma_hat > 0

    def test_sigma_hat_equals_mr_over_d2(self):
        r = self.result
        expected = r.r2_mR / 1.128
        assert abs(r.sigma_hat - expected) < 1e-10


class TestMaximumInformationSDS2:
    """SDS 2 — unbalanced design."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.study = _make_study(2)
        self.result = self.study.maximum_information()

    def test_returns_result(self):
        assert isinstance(self.result, MaximumInformationResult)

    def test_all_fields_populated(self):
        r = self.result
        assert r.n > 0
        assert np.isfinite(r.sigma_hat)


class TestMaximumInformationSDS3:
    """SDS 3 — sparse design."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.study = _make_study(3)
        self.result = self.study.maximum_information()

    def test_returns_result(self):
        assert isinstance(self.result, MaximumInformationResult)

    def test_all_fields_populated(self):
        r = self.result
        assert r.n > 0
        assert np.isfinite(r.sigma_hat)


# ---------------------------------------------------------------------------
# No-VAS guard
# ---------------------------------------------------------------------------

class TestMaximumInformationNoVAS:
    """Study without time → no VAS residuals → should raise."""

    def test_raises_without_time(self):
        df = pd.read_csv(VALIDATION_CSV)
        pb = ProcessBehavior(df)
        study = pb.formulate(
            response="PM SDS 1",
            factors=["FACTOR 1", "FACTOR 2"],
        )
        with pytest.raises(ValidationError, match="R2 residuals"):
            study.maximum_information()


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

class TestPresentation:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.study = _make_study(1)
        self.result = self.study.maximum_information()

    def test_as_dict_structure(self):
        d = self.result.as_dict()
        expected_keys = {"n", "r2_mean", "r2_mR", "sigma_hat", "upl", "lpl", "n_signals"}
        assert set(d.keys()) == expected_keys

    def test_as_dict_values_rounded(self):
        d = self.result.as_dict(round_to=2)
        # Check that values are actually rounded
        for key in ("r2_mean", "r2_mR", "sigma_hat", "upl", "lpl"):
            val = d[key]
            assert val == round(val, 2)

    def test_repr_contains_key_info(self):
        s = repr(self.result)
        assert "MaximumInformationResult" in s
        assert "sigma_hat" in s
        assert "NPL" in s
        assert "Signals" in s


# ---------------------------------------------------------------------------
# Plot smoke tests
# ---------------------------------------------------------------------------

class TestPlotting:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.study = _make_study(1)
        self.result = self.study.maximum_information()

    def test_plot_combined(self):
        fig = self.result.plot()
        assert isinstance(fig, go.Figure)

    def test_plot_xmr(self):
        fig = self.result.plot(view="xmr")
        assert isinstance(fig, go.Figure)

    def test_plot_histogram(self):
        fig = self.result.plot(view="histogram")
        assert isinstance(fig, go.Figure)

    def test_plot_invalid_view(self):
        with pytest.raises(ValueError, match="view must be"):
            self.result.plot(view="invalid")

    def test_plot_custom_title(self):
        fig = self.result.plot(title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_plot_with_string_theme(self):
        fig = self.result.plot(theme="ggplot")
        assert isinstance(fig, go.Figure)

    def test_plot_with_bins(self):
        fig = self.result.plot(bins=15)
        assert isinstance(fig, go.Figure)
