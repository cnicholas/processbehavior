"""
Maximum Information Analysis — Bishop.

Examines R2 residuals (the noise floor / within-cell variation) via two views:
an XmR process behavior chart ("Is the noise floor stable?") and a percentage
histogram ("What does the noise floor distribution look like?").

Together they give the analyst full insight into irreducible variation.

References
----------
Bishop, T.  *Variance Analysis System*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .exceptions import ValidationError
from .spc_constants import R_UPPER_LIMIT_MULTIPLIER, XMR_LIMIT_MULTIPLIER

if TYPE_CHECKING:
    from .analysis_dataset import AnalysisDataSet


# ============================================================================
# Data Class
# ============================================================================

@dataclass(frozen=True)
class MaximumInformationResult:
    """
    Result of a Maximum Information Analysis on R2 residuals.

    Attributes
    ----------
    n : int
        Number of R2 values.
    r2_mean : float
        Mean of R2 (~0).
    r2_mR : float
        Average moving range of R2.
    sigma_hat : float
        mR / d2 — noise floor sigma estimate.
    upl : float
        Upper natural process limit (R2 mean + 2.66 * mR).
    lpl : float
        Lower natural process limit (R2 mean - 2.66 * mR).
    n_signals : int
        Points beyond limits on XmR.
    round_to : int
        Decimal places for display.
    """

    n: int
    r2_mean: float
    r2_mR: float
    sigma_hat: float
    upl: float
    lpl: float
    n_signals: int

    # Internal — for plotting
    _xmr_chart_info: dict
    _histogram_chart_info: dict
    _response_name: str

    round_to: int = 3

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        view: Literal["combined", "xmr", "histogram"] = "combined",
        bins: int = 10,
        theme=None,
        width: int = 900,
        height: int = 700,
        title: str | None = None,
    ):
        """
        Plot the maximum information analysis.

        Parameters
        ----------
        view : str, default ``'combined'``
            ``'combined'`` — XmR (top) + percentage histogram (bottom).
            ``'xmr'`` — XmR of R2 only.
            ``'histogram'`` — Percentage histogram only.
        bins : int, default 10
            Number of histogram bins.
        theme : ChartTheme, optional
            Visual theme (default: processbehavior theme).
        width, height : int
            Figure dimensions in pixels.
        title : str, optional
            Override chart title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from .plotting.maximum_information_chart import create_maximum_information_chart

        return create_maximum_information_chart(
            self,
            view=view,
            bins=bins,
            theme=theme,
            width=width,
            height=height,
            title=title,
        )

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def as_dict(self, round_to: int | None = None) -> dict:
        """Return results as a dict with values rounded for display."""
        r = round_to if round_to is not None else self.round_to

        def _r(v: float) -> float:
            if not np.isfinite(v):
                return v
            return round(v, r)

        return {
            "n": self.n,
            "r2_mean": _r(self.r2_mean),
            "r2_mR": _r(self.r2_mR),
            "sigma_hat": _r(self.sigma_hat),
            "upl": _r(self.upl),
            "lpl": _r(self.lpl),
            "n_signals": self.n_signals,
        }

    def __repr__(self) -> str:
        d = self.as_dict()
        lines = [
            "MaximumInformationResult:",
            f"  R2 residuals (noise floor): n={d['n']}",
            f"  Mean={d['r2_mean']}, mR={d['r2_mR']}, sigma_hat={d['sigma_hat']}",
            f"  NPL: [{d['lpl']}, {d['upl']}]",
            f"  Signals: {d['n_signals']}",
        ]
        return "\n".join(lines)


# ============================================================================
# Pure Function
# ============================================================================

# d2 for n=2 (moving range of consecutive pairs)
_D2 = 1.128


def assess_maximum_information(
    ads: AnalysisDataSet,
    round_to: int = 3,
) -> MaximumInformationResult:
    """
    Perform Maximum Information Analysis on R2 residuals.

    Parameters
    ----------
    ads : AnalysisDataSet
        Must have VAS residuals with R2 column.
    round_to : int, default 3
        Decimal places for presentation.

    Returns
    -------
    MaximumInformationResult

    Raises
    ------
    ValidationError
        If R2 residuals are not available.
    """
    if not ads.has_vas_residuals or "R2" not in ads.analysis_dataset.columns:
        raise ValidationError(
            "Maximum information analysis requires R2 residuals. "
            "R2 is only available for designs with factors and time "
            f"(current SDS: {ads.observed_design_state})."
        )

    df = ads.analysis_dataset
    r2_values = df["R2"].dropna().to_numpy(dtype=float)
    n = len(r2_values)

    if n < 2:
        raise ValidationError(
            f"Maximum information analysis requires at least 2 R2 values, got {n}."
        )

    # --- XmR statistics on R2 ---
    r2_mean = float(np.mean(r2_values))
    mr_values = np.abs(np.diff(r2_values))
    r2_mR = float(np.mean(mr_values))
    sigma_hat = r2_mR / _D2

    # Limits: mean ± E2 * mR
    upl = r2_mean + XMR_LIMIT_MULTIPLIER * r2_mR
    lpl = r2_mean - XMR_LIMIT_MULTIPLIER * r2_mR

    # Count signals (beyond limits)
    beyond = (r2_values > upl) | (r2_values < lpl)
    n_signals = int(np.sum(beyond))

    # --- Build chart info dicts for renderers ---

    # XmR chart info — matches the format render_control_chart expects via RenderContext
    # We build a simplified dict for use by the maximum_information_chart module
    mr_series = np.empty(n, dtype=float)
    mr_series[0] = np.nan
    mr_series[1:] = mr_values

    # R chart limits
    r_upl = R_UPPER_LIMIT_MULTIPLIER * r2_mR

    xmr_data = pd.DataFrame({
        "R2": r2_values,
        "mr": mr_series,
        "obs": np.arange(1, n + 1),
        "beyond_limits": beyond.astype(int),
    })

    # Mark R beyond limits
    r_beyond = np.zeros(n, dtype=int)
    r_beyond[0] = 0  # first has NaN mR
    r_beyond[1:] = (mr_values > r_upl).astype(int)
    xmr_data["r_beyond_limits"] = r_beyond

    xmr_chart_info = {
        "data": xmr_data,
        "statistics": {
            "x_mean": r2_mean,
            "x_upl": upl,
            "x_lpl": lpl,
            "mR": r2_mR,
            "r_upl": r_upl,
        },
        "metadata": {
            "value_col": "R2",
            "mr_col": "mr",
            "x_col": "obs",
        },
    }

    # Histogram chart info
    response_name = ads.spec.response_var
    histogram_chart_info = {
        "data": pd.DataFrame({"R2": r2_values}),
        "statistics": {
            "mean": r2_mean,
            "n": n,
        },
        "metadata": {
            "value_col": "R2",
            "chart_type": "Histogram",
        },
    }

    return MaximumInformationResult(
        n=n,
        r2_mean=r2_mean,
        r2_mR=r2_mR,
        sigma_hat=sigma_hat,
        upl=upl,
        lpl=lpl,
        n_signals=n_signals,
        _xmr_chart_info=xmr_chart_info,
        _histogram_chart_info=histogram_chart_info,
        _response_name=response_name,
        round_to=round_to,
    )
