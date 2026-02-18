"""
Process Capability Analysis — Wheeler/Bishop Chapter 16.

Computes capability indices given specification limits. Core logic lives as
pure module-level functions (composition pattern matching ResidualCalculator /
EffectsCalculator). Power users can import ``assess_capability()`` directly.

Current Capability (how the process IS performing):
    S = std(Y, ddof=1)
    σ̂ = S / c4(N)
    Pp  = (USL − LSL) / 6σ̂
    Ppk = min((Ȳ − LSL) / 3σ̂, (USL − Ȳ) / 3σ̂)

Potential Capability (achievable by removing assignable causes):
    S_R2 = std(R2, ddof=1)
    σ̂_R2 = S_R2 / c4(N_R2)
    Cp  = (USL − LSL) / 6σ̂_R2
    Cpk = min((Ȳ − LSL) / 3σ̂_R2, (USL − Ȳ) / 3σ̂_R2)

References
----------
Wheeler, D.J. & Bishop, T.  *Variance Analysis System* — Chapter 16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import ValidationError
from .spc_constants import c4

if TYPE_CHECKING:
    from .analysis_dataset import AnalysisDataSet
    from .study import Study


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class SpecLimits:
    """
    Specification limits for capability analysis.

    At least one of ``usl`` or ``lsl`` must be provided.

    Parameters
    ----------
    usl : float | None
        Upper specification limit.
    lsl : float | None
        Lower specification limit.
    target : float | None
        Target value (optional).  When both limits are present,
        must satisfy ``lsl <= target <= usl``.
    """

    usl: float | None = None
    lsl: float | None = None
    target: float | None = None

    def __post_init__(self) -> None:
        if self.usl is None and self.lsl is None:
            raise ValidationError(
                "At least one specification limit (usl or lsl) must be provided."
            )

        if self.usl is not None and self.lsl is not None and self.lsl >= self.usl:
            raise ValidationError(
                f"LSL ({self.lsl}) must be less than USL ({self.usl})."
            )

        if (
            self.target is not None
            and self.usl is not None
            and self.lsl is not None
            and not (self.lsl <= self.target <= self.usl)
        ):
            raise ValidationError(
                f"Target ({self.target}) must be between "
                f"LSL ({self.lsl}) and USL ({self.usl})."
            )

    @property
    def is_two_sided(self) -> bool:
        """True when both USL and LSL are specified."""
        return self.usl is not None and self.lsl is not None


@dataclass(frozen=True)
class CapabilityResult:
    """
    Result of a process capability assessment (Wheeler/Bishop Ch. 16).

    Raw ``float`` values are stored unrounded.  Use :meth:`as_dict` for a
    rounded dictionary or ``repr()`` for a formatted summary.

    Attributes
    ----------
    specs : SpecLimits
        Specification limits used for the assessment.
    n : int
        Total valid (non-NaN) observations.
    y_bar : float
        Grand mean of valid observations.
    s : float
        Sample standard deviation (ddof=1).
    sigma_hat : float
        Unbiased sigma estimate: ``s / c4(n)``.
    pp, ppk_lower, ppk_upper, ppk : float | None
        Current capability indices (from overall sigma).
    sigma_hat_r2 : float | None
        Unbiased sigma from R2 residuals.
    cp, cpk_lower, cpk_upper, cpk : float | None
        Potential capability indices (from R2 sigma).
    potential_unavailable_reason : str | None
        Why Cp/Cpk are unavailable (if applicable).
    z_lower, z_upper : float | None
        Distance to spec in sigma units.
    n_below_lsl, n_above_usl, n_outside : int | None
        Empirical counts outside specs.
    pct_below_lsl, pct_above_usl, pct_outside : float | None
        Empirical percents outside specs.
    stability_evaluated : bool
        Always False in v1 — stability integration deferred.
    stability_warning : str | None
        Always None in v1.
    round_to : int
        Decimal places used by ``__repr__`` and ``as_dict``.
    """

    # Input context
    specs: SpecLimits
    n: int
    y_bar: float
    s: float
    sigma_hat: float

    # Current capability (from overall sigma)
    pp: float | None
    ppk_lower: float | None
    ppk_upper: float | None
    ppk: float | None

    # Potential capability (from R2 residual sigma)
    sigma_hat_r2: float | None
    cp: float | None
    cpk_lower: float | None
    cpk_upper: float | None
    cpk: float | None
    potential_unavailable_reason: str | None

    # Z-scores
    z_lower: float | None
    z_upper: float | None

    # Empirical percent outside
    n_below_lsl: int | None
    n_above_usl: int | None
    n_outside: int | None
    pct_below_lsl: float | None
    pct_above_usl: float | None
    pct_outside: float | None

    # Stability context
    stability_evaluated: bool = False
    stability_warning: str | None = None

    # Presentation
    round_to: int = 3

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, values, *, theme=None, show_potential=True,
             x_label=None, nbins=None, histnorm="", width=900, height=500,
             title=None):
        """
        Create a capability histogram with spec lines and index annotations.

        Parameters
        ----------
        values : array-like
            Response values to histogram.
        theme : ChartTheme, optional
            Visual theme (default: processbehavior theme).
        show_potential : bool, default True
            Show Cp/Cpk in the annotation box when available.
        x_label, nbins, histnorm, width, height, title
            Passed through to ``create_capability_chart()``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        from .plotting.capability_chart import create_capability_chart

        return create_capability_chart(
            self, values, theme=theme, show_potential=show_potential,
            x_label=x_label, nbins=nbins, histnorm=histnorm,
            width=width, height=height, title=title,
        )

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def as_dict(self, round_to: int | None = None) -> dict:
        """Return results as a dict with values rounded for display."""
        r = round_to if round_to is not None else self.round_to

        def _r(v: float | None) -> float | None:
            if v is None:
                return None
            if not np.isfinite(v):
                return v
            return round(v, r)

        return {
            "usl": self.specs.usl,
            "lsl": self.specs.lsl,
            "target": self.specs.target,
            "n": self.n,
            "y_bar": _r(self.y_bar),
            "s": _r(self.s),
            "sigma_hat": _r(self.sigma_hat),
            "pp": _r(self.pp),
            "ppk_lower": _r(self.ppk_lower),
            "ppk_upper": _r(self.ppk_upper),
            "ppk": _r(self.ppk),
            "sigma_hat_r2": _r(self.sigma_hat_r2),
            "cp": _r(self.cp),
            "cpk_lower": _r(self.cpk_lower),
            "cpk_upper": _r(self.cpk_upper),
            "cpk": _r(self.cpk),
            "potential_unavailable_reason": self.potential_unavailable_reason,
            "z_lower": _r(self.z_lower),
            "z_upper": _r(self.z_upper),
            "n_below_lsl": self.n_below_lsl,
            "n_above_usl": self.n_above_usl,
            "n_outside": self.n_outside,
            "pct_below_lsl": _r(self.pct_below_lsl),
            "pct_above_usl": _r(self.pct_above_usl),
            "pct_outside": _r(self.pct_outside),
        }

    def __repr__(self) -> str:
        """Formatted summary using self.round_to."""
        d = self.as_dict()
        lines = ["CapabilityResult:"]
        lines.append(f"  Specs: LSL={d['lsl']}, USL={d['usl']}, Target={d['target']}")
        lines.append(f"  N={d['n']}, Ybar={d['y_bar']}, S={d['s']}, sigma_hat={d['sigma_hat']}")
        lines.append("")

        lines.append("  Current Capability (overall sigma):")
        lines.append(f"    Pp={d['pp']}  Ppk={d['ppk']}  (lower={d['ppk_lower']}, upper={d['ppk_upper']})")

        lines.append("")
        if self.potential_unavailable_reason:
            lines.append(f"  Potential Capability: {self.potential_unavailable_reason}")
        else:
            lines.append("  Potential Capability (R2 sigma):")
            lines.append(
                f"    Cp={d['cp']}  Cpk={d['cpk']}  "
                f"(lower={d['cpk_lower']}, upper={d['cpk_upper']})"
            )

        lines.append("")
        lines.append("  Z-scores:")
        lines.append(f"    Z_lower={d['z_lower']}, Z_upper={d['z_upper']}")

        lines.append("")
        lines.append("  Empirical:")
        lines.append(
            f"    Below LSL: {d['n_below_lsl']} ({d['pct_below_lsl']}%)"
            f"  Above USL: {d['n_above_usl']} ({d['pct_above_usl']}%)"
            f"  Total outside: {d['n_outside']} ({d['pct_outside']}%)"
        )

        if not self.stability_evaluated:
            lines.append("")
            lines.append(
                "  Note: Stability not assessed; run study.execute() "
                "and review signals before interpreting indices."
            )

        return "\n".join(lines)


# ============================================================================
# Pure Functions
# ============================================================================

def compute_sigma_hat(values: np.ndarray) -> tuple[float, float]:
    """
    Compute sample std dev (ddof=1) and unbiased sigma estimate.

    Parameters
    ----------
    values : array-like
        Non-NaN numeric values.  Must have len >= 2.

    Returns
    -------
    (s, sigma_hat) : tuple[float, float]
        s = np.std(values, ddof=1)
        sigma_hat = s / c4(len(values))

    Notes
    -----
    Wheeler/Bishop Ch. 16 — ddof=1 is mandatory.  The c4 correction
    removes the small-sample bias inherent in the sample standard
    deviation.
    """
    n = len(values)
    s = float(np.std(values, ddof=1))
    sigma_hat = s / c4(n)
    return s, sigma_hat


def compute_capability_indices(
    y_bar: float,
    sigma_hat: float,
    specs: SpecLimits,
) -> dict:
    """
    Compute Pp/Ppk (or Cp/Cpk) from a mean, sigma estimate, and spec limits.

    Parameters
    ----------
    y_bar : float
        Process mean (grand mean of observations).
    sigma_hat : float
        Unbiased sigma estimate.  When zero (all identical values),
        indices are returned as ``float('inf')``.
    specs : SpecLimits
        Specification limits.

    Returns
    -------
    dict
        Keys: pp, ppk_lower, ppk_upper, ppk, z_lower, z_upper.
        ``None`` where the relevant limit is absent.
    """

    def _safe_div(numerator: float, denominator: float) -> float:
        """Divide, returning inf when denominator is zero."""
        if denominator == 0.0:
            return float("inf")
        return numerator / denominator

    pp = None
    ppk_lower = None
    ppk_upper = None
    z_lower = None
    z_upper = None

    if specs.lsl is not None:
        ppk_lower = _safe_div(y_bar - specs.lsl, 3 * sigma_hat)
        z_lower = _safe_div(y_bar - specs.lsl, sigma_hat)

    if specs.usl is not None:
        ppk_upper = _safe_div(specs.usl - y_bar, 3 * sigma_hat)
        z_upper = _safe_div(specs.usl - y_bar, sigma_hat)

    if specs.is_two_sided:
        pp = _safe_div(specs.usl - specs.lsl, 6 * sigma_hat)
        assert ppk_lower is not None and ppk_upper is not None
        ppk = min(ppk_lower, ppk_upper)
    elif ppk_lower is not None:
        ppk = ppk_lower
    else:
        ppk = ppk_upper

    return {
        "pp": pp,
        "ppk_lower": ppk_lower,
        "ppk_upper": ppk_upper,
        "ppk": ppk,
        "z_lower": z_lower,
        "z_upper": z_upper,
    }


def compute_pct_outside(values: np.ndarray, specs: SpecLimits) -> dict:
    """
    Compute empirical counts and percents outside specification limits.

    Parameters
    ----------
    values : array-like
        Non-NaN numeric values.
    specs : SpecLimits
        Specification limits.

    Returns
    -------
    dict
        Keys: n_below_lsl, n_above_usl, n_outside,
        pct_below_lsl, pct_above_usl, pct_outside.
        ``None`` where the relevant limit is absent.
    """
    n = len(values)

    n_below = None
    n_above = None
    pct_below = None
    pct_above = None

    if specs.lsl is not None:
        n_below = int(np.sum(values < specs.lsl))
        pct_below = n_below / n * 100

    if specs.usl is not None:
        n_above = int(np.sum(values > specs.usl))
        pct_above = n_above / n * 100

    # Total outside
    if n_below is not None and n_above is not None:
        n_outside = n_below + n_above
        pct_outside = pct_below + pct_above
    elif n_below is not None:
        n_outside = n_below
        pct_outside = pct_below
    else:
        n_outside = n_above
        pct_outside = pct_above

    return {
        "n_below_lsl": n_below,
        "n_above_usl": n_above,
        "n_outside": n_outside,
        "pct_below_lsl": pct_below,
        "pct_above_usl": pct_above,
        "pct_outside": pct_outside,
    }


# ============================================================================
# Coercion
# ============================================================================

def _coerce_ads(source: Study | AnalysisDataSet) -> AnalysisDataSet:
    """
    Accept Study or AnalysisDataSet and return ADS.

    Keeps the public ``assess_capability`` function stable if Study
    internals shift.
    """
    # Avoid circular import by checking attribute
    if hasattr(source, "_ads"):
        return source._ads
    return source


# ============================================================================
# Orchestrator
# ============================================================================

def assess_capability(
    source: Study | AnalysisDataSet,
    specs: SpecLimits,
    round_to: int = 3,
) -> CapabilityResult:
    """
    Assess process capability against specification limits.

    This is the main entry point for capability analysis.  It accepts
    either a :class:`Study` or an :class:`AnalysisDataSet` and returns
    a :class:`CapabilityResult`.

    Parameters
    ----------
    source : Study | AnalysisDataSet
        Data source containing the response variable (and optionally
        R2 residuals for potential capability).
    specs : SpecLimits
        Specification limits to assess against.
    round_to : int, default 3
        Decimal places for presentation in ``__repr__`` / ``as_dict``.

    Returns
    -------
    CapabilityResult
        Frozen dataclass with all capability indices.

    Raises
    ------
    ValidationError
        If fewer than 2 valid observations are available.
    """
    ads = _coerce_ads(source)
    response_var = ads.spec.response_var
    df = ads.analysis_dataset

    # --- Extract valid Y values ---
    y_values = df[response_var].dropna().to_numpy(dtype=float)
    n = len(y_values)

    if n < 2:
        raise ValidationError(
            f"Capability analysis requires at least 2 valid observations, got {n}."
        )

    # --- Current capability ---
    y_bar = float(np.mean(y_values))
    s, sigma_hat = compute_sigma_hat(y_values)

    current = compute_capability_indices(y_bar, sigma_hat, specs)
    outside = compute_pct_outside(y_values, specs)

    # --- Potential capability (from R2 residuals) ---
    sigma_hat_r2 = None
    cp = None
    cpk_lower = None
    cpk_upper = None
    cpk = None
    potential_unavailable_reason = None

    if ads.has_vas_residuals and "R2" in df.columns:
        r2_values = df["R2"].dropna().to_numpy(dtype=float)
        n_r2 = len(r2_values)

        if n_r2 >= 2:
            _, sigma_hat_r2 = compute_sigma_hat(r2_values)
            pot = compute_capability_indices(y_bar, sigma_hat_r2, specs)
            cp = pot["pp"]
            cpk_lower = pot["ppk_lower"]
            cpk_upper = pot["ppk_upper"]
            cpk = pot["ppk"]
        else:
            potential_unavailable_reason = (
                f"Too few R2 residual values ({n_r2}) for potential capability; "
                f"need at least 2"
            )
    else:
        potential_unavailable_reason = (
            f"R2 residuals not available (SDS {ads.sampling_design_state}); "
            f"Cp/Cpk require VAS residuals from a factorial+time design"
        )

    return CapabilityResult(
        specs=specs,
        n=n,
        y_bar=y_bar,
        s=s,
        sigma_hat=sigma_hat,
        pp=current["pp"],
        ppk_lower=current["ppk_lower"],
        ppk_upper=current["ppk_upper"],
        ppk=current["ppk"],
        sigma_hat_r2=sigma_hat_r2,
        cp=cp,
        cpk_lower=cpk_lower,
        cpk_upper=cpk_upper,
        cpk=cpk,
        potential_unavailable_reason=potential_unavailable_reason,
        z_lower=current["z_lower"],
        z_upper=current["z_upper"],
        n_below_lsl=outside["n_below_lsl"],
        n_above_usl=outside["n_above_usl"],
        n_outside=outside["n_outside"],
        pct_below_lsl=outside["pct_below_lsl"],
        pct_above_usl=outside["pct_above_usl"],
        pct_outside=outside["pct_outside"],
        round_to=round_to,
    )
