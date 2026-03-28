"""
Taguchi Loss Function Analysis — Wheeler/Bishop Chapter 15.

Decomposes expected loss into 5 components that identify where variation
comes from: centering, unexplained (within-cell), process design conditions
(PDC), time, and PDC×time interaction. When multiple factors exist, PDC
further decomposes into per-factor main effect losses and factor interaction
loss.

References
----------
Wheeler, D.J. & Bishop, T.  *Variance Analysis System* — Chapter 15.
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
# Data Class
# ============================================================================

@dataclass(frozen=True)
class LossResult:
    """
    Result of a Taguchi Loss Function analysis (Wheeler/Bishop Ch. 15).

    Decomposes expected loss into 5 components: centering, unexplained,
    PDC, time, and PDC×time interaction. When multiple factors exist,
    PDC further decomposes into per-factor losses and factor interaction.

    Attributes
    ----------
    target : float
        Target value used for the analysis.
    target_is_default : bool
        True when target defaulted to grand mean (centering = 0).
    y_bar : float
        Grand mean of all observations.
    n : int
        Total observations.
    K : int
        Number of PDC levels (rsg_key groups).
    T_periods : int
        Number of time periods.
    sds : int
        Analytical SDS.
    centering, unexplained, pdc, time, interaction : float
        The 5 loss components (squared-loss units).
    total : float
        Sum of 5 components.
    pct_centering, pct_unexplained, pct_pdc, pct_time, pct_interaction : float
        Each component as percentage of total.
    pdc_by_factor : dict[str, float] | None
        Per-factor loss decomposition (multi-factor only).
    pdc_factor_interaction : float | None
        PDC minus sum of per-factor losses (multi-factor only).
    round_to : int
        Decimal places for display.
    """

    # Context
    target: float
    target_is_default: bool
    y_bar: float
    n: int
    K: int
    T_periods: int
    sds: int

    # 5 loss components (squared-loss units)
    centering: float
    unexplained: float
    pdc: float
    time: float
    interaction: float
    total: float

    # Pareto (percentages of total)
    pct_centering: float
    pct_unexplained: float
    pct_pdc: float
    pct_time: float
    pct_interaction: float

    # PDC decomposition by factor (multi-factor only)
    pdc_by_factor: dict[str, float] | None
    pdc_factor_interaction: float | None

    # Presentation
    round_to: int = 3

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, *, structured: bool = False, orientation: str = 'vertical',
             theme=None, width: int = 700, height: int = 400,
             title: str | None = None):
        """
        Pareto bar chart of loss components.

        Parameters
        ----------
        structured : bool, default False
            False: 5 bars (MEAN, UNEXPLAINED, PDC, PT, PDCxPT INT).
            True: PDC expanded into per-factor bars + PDF INT.
        orientation : str, default 'vertical'
            ``'vertical'`` or ``'horizontal'``.
        theme : ChartTheme, optional
            Visual theme.
        width, height : int
            Figure dimensions.
        title : str, optional
            Chart title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        from .plotting.themes import ChartTheme, get_theme

        if theme is None:
            theme = ChartTheme()
        elif isinstance(theme, str):
            theme = get_theme(theme)

        # Build (label, pct) pairs
        if structured and self.pdc_by_factor and len(self.pdc_by_factor) >= 2:
            items = [
                ("MEAN", self.pct_centering),
                ("UNEXPLAINED", self.pct_unexplained),
            ]
            for factor, loss_val in self.pdc_by_factor.items():
                pct = (loss_val / self.total * 100) if self.total > 0 else 0.0
                items.append((factor, pct))
            pdc_int_pct = (
                (self.pdc_factor_interaction / self.total * 100)
                if self.total > 0 else 0.0
            )
            items.append(("PDF INT", pdc_int_pct))
            items.append(("PT", self.pct_time))
            items.append(("PDCxPT INT", self.pct_interaction))
        else:
            items = [
                ("MEAN", self.pct_centering),
                ("UNEXPLAINED", self.pct_unexplained),
                ("PDC", self.pct_pdc),
                ("PT", self.pct_time),
                ("PDCxPT INT", self.pct_interaction),
            ]

        # Pareto order (descending)
        items.sort(key=lambda x: x[1], reverse=True)
        labels = [x[0] for x in items]
        values = [x[1] for x in items]

        r = self.round_to
        text = [f"{round(v, r)}%" for v in values]
        horizontal = orientation.startswith('h')
        axis_max = max(values) * 1.1 if values else 100

        value_label = "Contribution to Average Loss (%)"
        component_label = "Components Contributing to Average Loss"

        if horizontal:
            fig = go.Figure(go.Bar(
                y=labels[::-1],
                x=values[::-1],
                orientation='h',
                text=text[::-1],
                textposition='outside',
            ))
            fig.update_layout(
                xaxis_title=value_label,
                yaxis_title=component_label,
                xaxis_range=[0, axis_max],
                width=width,
                height=height,
                margin=dict(l=120),
            )
        else:
            fig = go.Figure(go.Bar(
                x=labels,
                y=values,
                orientation='v',
                text=text,
                textposition='outside',
            ))
            fig.update_layout(
                yaxis_title=value_label,
                xaxis_title=component_label,
                yaxis_range=[0, axis_max],
                width=width,
                height=height,
                margin=dict(b=80),
            )

        layout = theme.to_layout_dict()
        layout['title']['text'] = title or "Taguchi Loss Function Analysis"
        fig.update_layout(**layout)

        return fig

    # ------------------------------------------------------------------
    # Presentation helpers
    # ------------------------------------------------------------------

    def as_dict(self, round_to: int | None = None) -> dict:
        """Return results as a dict with values rounded for display."""
        r = round_to if round_to is not None else self.round_to

        d = {
            "target": round(self.target, r),
            "target_is_default": self.target_is_default,
            "y_bar": round(self.y_bar, r),
            "n": self.n,
            "K": self.K,
            "T_periods": self.T_periods,
            "sds": self.sds,
            "centering": round(self.centering, r),
            "unexplained": round(self.unexplained, r),
            "pdc": round(self.pdc, r),
            "time": round(self.time, r),
            "interaction": round(self.interaction, r),
            "total": round(self.total, r),
            "pct_centering": round(self.pct_centering, r),
            "pct_unexplained": round(self.pct_unexplained, r),
            "pct_pdc": round(self.pct_pdc, r),
            "pct_time": round(self.pct_time, r),
            "pct_interaction": round(self.pct_interaction, r),
        }

        if self.pdc_by_factor is not None:
            d["pdc_by_factor"] = {
                k: round(v, r) for k, v in self.pdc_by_factor.items()
            }
            d["pdc_factor_interaction"] = round(self.pdc_factor_interaction, r)

        return d

    def __repr__(self) -> str:
        """Formatted summary."""
        d = self.as_dict()
        r = self.round_to
        lines = ["LossResult:"]
        tgt_note = " (default=grand mean)" if self.target_is_default else ""
        lines.append(f"  Target={d['target']}{tgt_note}, Ybar={d['y_bar']}, N={d['n']}")
        lines.append(f"  K={d['K']} PDC levels, T={d['T_periods']} periods, SDS={d['sds']}")
        lines.append("")
        lines.append(f"  Expected Loss (EL) = {d['total']}")
        lines.append("")

        # Pareto-ordered components
        components = [
            ("MEAN", d["pct_centering"]),
            ("UNEXPLAINED", d["pct_unexplained"]),
            ("PDC", d["pct_pdc"]),
            ("PT", d["pct_time"]),
            ("PDCxPT INT", d["pct_interaction"]),
        ]
        components.sort(key=lambda x: x[1], reverse=True)

        lines.append("  Loss Decomposition:")
        for label, pct in components:
            lines.append(f"    {label:>15s}: {round(pct, r)}%")

        if self.pdc_by_factor and len(self.pdc_by_factor) >= 2:
            lines.append("")
            lines.append("  PDC Decomposition:")
            pdc_items = sorted(
                self.pdc_by_factor.items(), key=lambda x: x[1], reverse=True
            )
            for factor, loss_val in pdc_items:
                pct = (loss_val / self.total * 100) if self.total > 0 else 0.0
                lines.append(f"    {factor:>15s}: {round(pct, r)}%")
            pdc_int_pct = (
                (self.pdc_factor_interaction / self.total * 100)
                if self.total > 0 else 0.0
            )
            lines.append(f"    {'PDF INT':>15s}: {round(pdc_int_pct, r)}%")

        return "\n".join(lines)


# ============================================================================
# Pure Functions
# ============================================================================

def _compute_centering(y_bar: float, target: float) -> float:
    """Eq 15.13 term 1: (Ȳ.. - T)²."""
    return (y_bar - target) ** 2


def _compute_unexplained_replicated(df, response_var: str) -> float:
    """
    Eq 15.16/15.17: within-cell variance for fully replicated designs.

    unexplained = (1/KT) Σ (S_kt / c4(n_kt))²
    """
    cell_stats = df.groupby('cell_key', observed=True)[response_var].agg(['std', 'count'])
    cell_stats['sigma_sq'] = (
        cell_stats['std'] / cell_stats['count'].apply(c4)
    ) ** 2
    return float(cell_stats['sigma_sq'].mean())


def _compute_unexplained_pooled(df) -> float:
    """
    Eq 15.18/15.19: pooled sigma from R2 for ADS 2/3.

    Uses std(R2, ddof=1) / 0.7 then squares (Wheeler/Bishop Eq 15.18).
    """
    r2 = df['R2'].dropna().to_numpy(dtype=float)
    sigma_hat = np.std(r2, ddof=1) / 0.7
    return float(sigma_hat ** 2)


def _compute_pdc(df, rsg_var_name: str) -> float:
    """Eq 15.20: (1/K) Σ (Ȳ_k. - Ȳ..)²."""
    ybar = df['Ybar'].iloc[0]
    ybar_k = df.groupby(rsg_var_name, observed=True)['Ybar_k'].first()
    rho_k = ybar_k - ybar
    return float((rho_k ** 2).mean())


def _compute_time(df, time_var: str) -> float:
    """Eq 15.13 term 4: (1/T) Σ (Ȳ_.t - Ȳ..)²."""
    ybar = df['Ybar'].iloc[0]
    ybar_t = df.groupby(time_var, observed=True)['Ybar_t'].first()
    tau_t = ybar_t - ybar
    return float((tau_t ** 2).mean())


def _compute_interaction(df, time_var: str) -> float:
    """Eq 15.13 term 5: (1/KT) Σ (ρτ_kt)²."""
    ybar = df['Ybar'].iloc[0]
    cell = df.groupby('cell_key', observed=True)[['Ybar_kt', 'Ybar_k', 'Ybar_t']].first()
    rho_tau = cell['Ybar_kt'] - cell['Ybar_k'] - cell['Ybar_t'] + ybar
    return float((rho_tau ** 2).mean())


def _compute_pdc_decomposition(
    effects: dict, rsg_vars: tuple[str, ...], pdc_total: float
) -> tuple[dict[str, float], float]:
    """
    Decompose PDC into per-factor main effect losses and factor interaction.

    Per-factor: Loss(Fm) = (1/Im) Σ [Main_Effect_i]²
    Factor interaction = PDC - Σ Loss(Fm)
    """
    pdc_by_factor: dict[str, float] = {}
    for factor in rsg_vars:
        me = effects[factor]['Main_Effect']
        pdc_by_factor[factor] = float((me ** 2).mean())

    pdc_factor_interaction = pdc_total - sum(pdc_by_factor.values())
    return pdc_by_factor, pdc_factor_interaction


def _compute_pareto(
    centering: float, unexplained: float, pdc: float,
    time_loss: float, interaction: float, total: float,
) -> tuple[float, float, float, float, float]:
    """Each component as percentage of total. Zero-safe."""
    if total == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        centering / total * 100,
        unexplained / total * 100,
        pdc / total * 100,
        time_loss / total * 100,
        interaction / total * 100,
    )


# ============================================================================
# Coercion
# ============================================================================

def _coerce_ads(source: Study | AnalysisDataSet) -> AnalysisDataSet:
    """Accept Study or AnalysisDataSet and return ADS."""
    if hasattr(source, "_ads"):
        return source._ads
    return source


# ============================================================================
# Orchestrator
# ============================================================================

def assess_loss(
    source: Study | AnalysisDataSet,
    target: float | None = None,
    round_to: int = 3,
) -> LossResult:
    """
    Assess Taguchi Loss Function decomposition (Wheeler/Bishop Ch. 15).

    Parameters
    ----------
    source : Study | AnalysisDataSet
        Data source with VAS residuals computed.
    target : float, optional
        Target value. Defaults to grand mean (centering = 0).
    round_to : int, default 3
        Decimal places for presentation.

    Returns
    -------
    LossResult
        Frozen dataclass with 5-component loss decomposition.

    Raises
    ------
    ValidationError
        If VAS residuals are not available (requires factors + time).
    """
    ads = _coerce_ads(source)

    if not ads.has_vas_residuals:
        raise ValidationError(
            "Loss function analysis requires VAS residuals "
            "(factors + time design). "
            f"Current design (SDS {ads.observed_design_state}) "
            "does not have VAS residuals."
        )

    df = ads.analysis_dataset
    spec = ads.spec
    response_var = spec.response_var
    rsg_var_name = spec.rsg_var_name
    time_var = spec.time_var
    rsg_vars = spec.rsg_vars

    # Context
    y_bar = float(df['Ybar'].iloc[0])
    n = len(df)
    K = df[rsg_var_name].nunique()
    T_periods = df[time_var].nunique()
    sds = ads.analytical_design_state.sds

    # Default target
    target_is_default = target is None
    if target_is_default:
        target = y_bar

    # --- 5 components ---
    centering = _compute_centering(y_bar, target)

    # Unexplained: branch on cell replication
    if ads._structure_stats.n_cell_min >= 2:
        unexplained = _compute_unexplained_replicated(df, response_var)
    else:
        unexplained = _compute_unexplained_pooled(df)

    pdc = _compute_pdc(df, rsg_var_name)
    time_loss = _compute_time(df, time_var)
    interaction = _compute_interaction(df, time_var)

    total = centering + unexplained + pdc + time_loss + interaction

    # Pareto
    pct = _compute_pareto(centering, unexplained, pdc, time_loss, interaction, total)

    # PDC decomposition by factor
    pdc_by_factor = None
    pdc_factor_interaction = None
    if rsg_vars and len(rsg_vars) >= 2:
        pdc_by_factor, pdc_factor_interaction = _compute_pdc_decomposition(
            ads.effects, rsg_vars, pdc
        )
    elif rsg_vars and len(rsg_vars) == 1:
        pdc_by_factor = {rsg_vars[0]: pdc}
        pdc_factor_interaction = 0.0

    return LossResult(
        target=target,
        target_is_default=target_is_default,
        y_bar=y_bar,
        n=n,
        K=K,
        T_periods=T_periods,
        sds=sds,
        centering=centering,
        unexplained=unexplained,
        pdc=pdc,
        time=time_loss,
        interaction=interaction,
        total=total,
        pct_centering=pct[0],
        pct_unexplained=pct[1],
        pct_pdc=pct[2],
        pct_time=pct[3],
        pct_interaction=pct[4],
        pdc_by_factor=pdc_by_factor,
        pdc_factor_interaction=pdc_factor_interaction,
        round_to=round_to,
    )
