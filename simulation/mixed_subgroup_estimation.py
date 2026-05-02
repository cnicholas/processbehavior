"""
Monte Carlo Simulation: Range-Based Sigma Estimation Under Mixed Subgroup Sizes

Investigates how range-based within-subgroup sigma estimation degrades when
data contains many singletons (n=1) mixed with real subgroups (n≥2).

Model:  y_ij = μ + α_i + ε_ij
        α_i ~ N(0, σ_between²)
        ε_ij ~ N(0, σ_within²)

Three estimators compared:
    1. Range-based (Wheeler): mean(R_i / d2(n_i)) for subgroups with n≥2
    2. GLM residual: sqrt(MSE) from one-way ANOVA
    3. Naive MR-mixed: moving ranges for singletons + subgroup ranges for n≥2

Experimental factors (5 × 3 × 4 = 60 scenarios):
    - Singleton proportion: 10%, 30%, 50%, 70%, 90%
    - Non-singleton size distribution: small (2–3), moderate (2–5), wide (2–15)
    - SNR (σ_between / σ_within): 0, 0.5, 1.0, 2.0

References
----------
Wheeler, D. J. & Chambers, D. S. (1992). Understanding Statistical Process
    Control.
Wheeler, D. J. (1995). Advanced Topics in Statistical Process Control.
Bishop, T. & Wheeler, D. J. (2024). Variance Analysis System (VAS).

Author: Nicholas Hollingsworth
Date: 2025-02
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# Constants
# ============================================================================

MU = 100.0
SIGMA_WITHIN = 5.0
K = 50  # number of subgroups per dataset
N_REPS = 1000  # replications per scenario

SINGLETON_PROPORTIONS = [0.10, 0.30, 0.50, 0.70, 0.90]

SIZE_DISTRIBUTIONS = {
    "small (2–3)": (2, 3),
    "moderate (2–5)": (2, 5),
    "wide (2–15)": (2, 15),
}

SNR_VALUES = [0.0, 0.5, 1.0, 2.0]  # σ_between / σ_within

OUTPUT_DIR = Path(__file__).parent / "output"

# Tabulated d2 constants: expected value of range for n standard normal obs.
# Source: Wheeler (1995), Table A.1; also ASTM E2587.
_D2_TABLE = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
    11: 3.173,
    12: 3.258,
    13: 3.336,
    14: 3.407,
    15: 3.472,
}


def d2(n: int) -> float:
    """Unbiasing constant for range-based sigma estimation."""
    if n < 2:
        raise ValueError(f"d2 requires n >= 2, got {n}")
    if n not in _D2_TABLE:
        raise ValueError(f"d2 not tabulated for n={n} (max 15)")
    return _D2_TABLE[n]


# ============================================================================
# Data Generation
# ============================================================================

def generate_subgroup_sizes(
    rng: np.random.Generator,
    k: int,
    singleton_prop: float,
    size_range: tuple[int, int],
) -> np.ndarray:
    """Generate K subgroup sizes with a controlled singleton proportion.

    Parameters
    ----------
    rng : np.random.Generator
    k : int
        Number of subgroups.
    singleton_prop : float
        Fraction of subgroups that are singletons (n=1).
    size_range : tuple[int, int]
        (min_size, max_size) for non-singleton subgroups (inclusive).

    Returns
    -------
    np.ndarray of int, shape (k,)
    """
    n_singletons = int(round(k * singleton_prop))
    n_grouped = k - n_singletons

    sizes = np.ones(k, dtype=int)
    if n_grouped > 0:
        lo, hi = size_range
        sizes[n_singletons:] = rng.integers(lo, hi + 1, size=n_grouped)

    rng.shuffle(sizes)
    return sizes


def generate_dataset(
    rng: np.random.Generator,
    sizes: np.ndarray,
    sigma_between: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate observations from y_ij = μ + α_i + ε_ij.

    Returns
    -------
    y : np.ndarray
        All observations, concatenated.
    group_labels : np.ndarray
        Subgroup index for each observation.
    """
    groups = []
    labels = []
    for i, n_i in enumerate(sizes):
        alpha_i = rng.normal(0, sigma_between) if sigma_between > 0 else 0.0
        eps = rng.normal(0, SIGMA_WITHIN, size=n_i)
        groups.append(MU + alpha_i + eps)
        labels.append(np.full(n_i, i, dtype=int))

    return np.concatenate(groups), np.concatenate(labels)


# ============================================================================
# Estimation Methods
# ============================================================================

def estimate_range_based(
    y: np.ndarray,
    group_labels: np.ndarray,
    sizes: np.ndarray,
) -> float | None:
    """Wheeler range-based: mean(R_i / d2(n_i)) for subgroups with n≥2.

    Returns None if no subgroups with n≥2 exist.
    """
    estimates = []
    for i, n_i in enumerate(sizes):
        if n_i < 2:
            continue
        obs = y[group_labels == i]
        r_i = obs.max() - obs.min()
        estimates.append(r_i / d2(n_i))

    if not estimates:
        return None
    return float(np.mean(estimates))


def estimate_glm_residual(
    y: np.ndarray,
    group_labels: np.ndarray,
    sizes: np.ndarray,
) -> float | None:
    """sqrt(MSE) from one-way ANOVA (only subgroups with n≥2 contribute df).

    Uses the within-group sum of squares / within-group df.
    Singletons contribute 0 df, so they're naturally excluded.
    Returns None if total within-df is 0.
    """
    ss_within = 0.0
    df_within = 0
    for i, n_i in enumerate(sizes):
        if n_i < 2:
            continue
        obs = y[group_labels == i]
        group_mean = obs.mean()
        ss_within += np.sum((obs - group_mean) ** 2)
        df_within += n_i - 1

    if df_within == 0:
        return None
    return float(np.sqrt(ss_within / df_within))


def estimate_naive_mr_mixed(
    y: np.ndarray,
    group_labels: np.ndarray,
    sizes: np.ndarray,
) -> float:
    """Naive mixed estimator: MR for singletons + subgroup ranges for n≥2.

    For singletons, uses moving ranges between consecutive singleton values
    (in data order). For subgroups with n≥2, uses R_i / d2(n_i).
    The final estimate is the weighted average.

    This estimator is BIASED when σ_between > 0 because moving ranges between
    singletons from different subgroups capture both within and between
    variation.
    """
    # Collect singleton values in order, and range-based estimates from groups
    singleton_values = []
    range_estimates = []

    for i, n_i in enumerate(sizes):
        if n_i == 1:
            singleton_values.append(float(y[group_labels == i][0]))
        else:
            obs = y[group_labels == i]
            r_i = obs.max() - obs.min()
            range_estimates.append(r_i / d2(n_i))

    # Moving ranges between consecutive singletons
    mr_estimates = []
    if len(singleton_values) >= 2:
        sv = np.array(singleton_values)
        mrs = np.abs(np.diff(sv))
        # Each MR is a range of n=2, so divide by d2(2)
        mr_sigma = float(np.mean(mrs)) / d2(2)
        mr_estimates.append(mr_sigma)

    all_estimates = range_estimates + mr_estimates
    if not all_estimates:
        return float("nan")
    return float(np.mean(all_estimates))


# ============================================================================
# Simulation Engine
# ============================================================================

def run_scenario(
    singleton_prop: float,
    size_label: str,
    size_range: tuple[int, int],
    snr: float,
    n_reps: int,
    seed_base: int,
) -> dict:
    """Run n_reps replications for one scenario. Returns summary metrics."""
    sigma_between = snr * SIGMA_WITHIN
    sigma_true = SIGMA_WITHIN

    range_estimates = []
    glm_estimates = []
    naive_estimates = []

    for rep in range(n_reps):
        rng = np.random.default_rng(seed_base + rep)
        sizes = generate_subgroup_sizes(rng, K, singleton_prop, size_range)
        y, labels = generate_dataset(rng, sizes, sigma_between)

        r_est = estimate_range_based(y, labels, sizes)
        g_est = estimate_glm_residual(y, labels, sizes)
        n_est = estimate_naive_mr_mixed(y, labels, sizes)

        if r_est is not None:
            range_estimates.append(r_est)
        if g_est is not None:
            glm_estimates.append(g_est)
        naive_estimates.append(n_est)

    def metrics(estimates: list[float], label: str) -> dict:
        arr = np.array(estimates)
        n_valid = len(arr)
        if n_valid == 0:
            return {
                "method": label,
                "n_valid": 0,
                "mean_estimate": np.nan,
                "bias_pct": np.nan,
                "rmse": np.nan,
                "ratio": np.nan,
                "std_estimate": np.nan,
            }
        bias = arr - sigma_true
        return {
            "method": label,
            "n_valid": n_valid,
            "mean_estimate": float(np.mean(arr)),
            "bias_pct": float(np.mean(bias) / sigma_true * 100),
            "rmse": float(np.sqrt(np.mean(bias**2))),
            "ratio": float(np.mean(arr) / sigma_true),
            "std_estimate": float(np.std(arr, ddof=1)),
        }

    results = []
    for est_list, label in [
        (range_estimates, "Range-based"),
        (glm_estimates, "GLM residual"),
        (naive_estimates, "Naive MR-mixed"),
    ]:
        m = metrics(est_list, label)
        m.update({
            "singleton_pct": int(singleton_prop * 100),
            "size_dist": size_label,
            "snr": snr,
        })
        results.append(m)

    return results


def run_all_scenarios() -> pd.DataFrame:
    """Run all 60 scenarios and return results DataFrame."""
    all_results = []
    total = len(SINGLETON_PROPORTIONS) * len(SIZE_DISTRIBUTIONS) * len(SNR_VALUES)
    done = 0

    for sp in SINGLETON_PROPORTIONS:
        for size_label, size_range in SIZE_DISTRIBUTIONS.items():
            for snr in SNR_VALUES:
                # Deterministic seed per scenario for reproducibility
                # Note: Python's hash() is randomized across invocations for
                # strings (since 3.3), so we use hashlib for true repeatability.
                key = f"{sp}|{size_label}|{snr}"
                seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)
                results = run_scenario(sp, size_label, size_range, snr, N_REPS, seed)
                all_results.extend(results)
                done += 1
                pct = done / total * 100
                print(f"  [{done:2d}/{total}] {pct:5.1f}%  "
                      f"singleton={int(sp*100):2d}%  size={size_label:<16s}  "
                      f"SNR={snr:.1f}")

    return pd.DataFrame(all_results)


# ============================================================================
# Visualization
# ============================================================================

def make_bias_heatmaps(df: pd.DataFrame) -> go.Figure:
    """Heatmap: singleton % vs size distribution, colored by bias. One per SNR."""
    methods = ["Range-based", "GLM residual", "Naive MR-mixed"]
    snr_vals = sorted(df["snr"].unique())
    n_snr = len(snr_vals)

    fig = make_subplots(
        rows=len(methods),
        cols=n_snr,
        subplot_titles=[
            f"{m} — SNR={s}" for m in methods for s in snr_vals
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    # Find global color range for consistent scale
    bias_vals = df["bias_pct"].dropna()
    zmax = max(abs(bias_vals.min()), abs(bias_vals.max()), 1.0)

    for row_idx, method in enumerate(methods, 1):
        for col_idx, snr in enumerate(snr_vals, 1):
            sub = df[(df["method"] == method) & (df["snr"] == snr)]
            pivot = sub.pivot_table(
                index="singleton_pct", columns="size_dist",
                values="bias_pct", aggfunc="first",
            )
            # Order columns consistently
            col_order = [c for c in SIZE_DISTRIBUTIONS if c in pivot.columns]
            pivot = pivot[col_order]

            show_colorbar = (row_idx == 1 and col_idx == n_snr)
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[c.replace("(", "\n(") for c in pivot.columns],
                    y=[f"{v}%" for v in pivot.index],
                    colorscale="RdBu_r",
                    zmid=0,
                    zmin=-zmax,
                    zmax=zmax,
                    text=np.round(pivot.values, 1),
                    texttemplate="%{text:.1f}%",
                    showscale=show_colorbar,
                    colorbar=dict(title="Bias %") if show_colorbar else None,
                ),
                row=row_idx, col=col_idx,
            )

    fig.update_layout(
        title="Bias (%) in σ̂_within Estimation: Singleton Proportion × Size Distribution",
        height=300 * len(methods) + 100,
        width=300 * n_snr + 100,
    )
    return fig


def make_bias_line_charts(df: pd.DataFrame) -> go.Figure:
    """Line chart: singleton % on x, bias on y, one line per size dist."""
    methods = ["Range-based", "GLM residual", "Naive MR-mixed"]
    snr_vals = sorted(df["snr"].unique())
    n_snr = len(snr_vals)

    fig = make_subplots(
        rows=len(methods),
        cols=n_snr,
        subplot_titles=[
            f"{m} — SNR={s}" for m in methods for s in snr_vals
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    colors = {"small (2–3)": "#1f77b4", "moderate (2–5)": "#ff7f0e", "wide (2–15)": "#2ca02c"}

    for row_idx, method in enumerate(methods, 1):
        for col_idx, snr in enumerate(snr_vals, 1):
            sub = df[(df["method"] == method) & (df["snr"] == snr)]
            for size_label in SIZE_DISTRIBUTIONS:
                s = sub[sub["size_dist"] == size_label].sort_values("singleton_pct")
                show_legend = (row_idx == 1 and col_idx == 1)
                fig.add_trace(
                    go.Scatter(
                        x=s["singleton_pct"],
                        y=s["bias_pct"],
                        mode="lines+markers",
                        name=size_label,
                        legendgroup=size_label,
                        showlegend=show_legend,
                        line=dict(color=colors[size_label]),
                    ),
                    row=row_idx, col=col_idx,
                )

            fig.update_xaxes(title_text="Singleton %", row=row_idx, col=col_idx)
            fig.update_yaxes(title_text="Bias %", row=row_idx, col=col_idx)

    fig.update_layout(
        title="Bias (%) vs Singleton Proportion by Estimation Method and SNR",
        height=350 * len(methods) + 100,
        width=300 * n_snr + 100,
    )
    return fig


def make_rmse_line_charts(df: pd.DataFrame) -> go.Figure:
    """Line chart: singleton % on x, RMSE on y, one line per size dist."""
    methods = ["Range-based", "GLM residual", "Naive MR-mixed"]
    snr_vals = sorted(df["snr"].unique())
    n_snr = len(snr_vals)

    fig = make_subplots(
        rows=len(methods),
        cols=n_snr,
        subplot_titles=[
            f"{m} — SNR={s}" for m in methods for s in snr_vals
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    colors = {"small (2–3)": "#1f77b4", "moderate (2–5)": "#ff7f0e", "wide (2–15)": "#2ca02c"}

    for row_idx, method in enumerate(methods, 1):
        for col_idx, snr in enumerate(snr_vals, 1):
            sub = df[(df["method"] == method) & (df["snr"] == snr)]
            for size_label in SIZE_DISTRIBUTIONS:
                s = sub[sub["size_dist"] == size_label].sort_values("singleton_pct")
                show_legend = (row_idx == 1 and col_idx == 1)
                fig.add_trace(
                    go.Scatter(
                        x=s["singleton_pct"],
                        y=s["rmse"],
                        mode="lines+markers",
                        name=size_label,
                        legendgroup=size_label,
                        showlegend=show_legend,
                        line=dict(color=colors[size_label]),
                    ),
                    row=row_idx, col=col_idx,
                )

            fig.update_xaxes(title_text="Singleton %", row=row_idx, col=col_idx)
            fig.update_yaxes(title_text="RMSE", row=row_idx, col=col_idx)

    fig.update_layout(
        title="RMSE vs Singleton Proportion by Estimation Method and SNR",
        height=350 * len(methods) + 100,
        width=300 * n_snr + 100,
    )
    return fig


def make_precision_chart(df: pd.DataFrame) -> go.Figure:
    """Std of estimates vs singleton %, showing precision degradation."""
    methods = ["Range-based", "GLM residual"]
    snr_vals = sorted(df["snr"].unique())
    n_snr = len(snr_vals)

    fig = make_subplots(
        rows=1,
        cols=n_snr,
        subplot_titles=[f"SNR={s}" for s in snr_vals],
        horizontal_spacing=0.06,
    )

    method_colors = {"Range-based": "#1f77b4", "GLM residual": "#ff7f0e"}
    dashes = {"small (2–3)": "solid", "moderate (2–5)": "dash", "wide (2–15)": "dot"}

    for col_idx, snr in enumerate(snr_vals, 1):
        for method in methods:
            sub = df[(df["method"] == method) & (df["snr"] == snr)]
            for size_label in SIZE_DISTRIBUTIONS:
                s = sub[sub["size_dist"] == size_label].sort_values("singleton_pct")
                show_legend = (col_idx == 1)
                fig.add_trace(
                    go.Scatter(
                        x=s["singleton_pct"],
                        y=s["std_estimate"],
                        mode="lines+markers",
                        name=f"{method} / {size_label}",
                        legendgroup=f"{method}_{size_label}",
                        showlegend=show_legend,
                        line=dict(
                            color=method_colors[method],
                            dash=dashes[size_label],
                        ),
                    ),
                    row=1, col=col_idx,
                )

        fig.update_xaxes(title_text="Singleton %", row=1, col=col_idx)
        fig.update_yaxes(title_text="Std(σ̂)", row=1, col=col_idx)

    fig.update_layout(
        title="Estimation Precision (Std of σ̂) vs Singleton Proportion",
        height=450,
        width=300 * n_snr + 200,
    )
    return fig


# ============================================================================
# Threshold Analysis
# ============================================================================

def threshold_analysis(df: pd.DataFrame) -> str:
    """Identify where RMSE exceeds meaningful thresholds."""
    lines = []
    lines.append("=" * 72)
    lines.append("THRESHOLD ANALYSIS: Where does estimation quality degrade?")
    lines.append("=" * 72)

    # For range-based and GLM, look at RMSE and precision
    thresholds = [0.5, 1.0, 1.5]  # RMSE thresholds

    for method in ["Range-based", "GLM residual"]:
        lines.append(f"\n--- {method} ---")
        sub = df[df["method"] == method].copy()

        for thresh in thresholds:
            exceeds = sub[sub["rmse"] > thresh]
            if exceeds.empty:
                lines.append(f"  RMSE > {thresh:.1f}: Never exceeded")
            else:
                lines.append(f"  RMSE > {thresh:.1f}:")
                for _, row in exceeds.iterrows():
                    lines.append(
                        f"    singleton={row['singleton_pct']:2.0f}%  "
                        f"size={row['size_dist']:<16s}  "
                        f"SNR={row['snr']:.1f}  "
                        f"RMSE={row['rmse']:.3f}  "
                        f"std={row['std_estimate']:.3f}"
                    )

    # For naive MR, look at bias
    lines.append("\n--- Naive MR-mixed (bias focus) ---")
    naive = df[df["method"] == "Naive MR-mixed"].copy()
    for bias_thresh in [5.0, 10.0, 20.0]:
        exceeds = naive[naive["bias_pct"].abs() > bias_thresh]
        if exceeds.empty:
            lines.append(f"  |Bias| > {bias_thresh:.0f}%: Never exceeded")
        else:
            lines.append(f"  |Bias| > {bias_thresh:.0f}%:")
            for _, row in exceeds.iterrows():
                lines.append(
                    f"    singleton={row['singleton_pct']:2.0f}%  "
                    f"size={row['size_dist']:<16s}  "
                    f"SNR={row['snr']:.1f}  "
                    f"bias={row['bias_pct']:+.1f}%"
                )

    return "\n".join(lines)


# ============================================================================
# Summary Report
# ============================================================================

def generate_summary(df: pd.DataFrame) -> str:
    """Generate a text summary of findings."""
    lines = []
    lines.append("=" * 72)
    lines.append("SIMULATION STUDY: Range-Based Sigma Estimation")
    lines.append("             Under Mixed Subgroup Sizes")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Parameters: μ={MU}, σ_within={SIGMA_WITHIN}, K={K}, "
                 f"replications={N_REPS}")
    lines.append(f"Scenarios:  {len(SINGLETON_PROPORTIONS)} singleton props × "
                 f"{len(SIZE_DISTRIBUTIONS)} size dists × "
                 f"{len(SNR_VALUES)} SNR values = "
                 f"{len(SINGLETON_PROPORTIONS) * len(SIZE_DISTRIBUTIONS) * len(SNR_VALUES)}")
    lines.append("")

    # Overall bias summary per method
    lines.append("-" * 72)
    lines.append("BIAS SUMMARY (% relative to σ_true)")
    lines.append("-" * 72)
    for method in ["Range-based", "GLM residual", "Naive MR-mixed"]:
        sub = df[df["method"] == method]
        lines.append(f"\n  {method}:")
        lines.append(f"    Mean bias:  {sub['bias_pct'].mean():+.2f}%")
        lines.append(f"    Max |bias|: {sub['bias_pct'].abs().max():.2f}%")
        lines.append(f"    Mean RMSE:  {sub['rmse'].mean():.3f}")
        lines.append(f"    Max RMSE:   {sub['rmse'].max():.3f}")

    # SNR=0 check (no between-group variation — all methods should be unbiased)
    lines.append("")
    lines.append("-" * 72)
    lines.append("SANITY CHECK: SNR=0 (no between-subgroup variation)")
    lines.append("-" * 72)
    snr0 = df[df["snr"] == 0.0]
    for method in ["Range-based", "GLM residual", "Naive MR-mixed"]:
        sub = snr0[snr0["method"] == method]
        lines.append(f"  {method}: mean bias = {sub['bias_pct'].mean():+.2f}%, "
                     f"max |bias| = {sub['bias_pct'].abs().max():.2f}%")

    # Precision degradation at high singleton proportions
    lines.append("")
    lines.append("-" * 72)
    lines.append("PRECISION DEGRADATION: Range-based std(σ̂) by singleton %")
    lines.append("-" * 72)
    rb = df[df["method"] == "Range-based"]
    for sp in SINGLETON_PROPORTIONS:
        sub = rb[rb["singleton_pct"] == int(sp * 100)]
        mean_std = sub["std_estimate"].mean()
        n_subgroups = int(K * (1 - sp))
        lines.append(f"  {int(sp*100):2d}% singletons → "
                     f"{n_subgroups:2d} subgroups with n≥2 → "
                     f"mean std(σ̂) = {mean_std:.3f}")

    # Naive MR bias by SNR
    lines.append("")
    lines.append("-" * 72)
    lines.append("NAIVE MR-MIXED: Bias contamination by SNR")
    lines.append("-" * 72)
    naive = df[df["method"] == "Naive MR-mixed"]
    for snr in SNR_VALUES:
        sub = naive[naive["snr"] == snr]
        lines.append(f"  SNR={snr:.1f}: mean bias = {sub['bias_pct'].mean():+.2f}%, "
                     f"max |bias| = {sub['bias_pct'].abs().max():.2f}%")

    # Effective subgroup counts
    lines.append("")
    lines.append("-" * 72)
    lines.append("EFFECTIVE SUBGROUP COUNTS (K=50)")
    lines.append("-" * 72)
    for sp in SINGLETON_PROPORTIONS:
        n_grouped = int(K * (1 - sp))
        lines.append(f"  {int(sp*100):2d}% singletons → {n_grouped} subgroups "
                     f"contributing to range-based estimate")

    lines.append("")
    lines.append(threshold_analysis(df))

    lines.append("")
    lines.append("=" * 72)
    lines.append("KEY FINDINGS")
    lines.append("=" * 72)
    lines.append("")

    # Determine findings from data
    rb_snr0 = df[(df["method"] == "Range-based") & (df["snr"] == 0.0)]
    glm_snr0 = df[(df["method"] == "GLM residual") & (df["snr"] == 0.0)]
    rb_unbiased = rb_snr0["bias_pct"].abs().max() < 5.0
    glm_unbiased = glm_snr0["bias_pct"].abs().max() < 5.0

    lines.append(f"1. Range-based estimator unbiased at SNR=0: "
                 f"{'YES' if rb_unbiased else 'NO'} "
                 f"(max |bias| = {rb_snr0['bias_pct'].abs().max():.2f}%)")
    lines.append(f"2. GLM residual estimator unbiased at SNR=0: "
                 f"{'YES' if glm_unbiased else 'NO'} "
                 f"(max |bias| = {glm_snr0['bias_pct'].abs().max():.2f}%)")

    naive_snr0 = df[(df["method"] == "Naive MR-mixed") & (df["snr"] == 0.0)]
    naive_snr2 = df[(df["method"] == "Naive MR-mixed") & (df["snr"] == 2.0)]
    lines.append(f"3. Naive MR-mixed bias at SNR=0: "
                 f"{naive_snr0['bias_pct'].mean():+.2f}% (should be ~0)")
    lines.append(f"   Naive MR-mixed bias at SNR=2: "
                 f"{naive_snr2['bias_pct'].mean():+.2f}% (should be >0, contaminated)")

    rb_90 = df[(df["method"] == "Range-based") & (df["singleton_pct"] == 90)]
    rb_10 = df[(df["method"] == "Range-based") & (df["singleton_pct"] == 10)]
    lines.append(f"4. Range-based precision at 90% singletons: "
                 f"mean std = {rb_90['std_estimate'].mean():.3f}")
    lines.append(f"   Range-based precision at 10% singletons: "
                 f"mean std = {rb_10['std_estimate'].mean():.3f}")
    lines.append(f"   Degradation ratio: "
                 f"{rb_90['std_estimate'].mean() / rb_10['std_estimate'].mean():.1f}x")

    lines.append("")
    lines.append("5. The degradation is a PRECISION problem, not a BIAS problem.")
    lines.append("   As singleton proportion increases, fewer subgroups contribute")
    lines.append("   to the range-based estimate, increasing its variability.")
    lines.append("   The estimate remains unbiased throughout.")

    return "\n".join(lines)


# ============================================================================
# Full Results Table
# ============================================================================

def format_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format the results DataFrame for CSV output."""
    col_order = [
        "singleton_pct", "size_dist", "snr", "method",
        "n_valid", "mean_estimate", "bias_pct", "rmse", "ratio", "std_estimate",
    ]
    out = df[col_order].copy()
    out = out.sort_values(["method", "snr", "size_dist", "singleton_pct"])
    return out.reset_index(drop=True)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("Monte Carlo Simulation: Range-Based Sigma Estimation")
    print("                Under Mixed Subgroup Sizes")
    print("=" * 72)
    print(f"\nParameters: μ={MU}, σ_within={SIGMA_WITHIN}, K={K}, reps={N_REPS}")
    print(f"Total scenarios: {len(SINGLETON_PROPORTIONS) * len(SIZE_DISTRIBUTIONS) * len(SNR_VALUES)}")
    print()

    t0 = time.time()
    df = run_all_scenarios()
    elapsed = time.time() - t0
    print(f"\nSimulation completed in {elapsed:.1f}s")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save results CSV
    results = format_results_table(df)
    csv_path = OUTPUT_DIR / "results.csv"
    results.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Results saved to {csv_path}")

    # Generate and save summary
    summary = generate_summary(df)
    summary_path = OUTPUT_DIR / "summary.txt"
    summary_path.write_text(summary)
    print(f"Summary saved to {summary_path}")
    print()
    print(summary)

    # Generate charts
    print("\nGenerating charts...")

    fig_heatmap = make_bias_heatmaps(df)
    fig_heatmap.write_html(OUTPUT_DIR / "bias_heatmaps.html")
    print(f"  → {OUTPUT_DIR / 'bias_heatmaps.html'}")

    fig_bias = make_bias_line_charts(df)
    fig_bias.write_html(OUTPUT_DIR / "bias_lines.html")
    print(f"  → {OUTPUT_DIR / 'bias_lines.html'}")

    fig_rmse = make_rmse_line_charts(df)
    fig_rmse.write_html(OUTPUT_DIR / "rmse_lines.html")
    print(f"  → {OUTPUT_DIR / 'rmse_lines.html'}")

    fig_prec = make_precision_chart(df)
    fig_prec.write_html(OUTPUT_DIR / "precision.html")
    print(f"  → {OUTPUT_DIR / 'precision.html'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
