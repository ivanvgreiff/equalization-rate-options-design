"""Plotting utilities for DDX analysis outputs."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ddx.utils.units import INTERVALS_PER_YEAR, to_apr_pct, to_pct_notional


def plot_funding_timeline(
    timestamps,
    funding_cf: np.ndarray,
    title: str = "Funding Rate Timeline",
    save_path: str | Path | None = None,
    rolling_window: int | None = 90,
    show_apr: bool = True,
) -> plt.Figure:
    """Plot funding CF over time with negative regions highlighted."""
    y = to_apr_pct(funding_cf) if show_apr else funding_cf
    ylabel = "APR (%)" if show_apr else "Per-interval CF"

    fig, axes = plt.subplots(2 if rolling_window else 1, 1,
                             figsize=(14, 8 if rolling_window else 5),
                             sharex=True)
    if not rolling_window:
        axes = [axes]

    ax = axes[0]
    ax.plot(timestamps, y, linewidth=0.3, alpha=0.7, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.fill_between(timestamps, y, 0, where=y < 0, alpha=0.3, color="red",
                    label="Negative funding")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)

    if rolling_window and len(funding_cf) >= rolling_window:
        import pandas as pd
        roll = pd.Series(funding_cf).rolling(rolling_window, min_periods=rolling_window).mean()
        roll_y = to_apr_pct(roll) if show_apr else roll
        axes[1].plot(timestamps, roll_y, linewidth=1.0, color="darkblue")
        axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
        axes[1].set_ylabel(f"Rolling {rolling_window}-interval Mean ({ylabel})")
        axes[1].set_xlabel("Date")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_streak_distributions(
    streak_data: dict[float, np.ndarray],
    title_prefix: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Multi-panel histogram of streak lengths with geometric fit overlay.

    streak_data : {threshold_b: array_of_streak_lengths}
    """
    thresholds = sorted(streak_data.keys())
    n = len(thresholds)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, b in enumerate(thresholds):
        ax = axes[idx // cols][idx % cols]
        streaks = streak_data[b]
        if len(streaks) == 0:
            ax.set_title(f"b={b:.5f} APR={to_apr_pct(b):.2f}%\n(no streaks)")
            continue

        max_len = int(np.max(streaks))
        bins = np.arange(0.5, max_len + 1.5, 1)
        ax.hist(streaks, bins=bins, density=True, alpha=0.7, color="steelblue",
                edgecolor="white", label="Empirical")

        # Geometric fit: P(X=k) = (1-p)^{k-1} * p where p = 1/mean
        mean_len = np.mean(streaks)
        p_geom = 1.0 / mean_len if mean_len > 0 else 1.0
        k_range = np.arange(1, min(max_len + 1, 50))
        geom_pmf = (1 - p_geom) ** (k_range - 1) * p_geom
        ax.plot(k_range, geom_pmf, "r-o", markersize=3, linewidth=1.2,
                label=f"Geometric(p={p_geom:.3f})")

        b_apr = to_apr_pct(b)
        ax.set_title(f"b={b:.5f} (APR={b_apr:.1f}%)\nmean={mean_len:.1f}, max={max_len}")
        ax.set_xlabel("Streak length (intervals)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"{title_prefix}Streak Length Distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_regime_markers(
    timestamps,
    rolling_mean: np.ndarray,
    rolling_pct_neg: np.ndarray,
    rolling_p01: np.ndarray,
    title: str = "Rolling Regime Markers (30d)",
    event_dates: list | None = None,
    save_path: str | Path | None = None,
    show_apr: bool = True,
) -> plt.Figure:
    """Three-panel time series of rolling regime indicators."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    if show_apr:
        mean_display = to_apr_pct(rolling_mean)
        p01_display = to_apr_pct(rolling_p01)
        unit = "APR (%)"
    else:
        mean_display = rolling_mean
        p01_display = rolling_p01
        unit = "per-interval"

    axes[0].plot(timestamps, mean_display, linewidth=0.8, color="darkblue")
    axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_ylabel(f"Rolling Mean {unit}")
    axes[0].set_title(title)

    axes[1].plot(timestamps, rolling_pct_neg * 100, linewidth=0.8, color="darkred")
    axes[1].set_ylabel("% Negative Intervals")
    axes[1].set_ylim(0, 100)

    axes[2].plot(timestamps, p01_display, linewidth=0.8, color="purple")
    axes[2].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[2].set_ylabel(f"Rolling 1st Percentile {unit}")
    axes[2].set_xlabel("Date")

    if event_dates:
        for ax in axes:
            for dt, label in event_dates:
                ax.axvline(dt, color="gray", linewidth=0.7, linestyle=":", alpha=0.7)
                if ax == axes[0]:
                    ax.text(dt, ax.get_ylim()[1] * 0.95, label, fontsize=6,
                            rotation=90, va="top", ha="right", alpha=0.7)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_distribution_analysis(
    funding_cf: np.ndarray,
    title_prefix: str = "",
    save_path: str | Path | None = None,
    show_apr: bool = True,
) -> plt.Figure:
    """Three-panel: f_i histogram, l_i histogram, QQ plot."""
    from scipy import stats as sp_stats

    unit = "APR (%)" if show_apr else "per-interval"
    f_scaled = to_apr_pct(funding_cf) if show_apr else funding_cf
    l_i = to_apr_pct(np.maximum(0.0, -funding_cf)) if show_apr else np.maximum(0.0, -funding_cf)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(f_scaled, bins=150, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    median_apr = float(np.median(f_scaled))
    ax.annotate(f"Median = {median_apr:.2f}", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))
    ax.set_xlabel(f"Funding Rate ({unit})")
    ax.set_ylabel("Density")
    ax.set_title(f"{title_prefix}f_i Distribution")

    ax = axes[1]
    l_pos = l_i[l_i > 0]
    if len(l_pos) > 0:
        ax.hist(l_pos, bins=100, density=True, alpha=0.7, color="indianred", edgecolor="none")
    ax.set_xlabel(f"Per-Interval Loss ({unit})")
    ax.set_ylabel("Density")
    ax.set_title(f"{title_prefix}l_i Distribution (losses only)")

    ax = axes[2]
    sp_stats.probplot(f_scaled, dist="norm", plot=ax)
    ax.set_title(f"{title_prefix}Normal QQ Plot")
    ax.get_lines()[0].set_markersize(1)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_top_episodes(
    episodes: list[dict],
    timestamps,
    metric: str = "duration",
    n: int = 20,
    title: str = "Top Episodes",
    save_path: str | Path | None = None,
    show_apr: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of top-N episodes by a given metric."""
    import pandas as pd

    sorted_eps = sorted(episodes, key=lambda e: e[metric], reverse=True)[:n]
    sorted_eps.reverse()

    labels = []
    values = []
    for ep in sorted_eps:
        ts = pd.Timestamp(timestamps[ep["start_idx"]])
        date_str = ts.strftime("%Y-%m-%d")
        labels.append(f"{date_str} ({ep['duration']} int)")
        if metric == "total_loss":
            v = ep["total_loss"]
            if show_apr:
                v = to_pct_notional(v)
            values.append(v)
        elif metric == "mean_severity":
            v = ep["mean_severity"]
            if show_apr:
                v = to_apr_pct(v)
            values.append(v)
        elif metric == "duration":
            values.append(ep["duration"])
        else:
            values.append(ep[metric])

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.3)))
    colors = ["indianred" if v > np.median(values) else "steelblue" for v in values]
    ax.barh(range(len(values)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    if metric == "total_loss":
        unit = " (% of notional)" if show_apr else ""
        ax.set_xlabel(f"Total Loss{unit}")
    elif metric == "mean_severity":
        unit = " APR (%)" if show_apr else ""
        ax.set_xlabel(f"Mean Severity{unit}")
    elif metric == "duration":
        ax.set_xlabel("Duration (intervals)")
    else:
        ax.set_xlabel(metric)
    ax.set_title(title)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_premium_decomposition(
    param_values: list,
    decompositions: list[dict],
    param_name: str = "Parameter",
    title: str = "Premium Decomposition",
    save_path: str | Path | None = None,
    param_labels: list[str] | None = None,
) -> plt.Figure:
    """Stacked bar chart of premium decomposition across a parameter sweep.

    decompositions : list of dicts with keys 'pure', 'risk_load', 'capital_charge'.
    """
    pures = [to_pct_notional(d["pure"]) for d in decompositions]
    risk_loads = [to_pct_notional(d["risk_load"]) for d in decompositions]
    cap_charges = [to_pct_notional(d["capital_charge"]) for d in decompositions]

    x = np.arange(len(param_values))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(8, len(param_values) * 1.5), 5))
    ax.bar(x, pures, width, label="Pure premium", color="#2196F3")
    ax.bar(x, risk_loads, width, bottom=pures, label="Risk load", color="#FF9800")
    bottoms = [p + r for p, r in zip(pures, risk_loads)]
    ax.bar(x, cap_charges, width, bottom=bottoms, label="Capital charge", color="#4CAF50")

    labels = param_labels if param_labels else [str(v) for v in param_values]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Premium (% of notional)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_premium_curve_with_ci(
    param_values: list,
    premiums: list[float],
    ci_lower: list[float] | None = None,
    ci_upper: list[float] | None = None,
    param_name: str = "Parameter",
    title: str = "Premium Curve",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Line plot of premium vs parameter with optional CI band."""
    prems_pct = [to_pct_notional(p) for p in premiums]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_values, prems_pct, "o-", color="#2196F3", linewidth=2, markersize=6)

    if ci_lower is not None and ci_upper is not None:
        lo = [to_pct_notional(v) for v in ci_lower]
        hi = [to_pct_notional(v) for v in ci_upper]
        ax.fill_between(param_values, lo, hi, alpha=0.2, color="#2196F3", label="90% CI")
        ax.legend(fontsize=8)

    ax.set_xlabel(param_name)
    ax.set_ylabel("Total Premium (% of notional)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cross_product_comparison(
    products: list[str],
    premiums: list[float],
    cvar_improvements: list[float],
    sharpness_vals: list[float],
    title: str = "Cross-Product Comparison (30d)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing premium, CVaR improvement, and sharpness."""
    x = np.arange(len(products))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    prems_pct = [to_pct_notional(p) for p in premiums]
    cvar_pct = [to_pct_notional(c) for c in cvar_improvements]

    axes[0].bar(x - width / 2, prems_pct, width, label="Premium", color="#2196F3")
    axes[0].bar(x + width / 2, cvar_pct, width, label="CVaR improvement", color="#FF5722")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(products, fontsize=8, rotation=20, ha="right")
    axes[0].set_ylabel("% of notional")
    axes[0].set_title("Premium vs CVaR Improvement")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, sharpness_vals, 0.5, color="#9C27B0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(products, fontsize=8, rotation=20, ha="right")
    axes[1].set_ylabel("Sharpness (|CVaR impr.| / premium)")
    axes[1].set_title("Hedge Sharpness")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hedge_frontier(
    results: dict,
    x_key: str = "premium",
    y_key: str = "cvar_01",
    title: str = "Hedge Efficiency Frontier",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot tail-risk vs cost frontier for different hedge strategies."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, vals in results.items():
        ax.scatter(vals[x_key], vals[y_key], s=80, zorder=3)
        ax.annotate(name, (vals[x_key], vals[y_key]), fontsize=9, ha="left")
    ax.set_xlabel(f"{x_key} (bps)")
    ax.set_ylabel(f"{y_key} (bps)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
