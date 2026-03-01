"""Plotting functions for hedge-efficiency frontiers (Phase 7).

Separate from plots.py to maintain single-responsibility:
  - plots.py serves Phases 2–6.5 (descriptives, calibration, premium surfaces)
  - frontier_plots.py serves Phase 7+ (frontiers, walk-forward, nonstationarity)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGY_COLORS = {
    "Unhedged": "#888888",
    "Swap (EWMA)": "#1f77b4",
    "Swap (mean)": "#aec7e8",
    "Floor d=0.0001": "#2ca02c",
    "Floor d=0.0003": "#98df8a",
    "DAF m=3": "#d62728",
    "DAF m=2": "#ff9896",
    "ASL q90": "#9467bd",
    "ASL q95": "#c5b0d5",
}

STRATEGY_MARKERS = {
    "Unhedged": "s",
    "Swap (EWMA)": "^",
    "Swap (mean)": "v",
    "Floor d=0.0001": "o",
    "Floor d=0.0003": "D",
    "DAF m=3": "P",
    "DAF m=2": "X",
    "ASL q90": "*",
    "ASL q95": "h",
}


def plot_efficiency_frontier(
    frontier_df: pd.DataFrame,
    x_col: str = "cvar_01_loss",
    y_col: str = "premium",
    h_col: str = "h",
    strategy_col: str = "strategy",
    bands: dict | None = None,
    title: str = "Hedge-Efficiency Frontier (30d, Loss-Only Lens)",
    xlabel: str = "Risk: CVaR₁% of reserve draw (% notional)",
    ylabel: str = "Cost: Premium (% notional)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot cost-vs-risk frontier with optional tri-band uncertainty.

    Parameters
    ----------
    frontier_df : DataFrame with columns [strategy, h, premium, cvar_01_loss, ...].
    bands : Optional dict mapping strategy -> {
        'bootstrap': (lower, upper),
        'era': (lower, upper),
        'model': (lower, upper),
    } for the metric at h=1.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    strategies = frontier_df[strategy_col].unique()
    for strat in strategies:
        sdf = frontier_df[frontier_df[strategy_col] == strat].sort_values(h_col)
        color = STRATEGY_COLORS.get(strat, "#333333")
        marker = STRATEGY_MARKERS.get(strat, "o")
        ax.plot(
            sdf[x_col] * 100, sdf[y_col] * 100,
            marker=marker, label=strat, color=color,
            linewidth=1.5, markersize=6, alpha=0.85,
        )
        h1 = sdf[sdf[h_col] == 1.0]
        if len(h1) > 0:
            ax.annotate(
                "h=1", (h1[x_col].values[0] * 100, h1[y_col].values[0] * 100),
                fontsize=7, alpha=0.7,
            )

    if bands:
        for strat, band_dict in bands.items():
            color = STRATEGY_COLORS.get(strat, "#333333")
            for band_name, (lo, hi) in band_dict.items():
                alpha_val = {"bootstrap": 0.15, "era": 0.10, "model": 0.08}.get(band_name, 0.1)
                strat_h1 = frontier_df[
                    (frontier_df[strategy_col] == strat) & (frontier_df[h_col] == 1.0)
                ]
                if len(strat_h1) > 0:
                    y_center = strat_h1[y_col].values[0] * 100
                    x_center = strat_h1[x_col].values[0] * 100
                    ax.errorbar(
                        x_center, y_center,
                        xerr=[[x_center - lo[0] * 100], [hi[0] * 100 - x_center]],
                        yerr=[[y_center - lo[1] * 100], [hi[1] * 100 - y_center]],
                        fmt="none", color=color, alpha=0.4, capsize=3,
                    )

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_walkforward_timeseries(
    wf_df: pd.DataFrame,
    metric: str = "reserve_draw_hedged",
    strategies: list[str] | None = None,
    title: str = "Walk-Forward: Realized Reserve Draw",
    ylabel: str = "Reserve Draw (% notional)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Time series of walk-forward realized metrics."""
    fig, ax = plt.subplots(figsize=(14, 6))

    if strategies is None:
        strategies = wf_df["strategy"].unique().tolist()

    for strat in strategies:
        sdf = wf_df[wf_df["strategy"] == strat].sort_values("t_start")
        color = STRATEGY_COLORS.get(strat, "#333333")
        ax.plot(
            sdf["month_idx"], sdf[metric] * 100,
            label=strat, color=color, linewidth=1.2, alpha=0.85,
        )

    ax.set_xlabel("Walk-Forward Month", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_drivers(
    center_timestamps: np.ndarray | pd.DatetimeIndex,
    driver_df: pd.DataFrame,
    title: str = "Rolling 2-Year Nonstationarity Drivers",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """4-panel chart showing time variation in key hedge-value drivers.

    driver_df should have columns: frac_neg, daf_activation, lambda_q90, lambda_q95.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    panels = [
        ("frac_neg", "Fraction Negative", "%", 100),
        ("daf_activation", "DAF m=3 Activation Rate", "%", 100),
        ("lambda_q90", "Λ₃₀d q90", "% notional", 100),
        ("lambda_q95", "Λ₃₀d q95", "% notional", 100),
    ]

    for ax, (col, label, unit, mult) in zip(axes.flat, panels):
        if col in driver_df.columns:
            ax.plot(center_timestamps, driver_df[col] * mult, linewidth=1.2, color="#1f77b4")
            ax.set_ylabel(f"{label} ({unit})", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_title(label, fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
