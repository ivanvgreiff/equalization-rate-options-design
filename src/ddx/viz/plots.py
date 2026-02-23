"""Plotting utilities for DDX analysis outputs."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_funding_timeline(
    timestamps,
    funding_cf: np.ndarray,
    title: str = "Funding Rate Timeline",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot funding CF over time with negative regions highlighted."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(timestamps, funding_cf, linewidth=0.5, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.fill_between(
        timestamps,
        funding_cf,
        0,
        where=funding_cf < 0,
        alpha=0.3,
        color="red",
        label="Negative funding",
    )
    ax.set_title(title)
    ax.set_ylabel("Per-interval CF")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_hedge_frontier(
    results: dict,
    x_key: str = "premium",
    y_key: str = "cvar_01",
    title: str = "Hedge Efficiency Frontier",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot tail-risk vs cost frontier for different hedge strategies.

    Parameters
    ----------
    results : dict mapping strategy name -> dict with x_key and y_key values
    """
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
