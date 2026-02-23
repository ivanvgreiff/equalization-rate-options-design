"""Risk metrics: VaR, CVaR, drawdown, episode statistics."""

from __future__ import annotations

import numpy as np


def var(x: np.ndarray, alpha: float = 0.01) -> float:
    """Value-at-Risk at level alpha (left tail).

    Returns the alpha-quantile: worst outcome at confidence (1 - alpha).
    """
    return float(np.quantile(x, alpha))


def cvar(x: np.ndarray, alpha: float = 0.01) -> float:
    """Conditional VaR (Expected Shortfall) at level alpha (left tail)."""
    threshold = np.quantile(x, alpha)
    tail = x[x <= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(np.mean(tail))


def cvar_right(x: np.ndarray, alpha: float = 0.01) -> float:
    """Right-tail CVaR: expected value in the top alpha fraction.

    For insurance payoffs, this captures the seller's worst-case claims.
    """
    threshold = np.quantile(x, 1 - alpha)
    tail = x[x >= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(np.mean(tail))


def prob_loss(x: np.ndarray) -> float:
    """Probability that x < 0."""
    return float(np.mean(x < 0))


def max_drawdown(cumulative: np.ndarray) -> float:
    """Max drawdown of a cumulative cashflow series."""
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def negative_episode_stats(funding_cf: np.ndarray) -> dict:
    """Compute statistics on negative-funding episodes.

    Returns dict with: count, mean_duration, max_duration,
    mean_severity, total_loss.
    """
    is_neg = funding_cf < 0
    episodes: list[list[float]] = []
    current: list[float] = []

    for i, neg in enumerate(is_neg):
        if neg:
            current.append(float(funding_cf[i]))
        else:
            if current:
                episodes.append(current)
                current = []
    if current:
        episodes.append(current)

    if not episodes:
        return {
            "count": 0,
            "mean_duration": 0.0,
            "max_duration": 0,
            "mean_severity_per_interval": 0.0,
            "total_loss": 0.0,
        }

    durations = [len(e) for e in episodes]
    severities = [sum(abs(v) for v in e) / len(e) for e in episodes]
    total_loss = sum(abs(v) for e in episodes for v in e)

    return {
        "count": len(episodes),
        "mean_duration": float(np.mean(durations)),
        "max_duration": max(durations),
        "mean_severity_per_interval": float(np.mean(severities)),
        "total_loss": total_loss,
    }
