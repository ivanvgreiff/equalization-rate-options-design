"""Risk metrics: VaR, CVaR, drawdown, episode statistics, loss-only lens."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Basic probability / drawdown
# ---------------------------------------------------------------------------

def prob_loss(x: np.ndarray) -> float:
    """Probability that x < 0."""
    return float(np.mean(x < 0))


def max_drawdown(cumulative: np.ndarray) -> float:
    """Max drawdown of a cumulative cashflow series."""
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    return float(np.max(dd)) if len(dd) > 0 else 0.0


# ---------------------------------------------------------------------------
# Loss-only / reserve-draw lens
# ---------------------------------------------------------------------------

def total_loss(funding_cf: np.ndarray) -> float:
    """Reserve draw over a window: sum of max(0, -f_i).

    This ignores positive funding entirely — it measures only the cumulative
    negative cashflow outflow, which is what an Ethena-like actor's reserve
    fund must cover.
    """
    return float(np.sum(np.maximum(0.0, -funding_cf)))


def hedged_loss(
    funding_cf: np.ndarray,
    payoff_fn,
    premium: float = 0.0,
    **payoff_kwargs,
) -> float:
    """Reserve draw after an option hedge, accounting for premium.

    hedged_loss = max(0, unhedged_loss - payoff) + premium

    The payoff offsets part of the raw loss.  Premium is an additional cost
    that the buyer pays regardless of outcome, so it adds to the effective
    reserve draw.
    """
    raw_loss = total_loss(funding_cf)
    payoff = payoff_fn(funding_cf, **payoff_kwargs)
    return max(0.0, raw_loss - payoff) + premium


# ---------------------------------------------------------------------------
# Episode statistics
# ---------------------------------------------------------------------------

def extract_episodes(funding_cf: np.ndarray) -> list[dict]:
    """Extract individual negative-funding episodes with metadata.

    Returns a list of dicts, each with:
        start_idx  : index of the first negative interval
        duration   : number of consecutive negative intervals
        total_loss : sum of |f_i| over the episode
        mean_severity : average |f_i| per interval in the episode
    """
    is_neg = funding_cf < 0
    episodes: list[dict] = []
    start = None

    for i, neg in enumerate(is_neg):
        if neg:
            if start is None:
                start = i
        else:
            if start is not None:
                segment = funding_cf[start:i]
                episodes.append({
                    "start_idx": start,
                    "duration": len(segment),
                    "total_loss": float(np.sum(np.abs(segment))),
                    "mean_severity": float(np.mean(np.abs(segment))),
                })
                start = None
    if start is not None:
        segment = funding_cf[start:]
        episodes.append({
            "start_idx": start,
            "duration": len(segment),
            "total_loss": float(np.sum(np.abs(segment))),
            "mean_severity": float(np.mean(np.abs(segment))),
        })

    return episodes


def negative_episode_stats(funding_cf: np.ndarray) -> dict:
    """Compute statistics on negative-funding episodes.

    Returns dict with: count, mean_duration, max_duration,
    mean_severity_per_interval, total_loss.
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
    ep_total_loss = sum(abs(v) for e in episodes for v in e)

    return {
        "count": len(episodes),
        "mean_duration": float(np.mean(durations)),
        "max_duration": max(durations),
        "mean_severity_per_interval": float(np.mean(severities)),
        "total_loss": ep_total_loss,
    }
