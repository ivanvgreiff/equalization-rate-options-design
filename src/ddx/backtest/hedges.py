"""Hedge strategy definitions for comparing DDX options vs linear hedges."""

from __future__ import annotations

import numpy as np

from ddx.risk.metrics import total_loss


# ---------------------------------------------------------------------------
# Unhedged / swap / option  — net-CF lens
# ---------------------------------------------------------------------------

def unhedged_cf(funding_cf: np.ndarray) -> float:
    """Unhedged total cashflow over a window."""
    return float(np.sum(funding_cf))


def swap_hedge_cf(
    funding_cf: np.ndarray,
    fixed_rate: float,
) -> float:
    """Swap hedge: lock funding at a fixed per-interval rate.

    Net CF = n * fixed_rate  (all variability removed).
    The 'cost' of the swap is implicit: you give up upside above fixed_rate.
    """
    return float(len(funding_cf) * fixed_rate)


def option_hedge_cf(
    funding_cf: np.ndarray,
    payoff_fn,
    premium: float,
    **payoff_kwargs,
) -> float:
    """Option hedge: underlying CF + option payoff - premium.

    Parameters
    ----------
    funding_cf   : per-interval CFs for the window
    payoff_fn    : DDX payoff function
    premium      : premium paid for the option
    payoff_kwargs: additional args to payoff_fn
    """
    underlying = float(np.sum(funding_cf))
    payoff = payoff_fn(funding_cf, **payoff_kwargs)
    return underlying + payoff - premium


# ---------------------------------------------------------------------------
# Loss-only lens helpers
# ---------------------------------------------------------------------------

def option_hedge_loss(
    funding_cf: np.ndarray,
    payoff_fn,
    premium: float,
    **payoff_kwargs,
) -> float:
    """Reserve draw (loss-only lens) after an option hedge.

    hedged_loss = max(0, raw_loss - payoff) + premium
    """
    raw = total_loss(funding_cf)
    payoff = payoff_fn(funding_cf, **payoff_kwargs)
    return max(0.0, raw - payoff) + premium


def swap_hedge_loss(
    funding_cf: np.ndarray,
    fixed_rate: float,
) -> float:
    """Reserve draw under a swap hedge.

    With a swap the buyer receives exactly n*k.  If k >= 0, the reserve draw
    is zero (no negative funding to cover).  If k < 0 the swap itself
    produces a loss of n*|k|.  The swap eliminates exposure to the actual
    funding series entirely.
    """
    n = len(funding_cf)
    net = n * fixed_rate
    return max(0.0, -net)


# ---------------------------------------------------------------------------
# Swap-rate estimators
# ---------------------------------------------------------------------------

def estimate_swap_rate_mean(
    funding_cf: np.ndarray,
    lookback: int,
    window_start: int,
) -> float:
    """Trailing arithmetic mean of funding CFs."""
    start = max(0, window_start - lookback)
    if start >= window_start:
        return 0.0
    return float(np.mean(funding_cf[start:window_start]))


def estimate_swap_rate_median(
    funding_cf: np.ndarray,
    lookback: int,
    window_start: int,
) -> float:
    """Trailing median — more robust to outliers."""
    start = max(0, window_start - lookback)
    if start >= window_start:
        return 0.0
    return float(np.median(funding_cf[start:window_start]))


def estimate_swap_rate_ewma(
    funding_cf: np.ndarray,
    lookback: int,
    window_start: int,
    halflife: int | None = None,
) -> float:
    """Exponentially weighted moving average — recency-weighted.

    halflife is in intervals; defaults to lookback // 2 if None.
    The decay factor per interval is  alpha = 1 - exp(-ln2 / halflife).
    """
    start = max(0, window_start - lookback)
    if start >= window_start:
        return 0.0
    segment = funding_cf[start:window_start]
    n = len(segment)
    if halflife is None:
        halflife = max(1, n // 2)
    decay = 1.0 - np.exp(-np.log(2) / halflife)
    # weights: most recent = highest weight
    weights = (1.0 - decay) ** np.arange(n - 1, -1, -1)
    return float(np.average(segment, weights=weights))


def estimate_swap_rate(
    funding_cf: np.ndarray,
    lookback: int,
    window_start: int,
    method: str = "mean",
    halflife: int | None = None,
) -> float:
    """Dispatch to the appropriate swap-rate estimator.

    method : "mean" | "median" | "ewma"
    """
    if method == "mean":
        return estimate_swap_rate_mean(funding_cf, lookback, window_start)
    elif method == "median":
        return estimate_swap_rate_median(funding_cf, lookback, window_start)
    elif method == "ewma":
        return estimate_swap_rate_ewma(
            funding_cf, lookback, window_start, halflife=halflife,
        )
    else:
        raise ValueError(f"Unknown swap method: {method!r}")
