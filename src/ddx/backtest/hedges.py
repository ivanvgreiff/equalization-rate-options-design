"""Hedge strategy definitions for comparing DDX options vs linear hedges."""

from __future__ import annotations

import numpy as np


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


def estimate_swap_rate(
    funding_cf: np.ndarray,
    lookback: int,
    window_start: int,
) -> float:
    """Estimate the 'fair' swap fixed rate as trailing mean.

    Uses the `lookback` intervals ending just before `window_start`.
    """
    start = max(0, window_start - lookback)
    if start >= window_start:
        return 0.0
    return float(np.mean(funding_cf[start:window_start]))
