"""Unit conversion utilities for funding-rate analysis.

Two fundamentally different quantities exist in the codebase:

1. **Per-interval rates** (f_i, l_i, d, b, k)
   - Represent a rate *per single 8-hour interval*
   - Annualize via: APR = rate * INTERVALS_PER_YEAR
   - Display as percentage: APR% = rate * INTERVALS_PER_YEAR * 100

2. **Per-window cumulative sums** (Lambda, episode total_loss, window payoffs)
   - Represent a *sum* of per-interval quantities over a window of n intervals
   - Already in "fraction of notional" units for the window
   - Display as: "% of notional per {horizon}" = sum * 100
   - If annualization is needed: "annualized % of notional" = sum * (INTERVALS_PER_YEAR / n) * 100

Misapplying per-interval annualization (multiply by 1095) to a cumulative sum
that already spans n intervals inflates the number by a factor of n.
"""

from __future__ import annotations

INTERVAL_HOURS: float = 8.0
INTERVALS_PER_YEAR: float = 365 * 24 / INTERVAL_HOURS  # 1095.0


def per_interval_to_apr(x: float) -> float:
    """Convert a per-interval rate to APR (decimal).

    Example: 0.0001 per 8h → 0.1095 APR (≈10.95%)
    """
    return x * INTERVALS_PER_YEAR


def per_interval_to_apr_pct(x: float) -> float:
    """Convert a per-interval rate to APR as a percentage.

    Example: 0.0001 per 8h → 10.95%
    """
    return x * INTERVALS_PER_YEAR * 100


def window_sum_to_pct_notional(x: float) -> float:
    """Convert a per-window cumulative sum to % of notional.

    Example: Lambda=0.005 over a 30d window → 0.5% of notional
    """
    return x * 100


def window_sum_to_annualized_pct(x: float, window_intervals: int) -> float:
    """Annualize a per-window cumulative sum and express as %.

    Divides by the window length to get a per-interval average,
    then annualizes that average.

    Example: Lambda=0.005 over 90 intervals → (0.005/90)*1095*100 = 6.08% APR
    """
    if window_intervals <= 0:
        return 0.0
    avg_per_interval = x / window_intervals
    return avg_per_interval * INTERVALS_PER_YEAR * 100


def window_intervals_for_days(days: int) -> int:
    """Number of 8h intervals in a given number of days."""
    return days * 24 // int(INTERVAL_HOURS)


# Shorthand aliases matching existing code style
to_apr = per_interval_to_apr
to_apr_pct = per_interval_to_apr_pct
to_pct_notional = window_sum_to_pct_notional
to_ann_pct = window_sum_to_annualized_pct
