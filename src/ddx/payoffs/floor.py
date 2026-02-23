"""Vanilla Funding Floor — Product 1.

Payoff = min(L, sum_i max(0, -f_i - d))

where f_i is per-interval CF (positive = buyer receives).
"""

from __future__ import annotations

import numpy as np


def vanilla_floor(
    funding_cf: np.ndarray,
    deductible: float = 0.0,
    cap: float | None = None,
) -> float:
    """Compute vanilla floor payoff over a single window.

    Parameters
    ----------
    funding_cf : array of per-interval CFs (buyer perspective)
    deductible : per-interval deductible d (>= 0)
    cap        : aggregate payout cap L (None = uncapped)

    Returns
    -------
    Payoff per $1 notional.
    """
    interval_payoffs = np.maximum(0.0, -funding_cf - deductible)
    total = float(np.sum(interval_payoffs))
    if cap is not None:
        total = min(total, cap)
    return total
