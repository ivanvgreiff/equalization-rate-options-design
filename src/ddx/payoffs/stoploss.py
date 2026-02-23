"""Aggregate Stop-Loss — Product 3.

Payoff = min(L, max(0, Lambda - D))
where Lambda = sum_i max(0, -f_i)  (total negative funding loss).
"""

from __future__ import annotations

import numpy as np


def aggregate_stop_loss(
    funding_cf: np.ndarray,
    deductible_D: float = 0.005,
    cap: float | None = None,
) -> float:
    """Product 3: reserve-layer stop-loss.

    Parameters
    ----------
    funding_cf    : per-interval CFs (buyer perspective)
    deductible_D  : aggregate deductible (fraction of notional)
    cap           : aggregate payout cap L
    """
    total_loss = float(np.sum(np.maximum(0.0, -funding_cf)))
    payoff = max(0.0, total_loss - deductible_D)
    if cap is not None:
        payoff = min(payoff, cap)
    return payoff
