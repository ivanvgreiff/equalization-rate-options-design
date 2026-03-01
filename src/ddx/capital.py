"""Capital efficiency metrics for hedge evaluation.

Implements the formal definitions from the Master Implementation Plan §7.0:
  - Definition A: reserve reduction per unit premium  (Eff_A)
  - Definition B: total economic cost = premium + cost-of-capital on reserve
  - Swap margin proxy for fair capital comparison
"""

from __future__ import annotations

import numpy as np

from ddx.risk.metrics import cvar, var


def reserve_requirement(
    losses: np.ndarray,
    alpha: float = 0.01,
    method: str = "cvar",
) -> float:
    """Required reserve capital R_alpha from a distribution of losses.

    Parameters
    ----------
    losses : array of per-window loss values (higher = worse).
    alpha  : tail probability level.
    method : "cvar" for CVaR (Expected Shortfall) or "var" for VaR.

    Returns
    -------
    R_alpha : the reserve level such that the tail risk constraint is met.
              For losses where higher = worse, this uses right-tail quantiles.
    """
    if len(losses) == 0:
        return 0.0
    if method == "cvar":
        threshold = np.quantile(losses, 1 - alpha)
        tail = losses[losses >= threshold]
        return float(np.mean(tail)) if len(tail) > 0 else float(threshold)
    elif method == "var":
        return float(np.quantile(losses, 1 - alpha))
    else:
        raise ValueError(f"Unknown method: {method!r}")


def efficiency_A(
    R_unhedged: float,
    R_hedged: float,
    premium: float,
) -> float:
    """Reserve reduction per unit premium.

    Eff_A(H) = (R_alpha(unhedged) - R_alpha(H)) / Premium(H)

    Higher is better.  Returns 0 if premium <= 0.
    """
    if premium <= 0:
        return 0.0
    return (R_unhedged - R_hedged) / premium


def total_economic_cost(
    premium: float,
    R_hedged: float,
    k: float,
    horizon_days: int = 30,
) -> float:
    """Total economic cost = premium + capital charge on residual reserve.

    Cost(H) = Premium(H) + k * (horizon_days / 365) * R_alpha(H)

    Parameters
    ----------
    premium      : option premium (% of notional).
    R_hedged     : required reserve after hedging (% of notional).
    k            : annual cost-of-capital rate (e.g. 0.10 for 10%).
    horizon_days : hedge horizon in calendar days.
    """
    return premium + k * (horizon_days / 365) * R_hedged


def swap_margin_proxy(
    swap_net_cfs: np.ndarray,
    alpha: float = 0.01,
) -> float:
    """Capital required for a swap hedge (margin proxy).

    M_alpha = CVaR_alpha(max(0, -X_swap))

    The swap can produce negative net cashflow if the fixed rate overshoots
    realized funding.  The margin proxy captures the tail of those losses.
    """
    swap_losses = np.maximum(0.0, -swap_net_cfs)
    if np.all(swap_losses == 0):
        return 0.0
    return reserve_requirement(swap_losses, alpha=alpha, method="cvar")
