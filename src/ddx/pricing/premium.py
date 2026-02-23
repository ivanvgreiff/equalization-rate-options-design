"""Premium decomposition: pure premium + risk load + capital charge."""

from __future__ import annotations

import numpy as np

from ddx.risk.metrics import cvar, cvar_right


def pure_premium(payoffs: np.ndarray) -> float:
    """Actuarial pure premium = E[payoff]."""
    return float(np.mean(payoffs))


def cvar_loaded_premium(
    payoffs: np.ndarray,
    lam: float = 0.35,
    alpha: float = 0.01,
) -> float:
    """Premium = E[Pi] + lambda * (CVaR_right(Pi) - E[Pi]).

    Uses right-tail CVaR because the seller's risk is large claims.

    Parameters
    ----------
    payoffs : array of realized payoffs across windows/paths
    lam     : risk-load multiplier
    alpha   : CVaR tail level (right tail)
    """
    pp = pure_premium(payoffs)
    cv = cvar_right(payoffs, alpha)
    return pp + lam * (cv - pp)


def target_sharpe_premium(
    payoffs: np.ndarray,
    target_sharpe: float = 0.75,
) -> float:
    """Premium such that seller's Sharpe >= target.

    Profit = Premium - Payoff
    Sharpe = E[Profit] / Std[Profit]
    => Premium = E[Pi] + S* * Std[Pi]
    """
    pp = pure_premium(payoffs)
    std = float(np.std(payoffs, ddof=1)) if len(payoffs) > 1 else 0.0
    return pp + target_sharpe * std


def capital_charge(
    payoffs: np.ndarray,
    cost_of_capital: float = 0.12,
    horizon_years: float = 30 / 365,
    alpha: float = 0.01,
) -> float:
    """Capital charge = k * C * T, where C = right-tail CVaR (max claim exposure)."""
    cv = cvar_right(payoffs, alpha)
    collateral = max(cv, 0.0)
    return cost_of_capital * collateral * horizon_years


def full_premium(
    payoffs: np.ndarray,
    lam: float = 0.35,
    cost_of_capital: float = 0.12,
    horizon_years: float = 30 / 365,
    alpha: float = 0.01,
) -> dict:
    """Compute all premium components and total.

    Returns dict with: pure, risk_load, capital_charge, total.
    """
    pp = pure_premium(payoffs)
    cv = cvar_right(payoffs, alpha)
    rl = lam * (cv - pp)
    cc = capital_charge(payoffs, cost_of_capital, horizon_years, alpha)
    return {
        "pure": pp,
        "risk_load": rl,
        "capital_charge": cc,
        "total": pp + rl + cc,
    }
