"""Premium decomposition: pure premium + risk load + capital charge.

Also provides alternative risk-adjusted pricing functionals (Wang distortion,
Esscher transform) that smooth tail weighting across the distribution rather
than relying on a hard CVaR quantile cutoff.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

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


def wang_distortion_premium(
    payoffs: np.ndarray,
    theta: float = 0.5,
) -> float:
    """Wang transform premium: distorted expectation of the payoff distribution.

    Applies the distortion g(u) = Phi(Phi^{-1}(u) + theta) to the survival
    function.  For theta > 0 this inflates survival probabilities, giving
    more weight to larger payoffs — a risk-averse pricing operator.

    Premium = sum_i x_(i) * [g(S_(i-1)) - g(S_i)]  where S_i = (n-i)/n.

    Parameters
    ----------
    payoffs : 1-D array of realized payoffs.
    theta   : Distortion parameter (>= 0). theta=0 gives pure premium.
              Typical insurance calibration: theta in [0.3, 1.0].
    """
    n = len(payoffs)
    if n == 0:
        return 0.0
    sorted_p = np.sort(payoffs)
    surv = (n - np.arange(n + 1)) / n
    g_surv = np.empty(n + 1)
    g_surv[0] = 1.0
    g_surv[-1] = 0.0
    inner = surv[1:-1]
    g_surv[1:-1] = norm.cdf(norm.ppf(inner) + theta)
    weights = g_surv[:-1] - g_surv[1:]
    return float(np.dot(sorted_p, weights))


def esscher_premium(
    payoffs: np.ndarray,
    theta: float = 1.0,
) -> float:
    """Esscher premium: exponentially tilted expectation.

    Premium = E[X * exp(theta * X)] / E[exp(theta * X)]

    A coherent premium principle from actuarial science that exponentially
    up-weights large payoffs. Uses the entire distribution (no hard quantile
    cutoff), making it smoother than CVaR-based loading.

    Parameters
    ----------
    payoffs : 1-D array of realized payoffs.
    theta   : Tilt parameter (>= 0). theta=0 gives pure premium.
    """
    if len(payoffs) == 0:
        return 0.0
    if theta == 0.0:
        return float(np.mean(payoffs))
    shifted = payoffs - np.max(payoffs)
    exp_vals = np.exp(theta * shifted)
    return float(np.sum(payoffs * exp_vals) / np.sum(exp_vals))


def compute_premium(
    payoffs: np.ndarray,
    method: str = "full",
    lam: float = 0.35,
    cost_of_capital: float = 0.12,
    horizon_years: float = 30 / 365,
    target_sharpe: float = 0.75,
    alpha: float = 0.01,
    wang_theta: float = 0.5,
    esscher_theta: float = 1.0,
) -> dict:
    """Dispatcher: compute premium under the specified method.

    method : "pure" | "full" | "target_sharpe" | "wang" | "esscher" | "all"

    Returns a dict that always contains at least:
        {"method": str, "premium": float, ...components}

    When method="all", the dict contains keys for every method.
    """
    pp = pure_premium(payoffs)

    if method == "pure":
        return {"method": "pure", "premium": pp, "pure": pp}

    fp = full_premium(payoffs, lam, cost_of_capital, horizon_years, alpha)

    if method == "full":
        return {"method": "full", "premium": fp["total"], **fp}

    tsp = target_sharpe_premium(payoffs, target_sharpe)

    if method == "target_sharpe":
        return {"method": "target_sharpe", "premium": tsp, "pure": pp,
                "target_sharpe_value": tsp}

    wp = wang_distortion_premium(payoffs, wang_theta)

    if method == "wang":
        return {"method": "wang", "premium": wp, "pure": pp,
                "wang_theta": wang_theta}

    ep = esscher_premium(payoffs, esscher_theta)

    if method == "esscher":
        return {"method": "esscher", "premium": ep, "pure": pp,
                "esscher_theta": esscher_theta}

    # method == "all"
    return {
        "method": "all",
        "premium": fp["total"],
        "premium_pure": pp,
        "premium_full": fp["total"],
        "premium_target_sharpe": tsp,
        "premium_wang": wp,
        "premium_esscher": ep,
        **fp,
    }
