"""Parameter calibration for DDX products.

Computes empirically grounded parameters from historical funding data:
- Conditional loss quantiles (l_i | l_i > 0) for Floor deductible d
- Rolling Lambda quantiles per horizon for ASL deductible D
- DAF activation frequency analysis for (b, m) pairs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ddx.backtest.rolling import rolling_windows, rolling_windows_regular
from ddx.risk.metrics import total_loss
from ddx.payoffs.distress import _run_lengths


def conditional_loss_quantiles(
    funding_cf: np.ndarray,
    quantiles: list[float] | None = None,
) -> dict:
    """Compute quantiles of per-interval loss l_i, conditional on l_i > 0.

    This avoids the "p50=0 because most intervals are non-negative" problem
    by conditioning on intervals that actually produce a loss.

    Returns dict with:
        n_total, n_negative, frac_negative,
        and q_XX keys for each requested quantile.
    """
    if quantiles is None:
        quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    losses = np.maximum(0.0, -funding_cf)
    positive_losses = losses[losses > 0]

    result = {
        "n_total": len(funding_cf),
        "n_negative": len(positive_losses),
        "frac_negative": len(positive_losses) / len(funding_cf) if len(funding_cf) > 0 else 0.0,
    }

    if len(positive_losses) == 0:
        for q in quantiles:
            result[f"q{int(q * 100):02d}"] = 0.0
        return result

    for q in quantiles:
        result[f"q{int(q * 100):02d}"] = float(np.quantile(positive_losses, q))

    result["mean"] = float(np.mean(positive_losses))
    result["std"] = float(np.std(positive_losses))
    return result


def lambda_quantiles_per_horizon(
    funding_cf: np.ndarray,
    is_regular: np.ndarray | None,
    horizon_intervals: int,
    quantiles: list[float] | None = None,
) -> dict:
    """Compute quantiles of rolling-window aggregate loss Lambda.

    Lambda = sum(max(0, -f_i)) over each rolling window of size horizon_intervals.
    These are in "fraction of notional per window" units.

    Returns dict with:
        n_windows, horizon_intervals, mean_lambda, std_lambda,
        and q_XX keys for each quantile.
    """
    if quantiles is None:
        quantiles = [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    if is_regular is not None:
        wins, _ = rolling_windows_regular(funding_cf, is_regular, horizon_intervals)
    else:
        wins = rolling_windows(funding_cf, horizon_intervals)

    if len(wins) == 0:
        result = {"n_windows": 0, "horizon_intervals": horizon_intervals}
        for q in quantiles:
            result[f"q{int(q * 100):02d}"] = 0.0
        return result

    lambdas = np.array([total_loss(w) for w in wins])

    result = {
        "n_windows": len(lambdas),
        "horizon_intervals": horizon_intervals,
        "mean_lambda": float(np.mean(lambdas)),
        "std_lambda": float(np.std(lambdas)),
    }

    for q in quantiles:
        result[f"q{int(q * 100):02d}"] = float(np.quantile(lambdas, q))

    return result


def daf_activation_analysis(
    funding_cf: np.ndarray,
    is_regular: np.ndarray | None,
    horizon_intervals: int,
    threshold_b: float,
    streak_m: int,
) -> dict:
    """Analyze DAF activation frequency and characteristics.

    For each rolling window of the given horizon, computes:
    - Whether the DAF activates at all in that window
    - How many intervals are "activated" (run length >= m)
    - The DAF payoff (with d=b for strike continuity)

    Returns dict with activation statistics.
    """
    if is_regular is not None:
        wins, _ = rolling_windows_regular(funding_cf, is_regular, horizon_intervals)
    else:
        wins = rolling_windows(funding_cf, horizon_intervals)

    if len(wins) == 0:
        return {
            "threshold_b": threshold_b,
            "streak_m": streak_m,
            "horizon_intervals": horizon_intervals,
            "n_windows": 0,
            "frac_windows_activated": 0.0,
        }

    n_activated_windows = 0
    activated_interval_counts = []
    payoffs = []

    for w in wins:
        runs = _run_lengths(w, threshold_b)
        active_mask = runs >= streak_m
        n_active = int(np.sum(active_mask))
        activated_interval_counts.append(n_active)

        if n_active > 0:
            n_activated_windows += 1

        interval_payoffs = active_mask.astype(np.float64) * np.maximum(0.0, -w - threshold_b)
        payoffs.append(float(np.sum(interval_payoffs)))

    payoffs_arr = np.array(payoffs)
    act_counts = np.array(activated_interval_counts)

    return {
        "threshold_b": threshold_b,
        "streak_m": streak_m,
        "horizon_intervals": horizon_intervals,
        "n_windows": len(wins),
        "frac_windows_activated": n_activated_windows / len(wins),
        "mean_activated_intervals": float(np.mean(act_counts)),
        "mean_payoff": float(np.mean(payoffs_arr)),
        "median_payoff": float(np.median(payoffs_arr)),
        "std_payoff": float(np.std(payoffs_arr)),
        "mean_payoff_when_active": (
            float(np.mean(payoffs_arr[payoffs_arr > 0]))
            if np.any(payoffs_arr > 0) else 0.0
        ),
    }


def compute_asl_deductible(
    funding_cf: np.ndarray,
    is_regular: np.ndarray | None,
    horizon_intervals: int,
    quantile: float = 0.90,
) -> float:
    """Compute the ASL deductible D as a quantile of rolling Lambda.

    D = q_p(Lambda_horizon) where Lambda = sum(max(0, -f_i)) per window.
    Returns D in per-interval-sum units (fraction of notional per window).
    """
    lq = lambda_quantiles_per_horizon(
        funding_cf, is_regular, horizon_intervals,
        quantiles=[quantile],
    )
    return lq[f"q{int(quantile * 100):02d}"]


def freeze_baseline_parameters(
    funding_cf: np.ndarray,
    is_regular: np.ndarray | None,
    horizons: list[dict],
    config: dict | None = None,
) -> dict:
    """Compute all frozen baseline parameters for the pruned product set.

    Parameters
    ----------
    funding_cf : the primary venue's funding CF array
    is_regular : regularity mask (or None)
    horizons   : list of {"name": "30d", "intervals": 90, ...}
    config     : calibration config (optional, uses defaults if None)

    Returns
    -------
    Dict with all frozen parameters, organized by product and horizon.
    """
    if config is None:
        config = {}
    cal = config.get("calibration", {})
    asl_q_base = cal.get("asl_deductible_quantile_baseline", 0.90)
    asl_q_sens = cal.get("asl_deductible_quantile_sensitivity", 0.95)
    lambda_qs = cal.get("lambda_quantiles", [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])
    cond_qs = cal.get("conditional_loss_quantiles", [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    cond_loss = conditional_loss_quantiles(funding_cf, cond_qs)

    params = {
        "conditional_loss_quantiles": cond_loss,
        "floor": {
            "benchmark": {"deductible_d": 0.0, "cap_L": None},
            "realistic": {"deductible_d": 0.0001, "cap_L": None},
            "realistic_alt": {"deductible_d": 0.0003, "cap_L": None},
        },
        "daf": {
            "baseline": {
                "threshold_b": 0.0001, "streak_m": 3,
                "deductible_d": 0.0001, "cap_L": None,
            },
            "sensitivity": {
                "threshold_b": 0.0001, "streak_m": 2,
                "deductible_d": 0.0001, "cap_L": None,
            },
        },
        "swap": {
            "primary": {"method": "ewma", "lookback": 90, "halflife": 45},
            "secondary": {"method": "mean", "lookback": 90},
        },
        "horizons": {},
    }

    for h in horizons:
        hname = h["name"]
        n_int = h["intervals"]

        lq = lambda_quantiles_per_horizon(
            funding_cf, is_regular, n_int, lambda_qs,
        )

        D_baseline = lq[f"q{int(asl_q_base * 100):02d}"]
        D_sensitivity = lq[f"q{int(asl_q_sens * 100):02d}"]

        daf_base = daf_activation_analysis(
            funding_cf, is_regular, n_int,
            threshold_b=0.0001, streak_m=3,
        )
        daf_sens = daf_activation_analysis(
            funding_cf, is_regular, n_int,
            threshold_b=0.0001, streak_m=2,
        )

        params["horizons"][hname] = {
            "intervals": n_int,
            "lambda_quantiles": lq,
            "asl": {
                "baseline": {"deductible_D": D_baseline},
                "sensitivity": {"deductible_D": D_sensitivity},
            },
            "daf_activation": {
                "baseline_m3": daf_base,
                "sensitivity_m2": daf_sens,
            },
        }

    return params
