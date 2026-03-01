"""Walk-forward out-of-sample evaluation engine for hedge strategies.

Implements the monthly trailing-window protocol from the Master Plan §7.6:
  For each evaluation month t:
    1. Take trailing W years of funding data up to t.
    2. Fit episode model on trailing window.
    3. Price each strategy using the trailing window.
    4. Apply hedge to the next horizon_intervals of realized data.
    5. Record outcomes.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ddx.backtest.hedges import estimate_swap_rate
from ddx.backtest.rolling import rolling_windows_regular
from ddx.capital import reserve_requirement
from ddx.models.cluster_semi_markov import (
    extract_episodes_and_clusters,
    fit_cluster_tail,
    simulate_semi_markov,
)
from ddx.pricing.premium import compute_premium
from ddx.risk.metrics import total_loss


def _price_option_strategy(
    payoff_fn: Callable,
    payoff_kwargs: dict,
    windows: np.ndarray,
    method: str = "full",
    **premium_kwargs,
) -> float:
    """Compute premium for an option strategy from rolling windows."""
    payoffs = np.array([payoff_fn(w, **payoff_kwargs) for w in windows])
    result = compute_premium(payoffs, method=method, **premium_kwargs)
    return result.get("premium", result.get("total", 0.0))


def walkforward_evaluate(
    funding_cf: np.ndarray,
    is_regular: np.ndarray,
    strategies: list[dict],
    horizon_intervals: int = 90,
    train_years: float = 3.0,
    step_intervals: int = 90,
    sim_n_paths: int = 50,
    sim_n_intervals: int = 10_000,
    g: int = 5,
    p_augment: float = 0.02,
    cap: float = 0.00375,
    alpha: float = 0.01,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Monthly walk-forward out-of-sample evaluation.

    Parameters
    ----------
    funding_cf     : Full Bybit funding CF array.
    is_regular     : Boolean array of interval regularity.
    strategies     : List of strategy dicts, each with keys:
                     'name', 'type' ('option'|'swap'|'unhedged'),
                     'payoff_fn' (for options), 'payoff_kwargs' (for options),
                     'swap_method' (for swaps), 'swap_lookback', 'swap_halflife'.
    horizon_intervals : Window size for evaluation (90 = 30d).
    train_years    : Trailing training window in years.
    step_intervals : Step size between evaluations (90 = monthly for 30d).
    sim_n_paths    : Simulation paths per walk-forward month (for model-based pricing).
    sim_n_intervals: Intervals per simulated path.
    g, p_augment, cap : Episode model hyperparameters.
    alpha          : Risk level for CVaR/VaR.
    rng            : Random number generator.

    Returns
    -------
    DataFrame with columns: month_idx, t_start, strategy, premium,
    payoff, reserve_draw_unhedged, reserve_draw_hedged, net_cf.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    intervals_per_year = int(365 * 24 / 8)
    train_len = int(train_years * intervals_per_year)
    n = len(funding_cf)

    min_start = train_len
    max_start = n - horizon_intervals

    if min_start >= max_start:
        return pd.DataFrame()

    eval_starts = list(range(min_start, max_start, step_intervals))

    records = []

    for month_idx, t in enumerate(eval_starts):
        train_cf = funding_cf[t - train_len : t]
        train_reg = is_regular[t - train_len : t]
        eval_cf = funding_cf[t : t + horizon_intervals]

        if len(eval_cf) < horizon_intervals:
            continue

        train_wins, _ = rolling_windows_regular(train_cf, train_reg, horizon_intervals)
        if len(train_wins) < 20:
            continue

        unhedged_loss = total_loss(eval_cf)
        unhedged_netcf = float(np.sum(eval_cf))

        ep_data = extract_episodes_and_clusters(train_cf, threshold_b=0.0001, gap_g=g)
        tail_fit = fit_cluster_tail(ep_data["clusters"], quantile_threshold=0.90)
        tail_params = tail_fit if tail_fit.get("fit_success", False) else None

        for strat in strategies:
            name = strat["name"]
            stype = strat["type"]

            if stype == "unhedged":
                records.append({
                    "month_idx": month_idx,
                    "t_start": t,
                    "strategy": name,
                    "premium": 0.0,
                    "payoff": 0.0,
                    "reserve_draw_unhedged": unhedged_loss,
                    "reserve_draw_hedged": unhedged_loss,
                    "net_cf": unhedged_netcf,
                })
                continue

            if stype == "swap":
                method = strat.get("swap_method", "ewma")
                lookback = strat.get("swap_lookback", 90)
                halflife = strat.get("swap_halflife", 45)
                fixed_rate = estimate_swap_rate(
                    train_cf, lookback, len(train_cf), method=method, halflife=halflife,
                )
                swap_netcf = float(len(eval_cf) * fixed_rate)
                swap_loss = max(0.0, -swap_netcf)
                records.append({
                    "month_idx": month_idx,
                    "t_start": t,
                    "strategy": name,
                    "premium": 0.0,
                    "payoff": 0.0,
                    "reserve_draw_unhedged": unhedged_loss,
                    "reserve_draw_hedged": swap_loss,
                    "net_cf": swap_netcf,
                })
                continue

            if stype == "option":
                payoff_fn = strat["payoff_fn"]
                payoff_kwargs = strat.get("payoff_kwargs", {})

                if "deductible_D" in payoff_kwargs and payoff_kwargs["deductible_D"] == "calibrate":
                    from ddx.calibration import compute_asl_deductible
                    payoff_kwargs = dict(payoff_kwargs)
                    payoff_kwargs["deductible_D"] = compute_asl_deductible(
                        train_cf, train_reg, horizon_intervals, quantile=strat.get("asl_quantile", 0.90),
                    )

                premium = _price_option_strategy(
                    payoff_fn, payoff_kwargs, train_wins,
                )
                payoff = payoff_fn(eval_cf, **payoff_kwargs)
                hedged_loss = max(0.0, unhedged_loss - payoff) + premium
                net_cf = unhedged_netcf + payoff - premium

                records.append({
                    "month_idx": month_idx,
                    "t_start": t,
                    "strategy": name,
                    "premium": premium,
                    "payoff": payoff,
                    "reserve_draw_unhedged": unhedged_loss,
                    "reserve_draw_hedged": hedged_loss,
                    "net_cf": net_cf,
                })

    return pd.DataFrame(records)
