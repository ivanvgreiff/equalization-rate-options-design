"""Phase 5: Compute hedge-efficiency frontier — DDX options vs swap benchmark.

Usage:
    python scripts/run_frontier.py --data data/processed/bitmex_xbtusd.parquet

Outputs frontier plots and summary tables.
"""

import argparse

import numpy as np

from ddx.data.io import load_processed
from ddx.backtest.rolling import rolling_windows
from ddx.backtest.hedges import unhedged_cf, swap_hedge_cf, option_hedge_cf, estimate_swap_rate
from ddx.payoffs.floor import vanilla_floor
from ddx.payoffs.stoploss import aggregate_stop_loss
from ddx.payoffs.distress import distress_activated_floor
from ddx.risk.metrics import var, cvar, prob_loss
from ddx.pricing.premium import pure_premium
from ddx.backtest.rolling import rolling_payoffs
from ddx.utils.config import load_analysis_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--horizon", default="30d")
    args = parser.parse_args()

    df = load_processed(args.data)
    cf = df["funding_cf"].values
    config = load_analysis_config()

    horizon_cfg = next(h for h in config["horizons"] if h["name"] == args.horizon)
    window = horizon_cfg["intervals"]
    lookback = config["swap_benchmark"]["lookback_intervals"]

    # Compute payoff distributions for premium estimation
    floor_payoffs = rolling_payoffs(cf, window, lambda w: vanilla_floor(w))
    floor_prem = pure_premium(floor_payoffs)

    daf_payoffs = rolling_payoffs(
        cf, window, lambda w: distress_activated_floor(w, streak_m=3)
    )
    daf_prem = pure_premium(daf_payoffs)

    asl_payoffs = rolling_payoffs(
        cf, window, lambda w: aggregate_stop_loss(w, deductible_D=0.003)
    )
    asl_prem = pure_premium(asl_payoffs)

    # Compute net CF distributions for each strategy
    windows = rolling_windows(cf, window)
    n_windows = len(windows)

    strategies = {
        "Unhedged": [],
        "Swap": [],
        "Vanilla Floor": [],
        "Distress-Activated Floor": [],
        "Aggregate Stop-Loss": [],
    }

    for idx in range(n_windows):
        w = windows[idx]
        window_start = idx

        strategies["Unhedged"].append(unhedged_cf(w))

        k = estimate_swap_rate(cf, lookback, window_start)
        strategies["Swap"].append(swap_hedge_cf(w, k))

        strategies["Vanilla Floor"].append(
            option_hedge_cf(w, vanilla_floor, floor_prem)
        )
        strategies["Distress-Activated Floor"].append(
            option_hedge_cf(w, distress_activated_floor, daf_prem, streak_m=3)
        )
        strategies["Aggregate Stop-Loss"].append(
            option_hedge_cf(w, aggregate_stop_loss, asl_prem, deductible_D=0.003)
        )

    print(f"=== Hedge Frontier ({args.horizon}, {n_windows} windows) ===\n")
    print(f"{'Strategy':<30} {'Mean':>10} {'Std':>10} {'VaR1%':>10} {'CVaR1%':>10} {'P(loss)':>10}")
    print("-" * 82)

    for name, vals in strategies.items():
        v = np.array(vals)
        print(
            f"{name:<30} {np.mean(v):>10.6f} {np.std(v):>10.6f} "
            f"{var(v):>10.6f} {cvar(v):>10.6f} {prob_loss(v):>10.4f}"
        )


if __name__ == "__main__":
    main()
