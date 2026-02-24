"""Compute hedge-efficiency frontier — DDX options vs swap benchmark.

Usage:
    python scripts/run_frontier.py --data data/processed/bitmex_xbtusd.parquet
    python scripts/run_frontier.py --data ... --horizon 30d --premium-method full

Outputs frontier plots and summary tables to reports/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ddx.data.io import load_processed
from ddx.backtest.rolling import rolling_windows, rolling_payoffs
from ddx.backtest.hedges import (
    unhedged_cf, swap_hedge_cf, option_hedge_cf, estimate_swap_rate,
    option_hedge_loss, swap_hedge_loss,
)
from ddx.payoffs.floor import vanilla_floor
from ddx.payoffs.stoploss import aggregate_stop_loss
from ddx.payoffs.distress import distress_activated_floor, soft_duration_cover
from ddx.risk.metrics import var, cvar, prob_loss, total_loss
from ddx.pricing.premium import compute_premium
from ddx.viz.plots import plot_hedge_frontier
from ddx.utils.config import load_analysis_config, load_contracts_config


def _first_param(contracts, config_key, param_name):
    """Get the first value from a contract parameter list."""
    return contracts["products"][config_key]["parameters"][param_name][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--horizon", default="30d")
    parser.add_argument("--premium-method", default="full",
                        choices=["pure", "full", "target_sharpe"])
    parser.add_argument("--swap-method", default="mean",
                        choices=["mean", "median", "ewma"])
    args = parser.parse_args()

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)

    df = load_processed(args.data)
    cf = df["funding_cf"].values
    config = load_analysis_config()
    contracts = load_contracts_config()

    horizon_cfg = next(h for h in config["horizons"] if h["name"] == args.horizon)
    window = horizon_cfg["intervals"]
    horizon_years = window * config["sampling"]["dt_years"]
    lookback = config["swap_benchmark"]["lookback_intervals"]
    swap_halflife = config["swap_benchmark"].get("ewma_halflife_intervals")

    prem_kwargs = {
        "method": args.premium_method,
        "lam": config["premium"]["risk_load_lambda"],
        "cost_of_capital": config["premium"]["cost_of_capital_annual"],
        "horizon_years": horizon_years,
        "target_sharpe": config["premium"]["target_sharpe"],
        "alpha": config["risk_metrics"]["cvar_alpha"],
    }

    # --- Read baseline params from config (first value in each list) ---
    floor_d = _first_param(contracts, "vanilla_floor", "deductible_d")
    daf_b = _first_param(contracts, "distress_activated_floor", "threshold_b")
    daf_m = _first_param(contracts, "distress_activated_floor", "streak_m")
    asl_D = _first_param(contracts, "aggregate_stop_loss", "deductible_D")
    sdc_b = _first_param(contracts, "soft_duration_cover", "threshold_b")
    sdc_m = _first_param(contracts, "soft_duration_cover", "streak_m")
    sdc_s = _first_param(contracts, "soft_duration_cover", "ramp_s")

    # --- Compute payoff distributions for premium estimation ---
    products = {
        "Vanilla Floor": {
            "fn": vanilla_floor, "kwargs": {"deductible": floor_d},
        },
        "Distress-Activated Floor": {
            "fn": distress_activated_floor,
            "kwargs": {"threshold_b": daf_b, "streak_m": daf_m},
        },
        "Aggregate Stop-Loss": {
            "fn": aggregate_stop_loss, "kwargs": {"deductible_D": asl_D},
        },
        "Soft-Duration Cover": {
            "fn": soft_duration_cover,
            "kwargs": {"threshold_b": sdc_b, "streak_m": sdc_m, "ramp_s": sdc_s},
        },
    }

    premiums = {}
    for name, spec in products.items():
        payoffs = rolling_payoffs(
            cf, window, lambda w, s=spec: s["fn"](w, **s["kwargs"])
        )
        prem_info = compute_premium(payoffs, **prem_kwargs)
        premiums[name] = prem_info["premium"]

    # --- Compute per-window metrics ---
    windows = rolling_windows(cf, window)
    n_windows = len(windows)

    rows = []
    for idx in range(n_windows):
        w = windows[idx]
        window_start = idx

        # Unhedged
        u_cf = unhedged_cf(w)
        u_loss = total_loss(w)

        # Swap
        k = estimate_swap_rate(cf, lookback, window_start,
                               method=args.swap_method, halflife=swap_halflife)
        s_cf = swap_hedge_cf(w, k)
        s_loss = swap_hedge_loss(w, k)

        row = {
            "window_start": idx,
            "Unhedged_cf": u_cf, "Unhedged_loss": u_loss,
            "Swap_cf": s_cf, "Swap_loss": s_loss,
        }

        for name, spec in products.items():
            prem = premiums[name]
            net_cf = option_hedge_cf(w, spec["fn"], prem, **spec["kwargs"])
            net_loss = option_hedge_loss(w, spec["fn"], prem, **spec["kwargs"])
            short_name = name.replace(" ", "_").replace("-", "")
            row[f"{short_name}_cf"] = net_cf
            row[f"{short_name}_loss"] = net_loss

        rows.append(row)

    df_results = pd.DataFrame(rows)

    # --- Compute summary stats ---
    alpha = config["risk_metrics"]["cvar_alpha"]
    summary_rows = []

    strategy_cols = {
        "Unhedged": ("Unhedged_cf", "Unhedged_loss", 0.0),
        f"Swap ({args.swap_method})": ("Swap_cf", "Swap_loss", 0.0),
    }
    for name in products:
        short = name.replace(" ", "_").replace("-", "")
        strategy_cols[name] = (f"{short}_cf", f"{short}_loss", premiums[name])

    unhedged_cvar = cvar(df_results["Unhedged_cf"].values, alpha)
    unhedged_loss_cvar = cvar_right_of = float(
        np.mean(np.sort(df_results["Unhedged_loss"].values)
                [-max(1, int(len(df_results) * alpha)):])
    )

    for strat_name, (cf_col, loss_col, prem) in strategy_cols.items():
        v_cf = df_results[cf_col].values
        v_loss = df_results[loss_col].values

        cvar_cf = cvar(v_cf, alpha)
        delta_cvar = cvar_cf - unhedged_cvar
        sharpness = delta_cvar / prem if prem > 0 else float("nan")

        summary_rows.append({
            "strategy": strat_name,
            "premium": prem,
            "mean_cf": float(np.mean(v_cf)),
            "std_cf": float(np.std(v_cf)),
            "var_1pct_cf": var(v_cf, alpha),
            "cvar_1pct_cf": cvar_cf,
            "prob_loss": prob_loss(v_cf),
            "mean_loss": float(np.mean(v_loss)),
            "cvar_1pct_loss": float(np.quantile(v_loss, 1 - alpha)),
            "sharpness_cf": sharpness,
        })

    df_summary = pd.DataFrame(summary_rows)

    # --- Print ---
    print(f"\n=== Hedge Frontier ({args.horizon}, {n_windows} windows, "
          f"premium={args.premium_method}, swap={args.swap_method}) ===\n")
    print(df_summary.to_string(index=False, float_format="%.6f"))

    # --- Save ---
    out_table = f"reports/tables/frontier_{args.horizon}.csv"
    df_summary.to_csv(out_table, index=False)
    print(f"\nSaved: {out_table}")

    # --- Plot ---
    frontier_data = {}
    for _, row in df_summary.iterrows():
        frontier_data[row["strategy"]] = {
            "premium": row["premium"] * 10_000,
            "cvar_01": row["cvar_1pct_cf"] * 10_000,
        }
    fig = plot_hedge_frontier(
        frontier_data,
        title=f"Hedge Frontier — {args.horizon} ({args.premium_method} premium)",
        save_path=f"reports/figures/frontier_{args.horizon}.png",
    )
    print(f"Saved: reports/figures/frontier_{args.horizon}.png")


if __name__ == "__main__":
    main()
