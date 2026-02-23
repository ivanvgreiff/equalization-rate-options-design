"""Phase 4: Compute premium surfaces across parameter grids.

Usage:
    python scripts/run_premium_grid.py --data data/processed/bitmex_xbtusd.parquet

Outputs premium tables to reports/tables/.
"""

import argparse

import numpy as np
import pandas as pd

from ddx.data.io import load_processed
from ddx.backtest.rolling import rolling_payoffs
from ddx.payoffs.floor import vanilla_floor
from ddx.payoffs.stoploss import aggregate_stop_loss
from ddx.pricing.premium import full_premium
from ddx.utils.config import load_analysis_config, load_contracts_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    df = load_processed(args.data)
    cf = df["funding_cf"].values
    analysis = load_analysis_config()
    contracts = load_contracts_config()

    results = []

    for horizon in analysis["horizons"]:
        name = horizon["name"]
        window = horizon["intervals"]
        horizon_years = window * analysis["sampling"]["dt_years"]

        # Vanilla floor grid
        for d in contracts["products"]["vanilla_floor"]["parameters"]["deductible_d"]:
            payoffs = rolling_payoffs(cf, window, lambda w, d=d: vanilla_floor(w, deductible=d))
            prem = full_premium(
                payoffs,
                lam=analysis["premium"]["risk_load_lambda"],
                cost_of_capital=analysis["premium"]["cost_of_capital_annual"],
                horizon_years=horizon_years,
            )
            results.append({
                "horizon": name,
                "product": "vanilla_floor",
                "deductible": d,
                "mean_payoff": float(np.mean(payoffs)),
                **{f"premium_{k}": v for k, v in prem.items()},
            })

        # Aggregate stop-loss grid
        for D in contracts["products"]["aggregate_stop_loss"]["parameters"]["deductible_D"]:
            payoffs = rolling_payoffs(
                cf, window, lambda w, D=D: aggregate_stop_loss(w, deductible_D=D)
            )
            prem = full_premium(
                payoffs,
                lam=analysis["premium"]["risk_load_lambda"],
                cost_of_capital=analysis["premium"]["cost_of_capital_annual"],
                horizon_years=horizon_years,
            )
            results.append({
                "horizon": name,
                "product": "aggregate_stop_loss",
                "deductible": D,
                "mean_payoff": float(np.mean(payoffs)),
                **{f"premium_{k}": v for k, v in prem.items()},
            })

    results_df = pd.DataFrame(results)
    out_path = "reports/tables/premium_surface.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Saved premium grid to {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
