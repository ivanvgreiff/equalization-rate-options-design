"""Compute premium surfaces across parameter grids for all products.

Usage:
    python scripts/run_premium_grid.py --data data/processed/bitmex_xbtusd.parquet
    python scripts/run_premium_grid.py --data ... --product floor
    python scripts/run_premium_grid.py --data ... --product all

Outputs per-product CSV files to reports/tables/.
"""

import argparse
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import pandas as pd

from ddx.data.io import load_processed
from ddx.backtest.rolling import rolling_payoffs
from ddx.payoffs.floor import vanilla_floor
from ddx.payoffs.distress import distress_activated_floor, soft_duration_cover
from ddx.payoffs.stoploss import aggregate_stop_loss
from ddx.pricing.premium import compute_premium
from ddx.utils.config import load_analysis_config, load_contracts_config


def _sweep_floor(cf, window, horizon_years, params, prem_kwargs):
    rows = []
    for d, cap in iterproduct(params["deductible_d"], params["cap_L"]):
        payoffs = rolling_payoffs(
            cf, window, lambda w, _d=d, _c=cap: vanilla_floor(w, deductible=_d, cap=_c)
        )
        prem = compute_premium(payoffs, horizon_years=horizon_years, **prem_kwargs)
        rows.append({"product": "floor", "deductible_d": d, "cap_L": cap,
                      "mean_payoff": float(np.mean(payoffs)), **prem})
    return rows


def _sweep_daf(cf, window, horizon_years, params, prem_kwargs):
    rows = []
    for b, m, d, cap in iterproduct(
        params["threshold_b"], params["streak_m"],
        params["deductible_d"], params["cap_L"],
    ):
        payoffs = rolling_payoffs(
            cf, window,
            lambda w, _b=b, _m=m, _d=d, _c=cap: distress_activated_floor(
                w, threshold_b=_b, streak_m=_m, deductible=_d, cap=_c,
            ),
        )
        prem = compute_premium(payoffs, horizon_years=horizon_years, **prem_kwargs)
        rows.append({"product": "daf", "threshold_b": b, "streak_m": m,
                      "deductible_d": d, "cap_L": cap,
                      "mean_payoff": float(np.mean(payoffs)), **prem})
    return rows


def _sweep_asl(cf, window, horizon_years, params, prem_kwargs):
    rows = []
    for D, cap in iterproduct(params["deductible_D"], params["cap_L"]):
        payoffs = rolling_payoffs(
            cf, window,
            lambda w, _D=D, _c=cap: aggregate_stop_loss(w, deductible_D=_D, cap=_c),
        )
        prem = compute_premium(payoffs, horizon_years=horizon_years, **prem_kwargs)
        rows.append({"product": "asl", "deductible_D": D, "cap_L": cap,
                      "mean_payoff": float(np.mean(payoffs)), **prem})
    return rows


def _sweep_sdc(cf, window, horizon_years, params, prem_kwargs):
    rows = []
    for b, m, s, d, cap in iterproduct(
        params["threshold_b"], params["streak_m"], params["ramp_s"],
        params["deductible_d"], params["cap_L"],
    ):
        payoffs = rolling_payoffs(
            cf, window,
            lambda w, _b=b, _m=m, _s=s, _d=d, _c=cap: soft_duration_cover(
                w, threshold_b=_b, streak_m=_m, ramp_s=_s, deductible=_d, cap=_c,
            ),
        )
        prem = compute_premium(payoffs, horizon_years=horizon_years, **prem_kwargs)
        rows.append({"product": "sdc", "threshold_b": b, "streak_m": m,
                      "ramp_s": s, "deductible_d": d, "cap_L": cap,
                      "mean_payoff": float(np.mean(payoffs)), **prem})
    return rows


SWEEPERS = {
    "floor": ("vanilla_floor", _sweep_floor),
    "daf": ("distress_activated_floor", _sweep_daf),
    "asl": ("aggregate_stop_loss", _sweep_asl),
    "sdc": ("soft_duration_cover", _sweep_sdc),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--product", default="all",
                        choices=["floor", "daf", "asl", "sdc", "all"])
    parser.add_argument("--premium-method", default="all",
                        choices=["pure", "full", "target_sharpe", "all"])
    args = parser.parse_args()

    Path("reports/tables").mkdir(parents=True, exist_ok=True)

    df = load_processed(args.data)
    cf = df["funding_cf"].values
    analysis = load_analysis_config()
    contracts = load_contracts_config()

    products_to_run = list(SWEEPERS.keys()) if args.product == "all" else [args.product]

    prem_kwargs = {
        "method": args.premium_method,
        "lam": analysis["premium"]["risk_load_lambda"],
        "cost_of_capital": analysis["premium"]["cost_of_capital_annual"],
        "target_sharpe": analysis["premium"]["target_sharpe"],
        "alpha": analysis["risk_metrics"]["cvar_alpha"],
    }

    for prod_key in products_to_run:
        config_key, sweep_fn = SWEEPERS[prod_key]
        params = contracts["products"][config_key]["parameters"]

        for horizon in analysis["horizons"]:
            h_name = horizon["name"]
            window = horizon["intervals"]
            horizon_years = window * analysis["sampling"]["dt_years"]

            print(f"Sweeping {prod_key} @ {h_name} ({window} intervals)...")
            rows = sweep_fn(cf, window, horizon_years, params, prem_kwargs)

            for r in rows:
                r["horizon"] = h_name

            out = pd.DataFrame(rows)
            out_path = f"reports/tables/premium_{prod_key}_{h_name}.csv"
            out.to_csv(out_path, index=False)
            print(f"  -> {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
