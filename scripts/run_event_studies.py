"""Stress-episode event studies.

Usage:
    python scripts/run_event_studies.py --data data/processed/bitmex_xbtusd.parquet

Uses events defined in configs/events.yaml.
Premium is estimated using only pre-event data (no lookahead).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddx.data.io import load_processed
from ddx.payoffs.floor import vanilla_floor
from ddx.payoffs.distress import distress_activated_floor, soft_duration_cover
from ddx.payoffs.stoploss import aggregate_stop_loss
from ddx.backtest.hedges import unhedged_cf, estimate_swap_rate
from ddx.backtest.rolling import rolling_payoffs
from ddx.risk.metrics import total_loss
from ddx.pricing.premium import compute_premium
from ddx.utils.config import (
    load_events_config, load_analysis_config, load_contracts_config,
)


def _first(contracts, key, param):
    return contracts["products"][key]["parameters"][param][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--premium-method", default="full",
                        choices=["pure", "full", "target_sharpe"])
    parser.add_argument("--swap-method", default="mean",
                        choices=["mean", "median", "ewma"])
    args = parser.parse_args()

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    Path("reports/markdown").mkdir(parents=True, exist_ok=True)

    df = load_processed(args.data)
    df = df.set_index("timestamp").sort_index()
    events = load_events_config()["events"]
    config = load_analysis_config()
    contracts = load_contracts_config()

    lookback = config["swap_benchmark"]["lookback_intervals"]
    swap_halflife = config["swap_benchmark"].get("ewma_halflife_intervals")

    # Baseline params from config
    floor_d = _first(contracts, "vanilla_floor", "deductible_d")
    daf_b = _first(contracts, "distress_activated_floor", "threshold_b")
    daf_m = _first(contracts, "distress_activated_floor", "streak_m")
    asl_D = _first(contracts, "aggregate_stop_loss", "deductible_D")
    sdc_b = _first(contracts, "soft_duration_cover", "threshold_b")
    sdc_m = _first(contracts, "soft_duration_cover", "streak_m")
    sdc_s = _first(contracts, "soft_duration_cover", "ramp_s")

    products = {
        "Floor": (vanilla_floor, {"deductible": floor_d}),
        "DAF": (distress_activated_floor,
                {"threshold_b": daf_b, "streak_m": daf_m}),
        "ASL": (aggregate_stop_loss, {"deductible_D": asl_D}),
        "SDC": (soft_duration_cover,
                {"threshold_b": sdc_b, "streak_m": sdc_m, "ramp_s": sdc_s}),
    }

    all_event_rows = []

    for event in events:
        name = event["name"]
        slug = name.lower().replace(" ", "_").replace("/", "_")

        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        train_end = pd.Timestamp(event["premium_train_end"])

        event_data = df.loc[start:end]
        if len(event_data) == 0:
            print(f"[{name}] No data — skipping")
            continue

        event_cf = event_data["funding_cf"].values
        pre_event = df.loc[:train_end]
        if len(pre_event) < 90:
            print(f"[{name}] Insufficient pre-event data — skipping")
            continue

        pre_cf = pre_event["funding_cf"].values
        window = len(event_cf)
        horizon_years = window * config["sampling"]["dt_years"]

        prem_kwargs = {
            "method": args.premium_method,
            "lam": config["premium"]["risk_load_lambda"],
            "cost_of_capital": config["premium"]["cost_of_capital_annual"],
            "horizon_years": horizon_years,
            "target_sharpe": config["premium"]["target_sharpe"],
            "alpha": config["risk_metrics"]["cvar_alpha"],
        }

        # Estimate premiums from pre-event data
        premiums = {}
        for pname, (fn, kw) in products.items():
            payoffs = rolling_payoffs(pre_cf, window, lambda w, f=fn, k=kw: f(w, **k))
            prem_info = compute_premium(payoffs, **prem_kwargs)
            premiums[pname] = prem_info["premium"]

        # Realized outcomes
        realized_cf = unhedged_cf(event_cf)
        raw_loss = total_loss(event_cf)
        swap_rate = estimate_swap_rate(
            pre_cf, lookback, len(pre_cf),
            method=args.swap_method, halflife=swap_halflife,
        )

        event_row = {
            "event": name, "start": str(start), "end": str(end),
            "intervals": len(event_cf),
            "unhedged_cf": realized_cf, "unhedged_loss": raw_loss,
            "swap_cf": len(event_cf) * swap_rate,
        }

        for pname, (fn, kw) in products.items():
            payoff = fn(event_cf, **kw)
            prem = premiums[pname]
            net_cf = realized_cf + payoff - prem
            net_loss = max(0.0, raw_loss - payoff) + prem
            event_row[f"{pname}_payoff"] = payoff
            event_row[f"{pname}_premium"] = prem
            event_row[f"{pname}_net_cf"] = net_cf
            event_row[f"{pname}_net_loss"] = net_loss

        all_event_rows.append(event_row)

        # --- Print ---
        print(f"\n{'='*60}")
        print(f"Event: {name}  ({start.date()} to {end.date()}, "
              f"{len(event_cf)} intervals)")
        print(f"  Unhedged CF:   {realized_cf:+.6f}   Loss: {raw_loss:.6f}")
        print(f"  Swap CF:       {len(event_cf) * swap_rate:+.6f}")
        for pname in products:
            p = event_row[f"{pname}_payoff"]
            pr = event_row[f"{pname}_premium"]
            nc = event_row[f"{pname}_net_cf"]
            nl = event_row[f"{pname}_net_loss"]
            print(f"  {pname:5s}: payoff={p:+.6f}  prem={pr:.6f}  "
                  f"net_cf={nc:+.6f}  net_loss={nl:.6f}")

        # --- Timeline figure ---
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(event_cf)), event_cf,
               color=["red" if v < 0 else "steelblue" for v in event_cf],
               width=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Funding CF — {name}")
        ax.set_xlabel("Interval index")
        ax.set_ylabel("Per-interval CF")
        fig.tight_layout()
        fig.savefig(f"reports/figures/event_{slug}_timeline.png", dpi=150)
        plt.close(fig)

    # --- Save cross-event table ---
    if all_event_rows:
        df_events = pd.DataFrame(all_event_rows)
        df_events.to_csv("reports/tables/event_cross_comparison.csv", index=False)
        print(f"\nSaved: reports/tables/event_cross_comparison.csv")

        # --- Markdown summary ---
        md_lines = ["# Event Study Summary\n"]
        for row in all_event_rows:
            md_lines.append(f"## {row['event']}\n")
            md_lines.append(f"- **Period**: {row['start']} to {row['end']} "
                            f"({row['intervals']} intervals)")
            md_lines.append(f"- **Unhedged CF**: {row['unhedged_cf']:+.6f}")
            md_lines.append(f"- **Unhedged Loss (reserve draw)**: "
                            f"{row['unhedged_loss']:.6f}")
            md_lines.append(f"- **Swap CF**: {row['swap_cf']:+.6f}\n")
            md_lines.append("| Product | Payoff | Premium | Net CF | Net Loss |")
            md_lines.append("|---------|--------|---------|--------|----------|")
            for pname in products:
                p = row[f"{pname}_payoff"]
                pr = row[f"{pname}_premium"]
                nc = row[f"{pname}_net_cf"]
                nl = row[f"{pname}_net_loss"]
                md_lines.append(f"| {pname} | {p:+.6f} | {pr:.6f} | "
                                f"{nc:+.6f} | {nl:.6f} |")
            md_lines.append("")

        md_path = "reports/markdown/event_study_summary.md"
        Path(md_path).write_text("\n".join(md_lines))
        print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
