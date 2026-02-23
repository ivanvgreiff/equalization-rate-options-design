"""Phase 6: Stress-episode event studies.

Usage:
    python scripts/run_event_studies.py --data data/processed/bitmex_xbtusd.parquet

Uses events defined in configs/events.yaml.
Premium is estimated using only pre-event data (no lookahead).
"""

import argparse

import numpy as np
import pandas as pd

from ddx.data.io import load_processed
from ddx.payoffs.floor import vanilla_floor
from ddx.payoffs.distress import distress_activated_floor
from ddx.payoffs.stoploss import aggregate_stop_loss
from ddx.backtest.hedges import unhedged_cf, estimate_swap_rate
from ddx.backtest.rolling import rolling_payoffs
from ddx.pricing.premium import pure_premium
from ddx.utils.config import load_events_config, load_analysis_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    df = load_processed(args.data)
    df = df.set_index("timestamp").sort_index()
    events = load_events_config()["events"]
    config = load_analysis_config()

    for event in events:
        print(f"\n{'='*60}")
        print(f"Event: {event['name']}")
        print(f"  {event['start']} to {event['end']}")
        print(f"  {event['description']}")

        start = pd.Timestamp(event["start"])
        end = pd.Timestamp(event["end"])
        train_end = pd.Timestamp(event["premium_train_end"])

        event_data = df.loc[start:end]
        if len(event_data) == 0:
            print("  [No data available for this event window]")
            continue

        event_cf = event_data["funding_cf"].values

        # Pre-event data for premium estimation
        pre_event = df.loc[:train_end]
        if len(pre_event) < 90:
            print("  [Insufficient pre-event data for premium estimation]")
            continue

        pre_cf = pre_event["funding_cf"].values
        lookback = config["swap_benchmark"]["lookback_intervals"]
        window = len(event_cf)

        # Estimate premiums from pre-event rolling windows
        floor_payoffs = rolling_payoffs(pre_cf, window, lambda w: vanilla_floor(w))
        floor_prem = pure_premium(floor_payoffs)

        daf_payoffs = rolling_payoffs(
            pre_cf, window, lambda w: distress_activated_floor(w, streak_m=3)
        )
        daf_prem = pure_premium(daf_payoffs)

        asl_payoffs = rolling_payoffs(
            pre_cf, window, lambda w: aggregate_stop_loss(w, deductible_D=0.003)
        )
        asl_prem = pure_premium(asl_payoffs)

        # Compute realized outcomes
        realized_cf = unhedged_cf(event_cf)
        floor_payoff = vanilla_floor(event_cf)
        daf_payoff = distress_activated_floor(event_cf, streak_m=3)
        asl_payoff = aggregate_stop_loss(event_cf, deductible_D=0.003)
        swap_rate = estimate_swap_rate(pre_cf, lookback, len(pre_cf))

        print(f"\n  Intervals in window: {len(event_cf)}")
        print(f"  Unhedged CF:         {realized_cf:+.6f}")
        print(f"  Swap hedge CF:       {len(event_cf) * swap_rate:+.6f}")
        print(f"  Floor payoff:        {floor_payoff:+.6f}  (premium: {floor_prem:.6f})")
        print(f"  DAF payoff:          {daf_payoff:+.6f}  (premium: {daf_prem:.6f})")
        print(f"  ASL payoff:          {asl_payoff:+.6f}  (premium: {asl_prem:.6f})")
        print(f"  Net CF (floor):      {realized_cf + floor_payoff - floor_prem:+.6f}")
        print(f"  Net CF (DAF):        {realized_cf + daf_payoff - daf_prem:+.6f}")
        print(f"  Net CF (ASL):        {realized_cf + asl_payoff - asl_prem:+.6f}")


if __name__ == "__main__":
    main()
