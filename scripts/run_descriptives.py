"""Phase 3: Compute descriptive funding analytics and regime statistics.

Usage:
    python scripts/run_descriptives.py --data data/processed/bitmex_xbtusd.parquet

Outputs to reports/figures/ and reports/tables/.
"""

import argparse

import numpy as np

from ddx.data.io import load_processed
from ddx.risk.metrics import negative_episode_stats
from ddx.viz.plots import plot_funding_timeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    df = load_processed(args.data)
    cf = df["funding_cf"].values

    # Overall stats
    print("=== Funding Rate Descriptives ===")
    print(f"  N intervals: {len(cf)}")
    print(f"  Date range:  {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print(f"  Mean CF:     {np.mean(cf):.6f}")
    print(f"  Std CF:      {np.std(cf):.6f}")
    print(f"  Min CF:      {np.min(cf):.6f}")
    print(f"  Max CF:      {np.max(cf):.6f}")
    print(f"  % negative:  {100 * np.mean(cf < 0):.2f}%")

    # Negative episode analysis
    stats = negative_episode_stats(cf)
    print(f"\n=== Negative Funding Episodes ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Timeline plot
    fig = plot_funding_timeline(
        df["timestamp"].values,
        cf,
        title="BitMEX XBTUSD Funding Rate History",
        save_path="reports/figures/funding_timeline.png",
    )
    print("\nSaved: reports/figures/funding_timeline.png")


if __name__ == "__main__":
    main()
