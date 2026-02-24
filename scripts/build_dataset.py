"""Build processed dataset from a raw venue funding-rate dump.

Supports multiple venues via the --venue flag, which routes to the
appropriate normalizer in ddx.data.preprocess.

Usage:
    python scripts/build_dataset.py --venue bitmex
    python scripts/build_dataset.py --venue deribit
    python scripts/build_dataset.py --venue bitmex --input data/raw/bitmex_xbtusd_raw.csv \
                                     --output data/processed/bitmex_xbtusd.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from ddx.data.preprocess import normalize_venue
from ddx.data.io import save_processed

VENUE_DEFAULTS = {
    "bitmex": {
        "input": "data/raw/bitmex_xbtusd_raw.csv",
        "output": "data/processed/bitmex_xbtusd.parquet",
    },
    "deribit": {
        "input": "data/raw/deribit_btcperp_raw.csv",
        "output": "data/processed/deribit_btcperp.parquet",
    },
    "bybit": {
        "input": "data/raw/bybit_btcusd_raw.csv",
        "output": "data/processed/bybit_btcusd.parquet",
    },
    "binance": {
        "input": "data/raw/binance_btcusd_raw.csv",
        "output": "data/processed/binance_btcusd.parquet",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Raw venue CSV -> processed parquet")
    parser.add_argument(
        "--venue",
        required=True,
        choices=list(VENUE_DEFAULTS),
        help="Venue name (determines which normalizer to use)",
    )
    parser.add_argument("--input", default=None, help="Path to raw CSV (default: venue-specific)")
    parser.add_argument("--output", default=None, help="Output path (default: venue-specific)")
    args = parser.parse_args()

    defaults = VENUE_DEFAULTS[args.venue]
    input_path = args.input or defaults["input"]
    output_path = args.output or defaults["output"]

    print(f"Building dataset for venue: {args.venue}")
    print(f"  Reading raw data from {input_path}")
    df_raw = pd.read_csv(input_path)
    print(f"  {len(df_raw)} raw rows")

    df = normalize_venue(args.venue, df_raw)
    print(f"  Processed: {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")

    n_irreg = (~df["is_regular"]).sum() if "is_regular" in df.columns else 0
    print(f"  Irregular intervals: {n_irreg}")

    save_processed(df, output_path)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
