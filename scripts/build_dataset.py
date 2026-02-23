"""Build processed dataset from raw BitMEX funding-rate dump.

Usage:
    python scripts/build_dataset.py --input data/raw/bitmex_funding.csv \
                                     --output data/processed/bitmex_xbtusd.parquet

You write the fetcher separately; this script just normalizes whatever CSV
you drop into data/raw/.
"""

import argparse
from pathlib import Path

import pandas as pd

from ddx.data.preprocess import normalize_bitmex
from ddx.data.io import save_processed


def main():
    parser = argparse.ArgumentParser(description="Raw BitMEX CSV -> processed parquet")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument(
        "--output",
        default="data/processed/bitmex_xbtusd.parquet",
        help="Output path",
    )
    args = parser.parse_args()

    print(f"Reading raw data from {args.input}")
    df_raw = pd.read_csv(args.input)
    print(f"  {len(df_raw)} rows")

    df = normalize_bitmex(df_raw)
    print(f"  Processed: {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")

    save_processed(df, args.output)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
